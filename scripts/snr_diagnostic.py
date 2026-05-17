"""
SNR diagnostic — compare three σ estimates for each of 149 BWH ROIs.

Three σ estimates on the *normalized* signal (S / S(0)):

1. σ_formula  — Stephan's legacy Gibbs formula:
                σ = 1 / (sqrt(v_count / 16) * 150)
                (from Paper3/code copy/.../models/old/gibbs.py:389,440)

2. σ_NUTS     — Posterior mean of σ from each .nc file
                (HalfCauchy(β=0.01) prior, no SNR centering)

3. σ_residual — Empirical: std of (s_norm - U @ R_MAP_ridge).
                Direct estimate of the noise on the ROI-mean signal.
                Should match σ_NUTS if NUTS inference is faithful.

4. σ_langkilde — Per-voxel SNR per Langkilde 2018 Eq.9 from biexp residuals,
                then σ_roi = 1 / (snr_voxel * sqrt(v_count)).

Outputs:
  results/biomarkers/snr_comparison.csv
  results/biomarkers/snr_formula_vs_nuts.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, "src")
from spectra_estimation_dmri.biomarkers.recompute import (
    B_VALUES_MS,
    B_VALUES_S_MM2,
    DIFFUSIVITIES,
    build_design_matrix,
    compute_map_spectrum,
    compute_spectra_id,
    load_dataset,
)
import arviz as az


NC_DIR = "results/inference_bwh_backup"
OUTPUT_DIR = "results/biomarkers"


def stephan_sigma(voxel_count: int) -> float:
    """σ = 1 / (sqrt(v_count/16) * 150) on normalized signal."""
    snr = np.sqrt(voxel_count / 16.0) * 150.0
    return 1.0 / snr


def map_residual_sigma(signal: np.ndarray, U: np.ndarray) -> float:
    """std of residual after ridge NNLS MAP fit (on normalized signal)."""
    S0 = signal[0] if signal[0] > 0 else 1.0
    s_norm = signal / S0
    R = compute_map_spectrum(signal, U)
    resid = s_norm - U @ R
    # dof correction: 15 obs, ~3-4 effective parameters
    dof = max(len(s_norm) - 4, 1)
    return float(np.sqrt(np.sum(resid**2) / dof))


def biexp(b, S0, f, D_slow, D_fast):
    return S0 * (f * np.exp(-b * D_slow) + (1 - f) * np.exp(-b * D_fast))


def langkilde_sigma(signal: np.ndarray, voxel_count: int) -> float:
    """Per-voxel SNR via Langkilde Eq.9 from biexp residuals, scaled to ROI."""
    S0_init = signal[0] if signal[0] > 0 else 1.0
    s_norm = signal / S0_init
    try:
        popt, _ = curve_fit(
            biexp,
            B_VALUES_MS,
            s_norm,
            p0=[1.0, 0.3, 0.3, 1.5],
            bounds=([0.5, 0.0, 0.0, 0.0], [1.5, 1.0, 2.0, 10.0]),
            maxfev=5000,
        )
        fit = biexp(B_VALUES_MS, *popt)
        resid = s_norm - fit
        n_p = 4
        chi2_n = np.sum(resid**2) / max(len(s_norm) - n_p - 1, 1)
        if chi2_n <= 0:
            return np.nan
        # Eq.9: SNR_voxel = S0 * sqrt(2 / (pi * chi2_n)). On normalized signal S0=1.
        snr_voxel = 1.0 * np.sqrt(2.0 / (np.pi * chi2_n))
        snr_roi = snr_voxel * np.sqrt(voxel_count)
        return 1.0 / snr_roi
    except Exception:
        return np.nan


def load_nuts_sigma(roi: dict) -> tuple:
    """Return (sigma_mean, sigma_std) from this ROI's .nc file."""
    sid = compute_spectra_id(roi["signal"], roi["b_values"], roi["snr"])
    nc_path = os.path.join(NC_DIR, f"{sid}.nc")
    if not os.path.exists(nc_path):
        return (np.nan, np.nan)
    idata = az.from_netcdf(nc_path)
    sigma = idata.posterior["sigma"].values.flatten()
    return (float(sigma.mean()), float(sigma.std()))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rois = load_dataset()
    U = build_design_matrix()

    rows = []
    for i, roi in enumerate(rois):
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(rois)}]")
        s_form = stephan_sigma(roi["voxel_count"])
        s_resid = map_residual_sigma(roi["signal"], U)
        s_lang = langkilde_sigma(roi["signal"], roi["voxel_count"])
        s_nuts_mean, s_nuts_std = load_nuts_sigma(roi)

        rows.append({
            "roi_id": roi["roi_id"],
            "patient": roi["patient"],
            "zone": roi["region"],
            "is_tumor": roi["is_tumor"],
            "voxel_count": roi["voxel_count"],
            "sigma_formula": s_form,
            "sigma_nuts_mean": s_nuts_mean,
            "sigma_nuts_std": s_nuts_std,
            "sigma_residual": s_resid,
            "sigma_langkilde": s_lang,
            "snr_formula": 1.0 / s_form if s_form > 0 else np.nan,
            "snr_nuts": 1.0 / s_nuts_mean if s_nuts_mean > 0 else np.nan,
            "snr_residual": 1.0 / s_resid if s_resid > 0 else np.nan,
            "snr_langkilde": 1.0 / s_lang if s_lang > 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "snr_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Summary
    print("\n=== σ comparison (normalized signal scale) ===")
    for col in ["sigma_formula", "sigma_nuts_mean", "sigma_residual", "sigma_langkilde"]:
        print(f"  {col:20s}  median={df[col].median():.5f}  "
              f"q25={df[col].quantile(0.25):.5f}  q75={df[col].quantile(0.75):.5f}")
    print("\n=== SNR (= 1/σ) ===")
    for col in ["snr_formula", "snr_nuts", "snr_residual", "snr_langkilde"]:
        print(f"  {col:18s}  median={df[col].median():.0f}  "
              f"q25={df[col].quantile(0.25):.0f}  q75={df[col].quantile(0.75):.0f}")

    # Correlations
    for ref, oth in [("sigma_nuts_mean", "sigma_formula"),
                     ("sigma_nuts_mean", "sigma_residual"),
                     ("sigma_nuts_mean", "sigma_langkilde"),
                     ("sigma_residual", "sigma_formula"),
                     ("sigma_residual", "sigma_langkilde")]:
        sub = df[[ref, oth]].dropna()
        r = sub[ref].corr(sub[oth])
        print(f"  Pearson r({ref:>20s}, {oth:>20s}) = {r:.3f}  (n={len(sub)})")

    # Scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    pairs = [
        ("sigma_formula",   "sigma_nuts_mean",  "Formula vs NUTS"),
        ("sigma_residual",  "sigma_nuts_mean",  "MAP-residual vs NUTS"),
        ("sigma_langkilde", "sigma_nuts_mean",  "Langkilde Eq.9 vs NUTS"),
        ("sigma_formula",   "sigma_residual",   "Formula vs MAP-residual"),
    ]
    for ax, (xc, yc, title) in zip(axes.flat, pairs):
        sub = df[[xc, yc, "is_tumor", "zone"]].dropna()
        for (zone, tumor), grp in sub.groupby(["zone", "is_tumor"]):
            label = f"{zone.upper()} {'tumor' if tumor else 'normal'}"
            ax.scatter(grp[xc], grp[yc], s=18, alpha=0.7, label=label)
        lim = max(sub[xc].max(), sub[yc].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="y=x")
        ax.set_xlabel(xc); ax.set_ylabel(yc); ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        r = sub[xc].corr(sub[yc])
        ax.text(0.98, 0.02, f"r = {r:.3f}, n={len(sub)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    fig.suptitle("σ estimate comparison across 149 BWH ROIs (normalized signal)", y=1.00)
    fig.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "snr_formula_vs_nuts.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
