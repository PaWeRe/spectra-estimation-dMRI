"""
Fixed-σ NUTS refit on 5 representative ROIs.

For each ROI, run two NUTS fits:
 (A) Free σ — HalfCauchy(β=0.01)  [baseline, matches the .nc files]
 (B) Fixed σ at σ_formula = 1/(sqrt(v_count/16)·150)  [Stephan's]
 (C) Fixed σ at σ_residual = std of MAP-ridge residuals

Compare per-bin posterior mean and std → identify whether σ pinning
materially changes the recovered spectrum.

Output: results/biomarkers/fixed_sigma_refit.csv  (per-ROI × per-bin × per-mode)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectra_estimation_dmri.biomarkers.recompute import (
    B_VALUES_MS,
    DIFFUSIVITIES,
    build_design_matrix,
    compute_map_spectrum,
)


OUTPUT_DIR = "results/biomarkers"
PICKS_CSV  = os.path.join(OUTPUT_DIR, "snr_refit_picks.csv")
SIGNAL_JSON = "src/spectra_estimation_dmri/data/bwh/signal_decays.json"


def load_signal_for_roi(roi_id: str) -> np.ndarray:
    """Look up raw signal_values for given roi_id (patient_zone_class)."""
    parts = roi_id.split("_")
    patient = parts[0]
    region  = parts[1]  # pz | tz
    cls     = parts[2]  # tumor | normal
    target_anat = f"{cls}_{region}"
    with open(SIGNAL_JSON) as f:
        data = json.load(f)
    for roi_name, roi in data[patient].items():
        if target_anat in roi["anatomical_region"]:
            return np.array(roi["signal_values"]), roi["v_count"]
    raise KeyError(roi_id)


def normalize(R):
    s = R.sum()
    return R / s if s > 0 else R


def nuts_fit(y_norm, U, lam=0.1, sigma_fixed=None,
             draws=1500, tune=600, chains=2, target_accept=0.95, seed=42):
    """Run NUTS with either free σ (HalfCauchy) or fixed σ."""
    import pymc as pm
    n_d = U.shape[1]
    sigma_R = 1.0 / np.sqrt(lam)
    with pm.Model():
        R = pm.HalfNormal("R", sigma=sigma_R, shape=n_d)
        if sigma_fixed is None:
            sigma = pm.HalfCauchy("sigma", beta=0.01)
        else:
            sigma = float(sigma_fixed)
        mu = pm.math.dot(U, R)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_norm)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, progressbar=False,
                          return_inferencedata=True, random_seed=seed)
    R_samps = idata.posterior["R"].values.reshape(-1, n_d)
    rs = R_samps.sum(axis=1, keepdims=True)
    R_norm = R_samps / np.maximum(rs, 1e-10)
    sigma_post = idata.posterior["sigma"].values.flatten() if sigma_fixed is None else None
    return {
        "R_mean": R_norm.mean(axis=0),
        "R_std":  R_norm.std(axis=0),
        "R_q05":  np.quantile(R_norm, 0.05, axis=0),
        "R_q95":  np.quantile(R_norm, 0.95, axis=0),
        "sigma_mean": float(sigma_post.mean()) if sigma_post is not None else sigma_fixed,
        "sigma_std":  float(sigma_post.std())  if sigma_post is not None else 0.0,
    }


def main():
    picks = pd.read_csv(PICKS_CSV)
    U = build_design_matrix()
    rows = []
    for _, p in picks.iterrows():
        roi_id = p["roi_id"]
        signal, v_count = load_signal_for_roi(roi_id)
        S0 = signal[0]
        y_norm = signal / S0
        sigma_form = float(p["sigma_formula"])
        sigma_resid = float(p["sigma_residual"])
        print(f"\n=== {roi_id}  v={v_count}  σ_form={sigma_form:.5f}  σ_resid={sigma_resid:.5f} ===")
        modes = [
            ("free",          None),
            ("fixed_formula", sigma_form),
            ("fixed_residual", sigma_resid),
        ]
        for mode, sigma_fixed in modes:
            print(f"  Fitting {mode}...")
            res = nuts_fit(y_norm, U, lam=0.1, sigma_fixed=sigma_fixed)
            for i, d in enumerate(DIFFUSIVITIES):
                rows.append({
                    "roi_id":  roi_id,
                    "mode":    mode,
                    "bin":     i,
                    "D":       d,
                    "R_mean":  res["R_mean"][i],
                    "R_std":   res["R_std"][i],
                    "R_q05":   res["R_q05"][i],
                    "R_q95":   res["R_q95"][i],
                    "sigma_mean": res["sigma_mean"],
                    "sigma_std":  res["sigma_std"],
                })

    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "fixed_sigma_refit.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # Quick summary: per-bin CV (std/mean) by mode, averaged across ROIs
    df["cv"] = df["R_std"] / np.maximum(df["R_mean"], 1e-6)
    print("\nMean per-bin posterior CV (std/mean) across 5 ROIs:")
    piv = df.groupby(["mode", "D"])["cv"].mean().unstack("D")
    print(piv.to_string(float_format="{:.2f}".format))

    print("\nMean per-bin posterior std across 5 ROIs:")
    piv2 = df.groupby(["mode", "D"])["R_std"].mean().unstack("D")
    print(piv2.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
