"""
Wider R-prior NUTS check — answer to meeting-prep Q6 / F8.

The intermediate bins (D = 0.50, 0.75, 1.00) have high posterior CV (> 0.8)
in the BWH cohort. Open question: is this because (a) the data don't
constrain those compartments at this b-grid + SNR (genuine ill-posedness),
or (b) the half-normal prior with σ_R = 1/√0.1 ≈ 3.16 is dragging mass
toward zero (prior shrinkage)?

Test: refit a small set of representative ROIs under several R-prior
widths σ_R ∈ {3.16, 10, 30, 100} (the first is the manuscript baseline;
the last is effectively non-informative). If intermediate-bin posterior
means stay close to current values under wider priors → data limit. If
they shift substantially upward → prior shrinkage is part of the story.

NUTS reduced settings are used for speed (800 draws, 400 tune, 2 chains)
matching `scripts/simulation_study.py`. Results align well with the full
4-chain × 2000-draw setting we use for the main cohort.

Outputs:
  results/simulation/wider_prior_check.csv
  results/simulation/wider_prior_check_summary.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectra_estimation_dmri.biomarkers.recompute import (
    B_VALUES_MS, DIFFUSIVITIES, build_design_matrix, load_dataset,
)


OUTPUT_DIR = "results/simulation"
SEED = 20260524

# Representative ROIs to refit. Mix of zone × tumour × extreme cases.
# (Identifiers match `results/biomarkers/features.csv`.)
TARGET_ROIS = [
    "new02_pz_tumor",    # PZ tumor, GGG=3 (4+3), moderate intermediate mass
    "new03_pz_tumor",    # PZ tumor, GGG=5 (4+3+5), highest grade
    "new01_pz_normal",   # PZ normal, high D=3.0 fraction
    "new02_pz_normal",   # PZ normal, large intermediate mass
    "new01_tz_tumor",    # TZ tumor
    "new01_tz_normal",   # TZ normal, very high D=3.0 fraction
    "new03_tz_normal",   # TZ normal, large intermediate mass
]

# σ_R values to compare. 3.16 is the manuscript baseline (= 1/√0.1).
SIGMA_R_VALUES = [3.16, 10.0, 30.0, 100.0]

# NUTS settings (reduced for runtime)
NUTS_DRAWS = 800
NUTS_TUNE = 400
NUTS_CHAINS = 2
NUTS_TARGET = 0.9


def fit_nuts(y_norm: np.ndarray, U: np.ndarray, sigma_R: float,
             seed: int) -> dict:
    """NUTS with explicit σ_R override."""
    import pymc as pm
    n_d = U.shape[1]
    with pm.Model():
        R = pm.HalfNormal("R", sigma=sigma_R, shape=n_d)
        sigma = pm.HalfCauchy("sigma", beta=0.01)
        mu = pm.math.dot(U, R)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_norm)
        idata = pm.sample(
            draws=NUTS_DRAWS, tune=NUTS_TUNE, chains=NUTS_CHAINS,
            target_accept=NUTS_TARGET, progressbar=False,
            random_seed=seed, return_inferencedata=True,
        )
    R_samples = idata.posterior["R"].values.reshape(-1, n_d)
    row_sums = R_samples.sum(axis=1, keepdims=True)
    R_norm = R_samples / np.maximum(row_sums, 1e-10)
    sigma_post = idata.posterior["sigma"].values.flatten()
    return {
        "R_mean": R_norm.mean(axis=0),
        "R_std":  R_norm.std(axis=0),
        "sigma_mean": float(sigma_post.mean()),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    U = build_design_matrix()
    rois = load_dataset()

    # Filter to target ROIs
    target_set = set(TARGET_ROIS)
    selected = [r for r in rois if r["roi_id"] in target_set]
    missing = target_set - {r["roi_id"] for r in selected}
    if missing:
        print(f"[WARN] Could not find ROIs: {sorted(missing)}")
    print(f"[wider_prior_check] {len(selected)} ROIs × {len(SIGMA_R_VALUES)} priors "
          f"= {len(selected) * len(SIGMA_R_VALUES)} NUTS runs")

    rows = []
    seed_base = SEED
    for r_idx, roi in enumerate(selected):
        s_norm = roi["signal"] / roi["signal"][0]  # divide by S(b=0)
        for prior_idx, sigma_R in enumerate(SIGMA_R_VALUES):
            seed = seed_base + r_idx * 100 + prior_idx
            t0 = time.time()
            print(f"  [{r_idx+1}/{len(selected)}] {roi['roi_id']}  σ_R={sigma_R}  "
                  f"snr={roi['snr']:.0f}", end="", flush=True)
            try:
                res = fit_nuts(s_norm, U, sigma_R=sigma_R, seed=seed)
                for i, d in enumerate(DIFFUSIVITIES):
                    rows.append({
                        "roi_id": roi["roi_id"],
                        "zone": roi["region"],
                        "is_tumor": roi["is_tumor"],
                        "ggg": roi["ggg"],
                        "sigma_R": sigma_R,
                        "bin": i,
                        "D": d,
                        "R_mean": res["R_mean"][i],
                        "R_std":  res["R_std"][i],
                        "CV": res["R_std"][i] / (res["R_mean"][i] + 1e-10),
                        "sigma_post": res["sigma_mean"],
                    })
                print(f"  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  FAILED: {e}")

    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "wider_prior_check.csv")
    df.to_csv(out, index=False)
    print(f"\n[wider_prior_check] Wrote {out} ({len(df)} rows)")

    # Summary: per (ROI, σ_R), show R_mean and CV at intermediate bins
    print("\n=== Intermediate bin posterior means by prior width ===")
    print(f"{'ROI':<20s} {'σ_R':>6s}  " +
          "  ".join(f"D={d:>4.2f}" for d in DIFFUSIVITIES))
    for roi_id in [r["roi_id"] for r in selected]:
        for sigma_R in SIGMA_R_VALUES:
            sub = df[(df["roi_id"] == roi_id) & (df["sigma_R"] == sigma_R)]
            if len(sub) == 0:
                continue
            means = sub.sort_values("bin")["R_mean"].values
            print(f"{roi_id:<20s} {sigma_R:>6.2f}  " +
                  "  ".join(f"{m:>5.3f}" for m in means))
        print()

    print("\n=== Intermediate bin CV by prior width ===")
    print(f"{'ROI':<20s} {'σ_R':>6s}  " +
          "  ".join(f"D={d:>4.2f}" for d in DIFFUSIVITIES))
    for roi_id in [r["roi_id"] for r in selected]:
        for sigma_R in SIGMA_R_VALUES:
            sub = df[(df["roi_id"] == roi_id) & (df["sigma_R"] == sigma_R)]
            if len(sub) == 0:
                continue
            cvs = sub.sort_values("bin")["CV"].values
            print(f"{roi_id:<20s} {sigma_R:>6.2f}  " +
                  "  ".join(f"{c:>5.2f}" for c in cvs))
        print()

    # Top-line numerical summary: how much do intermediate bins move?
    intermediate_bins = [1, 2, 3]  # D = 0.50, 0.75, 1.00
    summary_rows = []
    for roi_id in [r["roi_id"] for r in selected]:
        baseline = df[(df["roi_id"] == roi_id)
                      & (df["sigma_R"] == SIGMA_R_VALUES[0])
                      & (df["bin"].isin(intermediate_bins))]
        for sigma_R in SIGMA_R_VALUES[1:]:
            wider = df[(df["roi_id"] == roi_id)
                       & (df["sigma_R"] == sigma_R)
                       & (df["bin"].isin(intermediate_bins))]
            if len(wider) == 0 or len(baseline) == 0:
                continue
            mean_baseline = baseline.sort_values("bin")["R_mean"].values
            mean_wider = wider.sort_values("bin")["R_mean"].values
            ratio = (mean_wider + 1e-10) / (mean_baseline + 1e-10)
            summary_rows.append({
                "roi_id": roi_id,
                "sigma_R_wider": sigma_R,
                "intermediate_R_mean_baseline": mean_baseline.mean(),
                "intermediate_R_mean_wider": mean_wider.mean(),
                "ratio_wider_over_baseline": ratio.mean(),
            })

    summary_df = pd.DataFrame(summary_rows)
    out2 = os.path.join(OUTPUT_DIR, "wider_prior_check_summary.csv")
    summary_df.to_csv(out2, index=False)
    print(f"[wider_prior_check] Wrote {out2}")

    print("\n=== TL;DR: ratio of (wider-prior mean) / (baseline mean) at intermediate bins ===")
    print(f"{'ROI':<20s} {'σ_R':>6s}  {'baseline mean':>14s}  {'wider mean':>11s}  {'ratio':>6s}")
    for _, row in summary_df.iterrows():
        print(f"{row['roi_id']:<20s} {row['sigma_R_wider']:>6.2f}  "
              f"{row['intermediate_R_mean_baseline']:>14.4f}  "
              f"{row['intermediate_R_mean_wider']:>11.4f}  "
              f"{row['ratio_wider_over_baseline']:>6.2f}")


if __name__ == "__main__":
    main()
