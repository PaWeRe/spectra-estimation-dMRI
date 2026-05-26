"""
Re-fit MAP on the actual BWH cohort at a range of λ values.

The simulation says λ ≈ 1e-3 is the sweet spot for log-normal-shaped
spectra (the closest analog to real prostate tissue). The manuscript's
"MAP D=3.0 in PZ-normal = 0.24 vs NUTS 0.48" finding was computed at
λ = 0.1. Question: does the MAP-NUTS gap on D=3.0 collapse when MAP is
re-fitted at the tuned λ?

Outputs:
  results/biomarkers/map_lambda_bwh.csv (per-ROI MAP fractions per λ)
  results/biomarkers/map_lambda_bwh_summary.csv (per-zone × per-tumor means)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectra_estimation_dmri.biomarkers.recompute import (
    DIFFUSIVITIES, build_design_matrix, load_dataset,
)


OUTPUT_DIR = "results/biomarkers"
LAMBDAS = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]


def fit_map(y_norm: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    n_d = U.shape[1]
    A = U.T @ U + lam * np.eye(n_d)
    R = np.linalg.solve(A, U.T @ y_norm)
    R = np.maximum(R, 0.0)
    s = R.sum()
    return R / s if s > 0 else R


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    U = build_design_matrix()
    rois = load_dataset()
    print(f"=== Re-fitting MAP on {len(rois)} BWH ROIs at λ ∈ {LAMBDAS} ===")

    rows = []
    for roi in rois:
        s = roi["signal"]
        if s[0] <= 0:
            continue
        y = s / s[0]
        for lam in LAMBDAS:
            R = fit_map(y, U, lam)
            row = {"roi_id": roi["roi_id"],
                   "zone": roi["region"],
                   "is_tumor": roi["is_tumor"],
                   "ggg": roi["ggg"],
                   "lambda": lam}
            for i, d in enumerate(DIFFUSIVITIES):
                row[f"D_{d:.2f}"] = R[i]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "map_lambda_bwh.csv"), index=False)
    print(f"  Wrote {OUTPUT_DIR}/map_lambda_bwh.csv ({len(df)} rows)")

    # Per-zone × per-tumor summaries at key bins
    print("\n=== MAP fractions by λ (zone × tumour × D bin) ===")
    print("  NUTS posterior means for reference (from features.csv):")
    features = pd.read_csv("results/biomarkers/features.csv")
    for zone in ["pz", "tz"]:
        for is_tumor in [False, True]:
            sub_f = features[(features["zone"] == zone) & (features["is_tumor"] == is_tumor)]
            nuts_025 = sub_f["nuts_D_0.25"].mean()
            nuts_300 = sub_f["nuts_D_3.00"].mean()
            map_025_orig = sub_f["map_D_0.25"].mean()
            map_300_orig = sub_f["map_D_3.00"].mean()
            tlabel = "tumor " if is_tumor else "normal"
            print(f"\n  {zone.upper()} {tlabel}: features.csv MAP@λ=0.1: "
                  f"D=0.25={map_025_orig:.3f}, D=3.0={map_300_orig:.3f}  | "
                  f"NUTS: D=0.25={nuts_025:.3f}, D=3.0={nuts_300:.3f}")
            for lam in LAMBDAS:
                sub = df[(df["zone"] == zone) & (df["is_tumor"] == is_tumor)
                         & (df["lambda"] == lam)]
                d025 = sub["D_0.25"].mean()
                d300 = sub["D_3.00"].mean()
                d050 = sub["D_0.50"].mean()
                d075 = sub["D_0.75"].mean()
                d100 = sub["D_1.00"].mean()
                print(f"    λ={lam:>7.4f}  D=0.25={d025:.3f}  D=0.50={d050:.3f}  "
                      f"D=0.75={d075:.3f}  D=1.00={d100:.3f}  D=3.00={d300:.3f}")

    # Save summary
    summary_rows = []
    for zone in ["pz", "tz"]:
        for is_tumor in [False, True]:
            sub_f = features[(features["zone"] == zone) & (features["is_tumor"] == is_tumor)]
            row_summary = {"zone": zone, "is_tumor": is_tumor,
                           "n_rois": len(sub_f),
                           "NUTS_mean_D_0.25": sub_f["nuts_D_0.25"].mean(),
                           "NUTS_mean_D_3.00": sub_f["nuts_D_3.00"].mean()}
            for lam in LAMBDAS:
                sub = df[(df["zone"] == zone) & (df["is_tumor"] == is_tumor)
                         & (df["lambda"] == lam)]
                row_summary[f"MAP_mean_D_0.25_lam{lam}"] = sub["D_0.25"].mean()
                row_summary[f"MAP_mean_D_3.00_lam{lam}"] = sub["D_3.00"].mean()
            summary_rows.append(row_summary)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, "map_lambda_bwh_summary.csv"), index=False)
    print(f"\n  Wrote {OUTPUT_DIR}/map_lambda_bwh_summary.csv")

    # Manuscript-claim spot-check: "MAP D=3.0 in PZ-normal = 0.24" — does this
    # change at tuned λ?
    print("\n=== Manuscript claim check: MAP @ D=3.0 in PZ-normal vs NUTS ===")
    sub_pz_normal = features[(features["zone"] == "pz") & (~features["is_tumor"])]
    nuts_pz_normal_d3 = sub_pz_normal["nuts_D_3.00"].median()
    print(f"  NUTS PZ-normal D=3.0 median: {nuts_pz_normal_d3:.3f}")
    for lam in LAMBDAS:
        sub = df[(df["zone"] == "pz") & (~df["is_tumor"]) & (df["lambda"] == lam)]
        d3_median = sub["D_3.00"].median()
        print(f"  MAP @ λ={lam:>7.4f}: D=3.0 median = {d3_median:.3f}  "
              f"({100 * (nuts_pz_normal_d3 - d3_median) / nuts_pz_normal_d3:+.1f}% deviation from NUTS)")


if __name__ == "__main__":
    main()
