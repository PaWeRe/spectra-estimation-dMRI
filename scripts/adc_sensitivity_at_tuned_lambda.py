"""
Re-derive the "ADC sensitivity ≈ inverted LR coefficient" vector correlation
at tuned MAP λ.

The published Fig 4 finding (r ≈ −0.98) was computed using MAP at λ = 0.1.
The 2026-05-24 λ sweep showed that λ = 0.1 was substantially suboptimal:
at tuned λ ≈ 1e-3 to 1e-4 the MAP spectrum becomes concentrated at outer
bins (matching NUTS), and intermediate bins drop near zero.

Question: does the elegant correlation survive at tuned λ, or was it a
direct consequence of the smearing?

For each λ:
  1. Use MAP fractions from results/biomarkers/map_lambda_bwh.csv as features.
  2. Fit logistic regression on (PZ, tumor-vs-normal) and (TZ, tumor-vs-normal).
  3. Compute the ADC sensitivity vector ∂ADC/∂R_j at the average-tumor and
     average-normal spectrum (using the SAME λ's MAP estimate).
  4. Report Pearson and Spearman correlation between LR coefs (8) and
     sensitivity vectors (8).

Also reports NUTS reference for comparison.

Output: results/biomarkers/adc_sens_vs_lr_tuned_lambda.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from spectra_estimation_dmri.biomarkers.recompute import (
    DIFFUSIVITIES, compute_sensitivity,
)

D_COLS = [f"D_{d:.2f}" for d in DIFFUSIVITIES]


def correlate(spectrum_features: pd.DataFrame, is_tumor: pd.Series,
              feat_cols: list[str], C: float = 1.0) -> dict:
    """Fit LR; compute sensitivity at avg_tumor / avg_normal; return correlations."""
    X = spectrum_features[feat_cols].values
    y = is_tumor.astype(int).values
    if len(np.unique(y)) < 2:
        return None

    avg_tumor = X[y == 1].mean(axis=0)
    avg_normal = X[y == 0].mean(axis=0)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(X_s, y)
    coefs = clf.coef_[0]

    out = {"lr_coefs": coefs.tolist()}
    for label, spectrum in [("tumor", avg_tumor), ("normal", avg_normal)]:
        sens = compute_sensitivity(spectrum, b_max=1.0)
        r_p, p_p = stats.pearsonr(sens, coefs)
        r_s, p_s = stats.spearmanr(sens, coefs)
        out[f"r_pearson_{label}"] = float(r_p)
        out[f"r_spearman_{label}"] = float(r_s)
        out[f"sens_{label}"] = sens.tolist()
    return out


def main():
    # Load BWH MAP fractions at the swept λ values
    map_lambda = pd.read_csv("results/biomarkers/map_lambda_bwh.csv")
    features = pd.read_csv("results/biomarkers/features.csv")
    # Join zone/is_tumor (already in both — use map_lambda's directly)
    print(f"=== MAP at swept λ — vector correlations on PZ and TZ ===")
    print(f"{'λ':>8s}  {'zone':>4s}  "
          f"{'r_tumor':>8s}  {'r_normal':>9s}  {'rho_tumor':>10s}  "
          f"{'rho_normal':>11s}  | LR coef profile (tumor-pos minus normal-pos)")
    rows = []
    for lam in sorted(map_lambda["lambda"].unique()):
        sub_all = map_lambda[map_lambda["lambda"] == lam].copy()
        for zone in ["pz", "tz"]:
            sub = sub_all[sub_all["zone"] == zone].copy()
            res = correlate(sub, sub["is_tumor"], D_COLS, C=1.0)
            if res is None:
                continue
            row = {
                "estimator": f"MAP_lam={lam}",
                "zone": zone.upper(),
                "C": 1.0,
                **{k: v for k, v in res.items() if not isinstance(v, list)},
            }
            row["lr_coefs"] = res["lr_coefs"]
            row["sens_tumor"] = res["sens_tumor"]
            row["sens_normal"] = res["sens_normal"]
            rows.append(row)
            coef_str = " ".join(f"{c:+.2f}" for c in res["lr_coefs"])
            print(f"{lam:>8.4f}  {zone.upper():>4s}  "
                  f"{res['r_pearson_tumor']:>+8.3f}  "
                  f"{res['r_pearson_normal']:>+9.3f}  "
                  f"{res['r_spearman_tumor']:>+10.3f}  "
                  f"{res['r_spearman_normal']:>+11.3f}  | {coef_str}")

    # NUTS reference (from features.csv)
    print("\n=== NUTS reference (from features.csv) ===")
    nuts_cols = [f"nuts_{c}" for c in D_COLS]
    rename = dict(zip(nuts_cols, D_COLS))
    nuts_df = features.copy()
    nuts_df = nuts_df.rename(columns=rename)
    for zone in ["pz", "tz"]:
        sub = nuts_df[nuts_df["zone"] == zone].copy()
        res = correlate(sub, sub["is_tumor"], D_COLS, C=1.0)
        if res is None:
            continue
        rows.append({
            "estimator": "NUTS",
            "zone": zone.upper(),
            "C": 1.0,
            **{k: v for k, v in res.items() if not isinstance(v, list)},
            "lr_coefs": res["lr_coefs"],
            "sens_tumor": res["sens_tumor"],
            "sens_normal": res["sens_normal"],
        })
        coef_str = " ".join(f"{c:+.2f}" for c in res["lr_coefs"])
        print(f"{'NUTS':>8s}  {zone.upper():>4s}  "
              f"{res['r_pearson_tumor']:>+8.3f}  "
              f"{res['r_pearson_normal']:>+9.3f}  "
              f"{res['r_spearman_tumor']:>+10.3f}  "
              f"{res['r_spearman_normal']:>+11.3f}  | {coef_str}")

    # Save
    out = pd.DataFrame(rows)
    out.to_csv("results/biomarkers/adc_sens_vs_lr_tuned_lambda.csv", index=False)
    print("\nWrote results/biomarkers/adc_sens_vs_lr_tuned_lambda.csv")

    # MAP @ λ=0.1 from manuscript record (sanity check vs paper's reported r=-0.979)
    print("\n=== Sanity check: MAP @ λ=0.1 should reproduce paper's r ≈ −0.98 PZ ===")
    paper_match = out[(out["estimator"] == "MAP_lam=0.1") & (out["zone"] == "PZ")]
    if len(paper_match):
        print(f"  MAP λ=0.1 PZ: r_tumor={paper_match.iloc[0]['r_pearson_tumor']:+.3f}, "
              f"r_normal={paper_match.iloc[0]['r_pearson_normal']:+.3f}")
        print("  (Paper reports r ≈ −0.979 PZ tumor-op MAP. Match?)")


if __name__ == "__main__":
    main()
