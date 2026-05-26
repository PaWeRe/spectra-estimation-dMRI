"""
Figure 3 (v3): ROI-level scalar scatter of ADC vs the spectral-discriminant
score, demonstrating that ADC and the multivariate spectral classifier rank
patients near-identically (target r ~ -0.98 with bootstrap CI).

This is the central "Why ADC works" anchor of the paper. For each zone
(PZ, TZ) and each method (NUTS posterior mean, tuned-MAP at lambda=1e-3), we:

  1. Fit a single LogisticRegression on the 8 spectral fractions for that
     zone (tumor vs normal label) with C=1.0, class_weight='balanced',
     standardized features. No cross-validation -- the discriminant is a
     description of the cohort's optimal linear projection, not a held-out
     predictor.
  2. Read off the signed per-ROI discriminant via lr.decision_function(X_s).
  3. Bootstrap Pearson r between ADC and the discriminant across the ROIs
     of that zone (1000 resamples, percentile 95% CI).

v3 changes (2026-05-26):
  - MAP values pulled from regenerated features.csv (correct NNLS solver,
    written by biomarkers/recompute.py on 2026-05-26) rather than the stale
    map_lambda_bwh.csv from 2026-05-24.
  - Font sizes bumped to match Fig 2 (xtick/ytick/axis labels=17, title=15,
    legend=15).
  - Shared legend placed outside the panel grid (bottom-centred).

Layout: 2 rows (PZ, TZ) x 2 cols (NUTS, MAP @ lambda=1e-3) = 4 panels.

Output:
  paper/figures/fig3_v3.png
  paper/figures/fig3_v3.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REPO_ROOT = "/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI"
FEATURES_CSV = os.path.join(REPO_ROOT, "results/biomarkers/features.csv")
OUT_DIR = os.path.join(REPO_ROOT, "paper/figures")

DIFFUSIVITIES = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 20.00]
N_BOOT = 1000
RNG = np.random.default_rng(42)
LAMBDA_TUNED = 0.001

# --- Stephan-strict typography (match Fig 2 conventions) ---
mpl.rcParams.update({
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.labelsize": 17,
    "axes.titlesize": 15,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})

COLOR_TUMOR = "#d62728"   # red
COLOR_NORMAL = "#1f77b4"  # blue
COLOR_FIT = "black"


def bootstrap_pearson(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                      rng: np.random.Generator = RNG) -> tuple[float, float, float]:
    """Return (r_point, ci_lo, ci_hi) with percentile bootstrap CI."""
    r_point, _ = stats.pearsonr(x, y)
    n = len(x)
    boot_r = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            boot_r[b] = np.nan
            continue
        boot_r[b], _ = stats.pearsonr(xi, yi)
    boot_r = boot_r[np.isfinite(boot_r)]
    ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
    return float(r_point), float(ci_lo), float(ci_hi)


def discriminant(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit LR on standardized X; return signed decision-function per row."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000,
        solver="lbfgs", random_state=42,
    )
    clf.fit(X_s, y.astype(int))
    return clf.decision_function(X_s)


def build_method_dataframes() -> dict[str, pd.DataFrame]:
    """Return {'NUTS': df, 'MAP_tuned': df}, each with columns:
       roi_id, zone, is_tumor, adc, D_0.25 ... D_20.00.

    Both methods are now read straight from the regenerated features.csv so
    they reflect the corrected NNLS-augmented MAP solver.
    """
    features = pd.read_csv(FEATURES_CSV)

    base_cols = ["roi_id", "zone", "is_tumor", "adc"]
    feat_cols = [f"D_{d:.2f}" for d in DIFFUSIVITIES]

    nuts_cols = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
    nuts_df = features[base_cols + nuts_cols].rename(
        columns=dict(zip(nuts_cols, feat_cols))
    ).copy()

    map_cols = [f"map_D_{d:.2f}" for d in DIFFUSIVITIES]
    map_df = features[base_cols + map_cols].rename(
        columns=dict(zip(map_cols, feat_cols))
    ).copy()

    return {"NUTS": nuts_df, "MAP_tuned": map_df}


def per_panel(df_zone: pd.DataFrame) -> dict:
    """Given an already-zone-filtered df, fit the discriminant and compute
    bootstrap r vs ADC."""
    feat_cols = [f"D_{d:.2f}" for d in DIFFUSIVITIES]
    X = df_zone[feat_cols].values
    y = df_zone["is_tumor"].astype(int).values
    adc = df_zone["adc"].values * 1e3  # to um^2/ms
    score = discriminant(X, y)
    r, lo, hi = bootstrap_pearson(adc, score)
    return {"adc": adc, "score": score, "is_tumor": y.astype(bool),
            "r": r, "ci_lo": lo, "ci_hi": hi, "n": len(adc)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    method_dfs = build_method_dataframes()

    zones = [("pz", "PZ"), ("tz", "TZ")]
    methods = [("NUTS", "NUTS (posterior mean)"),
               ("MAP_tuned", f"MAP @ λ={LAMBDA_TUNED}")]

    results = {}
    for zkey, zlabel in zones:
        for mkey, mlabel in methods:
            df = method_dfs[mkey]
            sub = df[df["zone"] == zkey].copy()
            results[(zkey, mkey)] = per_panel(sub)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.0))
    for i, (zkey, zlabel) in enumerate(zones):
        for j, (mkey, mlabel) in enumerate(methods):
            ax = axes[i, j]
            res = results[(zkey, mkey)]
            adc, score, tumor = res["adc"], res["score"], res["is_tumor"]

            ax.scatter(adc[~tumor], score[~tumor], s=55, c=COLOR_NORMAL,
                       edgecolor="white", linewidth=0.7, label="Normal",
                       alpha=0.85)
            ax.scatter(adc[tumor], score[tumor], s=55, c=COLOR_TUMOR,
                       edgecolor="white", linewidth=0.7, label="Tumor",
                       alpha=0.85)

            # OLS regression line on full panel
            slope, intercept = np.polyfit(adc, score, 1)
            xline = np.linspace(adc.min(), adc.max(), 100)
            ax.plot(xline, slope * xline + intercept, color=COLOR_FIT,
                    lw=1.6, alpha=0.9, label="OLS fit")

            r, lo, hi, n = res["r"], res["ci_lo"], res["ci_hi"], res["n"]
            ax.set_title(
                f"{zlabel} — {mlabel}\n"
                f"r = {r:+.3f}  [{lo:+.3f}, {hi:+.3f}],  n = {n}",
            )
            ax.set_xlabel("ADC (μm²/ms)")
            ax.set_ylabel("Spectral-discriminant score")
            ax.grid(True, alpha=0.25, linewidth=0.5)

    # Shared figure-level legend, OUTSIDE the panel grid (below).
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=10,
               markerfacecolor=COLOR_TUMOR, markeredgecolor="white",
               label="Tumor"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=10,
               markerfacecolor=COLOR_NORMAL, markeredgecolor="white",
               label="Normal"),
        Line2D([0], [0], color=COLOR_FIT, lw=1.6, label="OLS fit"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        "ADC vs. spectral-discriminant score: ROI-level rank-equivalence "
        "across methods\n"
        "Bootstrap Pearson r, 1000 resamples, percentile 95% CI",
        fontsize=16,
    )
    # Leave room at top for suptitle and bottom for shared legend.
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])

    png_path = os.path.join(OUT_DIR, "fig3_v3.png")
    pdf_path = os.path.join(OUT_DIR, "fig3_v3.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Fig 3 v3 - ROI-level ADC vs spectral discriminant ===")
    print(f"{'zone':>5s}  {'method':>10s}  {'n':>4s}  "
          f"{'r':>8s}  {'95% CI low':>11s}  {'95% CI high':>12s}")
    for zkey, zlabel in zones:
        for mkey, _ in methods:
            res = results[(zkey, mkey)]
            print(f"{zlabel:>5s}  {mkey:>10s}  {res['n']:>4d}  "
                  f"{res['r']:>+8.3f}  {res['ci_lo']:>+11.3f}  "
                  f"{res['ci_hi']:>+12.3f}")

    print(f"\nWrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
