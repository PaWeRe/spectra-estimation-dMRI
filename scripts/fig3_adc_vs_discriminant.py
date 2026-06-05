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

v4 changes (2026-05-31):
  - Font sizes bumped to the fig_roc 2x2 scale (axis labels=20, ticks=18,
    panel titles=17, legend=17) so Fig 3 reads at the same apparent size as
    the other 2x2 grid in the manuscript.
  - Shared legend moved to the TOP of the figure (above the panel grid).
  - In-figure suptitle removed: the caption's first sentence is the title
    (MRM convention), matching fig_roc (Fig 2) and fig1 (Fig 1). The
    bootstrap methodology is described in the LaTeX caption.
  - All four panels share common x (ADC, identical physical quantity) and y
    (discriminant score) axes for a uniform grid.

Layout: 2 rows (PZ, TZ) x 2 cols (NUTS, MAP @ lambda=1e-3) = 4 panels.

Output:
  paper/figures/fig3_v3.png
  paper/figures/fig3_v3.pdf
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from spectra_estimation_dmri.visualization.paper_style import (  # noqa: E402
    apply_style,
    COLORS,
)


REPO_ROOT = str(ROOT)
FEATURES_CSV = os.path.join(REPO_ROOT, "results/biomarkers/features.csv")
OUT_DIR = os.path.join(REPO_ROOT, "paper/figures")

DIFFUSIVITIES = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 20.00]
N_BOOT = 1000
RNG = np.random.default_rng(42)
LAMBDA_TUNED = 0.001

# --- Shared manuscript typography (locks legend==title size, labels 20 /
# ticks 18 / title 17 / legend 17). Replaces the script's own rcParams. ---
apply_style("grid")

COLOR_TUMOR = COLORS["tumor"]    # red
COLOR_NORMAL = COLORS["normal"]  # blue
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
               ("MAP_tuned", r"MAP ($\lambda = 10^{-3}$)")]

    results = {}
    for zkey, zlabel in zones:
        for mkey, mlabel in methods:
            df = method_dfs[mkey]
            sub = df[df["zone"] == zkey].copy()
            results[(zkey, mkey)] = per_panel(sub)

    # Layout (Stefan 2026-06-03): zone = COLUMNS (PZ left, TZ right),
    # estimator = ROWS (NUTS top, MAP bottom). Matches the manuscript-wide
    # PZ-left / TZ-right convention set by Fig 1.
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    for i, (mkey, mlabel) in enumerate(methods):       # rows = estimator
        for j, (zkey, zlabel) in enumerate(zones):     # cols = zone
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
            if i == 1:  # bottom row only (x shared across rows)
                ax.set_xlabel("ADC (μm²/ms)")
            if j == 0:  # left column only (y shared across columns)
                ax.set_ylabel("Spectral-discriminant score")
            ax.grid(True, alpha=0.25, linewidth=0.5)

    # Shared figure-level legend, OUTSIDE the panel grid (at the TOP).
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=12,
               markerfacecolor=COLOR_TUMOR, markeredgecolor="white",
               label="Tumor"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=12,
               markerfacecolor=COLOR_NORMAL, markeredgecolor="white",
               label="Normal"),
        Line2D([0], [0], color=COLOR_FIT, lw=2.0, label="OLS fit"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=3, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.975))

    # No in-figure title: per MRM the caption's first sentence is the title,
    # matching fig_roc (Fig 2) and fig1 (Fig 1), which carry no suptitle.
    # Vertical spacing mirrors Fig 1: the legend->row1 gap and the row1->row2
    # gap are made slightly larger and roughly equal. subplots_adjust is used
    # directly (no tight_layout/constrained_layout, which would override it).
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.09, right=0.97,
                        hspace=0.31, wspace=0.12)

    png_path = os.path.join(OUT_DIR, "fig3_v4.png")
    pdf_path = os.path.join(OUT_DIR, "fig3_v4.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Fig 3 v4 - ROI-level ADC vs spectral discriminant ===")
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
