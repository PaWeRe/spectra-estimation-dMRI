"""
Generate publication-quality figures for the MRM paper.

Produces:
- fig_adc_discriminant.pdf: ADC vs spectral discriminant scatter (headline)
- fig_sensitivity.pdf: ADC sensitivity vs LR coefficients
- fig_identifiability.pdf: Per-component identifiability bars
- fig_map_nuts.pdf: MAP vs NUTS per-component scatter

Usage:
    uv run python scripts/generate_paper_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

# Style
mpl.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

DIFFUSIVITIES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]
D_LABELS = ["0.25", "0.50", "0.75", "1.00", "1.50", "2.00", "3.00", "20.0"]
OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv("results/biomarkers/features.csv")
    return df


# =========================================================================
# Figure 4: ADC vs Spectral Discriminant (HEADLINE)
# =========================================================================
def fig_adc_discriminant(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, zone, title in zip(axes, ["pz", "tz"], ["Peripheral Zone", "Transition Zone"]):
        zdf = df[df["zone"] == zone].copy()
        y = zdf["is_tumor"].astype(int).values
        adc = zdf["adc"].values

        # Use MAP features for discriminant
        feat_cols = [f"map_D_{d:.2f}" for d in DIFFUSIVITIES]
        X = zdf[feat_cols].values

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_s, y)
        disc = X_s @ clf.coef_[0] + clf.intercept_[0]

        r, p = stats.pearsonr(adc, disc)

        # Scatter
        tumor_mask = y == 1
        ax.scatter(adc[~tumor_mask], disc[~tumor_mask], c="#4292c6", s=30,
                   alpha=0.7, label="Normal", edgecolors="white", linewidth=0.3)
        ax.scatter(adc[tumor_mask], disc[tumor_mask], c="#ef3b2c", s=30,
                   alpha=0.7, label="Tumor", edgecolors="white", linewidth=0.3)

        # Regression line
        z = np.polyfit(adc, disc, 1)
        x_line = np.linspace(adc.min(), adc.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1)

        n = len(y)
        ax.set_xlabel("ADC (mm\u00b2/s)")
        ax.set_ylabel("Spectral discriminant score")
        ax.set_title(f"{title} (n={n})")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.text(0.05, 0.05, f"r = {r:.3f}\np < 10$^{{-{int(-np.log10(p))}}}$",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_adc_discriminant.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    print(f"  Saved {path}")
    plt.close()


# =========================================================================
# Figure 5: ADC Sensitivity vs LR Coefficients
# =========================================================================
def fig_sensitivity(df):
    from spectra_estimation_dmri.biomarkers.recompute import compute_sensitivity

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, zone, title in zip(axes, ["pz", "tz"], ["Peripheral Zone", "Transition Zone"]):
        zdf = df[df["zone"] == zone].copy()
        y = zdf["is_tumor"].astype(int).values

        feat_cols = [f"map_D_{d:.2f}" for d in DIFFUSIVITIES]
        X = zdf[feat_cols].values
        avg_tumor = X[y == 1].mean(axis=0)

        # LR coefficients (standardized)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_s, y)
        coefs = clf.coef_[0]

        # Sensitivity
        sens = compute_sensitivity(avg_tumor, b_max=1.0)

        # Normalize both for visual comparison
        sens_norm = sens / np.max(np.abs(sens))
        coefs_norm = coefs / np.max(np.abs(coefs))

        x = np.arange(len(DIFFUSIVITIES))
        width = 0.35

        bars1 = ax.bar(x - width/2, -sens_norm, width, label="$-\\partial$ADC/$\\partial R_j$ (normalized)",
                       color="#4292c6", alpha=0.8, edgecolor="white")
        bars2 = ax.bar(x + width/2, coefs_norm, width, label="LR coefficients (normalized)",
                       color="#ef3b2c", alpha=0.8, edgecolor="white")

        r, p = stats.pearsonr(sens, coefs)
        ax.set_xticks(x)
        ax.set_xticklabels(D_LABELS, fontsize=8)
        ax.set_xlabel("Diffusivity D (\u03bcm\u00b2/ms)")
        ax.set_ylabel("Normalized weight")
        ax.set_title(f"{title}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.text(0.05, 0.05, f"r = {r:.3f}",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_sensitivity.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    print(f"  Saved {path}")
    plt.close()


# =========================================================================
# Figure 6: Per-Component Identifiability
# =========================================================================
def fig_identifiability(df):
    ident = pd.read_csv("results/biomarkers/identifiability.csv")

    fig, ax = plt.subplots(figsize=(7, 4))

    cvs = ident["mean_CV"].values
    colors = ["#2ca02c" if cv < 0.40 else "#d62728" if cv > 0.70 else "#ff7f0e"
              for cv in cvs]

    x = np.arange(len(DIFFUSIVITIES))
    bars = ax.bar(x, cvs, color=colors, edgecolor="white", width=0.6, alpha=0.85)

    # Threshold line
    ax.axhline(0.40, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(DIFFUSIVITIES) - 0.5, 0.42, "CV = 0.40 threshold", fontsize=8,
            color="gray", ha="right")

    # Value labels
    for bar, cv in zip(bars, cvs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{cv:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(D_LABELS)
    ax.set_xlabel("Diffusivity D (\u03bcm\u00b2/ms)")
    ax.set_ylabel("Mean Coefficient of Variation (CV)")
    ax.set_title("Per-Component Identifiability from NUTS Posterior (n=149 ROIs)")
    ax.set_ylim(0, 1.0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.85, label="Well identified (CV < 0.40)"),
        Patch(facecolor="#ff7f0e", alpha=0.85, label="Moderate (0.40 \u2264 CV < 0.70)"),
        Patch(facecolor="#d62728", alpha=0.85, label="Poorly identified (CV \u2265 0.70)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_identifiability.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    print(f"  Saved {path}")
    plt.close()


# =========================================================================
# Figure 7: MAP vs NUTS Per-Component Scatter
# =========================================================================
def fig_map_nuts(df):
    components = [
        (0.25, "D = 0.25 (restricted)"),
        (1.00, "D = 1.00 (intermediate)"),
        (3.00, "D = 3.00 (free water)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (d, label) in zip(axes, components):
        map_col = f"map_D_{d:.2f}"
        nuts_col = f"nuts_D_{d:.2f}"

        map_vals = df[map_col].values
        nuts_vals = df[nuts_col].values

        tumor_mask = df["is_tumor"].values
        ax.scatter(map_vals[~tumor_mask], nuts_vals[~tumor_mask], c="#4292c6", s=20,
                   alpha=0.6, label="Normal", edgecolors="none")
        ax.scatter(map_vals[tumor_mask], nuts_vals[tumor_mask], c="#ef3b2c", s=20,
                   alpha=0.6, label="Tumor", edgecolors="none")

        r, _ = stats.pearsonr(map_vals, nuts_vals)

        # Identity line
        lim = max(map_vals.max(), nuts_vals.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

        ax.set_xlabel("MAP fraction")
        ax.set_ylabel("NUTS posterior mean fraction")
        ax.set_title(f"{label}")
        ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10,
                va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        ax.set_aspect("equal")

    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = OUTPUT_DIR / "fig_map_nuts.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    print(f"  Saved {path}")
    plt.close()


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    df = load_data()

    fig_adc_discriminant(df)
    fig_sensitivity(df)
    fig_identifiability(df)
    fig_map_nuts(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("fig_*.pdf")):
        print(f"  {f}")
