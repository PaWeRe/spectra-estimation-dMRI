"""
Generate publication-quality figures for the MRM paper.

Produces:
- fig_adc_discriminant.pdf: ADC vs spectral discriminant scatter (headline)
- fig_sensitivity.pdf: ADC sensitivity vs LR coefficients (with D=20 annotation)
- fig_identifiability.pdf: Per-component identifiability (no arbitrary threshold)
- fig_map_nuts.pdf: MAP vs NUTS per-component scatter
- fig_roc.pdf: ROC curves with final verified numbers (ADC/MAP/NUTS)

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
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
import sys

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

DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
D_LABELS = ["0.25", "0.50", "0.75", "1.00", "1.50", "2.00", "3.00", "20.0"]
D_COLS = [f"D_{d:.2f}" for d in DIFFUSIVITIES]
OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#4292c6"
RED = "#ef3b2c"
GRAY = "#888888"


def load_data():
    return pd.read_csv("results/biomarkers/features.csv")


def loocv_roc(X, y, C=1.0):
    """LOOCV predictions for ROC curve."""
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_tr, y[train_idx])
        y_pred[test_idx] = clf.predict_proba(X_te)[0, 1]
    return y_pred


# =========================================================================
# Figure: ADC vs Spectral Discriminant (HEADLINE)
# =========================================================================
def fig_adc_discriminant(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, zone, title in zip(axes, ["pz", "tz"], ["Peripheral Zone", "Transition Zone"]):
        zdf = df[df["zone"] == zone].copy()
        y = zdf["is_tumor"].astype(int).values
        adc = zdf["adc"].values

        feat_cols = [f"map_{c}" for c in D_COLS]
        X = zdf[feat_cols].values

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_s, y)
        disc = X_s @ clf.coef_[0] + clf.intercept_[0]

        r, p = stats.pearsonr(adc, disc)
        tumor_mask = y == 1

        ax.scatter(adc[~tumor_mask], disc[~tumor_mask], c=BLUE, s=30,
                   alpha=0.7, label="Normal", edgecolors="white", linewidth=0.3)
        ax.scatter(adc[tumor_mask], disc[tumor_mask], c=RED, s=30,
                   alpha=0.7, label="Tumor", edgecolors="white", linewidth=0.3)

        z = np.polyfit(adc, disc, 1)
        x_line = np.linspace(adc.min(), adc.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1)

        ax.set_xlabel("ADC (mm\u00b2/s)")
        ax.set_ylabel("Spectral discriminant score")
        ax.set_title(f"{title} (n={len(y)})")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.text(0.05, 0.05, f"r = {r:.3f}\np < 10$^{{-{int(-np.log10(p))}}}$",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    _save(fig, "fig_adc_discriminant")


# =========================================================================
# Figure: ADC Sensitivity vs LR Coefficients (with D=20 annotation)
# =========================================================================
def fig_sensitivity(df):
    from spectra_estimation_dmri.biomarkers.recompute import compute_sensitivity

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, zone, title in zip(axes, ["pz", "tz"], ["Peripheral Zone", "Transition Zone"]):
        zdf = df[df["zone"] == zone].copy()
        y = zdf["is_tumor"].astype(int).values

        feat_cols = [f"map_{c}" for c in D_COLS]
        X = zdf[feat_cols].values
        avg_tumor = X[y == 1].mean(axis=0)

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_s, y)
        coefs = clf.coef_[0]

        sens = compute_sensitivity(avg_tumor, b_max=1.0)

        # Normalize for visual comparison (negate sensitivity for alignment)
        sens_norm = -sens / np.max(np.abs(sens))
        coefs_norm = coefs / np.max(np.abs(coefs))

        x = np.arange(len(DIFFUSIVITIES))
        width = 0.35

        ax.bar(x - width/2, sens_norm, width,
               label=r"$-\partial$ADC/$\partial R_j$ (norm.)",
               color=BLUE, alpha=0.8, edgecolor="white")
        ax.bar(x + width/2, coefs_norm, width,
               label="LR coefficient (norm.)",
               color=RED, alpha=0.8, edgecolor="white")

        # Annotate D=20 discrepancy
        ax.annotate("ADC insensitive\nto D=20 at\nb \u2264 1000 s/mm\u00b2",
                     xy=(7, sens_norm[7]), xytext=(5.5, sens_norm[7] + 0.35),
                     fontsize=7, ha="center", color=GRAY,
                     arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))

        r, _ = stats.pearsonr(sens, coefs)
        # Correlation excluding D=20 (7 components sensitive to b<=1000)
        r7, _ = stats.pearsonr(sens[:7], coefs[:7])

        ax.set_xticks(x)
        ax.set_xticklabels(D_LABELS, fontsize=8)
        ax.set_xlabel("Diffusivity D (\u03bcm\u00b2/ms)")
        ax.set_ylabel("Normalized weight")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.text(0.05, 0.05, f"r = {r:.3f} (all 8)\nr = {r7:.3f} (D \u2264 3.0)",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    _save(fig, "fig_sensitivity")


# =========================================================================
# Figure: Per-Component Identifiability (continuous color, no threshold)
# =========================================================================
def fig_identifiability(df):
    ident = pd.read_csv("results/biomarkers/identifiability.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})

    # Left: CV bar chart with continuous colormap
    cvs = ident["mean_CV"].values
    cmap = mpl.cm.RdYlGn_r  # Red=high CV (bad), Green=low CV (good)
    norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
    colors = [cmap(norm(cv)) for cv in cvs]

    x = np.arange(len(DIFFUSIVITIES))
    bars = ax1.bar(x, cvs, color=colors, edgecolor="white", width=0.6, alpha=0.9)

    for bar, cv in zip(bars, cvs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{cv:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(D_LABELS)
    ax1.set_xlabel("Diffusivity D (\u03bcm\u00b2/ms)")
    ax1.set_ylabel("Posterior CV (std / mean)")
    ax1.set_title("Per-Component Identifiability")
    ax1.set_ylim(0, 1.0)
    # Add interpretation text
    ax1.text(0.02, 0.95, "Low CV = well constrained\nHigh CV = undetermined",
             transform=ax1.transAxes, fontsize=8, va="top", color=GRAY)

    # Right: Mean fraction vs mean std (2D view)
    fracs = ident["mean_fraction"].values
    stds = ident["mean_posterior_std"].values

    scatter = ax2.scatter(fracs, stds, c=cvs, cmap=cmap, norm=norm,
                          s=100, edgecolors="black", linewidth=0.5, zorder=3)
    for i, d in enumerate(DIFFUSIVITIES):
        ax2.annotate(f"D={d}", (fracs[i], stds[i]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)

    # Add CV=1 reference line (std = mean)
    lim = max(fracs.max(), stds.max()) * 1.15
    ax2.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1, label="CV = 1.0")
    ax2.set_xlabel("Mean fraction")
    ax2.set_ylabel("Mean posterior std")
    ax2.set_title("Fraction vs. Uncertainty")
    ax2.legend(fontsize=8, loc="upper left")
    plt.colorbar(scatter, ax=ax2, label="CV", shrink=0.8)

    plt.tight_layout()
    _save(fig, "fig_identifiability")


# =========================================================================
# Figure: MAP vs NUTS Per-Component Scatter
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

        ax.scatter(map_vals[~tumor_mask], nuts_vals[~tumor_mask], c=BLUE, s=20,
                   alpha=0.6, label="Normal", edgecolors="none")
        ax.scatter(map_vals[tumor_mask], nuts_vals[tumor_mask], c=RED, s=20,
                   alpha=0.6, label="Tumor", edgecolors="none")

        r, _ = stats.pearsonr(map_vals, nuts_vals)
        lim = max(map_vals.max(), nuts_vals.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("MAP fraction")
        ax.set_ylabel("NUTS posterior mean")
        ax.set_title(label)
        ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10,
                va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        ax.set_aspect("equal")

    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    _save(fig, "fig_map_nuts")


# =========================================================================
# Figure: ROC curves (final, verified numbers)
# =========================================================================
def fig_roc(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    map_cols = [f"map_{c}" for c in D_COLS]
    nuts_cols = [f"nuts_{c}" for c in D_COLS]

    tasks = [
        ("PZ Tumor Detection", df[df["zone"] == "pz"], "is_tumor", 81),
        ("TZ Tumor Detection", df[df["zone"] == "tz"], "is_tumor", 68),
    ]

    # GGG
    ggg_df = df[(df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)].copy()
    ggg_df["ggg_binary"] = (ggg_df["ggg"] >= 3).astype(int)
    tasks.append(("GGG Classification", ggg_df, "ggg_binary", 29))

    for ax, (title, tdf, label_col, n) in zip(axes, tasks):
        y = tdf[label_col].astype(int).values

        # ADC raw rank
        adc_vals = tdf["adc"].values
        adc_auc = roc_auc_score(y, adc_vals)
        if adc_auc < 0.5:
            adc_vals = -adc_vals
            adc_auc = 1 - adc_auc
        fpr_adc, tpr_adc, _ = roc_curve(y, adc_vals)
        ax.plot(fpr_adc, tpr_adc, color="#2171b5", linewidth=2,
                label=f"ADC raw ({adc_auc:.3f})")

        # ADC via LR
        adc_pred = loocv_roc(adc_vals.reshape(-1, 1), y, C=1.0)
        adc_lr_auc = roc_auc_score(y, adc_pred)
        fpr_al, tpr_al, _ = roc_curve(y, adc_pred)
        ax.plot(fpr_al, tpr_al, color="#6baed6", linewidth=1.5, linestyle="--",
                label=f"ADC LR ({adc_lr_auc:.3f})")

        # MAP Full LR
        X_map = tdf[map_cols].values
        map_pred = loocv_roc(X_map, y, C=1.0)
        map_auc = roc_auc_score(y, map_pred)
        fpr_m, tpr_m, _ = roc_curve(y, map_pred)
        ax.plot(fpr_m, tpr_m, color="#cb181d", linewidth=1.5,
                label=f"MAP 8-feat ({map_auc:.3f})")

        # NUTS Full LR
        X_nuts = tdf[nuts_cols].values
        nuts_pred = loocv_roc(X_nuts, y, C=1.0)
        nuts_auc = roc_auc_score(y, nuts_pred)
        fpr_n, tpr_n, _ = roc_curve(y, nuts_pred)
        ax.plot(fpr_n, tpr_n, color="#fc9272", linewidth=1.5, linestyle="--",
                label=f"NUTS 8-feat ({nuts_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title} (n={n})")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    _save(fig, "fig_roc")


# =========================================================================
# Figure: Mean Spectra by Tissue Type (boxplot)
# =========================================================================
def fig_spectra(df):
    nuts_cols = [f"nuts_{c}" for c in D_COLS]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (axes[0, 0], "pz", False, "Normal PZ"),
        (axes[0, 1], "tz", False, "Normal TZ"),
        (axes[1, 0], "pz", True, "Tumor PZ"),
        (axes[1, 1], "tz", True, "Tumor TZ"),
    ]

    for ax, zone, is_tumor, title in panels:
        mask = (df["zone"] == zone) & (df["is_tumor"] == is_tumor)
        sub = df[mask]
        n = len(sub)

        data = [sub[c].values for c in nuts_cols]

        bp = ax.boxplot(data, positions=range(len(DIFFUSIVITIES)), widths=0.5,
                        patch_artist=True, medianprops=dict(color="black", linewidth=1.5))

        color = RED if is_tumor else BLUE
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(range(len(DIFFUSIVITIES)))
        ax.set_xticklabels(D_LABELS, fontsize=8)
        ax.set_xlabel("Diffusivity D (\u03bcm\u00b2/ms)")
        ax.set_ylabel("Spectral fraction")
        ax.set_title(f"{title} (n={n})")
        ax.set_ylim(-0.02, 0.55)

    plt.tight_layout()
    _save(fig, "fig_spectra")


def _save(fig, name):
    for ext in [".pdf", ".png"]:
        fig.savefig(OUTPUT_DIR / f"{name}{ext}")
    print(f"  Saved {OUTPUT_DIR / name}.pdf")
    plt.close(fig)


# =========================================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    df = load_data()

    fig_spectra(df)
    fig_adc_discriminant(df)
    fig_sensitivity(df)
    fig_identifiability(df)
    fig_map_nuts(df)
    fig_roc(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("fig_*.pdf")):
        print(f"  {f}")
