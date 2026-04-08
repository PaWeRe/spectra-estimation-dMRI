"""
Generate publication-quality figures for the MRM paper.
Budget: 9 figures + 1 table = 10 items total.

Main figures:
1. fig_spectra_combined: Spectra (MAP+NUTS) + identifiability (merged)
2. fig_roc: ROC with individual components + ADC + Full LR
3. fig_adc_discriminant: ADC vs discriminant scatter
4. fig_sensitivity: ADC sensitivity vs LR coefficients
5. fig_map_nuts: MAP vs NUTS comparison (3 components)
6. fig_uncertainty: Uncertainty propagation (ISMRM-style, 3 panels)
7. fig_directions: Direction comparison (generated separately)
8. fig_pixelwise: Pixel-wise maps (existing)

Supplementary:
- fig_trace_simulation: NUTS convergence
- fig_robustness: Spectrum recovery across shapes

Usage:
    uv run python scripts/generate_paper_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# =========================================================================
# Global style — consistent across ALL figures
# =========================================================================
TUMOR_COLOR = "#d62728"
NORMAL_COLOR = "#1f77b4"
MAP_COLOR = "#2ca02c"
NUTS_COLOR = "#ff7f0e"
ADC_COLOR = "#1f77b4"
GRAY = "#888888"

DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
D_LABELS = ["0.25", "0.50", "0.75", "1.00", "1.50", "2.00", "3.00", "20.0"]
D_COLS = [f"D_{d:.2f}" for d in DIFFUSIVITIES]
OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def load_data():
    return pd.read_csv("results/biomarkers/features.csv")


def _save(fig, name):
    for ext in [".pdf", ".png"]:
        fig.savefig(OUTPUT_DIR / f"{name}{ext}")
    print(f"  Saved {name}")
    plt.close(fig)


def loocv_roc(X, y, C=1.0):
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(Xtr, y[tr]); y_pred[te] = clf.predict_proba(Xte)[0, 1]
    return y_pred


def bootstrap_auc_ci(y, y_pred, n_boot=2000, alpha=0.05):
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) > 1:
            aucs.append(roc_auc_score(y[idx], y_pred[idx]))
    return np.percentile(aucs, 100*alpha/2), np.percentile(aucs, 100*(1-alpha/2))


# =========================================================================
# Fig 1: Spectra + Identifiability Combined
# =========================================================================
def fig_spectra_combined(df):
    ident = pd.read_csv("results/biomarkers/identifiability.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    zones = [("pz", "Peripheral Zone"), ("tz", "Transition Zone")]

    for col_idx, (zone, zone_title) in enumerate(zones):
        zdf = df[df["zone"] == zone]

        for row_idx, (is_tumor, tissue_label) in enumerate([(False, "Normal"), (True, "Tumor")]):
            ax = axes[row_idx, col_idx]
            sub = zdf[zdf["is_tumor"] == is_tumor]
            n = len(sub)

            map_cols = [f"map_{c}" for c in D_COLS]
            nuts_cols = [f"nuts_{c}" for c in D_COLS]

            x = np.arange(len(DIFFUSIVITIES))
            width = 0.35

            # MAP boxplot
            map_data = [sub[c].values for c in map_cols]
            bp_map = ax.boxplot(map_data, positions=x - width/2, widths=width*0.85,
                                patch_artist=True, medianprops=dict(color="black", linewidth=1.2),
                                flierprops=dict(markersize=3), manage_ticks=False)
            for patch in bp_map["boxes"]:
                patch.set_facecolor(MAP_COLOR); patch.set_alpha(0.5)

            # NUTS boxplot
            nuts_data = [sub[c].values for c in nuts_cols]
            bp_nuts = ax.boxplot(nuts_data, positions=x + width/2, widths=width*0.85,
                                 patch_artist=True, medianprops=dict(color="black", linewidth=1.2),
                                 flierprops=dict(markersize=3), manage_ticks=False)
            for patch in bp_nuts["boxes"]:
                patch.set_facecolor(NUTS_COLOR); patch.set_alpha(0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(D_LABELS, fontsize=8)
            ax.set_xlabel("Diffusivity D (\u03bcm\u00b2/ms)")
            ax.set_ylabel("Spectral Fraction")
            ax.set_title(f"{tissue_label} {zone_title} (n={n})", fontweight="bold")
            ax.set_ylim(-0.02, 0.85)

            if row_idx == 0 and col_idx == 0:
                ax.legend([bp_map["boxes"][0], bp_nuts["boxes"][0]],
                          ["MAP", "NUTS"], loc="upper right", framealpha=0.9)

            # Add CV annotation on top row only (identifiability)
            if row_idx == 0 and col_idx == 0:
                for i, cv in enumerate(ident["mean_CV"].values):
                    color = MAP_COLOR if cv < 0.4 else TUMOR_COLOR if cv > 0.7 else NUTS_COLOR
                    ax.text(i, ax.get_ylim()[1] * 0.95, f"CV={cv:.2f}",
                            ha="center", fontsize=6.5, color=color, fontweight="bold")

    plt.tight_layout()
    _save(fig, "fig_spectra_combined")


# =========================================================================
# Fig 2: ROC Curves With Individual Components
# =========================================================================
def fig_roc(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    map_cols = [f"map_{c}" for c in D_COLS]

    # Component colors (consistent warm-to-cool)
    comp_cmap = plt.cm.coolwarm
    comp_colors = [comp_cmap(i / (len(DIFFUSIVITIES)-1)) for i in range(len(DIFFUSIVITIES))]

    tasks = [
        ("PZ: Tumor Detection", df[df["zone"] == "pz"], "is_tumor"),
        ("TZ: Tumor Detection", df[df["zone"] == "tz"], "is_tumor"),
    ]
    ggg_df = df[(df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)].copy()
    ggg_df["ggg_binary"] = (ggg_df["ggg"] >= 3).astype(int)
    tasks.append(("GGG: Grade Classification", ggg_df, "ggg_binary"))

    for ax, (title, tdf, label_col) in zip(axes, tasks):
        y = tdf[label_col].astype(int).values
        n = len(y)

        # Individual components (thin lines)
        for i, d in enumerate(DIFFUSIVITIES):
            feat_col = f"map_{D_COLS[i]}"
            vals = tdf[feat_col].values
            auc_val = roc_auc_score(y, vals)
            if auc_val < 0.5:
                vals = -vals; auc_val = 1 - auc_val
            fpr, tpr, _ = roc_curve(y, vals)
            ax.plot(fpr, tpr, color=comp_colors[i], linewidth=0.8, alpha=0.6,
                    label=f"D={d} ({auc_val:.2f})")

        # ADC raw rank (thick blue)
        adc_vals = tdf["adc"].values
        adc_auc = roc_auc_score(y, adc_vals)
        if adc_auc < 0.5: adc_vals = -adc_vals; adc_auc = 1 - adc_auc
        fpr_a, tpr_a, _ = roc_curve(y, adc_vals)
        ci_lo, ci_hi = bootstrap_auc_ci(y, adc_vals if adc_auc == roc_auc_score(y, adc_vals) else -adc_vals)
        ax.plot(fpr_a, tpr_a, color=ADC_COLOR, linewidth=2.5,
                label=f"ADC ({adc_auc:.3f} [{ci_lo:.2f}-{ci_hi:.2f}])")

        # MAP Full LR (thick green)
        X_map = tdf[map_cols].values
        map_pred = loocv_roc(X_map, y, C=1.0)
        map_auc = roc_auc_score(y, map_pred)
        ci_lo, ci_hi = bootstrap_auc_ci(y, map_pred)
        fpr_m, tpr_m, _ = roc_curve(y, map_pred)
        ax.plot(fpr_m, tpr_m, color=MAP_COLOR, linewidth=2,
                label=f"MAP 8-feat ({map_auc:.3f} [{ci_lo:.2f}-{ci_hi:.2f}])")

        # NUTS Full LR (thick orange)
        nuts_cols_list = [f"nuts_{c}" for c in D_COLS]
        X_nuts = tdf[nuts_cols_list].values
        nuts_pred = loocv_roc(X_nuts, y, C=1.0)
        nuts_auc = roc_auc_score(y, nuts_pred)
        ci_lo, ci_hi = bootstrap_auc_ci(y, nuts_pred)
        fpr_n, tpr_n, _ = roc_curve(y, nuts_pred)
        ax.plot(fpr_n, tpr_n, color=NUTS_COLOR, linewidth=2, linestyle="--",
                label=f"NUTS 8-feat ({nuts_auc:.3f} [{ci_lo:.2f}-{ci_hi:.2f}])")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{title} (n={n})", fontweight="bold", fontsize=11)
        ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    _save(fig, "fig_roc")


# =========================================================================
# Fig 3: ADC vs Spectral Discriminant
# =========================================================================
def fig_adc_discriminant(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, zone, title in zip(axes, ["pz", "tz"],
                                ["Peripheral Zone", "Transition Zone"]):
        zdf = df[df["zone"] == zone].copy()
        y = zdf["is_tumor"].astype(int).values
        adc = zdf["adc"].values

        # Convert ADC to um2/ms for cleaner axis (multiply by 1000 since it's in mm2/s)
        adc_display = adc * 1000

        feat_cols = [f"map_{c}" for c in D_COLS]
        X = zdf[feat_cols].values
        scaler = StandardScaler(); X_s = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_s, y)
        disc = X_s @ clf.coef_[0] + clf.intercept_[0]

        r, p = stats.pearsonr(adc_display, disc)
        tumor_mask = y == 1

        ax.scatter(adc_display[~tumor_mask], disc[~tumor_mask], c=NORMAL_COLOR, s=35,
                   alpha=0.7, label="Normal", edgecolors="white", linewidth=0.3)
        ax.scatter(adc_display[tumor_mask], disc[tumor_mask], c=TUMOR_COLOR, s=35,
                   alpha=0.7, label="Tumor", edgecolors="white", linewidth=0.3)

        z = np.polyfit(adc_display, disc, 1)
        x_line = np.linspace(adc_display.min(), adc_display.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, linewidth=1)

        ax.set_xlabel("ADC (\u03bcm\u00b2/ms)")
        ax.set_ylabel("Spectral Discriminant Score")
        ax.set_title(f"{title} (n={len(y)})", fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.text(0.05, 0.05, f"r = {r:.3f}\np < 10$^{{-{int(-np.log10(p))}}}$",
                transform=ax.transAxes, fontsize=10, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    _save(fig, "fig_adc_discriminant")


# =========================================================================
# Fig 4: ADC Sensitivity vs LR Coefficients
# =========================================================================
def fig_sensitivity(df):
    from spectra_estimation_dmri.biomarkers.recompute import compute_sensitivity

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, zone, title in zip(axes, ["pz", "tz"],
                                ["Peripheral Zone", "Transition Zone"]):
        zdf = df[df["zone"] == zone].copy()
        y = zdf["is_tumor"].astype(int).values
        feat_cols = [f"map_{c}" for c in D_COLS]
        X = zdf[feat_cols].values
        avg_tumor = X[y == 1].mean(axis=0)

        scaler = StandardScaler(); X_s = scaler.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_s, y)
        coefs = clf.coef_[0]
        sens = compute_sensitivity(avg_tumor, b_max=1.0)

        sens_norm = -sens / np.max(np.abs(sens))
        coefs_norm = coefs / np.max(np.abs(coefs))

        x = np.arange(len(DIFFUSIVITIES)); width = 0.35
        ax.bar(x - width/2, sens_norm, width,
               label=r"$-\partial$ADC/$\partial R_j$ (Normalized)",
               color=ADC_COLOR, alpha=0.7, edgecolor="white")
        ax.bar(x + width/2, coefs_norm, width,
               label="LR Coefficient (Normalized)",
               color=NUTS_COLOR, alpha=0.7, edgecolor="white")

        r, _ = stats.pearsonr(sens, coefs)
        r7, _ = stats.pearsonr(sens[:7], coefs[:7])

        ax.set_xticks(x); ax.set_xticklabels(D_LABELS, fontsize=8)
        ax.set_xlabel("Diffusivity D (\u03bcm\u00b2/ms)")
        ax.set_ylabel("Normalized Weight")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.text(0.05, 0.05, f"r = {r:.3f} (All 8)\nr = {r7:.3f} (D \u2264 3.0)",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    _save(fig, "fig_sensitivity")


# =========================================================================
# Fig 5: MAP vs NUTS Per-Component Scatter
# =========================================================================
def fig_map_nuts(df):
    components = [
        (0.25, "D = 0.25 (Restricted)"),
        (1.00, "D = 1.00 (Intermediate)"),
        (3.00, "D = 3.00 (Free Water)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (d, label) in zip(axes, components):
        map_vals = df[f"map_D_{d:.2f}"].values
        nuts_vals = df[f"nuts_D_{d:.2f}"].values
        tumor_mask = df["is_tumor"].values

        ax.scatter(map_vals[~tumor_mask], nuts_vals[~tumor_mask], c=NORMAL_COLOR, s=22,
                   alpha=0.6, label="Normal", edgecolors="none")
        ax.scatter(map_vals[tumor_mask], nuts_vals[tumor_mask], c=TUMOR_COLOR, s=22,
                   alpha=0.6, label="Tumor", edgecolors="none")

        r, _ = stats.pearsonr(map_vals, nuts_vals)
        lim = max(map_vals.max(), nuts_vals.max()) * 1.08
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("MAP Fraction")
        ax.set_ylabel("NUTS Posterior Mean Fraction")
        ax.set_title(label, fontweight="bold")
        ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10,
                va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        ax.set_aspect("equal")

    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    _save(fig, "fig_map_nuts")


# =========================================================================
# Fig 6: Uncertainty Propagation (ISMRM-style, 3 horizontal panels)
# =========================================================================
def fig_uncertainty(df):
    map_cols = [f"map_{c}" for c in D_COLS]
    nuts_cols = [f"nuts_{c}" for c in D_COLS]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tasks = [
        ("PZ: Tumor Detection", df[df["zone"] == "pz"], "is_tumor"),
        ("TZ: Tumor Detection", df[df["zone"] == "tz"], "is_tumor"),
    ]
    ggg_df = df[(df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)].copy()
    ggg_df["ggg_binary"] = (ggg_df["ggg"] >= 3).astype(int)
    tasks.append(("GGG: Grade Classification", ggg_df, "ggg_binary"))

    for ax, (title, tdf, label_col) in zip(axes, tasks):
        y = tdf[label_col].astype(int).values
        n = len(y)

        # Get NUTS LOOCV predictions with uncertainty
        X_nuts = tdf[nuts_cols].values
        # Also get per-ROI feature uncertainty (mean posterior std across components)
        nuts_std_cols = [f"nuts_std_{c}" for c in D_COLS]
        feature_unc = tdf[nuts_std_cols].mean(axis=1).values

        # LOOCV predictions
        nuts_pred = loocv_roc(X_nuts, y, C=1.0)

        # Sort by prediction
        sort_idx = np.argsort(nuts_pred)
        y_sorted = y[sort_idx]
        pred_sorted = nuts_pred[sort_idx]
        unc_sorted = feature_unc[sort_idx]

        # Normalize uncertainty for error bars (scale to prediction space)
        unc_display = unc_sorted * 2  # rough scaling for visualization

        for i in range(len(y_sorted)):
            is_correct = (pred_sorted[i] >= 0.5) == y_sorted[i]
            if y_sorted[i] == 1:
                color = TUMOR_COLOR if is_correct else TUMOR_COLOR
                marker = "s" if is_correct else "x"
            else:
                color = NORMAL_COLOR if is_correct else NORMAL_COLOR
                marker = "o" if is_correct else "x"
            alpha = 0.4 if not is_correct else 0.8
            ms = 5 if is_correct else 8

            ax.errorbar(i, pred_sorted[i], yerr=unc_display[i],
                        fmt=marker, color=color, alpha=alpha, markersize=ms,
                        ecolor=color, elinewidth=0.8, capsize=2)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Sample Index (Sorted By Prediction)")
        ax.set_ylabel("P(Positive Class)")
        ax.set_title(f"{title} (n={n})", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)

        # Compute misclassified uncertainty ratio
        correct_mask = ((pred_sorted >= 0.5) == y_sorted)
        if (~correct_mask).sum() > 0 and correct_mask.sum() > 0:
            ratio = unc_sorted[~correct_mask].mean() / unc_sorted[correct_mask].mean()
            ax.text(0.02, 0.95, f"Misclassified Uncertainty\nRatio: {ratio:.2f}x",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Add shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=NORMAL_COLOR, label="Normal (Correct)",
               markersize=6, linestyle="none"),
        Line2D([0], [0], marker="s", color=TUMOR_COLOR, label="Tumor (Correct)",
               markersize=6, linestyle="none"),
        Line2D([0], [0], marker="x", color=GRAY, label="Misclassified",
               markersize=8, linestyle="none", markeredgewidth=2),
    ]
    axes[1].legend(handles=legend_elements, loc="upper center",
                   fontsize=8, framealpha=0.9, ncol=3)

    plt.tight_layout()
    _save(fig, "fig_uncertainty")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("Generating paper figures (9 main + table)...")
    df = load_data()

    fig_spectra_combined(df)
    fig_roc(df)
    fig_adc_discriminant(df)
    fig_sensitivity(df)
    fig_map_nuts(df)
    fig_uncertainty(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("\nMain figures (6 generated here + directions + pixelwise = 8):")
    for f in sorted(OUTPUT_DIR.glob("fig_*.pdf")):
        print(f"  {f.name}")
    print("\n+ 1 table (AUC) = 9 items")
    print("+ 1 slot remaining for discretization/signal model")
