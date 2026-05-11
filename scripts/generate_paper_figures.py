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
    # Local font bump (~+50% over the global rcParams) per Stephan's review.
    with mpl.rc_context({
        "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        zones = [("pz", "Peripheral Zone"), ("tz", "Transition Zone")]
        legend_handles = None

        # Per-panel CV: mean across that panel's ROIs of (NUTS posterior std / mean).
        # Computed inline from features.csv columns nuts_std_D_<d> and nuts_D_<d>.
        nuts_mean_cols = [f"nuts_{c}" for c in D_COLS]
        nuts_std_cols = [f"nuts_std_{c}" for c in D_COLS]

        flier_style = dict(marker="o", markersize=4, markerfacecolor="none",
                           markeredgecolor="black", markeredgewidth=0.8, alpha=0.85)

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
                bp_map = ax.boxplot(
                    map_data, positions=x - width/2, widths=width*0.85,
                    patch_artist=True, medianprops=dict(color="black", linewidth=1.2),
                    flierprops=flier_style, manage_ticks=False,
                )
                for patch in bp_map["boxes"]:
                    patch.set_facecolor(MAP_COLOR); patch.set_alpha(0.5)

                # NUTS boxplot
                nuts_data = [sub[c].values for c in nuts_cols]
                bp_nuts = ax.boxplot(
                    nuts_data, positions=x + width/2, widths=width*0.85,
                    patch_artist=True, medianprops=dict(color="black", linewidth=1.2),
                    flierprops=flier_style, manage_ticks=False,
                )
                for patch in bp_nuts["boxes"]:
                    patch.set_facecolor(NUTS_COLOR); patch.set_alpha(0.5)

                ax.set_xticks(x)
                ax.set_xticklabels(D_LABELS)
                ax.set_xlabel("Diffusivity D (\u03bcm\u00b2/ms)")
                ax.set_ylabel("Spectral Fraction")
                ax.set_title(f"{tissue_label} {zone_title} (n={n})", fontweight="bold")
                ax.set_ylim(-0.02, 0.85)

                if legend_handles is None:
                    legend_handles = [bp_map["boxes"][0], bp_nuts["boxes"][0]]

                # Per-panel CV (NUTS posterior CV averaged over this panel's ROIs).
                cv_panel = (sub[nuts_std_cols].values
                            / np.maximum(sub[nuts_mean_cols].values, 1e-12)).mean(axis=0)
                for i, cv in enumerate(cv_panel):
                    color = MAP_COLOR if cv < 0.4 else TUMOR_COLOR if cv > 0.7 else NUTS_COLOR
                    ax.text(i, 0.81, f"CV={cv:.2f}",
                            ha="center", va="top", fontsize=9,
                            color=color, fontweight="bold")

        fig.legend(legend_handles, ["MAP", "NUTS"],
                   loc="upper center", ncol=2, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.965))
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
# Fig 7: Per-direction Comparison (Whole-ROI)
# =========================================================================
# Validates the trace-averaging approach: for a representative normal and
# tumor ROI from one demonstration patient, per-direction signal decays
# and per-direction MAP-estimated spectra agree within noise.
#
# Source data: pre-extracted Stephan tarball at /tmp/stephan_directions/.
# We use one patient/slice/zone pair (anonymized as "Demonstration Patient")
# with both a NormalPZ and a TumorPZ ROI from the same slice.
DIRECTIONS_DAT_DIR = Path("/tmp/stephan_directions/diffusion-spectrum-analysis")
DIRECTIONS_NORMAL_DAT = DIRECTIONS_DAT_DIR / "9283-Series12-Slice6-NormalPZ.dat"
DIRECTIONS_TUMOR_DAT = DIRECTIONS_DAT_DIR / "9283-Series12-Slice6-TumorPZ.dat"


def _parse_directions_dat(path):
    """Parse Stephan's .dat ROI-mean format.

    File layout (matches the project's pixel-binary pipeline; see
    `scripts/direction_comparison.py:68-107` for the analogue):

      - 46 image-mean rows, columns NR MEAN MAX MIN STDDEV AREA AREA FLUX.
      - Row 0 (image 1) = scanner reference / calibration scan (NOT
        protocol b=0). It is ~20% brighter than protocol b=0 and is dropped
        — same convention as the pixel-binary loader.
      - Rows 1-15 (images 2-16)  = direction 1, b descending 3500 -> 0.
      - Rows 16-30 (images 17-31) = direction 2, same descending order.
      - Rows 31-45 (images 32-46) = direction 3, same descending order.

    Each per-direction block ends with its own protocol b=0. After
    reversing, each direction yields a 15-point decay on the canonical
    b-grid [0, 250, ..., 3500] s/mm^2 — exactly matching the project
    grid in CLAUDE.md's critical parameters table.
    """
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0].isdigit():
                rows.append([float(x) for x in parts])
    arr = np.array(rows)              # shape (46, 8)
    means = arr[:, 1]                 # MEAN column

    # DROP image 1 (scanner reference). Reverse each 15-row block to get
    # b ascending 0 -> 3500. Each block's last raw row is protocol b=0.
    d1 = means[1:16][::-1]   # 15 values, b=[0, 250, ..., 3500]
    d2 = means[16:31][::-1]
    d3 = means[31:46][::-1]
    return d1, d2, d3        # each shape (15,)


def fig_directions():
    """Per-direction NUTS posterior spectra (MAP overlay) for one normal
    and one tumor ROI.

    Story: per-direction spectra agree within posterior uncertainty,
    validating the trace-averaging convention used elsewhere. The three
    per-direction posteriors collectively *are* the trace — no separate
    trace overlay is needed.

    Layout: 1x2 (spectra only — Normal left, Tumor right).
      Each panel shows three NUTS posteriors (one per gradient direction)
      as colored mean lines with 95% credible bands, plus the corresponding
      MAP point estimates as discrete jittered circular markers.

    NUTS cache: results from the 6 NUTS runs (3 directions x 2 ROIs) are
    cached to /tmp/fig_directions_nuts.npz on first run. To force a clean
    recompute (e.g. after changing inputs/parameters), simply delete the
    cache file:

        rm /tmp/fig_directions_nuts.npz

    Inputs are Stephan's whole-ROI mean .dat files for a single
    demonstration patient (one NormalPZ and one TumorPZ on the same slice).
    """
    # Canonical 15-point project b-grid (CLAUDE.md). Parser drops the
    # scanner-reference calibration so each direction has its own
    # protocol b=0 at index 0.
    b_values_smm2 = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750,
                              2000, 2250, 2500, 2750, 3000, 3250, 3500],
                             dtype=float)
    b_values_ms = b_values_smm2 / 1000.0  # convert s/mm^2 -> ms/um^2

    # ---- Parser sanity (per spec) ----
    # Load raw (un-normalized) decays per direction so we can verify the
    # calibration vs protocol-b=0 relationship and pass raw signals to
    # run_nuts_pixel (which normalizes internally).
    def _raw_means(path):
        rows = []
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("*"):
                    continue
                parts = s.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    rows.append([float(x) for x in parts])
        return np.array(rows)[:, 1]

    n_means = _raw_means(DIRECTIONS_NORMAL_DAT)
    t_means = _raw_means(DIRECTIONS_TUMOR_DAT)
    normal_calib = n_means[0]
    tumor_calib = t_means[0]
    normal_b0_trace = (n_means[15] + n_means[30] + n_means[45]) / 3.0
    tumor_b0_trace = (t_means[15] + t_means[30] + t_means[45]) / 3.0
    print("\n[fig_directions] Parser sanity")
    print("  Calibration vs protocol b=0:")
    print(f"    Normal: img1={normal_calib:.1f}, mean_b0={normal_b0_trace:.1f}, "
          f"ratio={normal_calib/normal_b0_trace:.3f}")
    print(f"    Tumor:  img1={tumor_calib:.1f}, mean_b0={tumor_b0_trace:.1f}, "
          f"ratio={tumor_calib/tumor_b0_trace:.3f}")
    # Per-direction b=0 agreement (should be within ~1-2%).
    for tag, arr, trace_b0 in [("Normal", n_means, normal_b0_trace),
                                ("Tumor ", t_means, tumor_b0_trace)]:
        rel = [abs(arr[idx] - trace_b0) / trace_b0 for idx in (15, 30, 45)]
        print(f"  {tag} per-direction b=0 |delta|/mean = "
              + ", ".join(f"{r*100:.2f}%" for r in rel))

    # ---- Load reversed per-direction decays (15 b-values, b ascending) ----
    n_d1, n_d2, n_d3 = _parse_directions_dat(DIRECTIONS_NORMAL_DAT)
    t_d1, t_d2, t_d3 = _parse_directions_dat(DIRECTIONS_TUMOR_DAT)
    # Trace decays still computed for parser sanity (not plotted).
    n_trace_raw = (n_d1 + n_d2 + n_d3) / 3.0
    t_trace_raw = (t_d1 + t_d2 + t_d3) / 3.0

    # Normalized per-direction signals (each by its own b=0). Used for both
    # the canonical MAP call and the closed-form parity check.
    n1 = n_d1 / n_d1[0]; n2 = n_d2 / n_d2[0]; n3 = n_d3 / n_d3[0]
    t1 = t_d1 / t_d1[0]; t2 = t_d2 / t_d2[0]; t3 = t_d3 / t_d3[0]

    # ---- MAP spectra via the project's canonical entry point ----
    # ProbabilisticModel.U_matrix() does np.exp(-np.outer(b_values, diffusivities)),
    # so b_values MUST be in ms/um^2 (= s/mm^2 / 1000) to match the
    # dimensionless decay model. Passing s/mm^2 here would produce all-but-
    # constant spectra — verified by the parity check below.
    from spectra_estimation_dmri.models.prob_model import ProbabilisticModel
    from types import SimpleNamespace

    def _project_map(signal_norm):
        prior_cfg = SimpleNamespace(type="ridge", strength=0.1)
        model = ProbabilisticModel(
            b_values=b_values_ms.tolist(),
            diffusivities=DIFFUSIVITIES.tolist(),
            prior_config=prior_cfg,
        )
        return model.map_estimate(signal_norm)

    n_specs = [_project_map(x) for x in (n1, n2, n3)]
    t_specs = [_project_map(x) for x in (t1, t2, t3)]

    # ---- One-shot parity check: project MAP vs old closed-form ridge ----
    # (Sanity per spec: should match to ~1e-10. If much bigger, units bug.)
    def _legacy_closed_form_map(signal_norm, ridge=0.1):
        U = np.exp(-np.outer(b_values_ms, DIFFUSIVITIES))
        spec = np.linalg.solve(
            U.T @ U + ridge * np.eye(U.shape[1]), U.T @ signal_norm
        )
        return np.maximum(spec, 0)

    legacy_n1 = _legacy_closed_form_map(n1, ridge=0.1)
    parity_delta = float(np.abs(legacy_n1 - n_specs[0]).max())
    print(f"\n[fig_directions] MAP parity check (project vs legacy closed-form):")
    print(f"  max |Delta| on Normal dir1: {parity_delta:.3e}")
    if parity_delta > 1e-3:
        raise RuntimeError(
            f"MAP parity check FAILED: max |Delta|={parity_delta:.3e}. "
            f"Likely a b-value unit error in ProbabilisticModel input."
        )

    # ---- NUTS posterior per direction (6 runs total, cached on disk) ----
    # Pass raw (un-normalized) signal; run_nuts_pixel normalizes internally.
    from spectra_estimation_dmri.pixelwise import run_nuts_pixel
    U_design = np.exp(-np.outer(b_values_ms, DIFFUSIVITIES))

    CACHE = Path("/tmp/fig_directions_nuts.npz")
    nuts_inputs = {
        "n_d1": n_d1, "n_d2": n_d2, "n_d3": n_d3,
        "t_d1": t_d1, "t_d2": t_d2, "t_d3": t_d3,
    }

    def _run_or_load_nuts():
        """Returns dict mapping key -> result dict (spectrum_mean, ..., r_hat_max)."""
        if CACHE.exists():
            print(f"[fig_directions] Loaded NUTS cache from {CACHE}")
            data = np.load(CACHE, allow_pickle=True)
            results = {}
            for key in nuts_inputs:
                r = {}
                for field in ("spectrum_mean", "spectrum_std",
                              "sigma_mean", "sigma_std", "snr", "r_hat_max"):
                    arr = data[f"{key}__{field}"]
                    # 0-d arrays -> scalar; 1-d arrays stay arrays.
                    r[field] = arr.item() if arr.ndim == 0 else arr
                results[key] = r
            return results

        print(f"[fig_directions] Cache miss — running 6 NUTS samplers (~3 min)")
        results = {}
        for key, raw_sig in nuts_inputs.items():
            results[key] = run_nuts_pixel(
                signal=raw_sig,
                U=U_design,
                ridge_strength=0.1,
                n_draws=2000, n_tune=200, n_chains=4, target_accept=0.95,
                random_seed=42,
            )
        # Persist for next run.
        flat = {}
        for key, r in results.items():
            for field, val in r.items():
                flat[f"{key}__{field}"] = np.asarray(val)
        np.savez(CACHE, **flat)
        print(f"[fig_directions] Saved NUTS cache to {CACHE}")
        return results

    nuts_results = _run_or_load_nuts()
    n_nuts = [nuts_results["n_d1"], nuts_results["n_d2"], nuts_results["n_d3"]]
    t_nuts = [nuts_results["t_d1"], nuts_results["t_d2"], nuts_results["t_d3"]]

    # ---- NUTS convergence printout (6 runs) ----
    print("\n[fig_directions] NUTS convergence (R-hat max) and D=0.25 posterior:")
    rhat_all = []
    for roi_name, dir_results in [("Normal", n_nuts), ("Tumor", t_nuts)]:
        for d, res in enumerate(dir_results, 1):
            rhat_all.append(float(res["r_hat_max"]))
            print(f"  {roi_name} dir{d}: R-hat_max={float(res['r_hat_max']):.3f}, "
                  f"D=0.25 mean={res['spectrum_mean'][0]:.3f} "
                  f"+- {res['spectrum_std'][0]:.3f}")
    print(f"  Overall R-hat max across 6 runs: {max(rhat_all):.3f}")

    # ---- MAP vs NUTS comparison ----
    print("\n[fig_directions] MAP vs NUTS posterior mean (max |Delta|):")
    clinical_idx = [i for i, d in enumerate(DIFFUSIVITIES) if d <= 3.0]
    for roi_name, maps, nuts_list in [
        ("Normal", n_specs, n_nuts), ("Tumor", t_specs, t_nuts)
    ]:
        for d_i, (m, n) in enumerate(zip(maps, nuts_list), 1):
            diff = np.abs(m - n["spectrum_mean"])
            clin_max = diff[clinical_idx].max()
            all_max = diff.max()
            print(f"  {roi_name} dir{d_i}: max |Delta| clinical(D<=3)={clin_max:.3f}, "
                  f"all-bins={all_max:.3f}")

    # Per-direction MAP spread (useful for caption).
    n_spread = np.std(np.stack(n_specs, axis=0), axis=0)
    t_spread = np.std(np.stack(t_specs, axis=0), axis=0)
    print(f"  Normal cross-direction MAP std (per bin): "
          + ", ".join(f"{v:.3f}" for v in n_spread))
    print(f"  Tumor  cross-direction MAP std (per bin): "
          + ", ".join(f"{v:.3f}" for v in t_spread))

    # ---- Plot (1x2: spectra only) ----
    # Three colorblind-safe-ish colors distinct from the tissue palette
    # (blue NORMAL_COLOR / red TUMOR_COLOR) and the inference palette
    # (green MAP_COLOR / orange NUTS_COLOR): purple, magenta, teal-green.
    DIR_COLORS = ["#5e3c99", "#d01c8b", "#1a9850"]
    dir_labels = ["Direction 1", "Direction 2", "Direction 3"]
    n_color = NORMAL_COLOR
    t_color = TUMOR_COLOR

    with mpl.rc_context({
        "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        x = np.arange(len(DIFFUSIVITIES), dtype=float)

        for ax, nuts_list, maps, accent, tissue_label in [
            (axes[0], n_nuts, n_specs, n_color, "Normal ROI"),
            (axes[1], t_nuts, t_specs, t_color, "Tumor ROI"),
        ]:
            for d_idx, (res, map_spec) in enumerate(zip(nuts_list, maps)):
                color = DIR_COLORS[d_idx]
                mu = np.asarray(res["spectrum_mean"])
                sd = np.asarray(res["spectrum_std"])
                lo = np.maximum(mu - 1.96 * sd, 0.0)
                hi = mu + 1.96 * sd

                # 95% credible band per direction.
                ax.fill_between(x, lo, hi, color=color, alpha=0.18,
                                linewidth=0)
                # NUTS posterior mean as a solid line (no markers, so it
                # reads as a continuous summary curve).
                ax.plot(x, mu, color=color, linewidth=1.8, alpha=0.95)

                # MAP markers — discrete circles ONLY (no connecting line),
                # x-jittered so all three directions are individually
                # visible at each D-bin instead of stacking on top of each
                # other.
                x_jit = x + (d_idx - 1) * 0.15
                ax.plot(x_jit, map_spec,
                        linestyle="none",
                        marker="o", markersize=7,
                        markerfacecolor=color,
                        markeredgecolor="black", markeredgewidth=0.8,
                        alpha=0.95, zorder=6)

            ax.set_xticks(x)
            ax.set_xticklabels(D_LABELS)
            ax.set_xlabel("Diffusivity D (μm$^{2}$/ms)")
            ax.set_ylabel("Spectral fraction R$_j$")
            ax.set_title(tissue_label, fontweight="bold", color=accent)
            ymax = max(
                max((np.asarray(r["spectrum_mean"])
                     + 1.96 * np.asarray(r["spectrum_std"])).max()
                    for r in nuts_list),
                max(s.max() for s in maps),
            ) * 1.18
            ax.set_ylim(0, ymax)

        # Figure-level legend: 3 directions + MAP marker + NUTS band.
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_handles = []
        for d_idx in range(3):
            legend_handles.append(
                Line2D([0], [0], color=DIR_COLORS[d_idx], linewidth=2.0,
                       marker="o", markersize=7,
                       markerfacecolor=DIR_COLORS[d_idx],
                       markeredgecolor="black", markeredgewidth=0.8,
                       label=dir_labels[d_idx])
            )
        legend_handles.append(
            Line2D([0], [0], color="black", linestyle="none",
                   marker="o", markersize=7,
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=0.8, label="MAP estimate")
        )
        legend_handles.append(
            Patch(facecolor="#888888", alpha=0.25, label="NUTS 95% CI")
        )
        fig.legend(legend_handles, [h.get_label() for h in legend_handles],
                   loc="upper center", ncol=5, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.02))

        fig.tight_layout(rect=(0, 0, 1, 0.93))
        _save(fig, "fig_directions")


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
    fig_directions()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("\nMain figures (6 generated here + directions + pixelwise = 8):")
    for f in sorted(OUTPUT_DIR.glob("fig_*.pdf")):
        print(f"  {f.name}")
    print("\n+ 1 table (AUC) = 9 items")
    print("+ 1 slot remaining for discretization/signal model")
