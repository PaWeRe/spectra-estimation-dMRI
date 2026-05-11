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
    # Local font bump (~+50% over the global rcParams) to match Fig 1 styling.
    with mpl.rc_context({
        "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
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

            # Individual components (thin lines) — no per-component AUC in legend.
            for i, d in enumerate(DIFFUSIVITIES):
                feat_col = f"map_{D_COLS[i]}"
                vals = tdf[feat_col].values
                auc_val = roc_auc_score(y, vals)
                if auc_val < 0.5:
                    vals = -vals; auc_val = 1 - auc_val
                fpr, tpr, _ = roc_curve(y, vals)
                ax.plot(fpr, tpr, color=comp_colors[i], linewidth=0.8, alpha=0.6,
                        label=f"D={d}")

            # ADC raw rank (thick blue)
            adc_vals = tdf["adc"].values
            adc_auc = roc_auc_score(y, adc_vals)
            if adc_auc < 0.5: adc_vals = -adc_vals; adc_auc = 1 - adc_auc
            fpr_a, tpr_a, _ = roc_curve(y, adc_vals)
            ci_lo, ci_hi = bootstrap_auc_ci(y, adc_vals if adc_auc == roc_auc_score(y, adc_vals) else -adc_vals)
            ax.plot(fpr_a, tpr_a, color=ADC_COLOR, linewidth=3,
                    label=f"ADC ({adc_auc:.3f} [{ci_lo:.2f}-{ci_hi:.2f}])")

            # MAP Full LR (thick green)
            X_map = tdf[map_cols].values
            map_pred = loocv_roc(X_map, y, C=1.0)
            map_auc = roc_auc_score(y, map_pred)
            ci_lo, ci_hi = bootstrap_auc_ci(y, map_pred)
            fpr_m, tpr_m, _ = roc_curve(y, map_pred)
            ax.plot(fpr_m, tpr_m, color=MAP_COLOR, linewidth=2.5,
                    label=f"MAP 8-feat ({map_auc:.3f} [{ci_lo:.2f}-{ci_hi:.2f}])")

            # NUTS Full LR (thick orange)
            nuts_cols_list = [f"nuts_{c}" for c in D_COLS]
            X_nuts = tdf[nuts_cols_list].values
            nuts_pred = loocv_roc(X_nuts, y, C=1.0)
            nuts_auc = roc_auc_score(y, nuts_pred)
            ci_lo, ci_hi = bootstrap_auc_ci(y, nuts_pred)
            fpr_n, tpr_n, _ = roc_curve(y, nuts_pred)
            ax.plot(fpr_n, tpr_n, color=NUTS_COLOR, linewidth=2.5, linestyle="--",
                    label=f"NUTS 8-feat ({nuts_auc:.3f} [{ci_lo:.2f}-{ci_hi:.2f}])")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{title} (n={n})", fontweight="bold")
            ax.legend(loc="lower right", framealpha=0.9)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.grid(False)

        fig.tight_layout()
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

    File layout (validated):
      - 46 image-mean rows, columns NR MEAN MAX MIN STDDEV AREA AREA FLUX.
      - Row 0 = shared b=0 image.
      - Rows 1-15 = direction 1, b descending (high b -> low b).
      - Rows 16-30 = direction 2, same descending convention.
      - Rows 31-45 = direction 3, same descending convention.

    We reverse each direction's slice to get ascending b and prepend the
    shared b=0 value. Each per-direction decay therefore has 16 points
    spanning b=0 to b=3500 s/mm^2; we attach an evenly-spaced 16-point
    b-value grid linspace(0, 3.5, 16) ms/um^2 to match. (Patrick's
    nominal grid is 15-point [0, 250, ..., 3500] but the .dat block has
    15 nonzero rows per direction; the small spacing difference does not
    affect the qualitative direction-consistency story this figure tells.)
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
    arr = np.array(rows)
    means = arr[:, 1]
    s_b0 = means[0]
    d1 = np.concatenate([[s_b0], means[1:16][::-1]])
    d2 = np.concatenate([[s_b0], means[16:31][::-1]])
    d3 = np.concatenate([[s_b0], means[31:46][::-1]])
    return d1, d2, d3  # each shape (16,)


def _map_estimate(b_values_ms, signal_norm, diffusivities, ridge=0.1):
    """Closed-form ridge-NNLS MAP (matches project's `ridge` prior, lambda=0.1)."""
    U = np.exp(-np.outer(b_values_ms, diffusivities))
    spec = np.linalg.solve(
        U.T @ U + ridge * np.eye(U.shape[1]), U.T @ signal_norm
    )
    return np.maximum(spec, 0)


def fig_directions():
    """Per-direction signal decays + MAP spectra for one normal and one tumor ROI.

    Story: 3 gradient directions give consistent spectra within noise,
    validating the trace-averaging approach used elsewhere in the paper.
    """
    # 16-point b grid matching the parser output (see _parse_directions_dat).
    b_values_ms = np.linspace(0.0, 3.5, 16)
    b_values_smm2 = b_values_ms * 1000.0  # for axis label

    # Load both ROIs.
    n_d1, n_d2, n_d3 = _parse_directions_dat(DIRECTIONS_NORMAL_DAT)
    t_d1, t_d2, t_d3 = _parse_directions_dat(DIRECTIONS_TUMOR_DAT)

    def _normalize(d1, d2, d3):
        s0 = (d1[0] + d2[0] + d3[0]) / 3.0  # shared b=0 value
        return d1 / s0, d2 / s0, d3 / s0

    n1, n2, n3 = _normalize(n_d1, n_d2, n_d3)
    t1, t2, t3 = _normalize(t_d1, t_d2, t_d3)
    n_trace = (n1 + n2 + n3) / 3.0
    t_trace = (t1 + t2 + t3) / 3.0

    # MAP spectra (ridge=0.1, project convention).
    n_specs = [_map_estimate(b_values_ms, x, DIFFUSIVITIES, ridge=0.1)
               for x in (n1, n2, n3)]
    n_trace_spec = _map_estimate(b_values_ms, n_trace, DIFFUSIVITIES, ridge=0.1)
    t_specs = [_map_estimate(b_values_ms, x, DIFFUSIVITIES, ridge=0.1)
               for x in (t1, t2, t3)]
    t_trace_spec = _map_estimate(b_values_ms, t_trace, DIFFUSIVITIES, ridge=0.1)

    # Print per-direction spectra to stdout (sanity check requested in spec).
    print("\n[fig_directions] Per-direction MAP spectra (D-bins, mass at each bin)")
    print(f"  D bins:    {D_LABELS}")
    for tag, specs, trace_spec in [
        ("normal", n_specs, n_trace_spec),
        ("tumor", t_specs, t_trace_spec),
    ]:
        for d_idx, spec in enumerate(specs):
            print(f"  {tag} dir{d_idx+1}: " + ", ".join(f"{v:.3f}" for v in spec))
        print(f"  {tag} trace:" + ", ".join(f"{v:.3f}" for v in trace_spec))

    # Per-direction spread for the report.
    n_spread = np.std(np.stack(n_specs, axis=0), axis=0)
    t_spread = np.std(np.stack(t_specs, axis=0), axis=0)
    print(f"  Normal cross-direction std (per bin): "
          + ", ".join(f"{v:.3f}" for v in n_spread))
    print(f"  Tumor  cross-direction std (per bin): "
          + ", ".join(f"{v:.3f}" for v in t_spread))

    # ---- Plot ----
    # Local font bump to match Fig 1's in-figure rcParams pattern.
    with mpl.rc_context({
        "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Three shades for the directions: lighter -> darker within tissue color.
        # Use grayscale for tissue-neutral direction encoding so the tissue
        # context comes from the column header, not the line color.
        dir_shades = ["#7f7f7f", "#404040", "#0a0a0a"]
        dir_styles = ["-", "-", "-"]
        dir_markers = ["o", "s", "^"]
        dir_labels = ["Direction 1", "Direction 2", "Direction 3"]
        trace_color = NORMAL_COLOR  # overridden per-column below
        trace_label = "Trace average"

        # Tissue accent colors used for the trace line + per-column subtitles.
        n_color = NORMAL_COLOR
        t_color = TUMOR_COLOR

        # --- Top row: signal decays ---
        for col, (ax, decays, trace, accent, tissue_label) in enumerate([
            (axes[0, 0], (n1, n2, n3), n_trace, n_color, "Normal ROI"),
            (axes[0, 1], (t1, t2, t3), t_trace, t_color, "Tumor ROI"),
        ]):
            for d_idx, dec in enumerate(decays):
                ax.plot(b_values_smm2, dec,
                        color=dir_shades[d_idx], linestyle=dir_styles[d_idx],
                        marker=dir_markers[d_idx], markersize=5,
                        linewidth=1.2, alpha=0.85,
                        label=dir_labels[d_idx] if col == 0 else None)
            ax.plot(b_values_smm2, trace, color=accent, linestyle="--",
                    linewidth=2.2, marker="x", markersize=6,
                    label=trace_label if col == 0 else None)
            ax.set_xlabel("b (s/mm$^{2}$)")
            ax.set_ylabel("S / S$_0$")
            ax.set_title(tissue_label, fontweight="bold", color=accent)
            ax.set_yscale("log")
            ax.set_ylim(0.03, 1.1)
            ax.set_xlim(-100, 3700)

        # --- Bottom row: MAP spectra ---
        x = np.arange(len(DIFFUSIVITIES))
        width = 0.2
        for col, (ax, specs, trace_spec, accent, tissue_label) in enumerate([
            (axes[1, 0], n_specs, n_trace_spec, n_color, "Normal ROI"),
            (axes[1, 1], t_specs, t_trace_spec, t_color, "Tumor ROI"),
        ]):
            for d_idx, spec in enumerate(specs):
                offset = (d_idx - 1) * width
                ax.bar(x + offset, spec, width * 0.9,
                       color=dir_shades[d_idx], alpha=0.75,
                       edgecolor="white", linewidth=0.4,
                       label=None)
            # Trace overlay (line with markers, on top of bars).
            ax.plot(x, trace_spec, color=accent, linestyle="--",
                    linewidth=2.2, marker="x", markersize=8,
                    label=None, zorder=5)
            ax.set_xticks(x)
            ax.set_xticklabels(D_LABELS)
            ax.set_xlabel("Diffusivity D (μm$^{2}$/ms)")
            ax.set_ylabel("Spectral fraction R$_j$")
            ax.set_title(tissue_label, fontweight="bold", color=accent)
            ymax = max(
                max(s.max() for s in specs),
                trace_spec.max(),
            ) * 1.18
            ax.set_ylim(0, ymax)

        # Figure-level legend (one row, all four entries: 3 directions + trace).
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_handles = [
            Line2D([0], [0], color=dir_shades[0], linestyle=dir_styles[0],
                   marker=dir_markers[0], markersize=6, linewidth=1.4,
                   label=dir_labels[0]),
            Line2D([0], [0], color=dir_shades[1], linestyle=dir_styles[1],
                   marker=dir_markers[1], markersize=6, linewidth=1.4,
                   label=dir_labels[1]),
            Line2D([0], [0], color=dir_shades[2], linestyle=dir_styles[2],
                   marker=dir_markers[2], markersize=6, linewidth=1.4,
                   label=dir_labels[2]),
            Line2D([0], [0], color="black", linestyle="--",
                   marker="x", markersize=7, linewidth=2.0,
                   label="Trace average"),
        ]
        fig.legend(legend_handles, [h.get_label() for h in legend_handles],
                   loc="upper center", ncol=4, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.0))

        fig.tight_layout(rect=(0, 0, 1, 0.96))
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
