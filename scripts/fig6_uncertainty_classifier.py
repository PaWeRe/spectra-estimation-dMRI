"""
Figure 6 (rework): Uncertainty-aware tumor classifier.

Narrative (Pillar 3 -- the exploratory "probability of cancer" idea, Sandy):
  A point-estimate classifier returns a single label (tumor / normal). The fully
  Bayesian NUTS posterior lets us do something a MAP point estimate cannot:
  propagate the ENTIRE posterior over the diffusivity spectrum through the
  trained classifier, turning one prediction into a *distribution* over the
  cancer probability P(tumor). We can then report not just "tumor: yes/no" but
  "P(cancer) = 0.82, 90% credible interval [0.61, 0.94]" -- and crucially the
  width of that interval is diagnostically meaningful:

    (i)  it widens near the decision boundary (the model flags the cases it is
         genuinely unsure about), and
    (ii) misclassified ROIs carry systematically wider intervals than correctly
         classified ones.

THE KEY METHODOLOGICAL POINT (vs the rejected-ISMRM version):
  The old figure drew error bars equal to the mean per-bin posterior std, scaled
  by an arbitrary x2 -- a feature-space heuristic, NOT a propagated predictive
  distribution. Here we do the propagation correctly: for each held-out ROI we
  fit the LOOCV classifier on the *posterior-mean* features of the other ROIs
  (identical pipeline to Fig 2 / Fig 3), then push all NUTS posterior DRAWS of
  the held-out ROI through that fixed classifier to obtain the posterior of
  P(tumor). Error bars are real 90% credible intervals on the probability.

Estimator: NUTS only -- this figure exists *because* NUTS gives a full posterior;
MAP cannot produce these intervals. 8 spectral fractions, C = 1.0, standardized,
matching the Fig 2 detection pipeline exactly so the labels agree with the ROC.

Detection only (PZ, TZ). GGG grade classification is NOT shown: N = 29 valid GGG
(9 high-grade) is too small for a credible classifier, the same reason it was
dropped from Fig 2. The third panel instead makes the figure's thesis explicit:
propagated uncertainty vs distance from the decision boundary, pooled across
zones, with misclassified ROIs highlighted.

Output:
  paper/figures/fig6_v1.png
  paper/figures/fig6_v1.pdf
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
from sklearn.model_selection import LeaveOneOut

import arviz as az

# Reuse the exact hashing + dataset loader that names the .nc files.
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "src"))
from spectra_estimation_dmri.biomarkers.recompute import (  # noqa: E402
    load_dataset, compute_spectra_id, DIFFUSIVITIES, NC_DIR, SIGNAL_JSON,
    METADATA_CSV,
)

REPO_ROOT = "/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI"
FEATURES_CSV = os.path.join(REPO_ROOT, "results/biomarkers/features.csv")
OUT_DIR = os.path.join(REPO_ROOT, "paper/figures")

NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
LR_C = 1.0
CI_LO, CI_HI = 5.0, 95.0           # 90% credible interval on P(tumor)
RNG = np.random.RandomState(42)

# --- Typography: match fig2 / fig3 2-up scale ---
mpl.rcParams.update({
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 17,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})

# --- Palette (project convention) ---
COLOR_TUMOR = "#d62728"    # red
COLOR_NORMAL = "#1f77b4"   # blue
COLOR_MISS = "#000000"     # black ring around misclassified
GRAY = "#888888"


def load_draws(rois, nc_dir):
    """roi_id -> (n_draws, 8) NUTS posterior draws, normalized to fractions."""
    var_names = [f"diff_{d:.2f}" for d in DIFFUSIVITIES]
    draws = {}
    for roi in rois:
        sid = compute_spectra_id(roi["signal"], roi["b_values"], roi["snr"])
        p = os.path.join(REPO_ROOT, nc_dir, f"{sid}.nc")
        if not os.path.exists(p):
            continue
        idata = az.from_netcdf(p)
        cols = [idata.posterior[vn].values.flatten() for vn in var_names]
        S = np.column_stack(cols)
        S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-10)
        draws[roi["roi_id"]] = S
    return draws


def propagate_zone(df_zone, draws, C=LR_C):
    """LOOCV: fit on posterior-mean features of the other ROIs, propagate the
    held-out ROI's full posterior draws through the fixed classifier.

    Returns a DataFrame (one row per ROI) with:
      p_point  -- P(tumor) at the posterior-mean features (the Fig-2 prediction)
      p_med    -- median of propagated P(tumor) draws
      p_lo,p_hi-- 90% credible interval on P(tumor)
      p_std    -- std of propagated P(tumor) draws
      ci_width -- p_hi - p_lo
    """
    df_zone = df_zone.reset_index(drop=True)
    y = df_zone["is_tumor"].astype(int).values
    Xmean = df_zone[NUTS_COLS].values
    roi_ids = df_zone["roi_id"].values
    n = len(y)

    p_point = np.zeros(n)
    p_med = np.zeros(n)
    p_lo = np.zeros(n)
    p_hi = np.zeros(n)
    p_std = np.zeros(n)
    z_std = np.zeros(n)   # logit-space spread (removes the sigmoid geometry)

    loo = LeaveOneOut()
    for tr, te in loo.split(Xmean):
        i = te[0]
        sc = StandardScaler().fit(Xmean[tr])
        clf = LogisticRegression(C=C, max_iter=2000, random_state=42,
                                 solver="lbfgs").fit(sc.transform(Xmean[tr]), y[tr])
        p_point[i] = clf.predict_proba(sc.transform(Xmean[i:i + 1]))[0, 1]
        D = draws[roi_ids[i]]
        probs = clf.predict_proba(sc.transform(D))[:, 1]
        p_med[i] = np.median(probs)
        p_lo[i], p_hi[i] = np.percentile(probs, [CI_LO, CI_HI])
        p_std[i] = probs.std()
        # logit of each draw = the classifier's linear predictor; its spread is
        # the "genuine" uncertainty before the sigmoid compresses it near 0/1.
        z = np.log(np.clip(probs, 1e-6, 1 - 1e-6)
                   / np.clip(1 - probs, 1e-6, 1 - 1e-6))
        z_std[i] = z.std()

    out = df_zone[["roi_id", "patient", "region", "is_tumor", "ggg", "gs",
                   "adc"]].copy()
    out["y"] = y
    out["p_point"] = p_point
    out["p_med"] = p_med
    out["p_lo"] = p_lo
    out["p_hi"] = p_hi
    out["p_std"] = p_std
    out["z_std"] = z_std
    out["ci_width"] = p_hi - p_lo
    out["dist"] = np.abs(p_point - 0.5)
    out["correct"] = (p_point >= 0.5).astype(int) == y
    return out


def diagnostics(res, label):
    print(f"\n=== {label}  (n = {len(res)}) ===")
    n_miss = int((~res["correct"]).sum())
    print(f"  misclassified: {n_miss}/{len(res)}")
    # Calibration claim 1: uncertainty vs distance from boundary
    r_w, p_w = stats.spearmanr(res["dist"], res["ci_width"])
    r_s, p_s = stats.spearmanr(res["dist"], res["p_std"])
    print(f"  Spearman rho(dist-to-boundary, CI width) = {r_w:+.3f} (p={p_w:.1e})")
    print(f"  Spearman rho(dist-to-boundary, P std)    = {r_s:+.3f} (p={p_s:.1e})")
    # Calibration claim 2: misclassified carry more uncertainty
    if n_miss > 0 and res["correct"].sum() > 0:
        unc_c = res.loc[res["correct"], "ci_width"]
        unc_m = res.loc[~res["correct"], "ci_width"]
        ratio = unc_m.mean() / unc_c.mean()
        t, pt = stats.ttest_ind(unc_m, unc_c, equal_var=False)
        print(f"  CI width  correct {unc_c.mean():.3f} | miss {unc_m.mean():.3f} "
              f"| ratio {ratio:.2f}x | Welch p={pt:.3f}")
        sc_ = res.loc[res["correct"], "p_std"]
        sm_ = res.loc[~res["correct"], "p_std"]
        print(f"  P std     correct {sc_.mean():.3f} | miss {sm_.mean():.3f} "
              f"| ratio {sm_.mean()/sc_.mean():.2f}x")
        # logit-space: removes the sigmoid geometry. If misclassified are wider
        # HERE too, the uncertainty signal is genuine (not just position).
        zc = res.loc[res["correct"], "z_std"]
        zm = res.loc[~res["correct"], "z_std"]
        rz, pz = stats.spearmanr(res["dist"], res["z_std"])
        print(f"  logit std correct {zc.mean():.3f} | miss {zm.mean():.3f} "
              f"| ratio {zm.mean()/zc.mean():.2f}x  ||  "
              f"rho(dist, logit std) = {rz:+.3f} (p={pz:.1e})")
    return r_w


def report_misclassified(pooled):
    """List misclassified ROIs, sorted by distance from boundary (most
    confidently wrong first) -- these are the failure mode of the uncertainty
    story (confident AND wrong, low interval width)."""
    miss = pooled[~pooled["correct"]].sort_values("dist", ascending=False)
    print(f"\n=== Misclassified ROIs ({len(miss)}), most-confident first ===")
    print(f"  {'patient':>10s} {'zone':>5s} {'truth':>7s} {'ggg':>4s} "
          f"{'gs':>6s} {'ADC':>6s} {'P(tum)':>7s} {'CIw':>6s} {'dist':>6s}")
    for _, r in miss.iterrows():
        truth = "tumor" if r["y"] == 1 else "normal"
        ggg = "" if pd.isna(r["ggg"]) else f"{int(r['ggg'])}"
        gs = "" if pd.isna(r["gs"]) else str(r["gs"])
        print(f"  {str(r['patient']):>10s} {r['region']:>5s} {truth:>7s} "
              f"{ggg:>4s} {gs:>6s} {r['adc']:6.3f} {r['p_point']:7.2f} "
              f"{r['ci_width']:6.2f} {r['dist']:6.2f}")
    # confidently wrong = far from boundary, narrow interval
    conf = miss[(miss["dist"] > 0.30)]
    print(f"  -> {len(conf)} 'confidently misclassified' (|P-0.5| > 0.30): "
          f"mean CIw {conf['ci_width'].mean():.3f} "
          f"vs all-correct {pooled.loc[pooled['correct'],'ci_width'].mean():.3f}")
    return miss


def plot_sorted_panel(ax, res, title):
    """Sorted-prediction panel with propagated 90% CIs as error bars."""
    r = res.sort_values("p_point").reset_index(drop=True)
    x = np.arange(len(r))
    y = r["y"].values
    p = r["p_point"].values
    lo = r["p_lo"].values
    hi = r["p_hi"].values
    correct = r["correct"].values

    yerr = np.vstack([np.maximum(p - lo, 0), np.maximum(hi - p, 0)])

    for cls, color, marker, lbl in [(0, COLOR_NORMAL, "o", "Normal"),
                                    (1, COLOR_TUMOR, "s", "Tumor")]:
        m = y == cls
        if m.sum() == 0:
            continue
        ax.errorbar(x[m], p[m], yerr=yerr[:, m], fmt=marker, color=color,
                    ecolor=color, elinewidth=1.3, capsize=2.5, markersize=7,
                    alpha=0.9, markeredgecolor="white", markeredgewidth=0.6,
                    linestyle="none", zorder=3)

    # misclassified: open black ring overlaid
    mm = ~correct
    if mm.sum() > 0:
        ax.scatter(x[mm], p[mm], s=190, facecolors="none",
                   edgecolors=COLOR_MISS, linewidths=1.8, zorder=4)

    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("ROI (sorted by predicted probability)")
    ax.set_ylabel("P(tumor)")
    ax.set_title(f"{title}  (n = {len(r)})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def plot_misclassified_panel(ax, pooled, rng=RNG):
    """Pooled: 90% CI width of P(tumor), correct vs misclassified ROIs.

    This is the figure's non-circular headline result: the propagated
    probability interval is systematically wider for the ROIs the classifier
    gets wrong. (Boundary-widening -- partly logistic-link geometry -- is shown
    directly by the fanning error bars in the PZ/TZ panels and quantified in the
    caption; we do not re-plot it here as a scatter.)
    """
    groups = [("Correct", pooled[pooled["correct"]]),
              ("Misclassified", pooled[~pooled["correct"]])]
    positions = [0, 1]

    data = [g["ci_width"].values for _, g in groups]
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color="0.15", linewidth=2),
                    whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"),
                    boxprops=dict(facecolor="0.92", edgecolor="0.4"), zorder=2)

    # jittered points colored by true tissue
    for pos, (_, g) in zip(positions, groups):
        for cls, color, marker in [(0, COLOR_NORMAL, "o"), (1, COLOR_TUMOR, "s")]:
            sub = g[g["y"] == cls]
            if len(sub) == 0:
                continue
            jit = rng.uniform(-0.16, 0.16, len(sub))
            ax.scatter(pos + jit, sub["ci_width"].values, c=color, marker=marker,
                       s=42, alpha=0.85, edgecolors="white", linewidths=0.5,
                       zorder=3)

    unc_c = groups[0][1]["ci_width"]
    unc_m = groups[1][1]["ci_width"]
    ratio = unc_m.mean() / unc_c.mean()
    _, pval = stats.ttest_ind(unc_m, unc_c, equal_var=False)
    star = "" if pval >= 0.05 else ("*" if pval >= 1e-2 else "**")

    # annotate the ratio with a bracket
    ytop = pooled["ci_width"].max()
    ax.plot([0, 0, 1, 1], [ytop * 1.04, ytop * 1.10, ytop * 1.10, ytop * 1.04],
            color="0.2", linewidth=1.3)
    ax.text(0.5, ytop * 1.13, f"{ratio:.1f}×{star}", ha="center", va="bottom",
            fontsize=16, color="0.15")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Correct\n(n = {len(unc_c)})",
                        f"Misclassified\n(n = {len(unc_m)})"])
    ax.set_ylabel("90% CI width of P(tumor)")
    ax.set_title("Pooled PZ + TZ")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.02, ytop * 1.28)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feats = pd.read_csv(FEATURES_CSV)
    rois = load_dataset(os.path.join(REPO_ROOT, SIGNAL_JSON),
                        os.path.join(REPO_ROOT, METADATA_CSV))
    print(f"Loading NUTS posterior draws for {len(rois)} ROIs ...")
    draws = load_draws(rois, NC_DIR)
    print(f"  loaded draws for {len(draws)} ROIs")

    zones = [("pz", "PZ: tumor detection"), ("tz", "TZ: tumor detection")]
    results = {}
    for zkey, _ in zones:
        dz = feats[feats["zone"] == zkey].copy()
        dz = dz[dz["roi_id"].isin(draws.keys())].copy()
        results[zkey] = propagate_zone(dz, draws)

    for zkey, zlabel in zones:
        diagnostics(results[zkey], zlabel)
    pooled = pd.concat([results["pz"], results["tz"]], ignore_index=True)
    diagnostics(pooled, "POOLED PZ + TZ")
    report_misclassified(pooled)

    # ---------- figure: 2x2 (PZ, TZ top; box plot bottom-left; 4th empty) ----
    # Minimal legend on top (matches Fig 2 / Fig 3 convention); the bottom-right
    # quadrant is left intentionally empty.
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.22, top=0.89,
                          width_ratios=[1, 1], height_ratios=[1, 1])
    ax_pz = fig.add_subplot(gs[0, 0])
    ax_tz = fig.add_subplot(gs[0, 1])
    ax_box = fig.add_subplot(gs[1, 0])
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis("off")

    plot_sorted_panel(ax_pz, results["pz"], "PZ: tumor detection")
    plot_sorted_panel(ax_tz, results["tz"], "TZ: tumor detection")
    plot_misclassified_panel(ax_box, pooled)

    legend_handles = [
        Line2D([0], [0], marker="o", color=COLOR_NORMAL, linestyle="none",
               markersize=11, markeredgecolor="white", label="Normal"),
        Line2D([0], [0], marker="s", color=COLOR_TUMOR, linestyle="none",
               markersize=11, markeredgecolor="white", label="Tumor"),
        Line2D([0], [0], marker="o", color="none", linestyle="none",
               markersize=15, markeredgecolor=COLOR_MISS, markeredgewidth=1.8,
               label="Misclassified"),
        Line2D([0], [0], color=GRAY, linestyle="--", linewidth=1.8,
               label="Decision boundary (0.5)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 0.965),
               columnspacing=2.0, handlelength=2.0, fontsize=17)

    png = os.path.join(OUT_DIR, "fig6_v1.png")
    pdf = os.path.join(OUT_DIR, "fig6_v1.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
