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
  paper/figures/fig6_v2.png
  paper/figures/fig6_v2.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
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
from spectra_estimation_dmri.visualization.paper_style import (  # noqa: E402
    apply_style, COLORS,
)

REPO_ROOT = "/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI"
FEATURES_CSV = os.path.join(REPO_ROOT, "results/biomarkers/features.csv")
OUT_DIR = os.path.join(REPO_ROOT, "paper/figures")

NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
LR_C = 1.0
CI_LO, CI_HI = 5.0, 95.0           # 90% credible interval on P(tumor)
RNG = np.random.RandomState(42)

# --- Shared manuscript typography (locks legend == title size, labels 20 /
# ticks 18 / title 17 / legend 17). Replaces the script's own rcParams. ---
# This figure OVERRIDES apply_style so that axis labels do NOT exceed the
# title/legend size: lead author wants label == title == legend == FONT_SIZE
# (the grid default ships labels at 20 > title 17, which crowded the panels).
apply_style("grid")
FONT_SIZE = 17
plt.rcParams.update({
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    # ticks slightly smaller so the harmonised label/title/legend size reads as
    # the dominant typography (kept below FONT_SIZE).
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

# --- Palette (lead-author redesign 2026-06-05, contrast pushed harder) ---
# VERY pale low-saturation pastel = correctly classified (recedes into the
# background); FULLY saturated + DARKER distinct shade = MISCLASSIFIED, drawn
# LARGER with a dark/black edge so misclassified ROIs unmistakably pop. Hue
# family is preserved (normal = blue, tumor = red) within each correctness
# class; the light/bold + size + edge separation carries the miss flag.
COLOR_NORMAL = "#cfe8f5"        # very pale blue   : correct normal
COLOR_TUMOR = "#fcd0cf"         # very pale red    : correct tumor
COLOR_NORMAL_MISS = COLORS["normal_dark"]  # deep navy (#0b3d61) : misclassified normal
COLOR_TUMOR_MISS = COLORS["tumor_dark"]    # deep maroon (#7f1416): misclassified tumor
MISS_EDGE = "#000000"           # black ring on misclassified markers
GRAY = "#666666"


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


def plot_sorted_panel(ax, res, zone_label, xlabel=True):
    """Sorted-prediction panel with propagated 90% CIs as error bars.

    Colour encodes BOTH true tissue (hue) and correctness (pale vs bold):
      VERY pale blue/red  = correctly classified normal/tumor (recede)
      deep navy/maroon    = misclassified  normal/tumor  (pop)
    Misclassified markers are drawn LARGER with a black edge so they pop hard
    against the pale correct cloud (lead author 2026-06-05).

    The title carries the case-count breakdown for the zone; the x-label is
    only drawn when ``xlabel`` is True (bottom panel only).
    """
    r = res.sort_values("p_point").reset_index(drop=True)
    x = np.arange(len(r))
    y = r["y"].values
    p = r["p_point"].values
    lo = r["p_lo"].values
    hi = r["p_hi"].values
    correct = r["correct"].values

    yerr = np.vstack([np.maximum(p - lo, 0), np.maximum(hi - p, 0)])

    # 4 visual classes: (true tissue) x (correct / misclassified).
    # Correct: small, pale, white edge, recede (zorder 2-3).
    # Misclassified: LARGE, fully saturated dark, BLACK edge, pop (zorder 5).
    groups = [
        (0, True, COLOR_NORMAL, "o"),        # correct normal  (very pale blue)
        (1, True, COLOR_TUMOR, "s"),         # correct tumor   (very pale red)
        (0, False, COLOR_NORMAL_MISS, "o"),  # miss normal (deep navy)
        (1, False, COLOR_TUMOR_MISS, "s"),   # miss tumor  (deep maroon)
    ]
    for cls, want_correct, color, marker in groups:
        m = (y == cls) & (correct == want_correct)
        if m.sum() == 0:
            continue
        if want_correct:
            ax.errorbar(x[m], p[m], yerr=yerr[:, m], fmt=marker, color=color,
                        ecolor=color, elinewidth=1.3, capsize=2.5,
                        markersize=7, alpha=0.95, markeredgecolor="white",
                        markeredgewidth=0.6, linestyle="none", zorder=3)
        else:
            ax.errorbar(x[m], p[m], yerr=yerr[:, m], fmt=marker, color=color,
                        ecolor=color, elinewidth=2.2, capsize=4.0,
                        markersize=14, alpha=1.0, markeredgecolor=MISS_EDGE,
                        markeredgewidth=1.6, linestyle="none", zorder=5)

    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.3, alpha=0.85)
    if xlabel:
        ax.set_xlabel("ROI (sorted by predicted probability)")
    ax.set_ylabel("P(tumor)")

    # --- case-count breakdown, one concise line per panel ---
    n = len(r)
    n_norm = int((y == 0).sum())
    n_tum = int((y == 1).sum())
    n_corr = int(correct.sum())
    n_miss = int((~correct).sum())
    ax.set_title(
        f"{zone_label} (n={n}): normal {n_norm}, tumor {n_tum} "
        f"— {n_corr} correct, {n_miss} misclassified")
    # pad x so the first/last points are not jammed against the spine.
    ax.set_xlim(-0.8, len(r) - 0.2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25, linewidth=0.5)


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

    # ---------- figure: 2 rows x 1 column, full width, PZ top / TZ bottom -----
    # (lead-author redesign 2026-06-05: dropped the CI-width violin row -- that
    # comparison is now described in the caption/text. Each sorted-prediction
    # panel spans the full figure width so individual ROI points are well
    # separated horizontally.) Legend on TOP, no suptitle.
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.32, top=0.87)
    ax_pz = fig.add_subplot(gs[0, 0])
    ax_tz = fig.add_subplot(gs[1, 0])

    # TOP panel: NO x-label (it crowds the middle); BOTTOM panel keeps it.
    plot_sorted_panel(ax_pz, results["pz"], "PZ", xlabel=False)
    plot_sorted_panel(ax_tz, results["tz"], "TZ", xlabel=True)

    # Legend on TOP (Fig 1 / 3 style). Correct = small pale pastel marker;
    # misclassified = larger fully-saturated dark marker with black edge, so the
    # legend mirrors the in-panel contrast. No "decision boundary" entry -- the
    # dashed 0.5 line stays in each panel but is self-explanatory.
    legend_handles = [
        Line2D([0], [0], marker="o", color=COLOR_NORMAL, linestyle="none",
               markersize=10, markeredgecolor="white", markeredgewidth=0.6,
               label="Normal (correct)"),
        Line2D([0], [0], marker="s", color=COLOR_TUMOR, linestyle="none",
               markersize=10, markeredgecolor="white", markeredgewidth=0.6,
               label="Tumor (correct)"),
        Line2D([0], [0], marker="o", color=COLOR_NORMAL_MISS, linestyle="none",
               markersize=13, markeredgecolor=MISS_EDGE, markeredgewidth=1.4,
               label="Normal (misclassified)"),
        Line2D([0], [0], marker="s", color=COLOR_TUMOR_MISS, linestyle="none",
               markersize=13, markeredgecolor=MISS_EDGE, markeredgewidth=1.4,
               label="Tumor (misclassified)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 0.965),
               columnspacing=1.6, handlelength=1.8, fontsize=FONT_SIZE)

    png = os.path.join(OUT_DIR, "fig6_v2.png")
    pdf = os.path.join(OUT_DIR, "fig6_v2.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
