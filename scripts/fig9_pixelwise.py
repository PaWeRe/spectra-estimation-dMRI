"""
Figure 9 (v1): Pixel-wise spectral decomposition -- feasibility demo on one
DWI slice (anonymized patient; raw source 8640, slice 6).

Scope (PROJECT_STATE Sec.1): a single-slice MAP/NUTS feasibility demo, NOT a
delivered per-voxel result. No per-voxel histology ground truth exists, so NO
quantitative per-voxel performance claim is made (Sec.9 parked items). What the
figure shows:
  - the ROI-level spectral classifiers push to the voxel level and reproduce
    ADC's spatial pattern (the "collapse" narrative, Fig 2);
  - the UNIQUE Bayesian deliverable -- a per-voxel uncertainty map. The full
    8-bin classifier, leaning on the poorly-identified intermediate bins, is
    spatially MORE uncertain than the collapsed 2-bin {D=0.25, 3.0} classifier
    built on the two well-identified outer bins.

Key per-voxel finding (2026-06-01): the ABSOLUTE voxel score skews tumor-like
(median voxel logit > median *tumor* ROI logit). A single voxel's low SNR
inflates the restricted (D=0.25) fraction -- the high-b noise floor mimics slow
decay -- biasing the ROI-trained classifier optimistically. This is shown
honestly (panels E,F = absolute, centred on the decision boundary) and is a
limitation for per-voxel deployment (reinforces ROI-level scope). Panel G shows
the SAME 2-bin score windowed to within-slice contrast (exactly as ADC maps are
always windowed), which recovers the red(tumor)/blue(normal) split and the
spatial concordance with ADC.

Display decisions (Patrick, 2026-06-01):
  - Classifier maps use the continuous LINEAR DISCRIMINANT SCORE (LR decision
    function), not P(tumor) (the sigmoid saturates, inflating contrast and
    collapsing sigma[P] -> 0 where P saturates, wrecking the uncertainty cmp).
  - Estimator = NUTS posterior mean (matches Figs 2/4); MAP near-identical.
  - Classifiers retrained from features.csv as in Fig 2 (NUTS features,
    StandardScaler, C=1) for the 2-bin {D=0.25, 3.0} and 8-bin models.
  - ADC grayscale, conventional orientation (low ADC = dark = tumor).

Layout: 3 x 3 (col 1 = ADC-comparison column).
  A anatomy(b=0)         B restricted frac D=0.25   C free-water frac D=3.0
  D ADC                  E 2-bin score (absolute)   F 8-bin score (absolute)
  G 2-bin score (windowed) H 2-bin score SD         I 8-bin score SD

PROVISIONAL: applies the PZ-trained classifier. Patient 8640's lesion zone is
not recoverable from the repo. Flip ZONE to "tz" once the label is supplied.

Output:
  paper/figures/fig9_v1.png
  paper/figures/fig9_v1.pdf
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

REPO_ROOT = "/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI"
sys.path.insert(0, REPO_ROOT)

from src.spectra_estimation_dmri.data.loaders import (
    load_binary_images,
    subsample_to_native,
    group_images_b0_plus_directions,
)
from src.spectra_estimation_dmri.pixelwise import assemble_map

DATA_FOLDER = os.path.join(REPO_ROOT, "src/spectra_estimation_dmri/data/8640-sl6-bin")
FEATURES_CSV = os.path.join(REPO_ROOT, "results/biomarkers/features.csv")
NUTS_NPZ = os.path.join(REPO_ROOT, "results/pixelwise/nuts_results.npz")
FAST_NPZ = os.path.join(REPO_ROOT, "results/pixelwise_all_fast.npz")
OUT_DIR = os.path.join(REPO_ROOT, "paper/figures")

DIFF = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 20.00]
NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFF]
OUTER_IDX = [0, 6]            # D=0.25 (restricted) and D=3.0 (free-water)
ZONE = "pz"                   # PROVISIONAL -- see module docstring
LR_C = 1.0
N_MC = 500
SEED = 42

mpl.rcParams.update({
    "axes.titlesize": 18,
    "font.family": "DejaVu Sans",
})

CMAP_UNC = "Purples"
CB_LABEL_FS = 17
CB_TICK_FS = 14
LETTER_FS = 19


# ---------------------------------------------------------------------------
# classifier training (identical recipe to Fig 2)
# ---------------------------------------------------------------------------
def train_lr(df_zone, cols):
    X = df_zone[cols].values
    y = df_zone["is_tumor"].astype(int).values
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=LR_C, max_iter=2000, random_state=SEED, solver="lbfgs")
    clf.fit(sc.transform(X), y)
    return sc, clf, X, y


def loocv_auc(X, y):
    loo = LeaveOneOut()
    pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        s = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=LR_C, max_iter=2000, random_state=SEED, solver="lbfgs")
        m.fit(s.transform(X[tr]), y[tr])
        pred[te] = m.predict_proba(s.transform(X[te]))[0, 1]
    return roc_auc_score(y, pred)


def voxel_score_and_sd(spec_mean, spec_std, cols_idx, sc, clf, n_mc=N_MC, seed=SEED):
    """Per-voxel linear discriminant score (logit) and its posterior SD.

    SD via Monte-Carlo: draw per-bin samples from the NUTS marginal N(mean, std)
    truncated at 0 (independence across bins -- only per-bin std is cached), push
    through the SAME standardiser + LR, take the std of the logit.
    """
    mean = spec_mean[:, cols_idx]
    std = spec_std[:, cols_idx]
    score_mean = clf.decision_function(sc.transform(mean))
    rng = np.random.RandomState(seed)
    n_px, p = mean.shape
    sd = np.zeros(n_px)
    for i in range(n_px):
        samp = np.maximum(rng.normal(mean[i], std[i], size=(n_mc, p)), 0.0)
        sd[i] = clf.decision_function(sc.transform(samp)).std()
    return score_mean, sd


def robust_z(v):
    """Within-slice robust z-score about the median (MAD-scaled)."""
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return (v - med) / (1.4826 * mad + 1e-9)


# ---------------------------------------------------------------------------
# panel helper -- identical image size on every panel (reserved cbar slot)
# ---------------------------------------------------------------------------
def panel(ax, b0, value_map, cmap, title, letter, vmin=None, vmax=None,
          add_cbar=True, cbar_label="", alpha=0.92):
    ax.imshow(b0, cmap="gray", interpolation="nearest")
    if value_map is not None:
        m = np.ma.masked_invalid(value_map)
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                       interpolation="nearest")
    ax.set_title(title, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.045, 0.955, letter, transform=ax.transAxes, fontsize=LETTER_FS,
            fontweight="bold", color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.55))
    # reserve an identical-size cbar slot on EVERY panel -> equal image sizes
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
    if add_cbar and value_map is not None:
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(cbar_label, fontsize=CB_LABEL_FS)
        cb.ax.tick_params(labelsize=CB_TICK_FS)
    else:
        cax.axis("off")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- anatomy (full-slice b=0) -------------------------------------------
    imgs = load_binary_images(DATA_FOLDER, shape=(256, 256), dtype=np.int16)
    imgs64 = subsample_to_native(imgs, factor=4)
    _, trace, _ = group_images_b0_plus_directions(imgs64, n_directions=3)
    b0 = trace[0]

    # --- per-voxel NUTS + ADC (same coords/mask) ----------------------------
    nuts = np.load(NUTS_NPZ)
    fast = np.load(FAST_NPZ)
    spec_mean, spec_std = nuts["spectrum_mean"], nuts["spectrum_std"]
    coords, mask = nuts["coords"], nuts["mask"]
    shape = mask.shape
    assert np.array_equal(coords, fast["coords"]), "NUTS/fast coords mismatch"
    adc = fast["adc"] * 1000.0           # mm^2/s -> um^2/ms (manuscript convention)

    # --- classifiers (Fig-2 recipe) -----------------------------------------
    feats = pd.read_csv(FEATURES_CSV)
    dfz = feats[feats["zone"] == ZONE].copy()
    cols8, cols2 = NUTS_COLS, [NUTS_COLS[i] for i in OUTER_IDX]
    sc8, clf8, X8, y8 = train_lr(dfz, cols8)
    sc2, clf2, X2, y2 = train_lr(dfz, cols2)

    score8, sd8 = voxel_score_and_sd(spec_mean, spec_std, list(range(8)), sc8, clf8)
    score2, sd2 = voxel_score_and_sd(spec_mean, spec_std, OUTER_IDX, sc2, clf2)
    s2_win = robust_z(score2)            # windowed view of the 2-bin score
    frac_restricted, frac_freewater = spec_mean[:, 0], spec_mean[:, 6]

    # --- shared scales -------------------------------------------------------
    abs_k = np.percentile(np.abs(np.concatenate([score2, score8])), 98)   # E,F
    win_k = np.percentile(np.abs(s2_win), 98)                              # G
    unc_max = np.nanpercentile(np.concatenate([sd2, sd8]), 98)            # H,I
    adc_lo, adc_hi = np.nanpercentile(adc, [2, 98])
    fr_hi = np.nanpercentile(frac_restricted, 98)
    fw_hi = np.nanpercentile(frac_freewater, 98)

    # --- assemble + crop to the gland ---------------------------------------
    def amap(v):
        return assemble_map(v, coords, shape)
    M = dict(adc=amap(adc), fr=amap(frac_restricted), fw=amap(frac_freewater),
             s2=amap(score2), s8=amap(score8), s2w=amap(s2_win),
             u2=amap(sd2), u8=amap(sd8))
    rr, cc = np.where(mask)
    pad = 6
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, shape[0])
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, shape[1])
    crop = lambda a: a[r0:r1, c0:c1]
    b0 = crop(b0)
    M = {k: crop(v) for k, v in M.items()}

    # --- diagnostics ---------------------------------------------------------
    auc8, auc2 = loocv_auc(X8, y8), loocv_auc(X2, y2)
    roi_tum_med = np.median(clf2.decision_function(sc2.transform(X2[y2 == 1])))
    print(f"\n=== Fig 9 ({ZONE.upper()} classifier, n={len(y8)}, "
          f"{int(y8.sum())} tumor ROIs) ===")
    print(f"  LOOCV AUC   2-bin = {auc2:.3f}   8-bin = {auc8:.3f}")
    print(f"  voxel score vs ADC   r(2-bin) = {np.corrcoef(adc, score2)[0,1]:+.3f}   "
          f"r(8-bin) = {np.corrcoef(adc, score8)[0,1]:+.3f}")
    print(f"  voxel logit median: 2-bin={np.median(score2):+.2f}  8-bin={np.median(score8):+.2f}"
          f"  (vs ROI tumor median {roi_tum_med:+.2f})  -> per-voxel tumor-skew")
    print(f"  voxel score SD median: 2-bin = {np.median(sd2):.3f}   "
          f"8-bin = {np.median(sd8):.3f}   "
          f"(ratio {np.median(sd8)/max(np.median(sd2),1e-9):.2f}x)")

    # --- figure 3 x 3 --------------------------------------------------------
    fig, ax = plt.subplots(3, 3, figsize=(16.5, 13.6))

    panel(ax[0, 0], b0, None, "gray", r"Anatomy ($b=0$)", "A", add_cbar=False)
    panel(ax[0, 1], b0, M["fr"], "Reds", r"Restricted fraction ($D=0.25$)", "B",
          vmin=0, vmax=fr_hi, cbar_label="fraction")
    panel(ax[0, 2], b0, M["fw"], "Blues", r"Free-water fraction ($D=3.0$)", "C",
          vmin=0, vmax=fw_hi, cbar_label="fraction")

    panel(ax[1, 0], b0, M["adc"], "gray", "ADC", "D", vmin=adc_lo, vmax=adc_hi,
          cbar_label=r"ADC ($\mu$m$^2$/ms)", alpha=1.0)
    panel(ax[1, 1], b0, M["s2"], "RdBu_r", "2-bin score, absolute", "E",
          vmin=-abs_k, vmax=abs_k, add_cbar=False)
    panel(ax[1, 2], b0, M["s8"], "RdBu_r", "8-bin score, absolute", "F",
          vmin=-abs_k, vmax=abs_k, cbar_label="discriminant score\n(logit; 0 = boundary)")

    panel(ax[2, 0], b0, M["s2w"], "RdBu_r", "2-bin score, windowed", "G",
          vmin=-win_k, vmax=win_k, cbar_label="within-slice score")
    panel(ax[2, 1], b0, M["u2"], CMAP_UNC, "2-bin score uncertainty", "H",
          vmin=0, vmax=unc_max, add_cbar=False)
    panel(ax[2, 2], b0, M["u8"], CMAP_UNC, "8-bin score uncertainty", "I",
          vmin=0, vmax=unc_max, cbar_label="posterior SD")

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "fig9_v1.png")
    pdf = os.path.join(OUT_DIR, "fig9_v1.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
