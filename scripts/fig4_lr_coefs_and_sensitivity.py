"""Fig. 4 (main, centrepiece) -- per-bin anatomy of the spectral discriminant.

Complement to Fig. 3 (Fig. 3 = "the spectral discriminant ranks ROIs like ADC";
Fig. 4 = "here is the per-bin structure of that discriminant, where its weight
actually lives, and how it relates to ADC").

Two panels (PZ, TZ). On a single unit-normalised axis (no dual-axis):

  - Bars  = standardised tumor-vs-normal logistic-regression coefficient w_std
            per diffusivity bin -- the SAME fit as Fig. 3 (NUTS posterior-mean
            features, C=1, balanced, standardised) -- normalised to max|w| = 1,
            with 95% bootstrap CIs (resample ROIs, refit). Bars are COLOURED +
            HATCHED by the bin's within-ROI posterior CV (colourblind-safe,
            shared with the S1 atlas via visualization.identifiability).
  - Markers = -dADC/dR_j (charcoal diamonds, NO connecting line) at the average
            tumour operating point, normalised to max|.| = 1 and negated so
            positive = tumour direction. ALSO given 95% bootstrap CIs (resample
            ROIs, recompute the operating point, recompute the gradient) so the
            two quantities are compared on equal, honest footing.

A slim strip below each panel shows the per-bin within-ROI CV as mean +/- std.

KEY (bootstrap) RESULT this figure must convey: only the two OUTER bins
(D = 0.25 with +weight, D = 3.0 with -weight) have coefficients whose 95% CI
excludes zero -- in both zones, 100% sign-stable. Every PZ intermediate bin, and
most TZ intermediate bins, have CIs straddling zero: the discriminant is
statistically a TWO-BIN detector. This is the "why ADC works" collapse, and it
is where ADC is most sensitive (matching, opposite sign). The intermediate bins'
biological (grading) signal is a separate, univariate finding (Fig. 5), not a
detection-coefficient claim.

Inputs:
  - results/biomarkers/features.csv  (NUTS fractions + std -> LR, CV, sensitivity)

Outputs:
  - paper/figures/fig4_v3.png  (300 dpi)
  - paper/figures/fig4_v3.pdf

Usage:
    uv run python scripts/fig4_lr_coefs_and_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from spectra_estimation_dmri.biomarkers.recompute import compute_sensitivity
from spectra_estimation_dmri.visualization.identifiability import (
    cv_color, cv_hatch, cv_legend_handles)

# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO / "results" / "biomarkers" / "features.csv"
OUT_DIR = REPO / "paper" / "figures"
OUT_PNG = OUT_DIR / "fig4_v3.png"
OUT_PDF = OUT_DIR / "fig4_v3.pdf"

D = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
DLAB = ["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "20"]
NUTS = [f"nuts_D_{d:.2f}" for d in D]
NSTD = [f"nuts_std_D_{d:.2f}" for d in D]
ZONES = [("pz", "PZ"), ("tz", "TZ")]

SENS_COLOR = "#3a3a3a"   # charcoal sensitivity markers
N_BOOT = 2000

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 18, "axes.titlesize": 16,
    "xtick.labelsize": 16, "ytick.labelsize": 15, "legend.fontsize": 15.5,
    "hatch.linewidth": 0.6,
})


def std_lr_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standardised tumor-vs-normal LR coefficient vector (Fig. 3's exact fit)."""
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                             solver="lbfgs", random_state=42)
    clf.fit(Xs, y.astype(int))
    return clf.coef_[0]


def unit(v: np.ndarray) -> float:
    m = np.max(np.abs(v))
    return m if m > 0 else 1.0


def analyze_zone(feat: pd.DataFrame, zkey: str, rng: np.random.Generator) -> dict:
    """Point estimates + 95% bootstrap CIs for LR coef and -dADC/dR, plus CV."""
    sub = feat[feat["zone"] == zkey]
    X = sub[NUTS].to_numpy(float)
    y = sub["is_tumor"].to_numpy()
    n = len(sub)

    w0 = std_lr_coef(X, y)
    g0 = compute_sensitivity(X[y == 1].mean(axis=0))   # dADC/dR at avg tumour
    sens0 = -g0                                         # plotted (tumour dir.)

    mean = X
    std = sub[NSTD].to_numpy(float)
    cv = np.divide(std, mean, out=np.full_like(mean, np.nan), where=mean > 1e-8)

    # display normalisation (fixed by point estimate)
    mw, ms = unit(w0), unit(sens0)
    Wb, Sb, Rb = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        if yb.sum() < 2 or (~yb.astype(bool)).sum() < 2:
            continue
        wb = std_lr_coef(Xb, yb)
        gb = compute_sensitivity(Xb[yb == 1].mean(axis=0))
        Wb.append(wb); Sb.append(-gb); Rb.append(stats.pearsonr(wb, gb)[0])
    Wb, Sb, Rb = np.array(Wb), np.array(Sb), np.array(Rb)

    w_lo, w_hi = np.percentile(Wb / mw, [2.5, 97.5], axis=0)
    s_lo, s_hi = np.percentile(Sb / ms, [2.5, 97.5], axis=0)
    r0 = stats.pearsonr(w0, g0)[0]
    r_lo, r_hi = np.percentile(Rb, [2.5, 97.5])

    return {
        "n": n,
        "w_n": w0 / mw, "w_lo": w_lo, "w_hi": w_hi,
        "s_n": sens0 / ms, "s_lo": s_lo, "s_hi": s_hi,
        "sig": (w_lo > 0) | (w_hi < 0),               # CI excludes zero
        "cv_mean": np.nanmean(cv, axis=0), "cv_std": np.nanstd(cv, axis=0),
        "r0": r0, "r_lo": r_lo, "r_hi": r_hi,
    }


def main() -> None:
    feat = pd.read_csv(FEATURES_CSV)
    rng = np.random.default_rng(42)
    x = np.arange(len(D))
    data = {zk: analyze_zone(feat, zk, rng) for zk, _ in ZONES}

    fig = plt.figure(figsize=(15.0, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], hspace=0.07, wspace=0.16,
                          left=0.075, right=0.985, top=0.85, bottom=0.115)
    cv_ymax = max(np.nanmax(data[z]["cv_mean"] + data[z]["cv_std"])
                  for z in ("pz", "tz")) * 1.12
    # symmetric y so the (wide) D=0.25 coef CI + the significance markers are not
    # clipped; pad above the largest CI extent across both panels.
    ci_extent = np.concatenate([data[z][k] for z in ("pz", "tz")
                                for k in ("w_lo", "w_hi", "s_lo", "s_hi")])
    ymax = float(np.nanmax(np.abs(ci_extent))) * 1.18
    bw = 0.34
    dx = 0.18

    for j, (zkey, zlabel) in enumerate(ZONES):
        z = data[zkey]
        ax = fig.add_subplot(gs[0, j])
        axc = fig.add_subplot(gs[1, j], sharex=ax)

        # outer "detection" bins, gently flagged
        for xi in (0, 6):
            ax.axvspan(xi - 0.5, xi + 0.5, color="0.5", alpha=0.07, zorder=0)
            axc.axvspan(xi - 0.5, xi + 0.5, color="0.5", alpha=0.07, zorder=0)

        # --- LR bars (colour+hatch = CV) with bootstrap CIs ---
        for i in range(len(D)):
            ax.bar(x[i] - dx, z["w_n"][i], width=bw,
                   color=cv_color(z["cv_mean"][i]), hatch=cv_hatch(z["cv_mean"][i]),
                   edgecolor="black", linewidth=0.6, zorder=2)
        ax.errorbar(x - dx, z["w_n"],
                    yerr=[z["w_n"] - z["w_lo"], z["w_hi"] - z["w_n"]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5, zorder=3)

        # --- ADC sensitivity markers (NO line) with bootstrap CIs ---
        ax.errorbar(x + dx, z["s_n"],
                    yerr=[z["s_n"] - z["s_lo"], z["s_hi"] - z["s_n"]],
                    fmt="D", ms=8, mfc="white", mec=SENS_COLOR, mew=1.8,
                    ecolor=SENS_COLOR, elinewidth=1.4, capsize=3, zorder=4)

        # significance markers: star at the CI extremum of bins whose coef CI
        # excludes zero (the two outer "stable detection weight" bins).
        for i in range(len(D)):
            if not z["sig"][i]:
                continue
            if z["w_n"][i] >= 0:
                ystar, va = z["w_hi"][i] + 0.05 * ymax, "bottom"
            else:
                ystar, va = z["w_lo"][i] - 0.05 * ymax, "top"
            ax.text(x[i] - dx, ystar, "*", ha="center", va=va,
                    fontsize=17, fontweight="bold", zorder=6)

        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlim(-0.6, len(D) - 0.4)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_title(f"{zlabel}   ($n = {z['n']}$) · NUTS", fontsize=17)
        # r + CI as a compact in-panel annotation (keeps the title short).
        ax.text(0.5, 0.965,
                f"$r(w,\\,\\partial$ADC$/\\partial R) = {z['r0']:+.2f}$  "
                f"$[{z['r_lo']:+.2f},\\,{z['r_hi']:+.2f}]$",
                transform=ax.transAxes, ha="center", va="top", fontsize=15,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        if j == 0:
            ax.set_ylabel(r"normalised weight  ($\max|\cdot| = 1$)")

        # --- CV strip ---
        axc.errorbar(x, z["cv_mean"], yerr=z["cv_std"], fmt="none",
                     ecolor="0.55", elinewidth=1.1, capsize=3, zorder=2)
        axc.scatter(x, z["cv_mean"], s=70, zorder=3, edgecolor="black",
                    linewidth=0.6, c=[cv_color(c) for c in z["cv_mean"]])
        for thr in (0.4, 0.8):
            axc.axhline(thr, color="0.7", lw=0.7, ls=":", zorder=1)
        axc.set_ylim(0, cv_ymax)
        axc.set_xticks(x)
        axc.set_xticklabels(DLAB)
        axc.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)")
        axc.grid(axis="y", alpha=0.18, linewidth=0.5)
        if j == 0:
            axc.set_ylabel("within-ROI\nCV")

    # --- shared top legend: row 1 = glyphs, row 2 = the 4 CV bands ---
    handles = [
        mpatches.Patch(facecolor="0.82", edgecolor="black",
                       label=r"LR coef (bars, NUTS)"),
        Line2D([0], [0], color=SENS_COLOR, marker="D", linestyle="None", ms=8,
               mfc="white", mew=1.8, label=r"$-\,\partial$ADC$/\partial R$"),
        Line2D([0], [0], color="black", marker="|", linestyle="None", ms=11,
               markeredgewidth=1.4, label="95% boot CI"),
        Line2D([0], [0], color="black", marker="*", linestyle="None", ms=12,
               label="coef CI excludes 0"),
    ] + cv_legend_handles()
    # CV colour+hatch meaning is described in the caption (not as a legend title).
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.995),
               columnspacing=1.6, handletextpad=0.5, fontsize=15.5)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG.relative_to(REPO)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO)}")

    # --- console summary ---
    for zkey, zlabel in ZONES:
        z = data[zkey]
        print(f"\n{zlabel} (n={z['n']}): r={z['r0']:+.3f} "
              f"[{z['r_lo']:+.3f}, {z['r_hi']:+.3f}]")
        print("  " + "  ".join(
            f"{d:>5g}{'*' if s else ' '}" for d, s in zip(D, z["sig"])))
        print("  CI excludes 0 (*) marks the bins with stable detection weight.")


if __name__ == "__main__":
    main()
