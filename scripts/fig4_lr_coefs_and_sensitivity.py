"""Fig. 4 (main, centrepiece) -- per-bin anatomy of the spectral discriminant.

Complement to Fig. 3 (Fig. 3 = "the spectral discriminant ranks ROIs like ADC";
Fig. 4 = "here is the per-bin structure of that discriminant, where its weight
actually lives, and how it relates to ADC").

Two panels (PZ, TZ). On a single unit-normalised axis (no dual-axis):

  - Bars  = tumor-vs-normal logistic-regression coefficient w per diffusivity
            bin -- the SAME fit as Fig. 3 (NUTS posterior-mean features, C=1,
            balanced) -- normalised to max|w| = 1, with 95% bootstrap CIs
            (resample ROIs, refit). Bars are COLOURED + HATCHED by the bin's
            within-ROI posterior CV (colourblind-safe, shared with the S1 atlas
            via visualization.identifiability).
  - Markers = -dADC/dR_j (charcoal diamonds, NO connecting line) at the average
            tumour operating point, normalised to max|.| = 1 and negated so
            positive = tumour direction. ALSO given 95% bootstrap CIs (resample
            ROIs, recompute the operating point, recompute the gradient) so the
            two quantities are compared on equal, honest footing.

Two coefficient versions are built (Stefan 2026-06-03; Patrick picks one at
review):

  - STANDARDIZED (fig4_std_v4): features z-scored before the LR fit, so the
    coefficient is the weight on standardised features (per-bin SD-normalised
    importance). This is Fig. 3's exact fit.
  - RAW (fig4_raw_v4): LR fit on the un-standardised fractions directly, so the
    coefficient is the weight on raw R_j. Standardising shrinks the
    intermediate bins (they have larger SDs); the raw version keeps the middle
    bins comparatively larger.

Both are normalised to max|w| = 1 for display only.

Inputs:
  - results/biomarkers/features.csv  (NUTS fractions + std -> LR, CV, sensitivity)

Outputs:
  - paper/figures/fig4_std_v4.{png,pdf}  (standardized coefficients)
  - paper/figures/fig4_raw_v4.{png,pdf}  (raw / unstandardized coefficients)

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
from spectra_estimation_dmri.visualization.paper_style import (
    apply_style, COLORS, DIFFUSIVITIES, DLABELS)

# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO / "results" / "biomarkers" / "features.csv"
OUT_DIR = REPO / "paper" / "figures"

D = DIFFUSIVITIES
DLAB = DLABELS
NUTS = [f"nuts_D_{d:.2f}" for d in D]
NSTD = [f"nuts_std_D_{d:.2f}" for d in D]
ZONES = [("pz", "PZ"), ("tz", "TZ")]

SENS_COLOR = COLORS["sensitivity"]   # charcoal sensitivity markers
N_BOOT = 2000

# Locks legend == title font size (Stefan); replaces this script's old per-file
# font rcParams.
apply_style("grid")
mpl.rcParams.update({"hatch.linewidth": 0.6})


def lr_coef(X: np.ndarray, y: np.ndarray, standardize: bool) -> np.ndarray:
    """Tumor-vs-normal LR coefficient vector.

    standardize=True : z-score features first -> weight on standardised features
                       (Fig. 3's exact fit).
    standardize=False: fit on the raw fractions directly -> weight on raw R_j.
    """
    Xf = StandardScaler().fit_transform(X) if standardize else X
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                             solver="lbfgs", random_state=42)
    clf.fit(Xf, y.astype(int))
    return clf.coef_[0]


def unit(v: np.ndarray) -> float:
    m = np.max(np.abs(v))
    return m if m > 0 else 1.0


def analyze_zone(feat: pd.DataFrame, zkey: str, standardize: bool,
                 rng: np.random.Generator) -> dict:
    """Point estimates + 95% bootstrap CIs for LR coef and -dADC/dR, plus CV."""
    sub = feat[feat["zone"] == zkey]
    X = sub[NUTS].to_numpy(float)
    y = sub["is_tumor"].to_numpy()
    n = len(sub)

    w0 = lr_coef(X, y, standardize)
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
        wb = lr_coef(Xb, yb, standardize)
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
        "cv_mean": np.nanmean(cv, axis=0),
        "r0": r0, "r_lo": r_lo, "r_hi": r_hi,
    }


def build_figure(data: dict, standardize: bool, out_stem: Path) -> None:
    """Render the two-panel figure for one coefficient version."""
    x = np.arange(len(D))

    fig = plt.figure(figsize=(15.0, 7.5))
    gs = fig.add_gridspec(1, 2, wspace=0.16,
                          left=0.075, right=0.985, top=0.80, bottom=0.135)
    # symmetric y so the (wide) D=0.25 coef CI is not clipped; pad above the
    # largest CI extent across both panels.
    ci_extent = np.concatenate([data[z][k] for z in ("pz", "tz")
                                for k in ("w_lo", "w_hi", "s_lo", "s_hi")])
    ymax = float(np.nanmax(np.abs(ci_extent))) * 1.12
    bw = 0.34
    dx = 0.18

    for j, (zkey, zlabel) in enumerate(ZONES):
        z = data[zkey]
        ax = fig.add_subplot(gs[0, j])

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

        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlim(-0.6, len(D) - 0.4)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(DLAB)
        ax.set_xlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
        ax.set_title(f"{zlabel}   ($n = {z['n']}$) · NUTS")
        # r + CI as a compact in-panel annotation (keeps the title short).
        ax.text(0.5, 0.965,
                f"$r(w,\\,\\partial$ADC$/\\partial R) = {z['r0']:+.2f}$  "
                f"$[{z['r_lo']:+.2f},\\,{z['r_hi']:+.2f}]$",
                transform=ax.transAxes, ha="center", va="top", fontsize=15,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        if j == 0:
            ax.set_ylabel(r"normalised weight  ($\max|\cdot| = 1$)")

    # --- shared top legend: row 1 = glyphs, row 2 = the 4 CV bands ---
    coef_label = (r"LR coef (bars, NUTS)" if standardize
                  else r"LR coef, raw (bars, NUTS)")
    handles = [
        mpatches.Patch(facecolor="0.82", edgecolor="black", label=coef_label),
        Line2D([0], [0], color=SENS_COLOR, marker="D", linestyle="None", ms=8,
               mfc="white", mew=1.8, label=r"$-\,\partial$ADC$/\partial R$"),
        Line2D([0], [0], color="black", marker="|", linestyle="None", ms=11,
               markeredgewidth=1.4, label="95% boot CI"),
    ] + cv_legend_handles()
    # CV colour+hatch meaning is described in the caption (not as a legend title).
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.995),
               columnspacing=1.6, handletextpad=0.5)

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png.relative_to(REPO)}")
    print(f"Wrote {out_pdf.relative_to(REPO)}")


def main() -> None:
    feat = pd.read_csv(FEATURES_CSV)

    for standardize, stem in [
        (True, OUT_DIR / "fig4_std_v4"),
        (False, OUT_DIR / "fig4_raw_v4"),
    ]:
        # fresh RNG per version so each bootstrap is reproducible & independent
        rng = np.random.default_rng(42)
        data = {zk: analyze_zone(feat, zk, standardize, rng) for zk, _ in ZONES}
        kind = "standardized" if standardize else "raw"
        print(f"\n=== {kind} coefficients ===")
        build_figure(data, standardize, stem)
        # --- console summary ---
        for zkey, zlabel in ZONES:
            z = data[zkey]
            print(f"{zlabel} (n={z['n']}): r={z['r0']:+.3f} "
                  f"[{z['r_lo']:+.3f}, {z['r_hi']:+.3f}]")
            print("  w_n: " + "  ".join(f"{w:+.2f}" for w in z["w_n"]))


if __name__ == "__main__":
    main()
