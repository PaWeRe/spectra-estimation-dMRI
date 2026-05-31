"""Supplementary figure: individual estimated spectra for EVERY ROI.

fig1_v3 styling adapted for the supplement: each panel is one ROI's NUTS
posterior-mean spectrum (bars) with posterior-std whiskers, coloured by the
per-bin WITHIN-ROI coefficient of variation (purple sequential = identifiability,
matching the Fig 4 scheme), and the tuned-MAP point estimate overlaid as a green
x marker. One grid per zone x tissue so every ROI is visible.

This is the un-conflated home for identifiability: each panel is a single ROI,
so the colour is purely within-ROI posterior CV (no cohort averaging).

Everything is read from results/biomarkers/features.csv (nuts_D_*, nuts_std_D_*,
map_D_*) -- no .nc reloading.

Outputs (paper/figures/):
    figS1_all_roi_pz_normal.{pdf,png}
    figS1_all_roi_pz_tumor.{pdf,png}
    figS1_all_roi_tz_normal.{pdf,png}
    figS1_all_roi_tz_tumor.{pdf,png}

Usage:
    uv run python scripts/figS1_all_roi_spectra.py
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

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "results" / "biomarkers" / "features.csv"
OUT = REPO / "paper" / "figures"

D = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
NUTS = [f"nuts_D_{d:.2f}" for d in D]
NSTD = [f"nuts_std_D_{d:.2f}" for d in D]
MAPC = [f"map_D_{d:.2f}" for d in D]

MAP_MARK = "#2ca02c"  # green x, matches the MAP colour convention
NCOLS = 7
YMAX = 0.9

# Purple sequential CV bands (identifiability) -- matches the planned Fig 4 scheme.
CV_BANDS = [(0.4, "#dadaeb"), (0.6, "#9e9ac8"), (0.8, "#756bb1"), (np.inf, "#54278f")]


def cv_color(cv: float) -> str:
    if not np.isfinite(cv):
        return "#d9d9d9"
    for hi, c in CV_BANDS:
        if cv < hi:
            return c
    return CV_BANDS[-1][1]


def make_grid(sub: pd.DataFrame, title: str, fname: str) -> None:
    n = len(sub)
    nrows = int(np.ceil(n / NCOLS))
    fig, axes = plt.subplots(
        nrows, NCOLS, figsize=(NCOLS * 2.0, nrows * 1.7 + 0.8), squeeze=False
    )
    x = np.arange(len(D))
    sub = sub.reset_index(drop=True)

    for k in range(nrows * NCOLS):
        ax = axes[k // NCOLS][k % NCOLS]
        if k >= n:
            ax.axis("off")
            continue
        row = sub.iloc[k]
        nm = row[NUTS].to_numpy(float)
        ns = row[NSTD].to_numpy(float)
        mp = row[MAPC].to_numpy(float)
        cv = np.divide(ns, nm, out=np.full_like(nm, np.nan), where=nm > 1e-8)
        colors = [cv_color(c) for c in cv]

        ax.bar(x, nm, color=colors, edgecolor="black", linewidth=0.4,
               yerr=ns, ecolor="0.4", error_kw=dict(elinewidth=0.6), capsize=1.0)
        ax.scatter(x, mp, marker="x", s=16, c=MAP_MARK, linewidths=1.1, zorder=5)
        ax.set_ylim(0, YMAX)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=6)
        if k % NCOLS != 0:
            ax.set_yticklabels([])
        ax.axhline(0, color="k", lw=0.4)

        ggg = pd.to_numeric(row.get("ggg", np.nan), errors="coerce")
        tag = f"#{int(row['gidx'])}"
        if np.isfinite(ggg) and ggg >= 1:
            tag += f" · GGG{int(ggg)}"
        ax.set_title(tag, fontsize=7.5)

    # Figure-level legend (CV bands + MAP marker + what the whisker is).
    handles = [
        mpatches.Patch(facecolor="#dadaeb", edgecolor="black", label="CV < 0.4"),
        mpatches.Patch(facecolor="#9e9ac8", edgecolor="black", label="0.4–0.6"),
        mpatches.Patch(facecolor="#756bb1", edgecolor="black", label="0.6–0.8"),
        mpatches.Patch(facecolor="#54278f", edgecolor="black", label="CV > 0.8"),
        Line2D([0], [0], marker="x", color=MAP_MARK, linestyle="None",
               markersize=8, markeredgewidth=1.5, label="MAP ($\\lambda=10^{-3}$)"),
        Line2D([0], [0], color="0.4", lw=1.2, label="NUTS mean ± posterior std"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=True,
               framealpha=0.95, fontsize=10, title="NUTS posterior CV (identifiability)",
               title_fontsize=10, bbox_to_anchor=(0.5, 0.999))
    fig.suptitle(title, fontsize=12, y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname} ({n} ROIs, {nrows}x{NCOLS})")


def main() -> None:
    df = pd.read_csv(FEAT)
    # Stable global ROI index across the four category figures (sorted by ADC
    # within each category), so each panel has a unique anonymised id.
    cats = [
        ("pz", False, "PZ normal", "figS1_all_roi_pz_normal"),
        ("pz", True, "PZ tumor", "figS1_all_roi_pz_tumor"),
        ("tz", False, "TZ normal", "figS1_all_roi_tz_normal"),
        ("tz", True, "TZ tumor", "figS1_all_roi_tz_tumor"),
    ]
    ordered = pd.concat(
        [df[(df.zone == z) & (df.is_tumor == t)].sort_values("adc", ascending=False)
         for z, t, _, _ in cats]
    ).reset_index(drop=True)
    ordered["gidx"] = ordered.index + 1

    print("Generating per-ROI spectra grids:")
    for z, t, title, fname in cats:
        sub = ordered[(ordered.zone == z) & (ordered.is_tumor == t)]
        make_grid(sub, f"Individual NUTS spectra — {title} (n={len(sub)})", fname)


if __name__ == "__main__":
    main()
