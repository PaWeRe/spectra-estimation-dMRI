"""Supplementary atlas: individual estimated spectra for EVERY ROI.

Multi-page PDF (2 columns/page, grouped by zone x tissue). Each panel is one
ROI's NUTS posterior-mean spectrum (bars) with posterior-std whiskers, coloured
by the per-bin WITHIN-ROI coefficient of variation (purple sequential =
identifiability, matching the Fig 4 scheme); the tuned-MAP point estimate is a
green x. Proper per-panel axes (diffusivity ticks + labels, gridlines).

This is the un-conflated home for identifiability: each panel is a single ROI,
so the colour is purely within-ROI posterior CV (no cohort averaging).

Everything is read from results/biomarkers/features.csv (nuts_D_*,
nuts_std_D_*, map_D_*) -- no .nc reloading.

MRM style: no figure title; panel titles give only the Gleason Grade Group (no
patient identifier). A light per-page header names the zone x tissue group.

Outputs (paper/figures/):
    figS1_all_roi_spectra.pdf          -- multi-page atlas (include via \\includepdf)
    figS1_all_roi_spectra_preview.png  -- first page, for quick review

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "results" / "biomarkers" / "features.csv"
OUT = REPO / "paper" / "figures"

D = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
DLAB = ["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "20"]
NUTS = [f"nuts_D_{d:.2f}" for d in D]
NSTD = [f"nuts_std_D_{d:.2f}" for d in D]
MAPC = [f"map_D_{d:.2f}" for d in D]

MAP_MARK = "#2ca02c"  # green x, matches the MAP colour convention
NCOLS, NROWS = 2, 6   # 12 ROIs / page
YMAX = 0.9

CV_BANDS = [(0.4, "#dadaeb"), (0.6, "#9e9ac8"), (0.8, "#756bb1"), (np.inf, "#54278f")]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 14, "axes.titlesize": 13,
    "xtick.labelsize": 9, "ytick.labelsize": 10, "legend.fontsize": 12,
})


def cv_color(cv: float) -> str:
    if not np.isfinite(cv):
        return "#d9d9d9"
    for hi, c in CV_BANDS:
        if cv < hi:
            return c
    return CV_BANDS[-1][1]


def panel_title(row: pd.Series) -> str:
    # Match the old Gibbs-output titles: Gleason score + GGG, no patient id.
    if not row["is_tumor"]:
        return ""  # normal: the per-page header already states the group
    ggg = pd.to_numeric(row.get("ggg", np.nan), errors="coerce")
    gs = row.get("gs", None)
    # "0+0" is the metadata placeholder for a tumor ROI with no real biopsy grade.
    gs_str = gs if (isinstance(gs, str) and "+" in gs and gs != "0+0") else None
    has_ggg = bool(np.isfinite(ggg) and ggg >= 1)
    if gs_str and has_ggg:
        return f"{gs_str} (GGG {int(ggg)})"
    if gs_str:
        return gs_str
    if has_ggg:
        return f"GGG {int(ggg)}"
    return "tumor, ungraded"


def draw_panel(ax, row, show_x, show_y) -> None:
    x = np.arange(len(D))
    nm = row[NUTS].to_numpy(float)
    ns = row[NSTD].to_numpy(float)
    mp = row[MAPC].to_numpy(float)
    cv = np.divide(ns, nm, out=np.full_like(nm, np.nan), where=nm > 1e-8)
    ax.bar(x, nm, color=[cv_color(c) for c in cv], edgecolor="black", linewidth=0.5,
           yerr=ns, ecolor="0.4", error_kw=dict(elinewidth=0.7), capsize=1.5)
    ax.scatter(x, mp, marker="x", s=26, c=MAP_MARK, linewidths=1.4, zorder=5)
    ax.set_ylim(0, YMAX)
    ax.set_xlim(-0.6, len(D) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(DLAB if show_x else [])
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.axhline(0, color="k", lw=0.4)
    if not show_y:
        ax.set_yticklabels([])
    title = panel_title(row)
    if title:
        ax.set_title(title, fontsize=12)


def legend_handles():
    return [
        mpatches.Patch(facecolor="#dadaeb", edgecolor="black", label="CV < 0.4"),
        mpatches.Patch(facecolor="#9e9ac8", edgecolor="black", label="0.4–0.6"),
        mpatches.Patch(facecolor="#756bb1", edgecolor="black", label="0.6–0.8"),
        mpatches.Patch(facecolor="#54278f", edgecolor="black", label="CV > 0.8"),
        Line2D([0], [0], marker="x", color=MAP_MARK, linestyle="None",
               markersize=9, markeredgewidth=1.6, label=r"MAP ($\lambda=10^{-3}$)"),
        Line2D([0], [0], color="0.4", lw=1.4, label="NUTS mean ± post. std"),
    ]


def make_page(pdf, chunk, header, first_preview=None):
    n = len(chunk)
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(8.5, 11.0), squeeze=False,
                             sharey=True)
    chunk = chunk.reset_index(drop=True)
    for k in range(NROWS * NCOLS):
        r, c = k // NCOLS, k % NCOLS
        ax = axes[r][c]
        if k >= n:
            ax.axis("off")
            continue
        # show x labels on the bottom-most filled panel of each column
        is_bottom = (k + NCOLS >= n) or (r == NROWS - 1)
        draw_panel(ax, chunk.iloc[k], show_x=is_bottom, show_y=(c == 0))
    fig.legend(handles=legend_handles(), loc="upper center", ncol=6, frameon=True,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.985),
               title="NUTS posterior CV (per-bin identifiability)", title_fontsize=11)
    fig.text(0.01, 0.992, header, ha="left", va="top", fontsize=11,
             color="0.35", fontstyle="italic")
    fig.supxlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)", fontsize=14)
    fig.supylabel(r"spectral fraction $R_j$", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    if first_preview is not None:
        fig.savefig(first_preview, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(FEAT)
    # Join the Gleason score (gs, e.g. "4+3") from metadata by patient id, which
    # is the roi_id prefix (e.g. "new02_pz_tumor" -> "new02").
    meta = pd.read_csv(REPO / "src" / "spectra_estimation_dmri" / "data" / "bwh" / "metadata.csv")
    gs_map = dict(zip(meta["patient_id"].astype(str), meta["gs"]))
    df["patient_id"] = df["roi_id"].astype(str).str.split("_").str[0]
    df["gs"] = df["patient_id"].map(gs_map)
    cats = [
        ("pz", False, "peripheral zone · normal"),
        ("pz", True, "peripheral zone · tumor"),
        ("tz", False, "transition zone · normal"),
        ("tz", True, "transition zone · tumor"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "figS1_all_roi_spectra.pdf"
    per = NROWS * NCOLS
    preview = OUT / "figS1_all_roi_spectra_preview.png"
    n_pages = 0
    with PdfPages(pdf_path) as pdf:
        for z, t, header in cats:
            sub = df[(df.zone == z) & (df.is_tumor == t)].sort_values(
                "adc", ascending=False)
            for start in range(0, len(sub), per):
                chunk = sub.iloc[start:start + per]
                pages_in_cat = int(np.ceil(len(sub) / per))
                pg = start // per + 1
                hdr = f"{header}   ({len(sub)} ROIs · page {pg}/{pages_in_cat})"
                make_page(pdf, chunk, hdr,
                          first_preview=preview if n_pages == 0 else None)
                n_pages += 1
    print(f"wrote {pdf_path.relative_to(REPO)} ({n_pages} pages)")
    print(f"wrote {preview.relative_to(REPO)} (page 1)")


if __name__ == "__main__":
    main()
