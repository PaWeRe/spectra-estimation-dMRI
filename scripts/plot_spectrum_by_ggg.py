"""Fig 5 — Diffusivity spectrum by Gleason Grade Group (two-panel layout).

Two panels side by side, with the Normal group as a shared neutral-grey
baseline in BOTH panels (Stefan 2026-06-03 decided layout). Each curve is the
mean NUTS posterior-mean spectrum over the 8 diffusivity bins
(D = 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0 um^2/ms):

  LEFT — "emergence" of tumour signal:
      * Normal   (n=109) — grey baseline (benign tissue).
      * GGG = 1  (n=8)   — resolved grade group 1.
      * GGG >= 2 (n=21)  — GGG2 + GGG>=3 pooled (clinically significant).

  RIGHT — "aggressiveness" among clinically significant tumours:
      * Normal   (n=109) — grey baseline (benign tissue).
      * GGG = 2  (n=12)  — resolved grade group 2.
      * GGG >= 3 (n=9)   — cumulative grade groups 3-5 (resolved high grades have
                            tiny n: GGG3=5, GGG4=2, GGG5=2).

Normal is a neutral grey baseline. The four tumour grade groups each get a
distinct shade of red (light salmon -> orange-red -> red -> dark maroon), and
every grade group keeps its own colour regardless of which panel it appears in
(so the left-panel GGG=1 and right-panel GGG=2 are NOT the same colour).

Bands are 95% CI of the group mean (Student-t, df = n-1) across ROIs; they
reflect between-ROI variability of the per-ROI posterior mean and do NOT
include within-ROI posterior uncertainty. Analysis is pooled across zones
(PZ + TZ together).

Tumor ROIs with GGG = 0 (n=7) or ungraded / missing GGG (n=4) — 11 ROIs total —
are excluded from this figure.

Spectra are NUTS posterior-mean spectra (nuts_D_*) from
results/biomarkers/features.csv.

Outputs:
    paper/figures/fig5_v5.{png,pdf}                  — paper-grade two-panel (main)
    results/biomarkers/spectrum_by_ggg.{png,pdf}     — archival copy

Usage:
    uv run python scripts/plot_spectrum_by_ggg.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from spectra_estimation_dmri.visualization.paper_style import (
    apply_style,
    DIFFUSIVITIES,
    set_diff_xaxis,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FEAT_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"

PAPER_PNG = REPO_ROOT / "paper" / "figures" / "fig5_v5.png"
PAPER_PDF = REPO_ROOT / "paper" / "figures" / "fig5_v5.pdf"
OUT_PNG = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.png"
OUT_PDF = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.pdf"

NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]

# Normal = neutral grey shared baseline, shared by BOTH panels (no grey key in
# COLORS). Each panel then uses a DISTINCT colour family so the left-panel
# groups (emergence) are unmistakably separate from the right-panel groups
# (aggressiveness):
#   LEFT  (emergence)      = WARM family, light->dark with grade.
#   RIGHT (aggressiveness) = COOL family, light->dark with grade.
GROUP_NORMAL = "#7f7f7f"  # neutral grey baseline (both panels)
# LEFT — warm (orange -> dark red).
C_GGG1 = "#fdae6b"     # light orange — GGG = 1   (left panel)
C_GGG_GE2 = "#cb181d"  # dark red     — GGG >= 2  (left panel)
# RIGHT — cool (purple).
C_GGG2 = "#9e9ac8"     # light purple — GGG = 2   (right panel)
C_GGG_GE3 = "#54278f"  # dark purple  — GGG >= 3  (right panel, most aggressive)


def group_stats(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = spec.shape[0]
    mean = spec.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    sem = spec.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * sem, mean + t_crit * sem


def plot_group(ax, x, sub: pd.DataFrame, label: str, color: str):
    spec = sub[NUTS_COLS].to_numpy(dtype=float)
    mean, lo, hi = group_stats(spec)
    if len(sub) >= 2:
        # Low-alpha band drawn UNDER the mean line so it does not obscure the
        # mean but stays distinguishable per group.
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, lw=0, zorder=1)
    (line,) = ax.plot(
        x, mean, "-o", color=color, lw=2.6, ms=7,
        label=f"{label}  (n={len(sub)})", zorder=3,
    )
    return line


def main() -> None:
    apply_style("grid")

    df = pd.read_csv(FEAT_CSV)
    df["ggg"] = pd.to_numeric(df["ggg"], errors="coerce")

    x = np.arange(len(DIFFUSIVITIES))
    is_tumor = df["is_tumor"] == True  # noqa: E712

    normal = df[~is_tumor]
    ggg1 = df[is_tumor & (df["ggg"] == 1)]
    ggg_ge2 = df[is_tumor & (df["ggg"] >= 2)]
    ggg2 = df[is_tumor & (df["ggg"] == 2)]
    ggg_ge3 = df[is_tumor & (df["ggg"] >= 3)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    ax_l, ax_r = axes

    # LEFT — emergence: Normal baseline + GGG=1 + GGG>=2.
    h_normal = plot_group(ax_l, x, normal, "Normal", GROUP_NORMAL)
    h_ggg1 = plot_group(ax_l, x, ggg1, "GGG = 1", C_GGG1)
    h_ggg_ge2 = plot_group(ax_l, x, ggg_ge2, "GGG ≥ 2", C_GGG_GE2)
    ax_l.set_title("Tumour emergence (GGG 1 vs ≥2)", fontsize=20, pad=8)

    # RIGHT — aggressiveness: Normal baseline + GGG=2 + GGG>=3.
    plot_group(ax_r, x, normal, "Normal", GROUP_NORMAL)
    h_ggg2 = plot_group(ax_r, x, ggg2, "GGG = 2", C_GGG2)
    h_ggg_ge3 = plot_group(ax_r, x, ggg_ge3, "GGG ≥ 3", C_GGG_GE3)
    ax_r.set_title("Aggressiveness (GGG 2 vs ≥3)", fontsize=20, pad=8)

    for ax in axes:
        set_diff_xaxis(ax, label=True)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.4)
        ax.margins(x=0.02)
    ax_l.set_ylabel(r"spectral fraction $R_j$")

    # SINGLE shared figure-level legend on top (matches Fig 1 / Fig 3). Normal
    # is listed ONCE; each grade group is its own distinct red. One row of 5
    # entries keeps the legend width within the two-subplot span.
    legend_handles = [h_normal, h_ggg1, h_ggg_ge2, h_ggg2, h_ggg_ge3]
    fig.legend(
        legend_handles, [h.get_label() for h in legend_handles],
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=5, frameon=True, framealpha=0.95, fontsize=20,
        handlelength=1.4, columnspacing=1.0, handletextpad=0.35,
    )

    # Vertical order is: top legend -> panel subtitle -> panel. The panel
    # subtitles (set_title, 20 pt) sit at top=0.80; the shared legend (also
    # 20 pt, equal to the titles) sits well above them at y=1.02, leaving a
    # clear gap so the legend does not overshadow the titles. No tight_layout
    # (it would override the legend->title gap).
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.07, right=0.97, wspace=0.08)

    PAPER_PNG.parent.mkdir(parents=True, exist_ok=True)
    for path in (PAPER_PNG, OUT_PNG):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    for path in (PAPER_PDF, OUT_PDF):
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    for path in (PAPER_PNG, PAPER_PDF, OUT_PNG, OUT_PDF):
        print(f"Wrote {path.relative_to(REPO_ROOT)}")

    # ---- Honest reporting table for the caption ----
    panels = [
        ("LEFT (emergence)", [("Normal", normal), ("GGG = 1", ggg1), ("GGG >= 2", ggg_ge2)]),
        ("RIGHT (aggressiveness)", [("Normal", normal), ("GGG = 2", ggg2), ("GGG >= 3", ggg_ge3)]),
    ]
    print("\nPer-group mean fraction R_j by diffusivity bin (NUTS posterior mean):")
    header = "  group       n  " + "".join(f"{d:>8g}" for d in DIFFUSIVITIES)
    for panel_label, groups in panels:
        print(f"\n {panel_label}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for label, sub in groups:
            spec = sub[NUTS_COLS].to_numpy(dtype=float)
            mean = spec.mean(axis=0)
            row = f"  {label:<10s} {len(sub):>3d}  " + "".join(f"{v:>8.3f}" for v in mean)
            print(row)

    n_ggg0 = int((is_tumor & (df["ggg"] == 0)).sum())
    n_ungraded = int((is_tumor & df["ggg"].isna()).sum())
    print(
        f"\nExcluded tumor ROIs: {n_ggg0 + n_ungraded} total "
        f"({n_ggg0} with GGG=0, {n_ungraded} ungraded/missing GGG)."
    )


if __name__ == "__main__":
    main()
