"""Fig 5 — Average diffusivity spectrum by Gleason Grade Group.

Three-curve overlay: normal (n=109), GGG=1 (n=8), GGG≥2 (n=21).
Bands are 95% CI of the mean (t-based) across ROIs in each group.

Spectra are NUTS posterior-mean spectra (nuts_D_*) from
results/biomarkers/features.csv.

Outputs:
    paper/figures/fig5_v2.{png,pdf}      — paper-grade overlay (main)
    results/biomarkers/spectrum_by_ggg.{png,pdf}        — archival copy
    results/biomarkers/spectrum_by_ggg_split.{png,pdf}  — 2-panel split view

Usage:
    uv run python scripts/plot_spectrum_by_ggg.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
FEAT_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"

PAPER_PNG = REPO_ROOT / "paper" / "figures" / "fig5_v2.png"
PAPER_PDF = REPO_ROOT / "paper" / "figures" / "fig5_v2.pdf"
OUT_PNG = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.png"
OUT_PDF = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.pdf"
OUT_SPLIT_PNG = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg_split.png"
OUT_SPLIT_PDF = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg_split.pdf"

DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]

# Match Fig 1 / Fig 3 paper styling.
mpl.rcParams.update({
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.labelsize": 17,
    "axes.titlesize": 15,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})

GROUP_NORMAL = "#555555"   # neutral baseline
GROUP_GGG1 = "#1f77b4"     # cool — low grade
GROUP_GGG_HI = "#d62728"   # warm — higher grade


def group_stats(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = spec.shape[0]
    mean = spec.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    sem = spec.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * sem, mean + t_crit * sem


def draw_curves(
    ax,
    x: np.ndarray,
    groups: list[tuple[str, pd.DataFrame, str]],
    *,
    show_legend: bool = True,
    band_alpha: float = 0.15,
) -> None:
    for label, sub, color in groups:
        spec = sub[NUTS_COLS].to_numpy(dtype=float)
        mean, lo, hi = group_stats(spec)
        ax.plot(
            x, mean, "-o", color=color, lw=2.5, ms=7,
            label=f"{label}  (n={len(sub)})",
        )
        if len(sub) >= 2:
            ax.fill_between(x, lo, hi, color=color, alpha=band_alpha)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in DIFFUSIVITIES])
    ax.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)")
    ax.set_ylabel("NUTS posterior-mean spectrum mass  $R_j$")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.4)
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)


def main() -> None:
    df = pd.read_csv(FEAT_CSV)
    df["ggg"] = pd.to_numeric(df["ggg"], errors="coerce")

    x = np.arange(len(DIFFUSIVITIES))
    normal = ("Normal", df[df["is_tumor"] == False], GROUP_NORMAL)
    ggg1 = ("GGG = 1", df[(df["is_tumor"] == True) & (df["ggg"] == 1)], GROUP_GGG1)
    ggg_hi = ("GGG ≥ 2", df[(df["is_tumor"] == True) & (df["ggg"] >= 2)], GROUP_GGG_HI)

    # Main paper figure: 3-curve overlay, legend outside.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    draw_curves(ax, x, [normal, ggg1, ggg_hi])
    fig.tight_layout()
    PAPER_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PAPER_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PAPER_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {PAPER_PDF.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")

    # Supplementary split: GGG=1-vs-normal | GGG≥2-vs-normal.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    draw_curves(axes[0], x, [normal, ggg1], show_legend=True, band_alpha=0.18)
    axes[0].set_title("GGG = 1 (low-grade tumor)")
    draw_curves(axes[1], x, [normal, ggg_hi], show_legend=True, band_alpha=0.18)
    axes[1].set_title("GGG ≥ 2 (intermediate/high-grade tumor)")
    fig.tight_layout()
    fig.savefig(OUT_SPLIT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_SPLIT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_SPLIT_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_SPLIT_PDF.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
