"""Fig 5 — Average diffusivity spectrum by Gleason Grade Group (two thresholds).

Two-panel overlay sharing a common y-axis, normal tissue as the gray baseline
in both panels:

  Panel A — tumor EMERGENCE boundary: Normal (n=109) | GGG=1 (n=8) | GGG>=2 (n=21).
      The detection axis: the free-water/lumen bin (D=3.0) collapses the instant
      tumor appears (even low grade), then barely moves.

  Panel B — tumor AGGRESSIVENESS boundary: Normal | GGG<=2 (n=20) | GGG>=3 (n=9).
      The grading axis: between favorable and unfavorable grade the lumen bin is
      already saturated, while the restricted-cellular (D=0.25) and
      glandular-epithelial (D=2.0) bins keep shifting.

Together the two thresholds show that detection signal fires once at onset
(outer free-water bin) while grading signal continues to track the
restricted + intermediate/lumen bins where ADC is least sensitive.

Spectra are NUTS posterior-mean spectra (nuts_D_*) from
results/biomarkers/features.csv. Bands are 95% CI of the group mean (t-based,
df = n-1) across ROIs; they reflect between-ROI variability of the per-ROI
posterior mean and do NOT include within-ROI posterior uncertainty.

Outputs:
    paper/figures/fig5_v3.{png,pdf}                  — paper-grade two-panel (main)
    results/biomarkers/spectrum_by_ggg.{png,pdf}     — archival copy

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

PAPER_PNG = REPO_ROOT / "paper" / "figures" / "fig5_v3.png"
PAPER_PDF = REPO_ROOT / "paper" / "figures" / "fig5_v3.pdf"
OUT_PNG = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.png"
OUT_PDF = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.pdf"

DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]

# Match Fig 2 / Fig 3 paper styling (apparent-size matched: this is a 1x2 grid).
mpl.rcParams.update({
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 17,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})

GROUP_NORMAL = "#555555"   # neutral baseline (normal tissue)
GROUP_LOW = "#1f77b4"      # cool — lower-grade tumor group
GROUP_HIGH = "#d62728"     # warm — higher-grade tumor group


def group_stats(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = spec.shape[0]
    mean = spec.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    sem = spec.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * sem, mean + t_crit * sem


def draw_panel(
    ax,
    x: np.ndarray,
    groups: list[tuple[str, pd.DataFrame, str]],
    *,
    title: str,
    show_ylabel: bool,
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
    if show_ylabel:
        ax.set_ylabel("NUTS posterior-mean spectrum mass  $R_j$")
    ax.set_title(title, pad=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.4)
    # Legend on TOP of the panel (consistent with Figs 2 & 3), placed in the
    # empty upper-left region (curves peak at the right, D=2-3).
    ax.legend(loc="upper left", framealpha=0.95, handlelength=1.8,
              borderaxespad=0.5)


def main() -> None:
    df = pd.read_csv(FEAT_CSV)
    df["ggg"] = pd.to_numeric(df["ggg"], errors="coerce")

    x = np.arange(len(DIFFUSIVITIES))
    is_tumor = df["is_tumor"] == True  # noqa: E712

    normal = ("Normal", df[~is_tumor], GROUP_NORMAL)
    # Panel A — emergence boundary (low grade vs any clinically significant).
    ggg1 = ("GGG = 1", df[is_tumor & (df["ggg"] == 1)], GROUP_LOW)
    ggg_ge2 = ("GGG ≥ 2", df[is_tumor & (df["ggg"] >= 2)], GROUP_HIGH)
    # Panel B — aggressiveness boundary (favorable vs unfavorable grade).
    ggg_le2 = ("GGG ≤ 2", df[is_tumor & (df["ggg"] >= 1) & (df["ggg"] <= 2)], GROUP_LOW)
    ggg_ge3 = ("GGG ≥ 3", df[is_tumor & (df["ggg"] >= 3)], GROUP_HIGH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    draw_panel(
        axes[0], x, [normal, ggg1, ggg_ge2],
        title="Tumor emergence  (GGG = 1 vs ≥ 2)", show_ylabel=True,
    )
    draw_panel(
        axes[1], x, [normal, ggg_le2, ggg_ge3],
        title="Tumor aggressiveness  (GGG ≤ 2 vs ≥ 3)", show_ylabel=False,
    )
    fig.tight_layout()

    PAPER_PNG.parent.mkdir(parents=True, exist_ok=True)
    for path in (PAPER_PNG, OUT_PNG):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    for path in (PAPER_PDF, OUT_PDF):
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    for path in (PAPER_PNG, PAPER_PDF, OUT_PNG, OUT_PDF):
        print(f"Wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
