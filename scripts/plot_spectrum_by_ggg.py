"""Plot the average diffusivity spectrum per Gleason Grade Group (GGG).

Uses NUTS posterior-mean spectra (nuts_D_*) from results/biomarkers/features.csv.
Groups: normal (is_tumor=False), then tumor ROIs by GGG ∈ {1,2,3,4,5}.
Shows pooled (PZ+TZ) and PZ-only panels. Bands are 95% CI of the mean
(t-based for n<30) across ROIs in each group.

GGG distribution (tumor ROIs, valid GGG only, n=29):
    GGG=1 pz=7 tz=1   GGG=2 pz=10 tz=2   GGG=3 pz=2 tz=3
    GGG=4 pz=1 tz=1   GGG=5 pz=1 tz=1

Note small-group caveat: GGG=4,5 have n=2 each, GGG=3 has n=5 total
(2 PZ + 3 TZ). CIs will be wide.

Outputs: results/biomarkers/spectrum_by_ggg.{png,pdf}

Usage:
    uv run python scripts/plot_spectrum_by_ggg.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
FEAT_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"
OUT_PNG = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.png"
OUT_PDF = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.pdf"
OUT_SPLIT_PNG = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg_split.png"
OUT_SPLIT_PDF = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg_split.pdf"

DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]

GROUP_COLORS = {
    "normal": "#777777",
    "GGG=1": "#1f77b4",
    "GGG=2": "#2ca02c",
    "GGG=3": "#ff7f0e",
    "GGG=4": "#d62728",
    "GGG=5": "#7f0000",
}


def group_stats(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = spec.shape[0]
    mean = spec.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    sem = spec.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * sem, mean + t_crit * sem


def draw_curves(ax, x: np.ndarray, groups: list[tuple[str, pd.DataFrame, str]]) -> None:
    for label, sub, color in groups:
        spec = sub[NUTS_COLS].to_numpy(dtype=float)
        mean, lo, hi = group_stats(spec)
        ax.plot(x, mean, "-o", color=color, lw=2.2, ms=6, label=f"{label}  (n={len(sub)})")
        if len(sub) >= 2:
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in DIFFUSIVITIES])
    ax.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)")
    ax.set_ylabel("NUTS posterior-mean spectrum mass  $R_j$")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.4)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)


def main() -> None:
    df = pd.read_csv(FEAT_CSV)
    df["ggg"] = pd.to_numeric(df["ggg"], errors="coerce")

    x = np.arange(len(DIFFUSIVITIES))
    normal = ("normal", df[df["is_tumor"] == False], GROUP_COLORS["normal"])
    ggg1 = ("GGG = 1", df[(df["is_tumor"] == True) & (df["ggg"] == 1)], "#1f77b4")
    ggg_hi = ("GGG ≥ 2", df[(df["is_tumor"] == True) & (df["ggg"] >= 2)], "#d62728")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    draw_curves(ax, x, [normal, ggg1, ggg_hi])
    fig.suptitle(
        "Average NUTS diffusivity spectrum: normal vs GGG=1 vs GGG≥2  ·  "
        f"source: {FEAT_CSV.relative_to(REPO_ROOT)}  ·  bands = 95% CI of mean",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    draw_curves(axes[0], x, [normal, ggg1])
    axes[0].set_title("GGG = 1 (low-grade tumor)")
    draw_curves(axes[1], x, [normal, ggg_hi])
    axes[1].set_title("GGG ≥ 2 (intermediate/high-grade tumor)")
    fig.suptitle(
        "Average NUTS diffusivity spectrum: GGG=1 vs GGG≥2 (normal reference)  ·  "
        f"source: {FEAT_CSV.relative_to(REPO_ROOT)}  ·  bands = 95% CI of mean",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_SPLIT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_SPLIT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_SPLIT_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_SPLIT_PDF.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
