"""Fig 5 — Grade-ordered ladder of average diffusivity spectra.

Single full-width panel: four mean NUTS posterior-mean spectra over the 8
diffusivity bins (D = 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0 um^2/ms),
ordered by increasing Gleason Grade Group:

  * Normal     (n=109) — gray shared baseline (benign tissue).
  * GGG = 1    (n=8)   — resolved grade group 1.
  * GGG = 2    (n=12)  — resolved grade group 2.
  * GGG >= 3   (n=9)   — cumulative grade groups 3-5 (resolved high grades have
                          tiny n: GGG3=5, GGG4=2, GGG5=2).

The three tumor groups use a grade-ordered sequential ramp (light -> dark
green/teal), deliberately avoiding the tumor-red / normal-blue and
MAP-green / NUTS-orange conventions used elsewhere in the paper. Normal is
neutral gray.

Bands are 95% CI of the group mean (Student-t, df = n-1) across ROIs; they
reflect between-ROI variability of the per-ROI posterior mean and do NOT
include within-ROI posterior uncertainty.

Tumor ROIs with GGG = 0 (n=7) or ungraded / missing GGG (n=4) — 11 ROIs total —
are excluded from this figure.

Spectra are NUTS posterior-mean spectra (nuts_D_*) from
results/biomarkers/features.csv.

Outputs:
    paper/figures/fig5_v4.{png,pdf}                  — paper-grade single panel (main)
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

PAPER_PNG = REPO_ROOT / "paper" / "figures" / "fig5_v4.png"
PAPER_PDF = REPO_ROOT / "paper" / "figures" / "fig5_v4.pdf"
OUT_PNG = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.png"
OUT_PDF = REPO_ROOT / "results" / "biomarkers" / "spectrum_by_ggg.pdf"

DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]

# Single-panel full-width: fonts large but a touch below the 1x2 grid sizes
# (single panel renders bigger at \textwidth, so don't over-size). Reference is
# "same size as Fig 2".
mpl.rcParams.update({
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.labelsize": 19,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})

# Normal = neutral gray. Tumor grades = sequential teal->dark-green ramp,
# colorblind-safe and distinct from red/blue and green/orange conventions.
GROUP_NORMAL = "#7f7f7f"   # neutral gray baseline (benign)
GGG1_COLOR = "#66c2a4"   # light teal
GGG2_COLOR = "#2ca25f"   # mid green
GGG3_COLOR = "#005824"   # dark green


def group_stats(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = spec.shape[0]
    mean = spec.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    sem = spec.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * sem, mean + t_crit * sem


def main() -> None:
    df = pd.read_csv(FEAT_CSV)
    df["ggg"] = pd.to_numeric(df["ggg"], errors="coerce")

    x = np.arange(len(DIFFUSIVITIES))
    is_tumor = df["is_tumor"] == True  # noqa: E712

    groups: list[tuple[str, pd.DataFrame, str]] = [
        ("Normal", df[~is_tumor], GROUP_NORMAL),
        ("GGG = 1", df[is_tumor & (df["ggg"] == 1)], GGG1_COLOR),
        ("GGG = 2", df[is_tumor & (df["ggg"] == 2)], GGG2_COLOR),
        ("GGG ≥ 3", df[is_tumor & (df["ggg"] >= 3)], GGG3_COLOR),
    ]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for label, sub, color in groups:
        spec = sub[NUTS_COLS].to_numpy(dtype=float)
        mean, lo, hi = group_stats(spec)
        ax.plot(
            x, mean, "-o", color=color, lw=2.6, ms=7,
            label=f"{label}  (n={len(sub)})",
        )
        if len(sub) >= 2:
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in DIFFUSIVITIES])
    ax.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)")
    ax.set_ylabel("NUTS posterior-mean spectrum mass  $R_j$")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.4)
    ax.margins(x=0.02)

    # Legend ON TOP, above the axes, in one row. No in-figure title (caption
    # carries the title — MRM convention).
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
        ncol=4, frameon=False, handlelength=1.8,
        columnspacing=1.4, borderaxespad=0.0,
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

    # ---- Honest reporting table for the caption ----
    print("\nPer-group mean fraction R_j by diffusivity bin (NUTS posterior mean):")
    header = "  group       n  " + "".join(f"{d:>8g}" for d in DIFFUSIVITIES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, sub, _ in groups:
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
    print(f"Tumor total = {int(is_tumor.sum())}; "
          f"included in figure = {8 + 12 + 9} (re-derived below).")
    incl = sum(len(s) for l, s, c in groups if l != "Normal")
    print(f"Included tumor ROIs (sum of GGG=1,2,>=3) = {incl}.")


if __name__ == "__main__":
    main()
