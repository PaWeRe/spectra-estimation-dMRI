"""Plot per-bin LR weights (w_raw) vs diffusivity D from lr_coef_decomp.csv.

Visualizes NUTS 8-bin LR coefficients across (zone × task) rows. Companion to
F4 in PROJECT_STATE — note this is per-bin weight magnitude, NOT the 8→2
collapse (which lives in bin_information_sweep.csv, F2).

Usage:
    uv run python scripts/plot_lr_weights_per_bin.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COEF_CSV = REPO_ROOT / "results" / "biomarkers" / "lr_coef_decomp.csv"
OUT_DIR = REPO_ROOT / "results" / "biomarkers"
OUT_PNG = OUT_DIR / "lr_weights_per_bin.png"
OUT_PDF = OUT_DIR / "lr_weights_per_bin.pdf"

DIFFUSIVITIES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]
RAW_COLS = [f"w_raw_D_{d:.2f}" for d in DIFFUSIVITIES]
STD_COLS = [f"w_std_D_{d:.2f}" for d in DIFFUSIVITIES]


def main() -> None:
    df = pd.read_csv(COEF_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    x = np.arange(len(DIFFUSIVITIES))
    width = 0.18
    colors = {
        ("pooled", "tumor_vs_normal"): "#1f77b4",
        ("pz", "tumor_vs_normal"): "#2ca02c",
        ("tz", "tumor_vs_normal"): "#ff7f0e",
        ("pooled", "ggg_ge_3"): "#d62728",
        ("pz", "ggg_ge_3"): "#9467bd",
    }
    label_map = {
        ("pooled", "tumor_vs_normal"): "pooled · tumor vs normal (n=149)",
        ("pz", "tumor_vs_normal"): "PZ · tumor vs normal (n=81)",
        ("tz", "tumor_vs_normal"): "TZ · tumor vs normal (n=68)",
        ("pooled", "ggg_ge_3"): "pooled · GGG≥3 (n=29)",
        ("pz", "ggg_ge_3"): "PZ · GGG≥3 (n=21)",
    }

    rows = list(df.iterrows())
    offsets = np.linspace(-(len(rows) - 1) / 2, (len(rows) - 1) / 2, len(rows)) * width

    for ax, cols, title in [
        (axes[0], RAW_COLS, "Raw-space LR weights  $w_{\\mathrm{raw}}$"),
        (axes[1], STD_COLS, "Standardized LR weights  $w_{\\mathrm{std}}$ (per-SD)"),
    ]:
        for (idx, row), off in zip(rows, offsets):
            key = (row["zone"], row["task"])
            vals = row[cols].to_numpy(dtype=float)
            ax.bar(
                x + off,
                vals,
                width,
                color=colors.get(key, "gray"),
                label=label_map.get(key, f"{row['zone']} · {row['task']}"),
                edgecolor="black",
                linewidth=0.4,
            )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:g}" for d in DIFFUSIVITIES])
        ax.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("LR coefficient")
    axes[1].set_ylabel("LR coefficient (per-SD)")
    axes[0].legend(fontsize=8, loc="best", frameon=True)
    fig.suptitle(
        "NUTS 8-bin Logistic-Regression weights per diffusivity bin  ·  "
        f"source: {COEF_CSV.relative_to(REPO_ROOT)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
