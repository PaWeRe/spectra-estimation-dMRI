"""Compare NUTS LR standardized weights (w_std) to the ADC sensitivity vector.

For PZ and TZ (tumor-vs-normal), overlay the per-bin standardized LR
coefficient against ∂ADC/∂R_j at the average-tumor and average-normal
operating points, and scatter w_std vs sensitivity to show the rank/linear
correlation directly.

Context: PROJECT_STATE F4b — the manuscript's r ≈ −0.98 finding was largely
a λ=0.1 artifact. Under NUTS (or MAP at tuned λ) the correlation drops to
r ≈ −0.79 to −0.88, with Spearman holding up slightly better than Pearson.

Source: results/biomarkers/adc_sensitivity.csv
Outputs: results/biomarkers/lr_weights_vs_adc_sensitivity.{png,pdf}

Usage:
    uv run python scripts/plot_lr_weights_vs_adc_sensitivity.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = REPO_ROOT / "results" / "biomarkers" / "adc_sensitivity.csv"
OUT_PNG = REPO_ROOT / "results" / "biomarkers" / "lr_weights_vs_adc_sensitivity.png"
OUT_PDF = REPO_ROOT / "results" / "biomarkers" / "lr_weights_vs_adc_sensitivity.pdf"

DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])


def parse_vec(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=float)


def main() -> None:
    df = pd.read_csv(SRC_CSV)
    df = df[df["features"] == "NUTS"].copy()
    df["sensitivity"] = df["sensitivity"].apply(parse_vec)
    df["lr_coefs"] = df["lr_coefs"].apply(parse_vec)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    x = np.arange(len(DIFFUSIVITIES))
    xlabels = [f"{d:g}" for d in DIFFUSIVITIES]

    for row_idx, zone in enumerate(["PZ", "TZ"]):
        sub = df[df["zone"] == zone]
        w = sub.iloc[0]["lr_coefs"]
        sens_tumor = sub[sub["operating_point"] == "tumor"].iloc[0]["sensitivity"]
        sens_normal = sub[sub["operating_point"] == "normal"].iloc[0]["sensitivity"]
        r_t = sub[sub["operating_point"] == "tumor"].iloc[0]["r_pearson"]
        rs_t = sub[sub["operating_point"] == "tumor"].iloc[0]["r_spearman"]
        r_n = sub[sub["operating_point"] == "normal"].iloc[0]["r_pearson"]
        rs_n = sub[sub["operating_point"] == "normal"].iloc[0]["r_spearman"]

        ax = axes[row_idx, 0]
        ax2 = ax.twinx()
        ax.bar(x - 0.18, w, 0.36, color="#1f77b4", label=r"NUTS LR $w_{\mathrm{std}}$", edgecolor="k", lw=0.4)
        ax2.bar(x + 0.18, -sens_tumor, 0.36, color="#d62728", label=r"$-\,\partial\mathrm{ADC}/\partial R_j$ (tumor op.)", edgecolor="k", lw=0.4, alpha=0.85)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)")
        ax.set_ylabel(r"LR $w_{\mathrm{std}}$ (NUTS)", color="#1f77b4")
        ax2.set_ylabel(r"$-\,\partial\mathrm{ADC}/\partial R_j$", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(f"{zone}  ·  tumor op.  ·  Pearson r = {r_t:+.2f}, Spearman ρ = {rs_t:+.2f}")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower left")

        ax = axes[row_idx, 1]
        ax2 = ax.twinx()
        ax.bar(x - 0.18, w, 0.36, color="#1f77b4", edgecolor="k", lw=0.4)
        ax2.bar(x + 0.18, -sens_normal, 0.36, color="#d62728", edgecolor="k", lw=0.4, alpha=0.85)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)")
        ax.set_ylabel(r"LR $w_{\mathrm{std}}$ (NUTS)", color="#1f77b4")
        ax2.set_ylabel(r"$-\,\partial\mathrm{ADC}/\partial R_j$", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(f"{zone}  ·  normal op.  ·  Pearson r = {r_n:+.2f}, Spearman ρ = {rs_n:+.2f}")

        ax = axes[row_idx, 2]
        ax.scatter(sens_tumor, w, s=110, c="#1f77b4", edgecolor="k", label="tumor op.", zorder=3)
        ax.scatter(sens_normal, w, s=70, c="white", edgecolor="#1f77b4", label="normal op.", zorder=2)
        for xi, yi, d in zip(sens_tumor, w, DIFFUSIVITIES):
            ax.annotate(f"{d:g}", (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)
        xs = np.linspace(min(sens_tumor.min(), sens_normal.min()), max(sens_tumor.max(), sens_normal.max()), 50)
        slope, intercept, *_ = stats.linregress(sens_tumor, w)
        ax.plot(xs, slope * xs + intercept, "--", color="#1f77b4", lw=1.0, alpha=0.6, label="OLS (tumor op.)")
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_xlabel(r"ADC sensitivity  $\partial\mathrm{ADC}/\partial R_j$")
        ax.set_ylabel(r"NUTS LR $w_{\mathrm{std}}$")
        ax.set_title(f"{zone}  ·  scatter (8 bins, labelled by D)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "NUTS LR standardized weights vs ADC sensitivity vector  ·  per zone (tumor-vs-normal)  ·  "
        f"source: {SRC_CSV.relative_to(REPO_ROOT)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
