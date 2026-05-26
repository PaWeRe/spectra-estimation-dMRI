"""Fig. 4 (main text) — LR coefficient profile per bin + ADC sensitivity overlay.

Replaces the old Fig. 4 (ADC-sensitivity vs LR-coef as a single regularizer-
dependent r = -0.98 plot). This version is the layered "why ADC works"
evidence assembled from NUTS posteriors and corrected MAP solver output:

  Panel (a) — Per-bin LR coefficients (NUTS-features, tumor-vs-normal),
              PZ and TZ side-by-side. Raw and standardized (per-SD) shown
              as two sub-panels. Outer bins (D = 0.25, D = 3.0) carry the
              strong opposing-sign weights; intermediates are small.

  Panel (b) — Per-bin LR coefficient profile (NUTS w_std) overlaid against
              the ADC sensitivity vector ∂ADC/∂R_j at the average-tumor
              operating point per zone. ADC sensitivity is smooth and
              roughly monotone in D; LR coefs are peakier at the outer
              bins; they anti-correlate (PZ r ≈ -0.79, TZ r ≈ -0.88).

Inputs:
  - results/biomarkers/lr_coef_decomp.csv   (per-bin LR weights, NUTS feats)
  - results/biomarkers/adc_sensitivity.csv  (∂ADC/∂R_j, NUTS features rows)

Outputs:
  - paper/figures/fig4_v1.png  (300 dpi)
  - paper/figures/fig4_v1.pdf

Usage:
    uv run python scripts/fig4_lr_coefs_and_sensitivity.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
LR_COEF_CSV = REPO_ROOT / "results" / "biomarkers" / "lr_coef_decomp.csv"
ADC_SENS_CSV = REPO_ROOT / "results" / "biomarkers" / "adc_sensitivity.csv"
OUT_DIR = REPO_ROOT / "paper" / "figures"
OUT_PNG = OUT_DIR / "fig4_v1.png"
OUT_PDF = OUT_DIR / "fig4_v1.pdf"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
N_BINS = len(DIFFUSIVITIES)
RAW_COLS = [f"w_raw_D_{d:.2f}" for d in DIFFUSIVITIES]
STD_COLS = [f"w_std_D_{d:.2f}" for d in DIFFUSIVITIES]

# Zone palette (PZ blue, TZ orange — matches manuscript convention)
ZONE_COLOR = {"pz": "#1f77b4", "tz": "#ff7f0e"}
ZONE_LABEL = {"pz": "PZ (n=81)", "tz": "TZ (n=68)"}
ZONE_ORDER = ["pz", "tz"]

# Sensitivity: red (tumor-direction; we plot -sens so that positive bars =
# "ADC decreases when this bin grows" = tumor-direction)
SENS_COLOR = "#d62728"

# Style spec (per Patrick's request)
FS_TITLE = 15
FS_AXIS = 17
FS_TICK = 17
FS_LEGEND = 15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_vec(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s), dtype=float)


def load_lr_coefs() -> dict[str, dict[str, np.ndarray]]:
    """Return {zone: {'raw': w_raw_vec, 'std': w_std_vec}} for tumor-vs-normal."""
    df = pd.read_csv(LR_COEF_CSV)
    df = df[df["task"] == "tumor_vs_normal"].copy()
    out: dict[str, dict[str, np.ndarray]] = {}
    for zone in ZONE_ORDER:
        row = df[df["zone"] == zone].iloc[0]
        out[zone] = {
            "raw": row[RAW_COLS].to_numpy(dtype=float),
            "std": row[STD_COLS].to_numpy(dtype=float),
            "n": int(row["n"]),
            "n_pos": int(row["n_pos"]),
            "n_neg": int(row["n_neg"]),
        }
    return out


def load_adc_sensitivity() -> dict[str, dict[str, float | np.ndarray]]:
    """Return {zone: {'sens_tumor':vec, 'sens_normal':vec, 'w_std':vec, r_p, r_s}}
    for the NUTS-features row at C=1.0."""
    df = pd.read_csv(ADC_SENS_CSV)
    df = df[(df["features"] == "NUTS") & (df["C"] == 1.0)].copy()
    df["sensitivity"] = df["sensitivity"].apply(parse_vec)
    df["lr_coefs"] = df["lr_coefs"].apply(parse_vec)
    out: dict[str, dict[str, float | np.ndarray]] = {}
    for zone in ["PZ", "TZ"]:
        sub = df[df["zone"] == zone]
        t = sub[sub["operating_point"] == "tumor"].iloc[0]
        n = sub[sub["operating_point"] == "normal"].iloc[0]
        out[zone.lower()] = {
            "sens_tumor": t["sensitivity"],
            "sens_normal": n["sensitivity"],
            "w_std": t["lr_coefs"],
            "r_pearson_tumor": float(t["r_pearson"]),
            "r_spearman_tumor": float(t["r_spearman"]),
            "r_pearson_normal": float(n["r_pearson"]),
            "r_spearman_normal": float(n["r_spearman"]),
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main() -> None:
    lr = load_lr_coefs()
    sens = load_adc_sensitivity()

    # ------- Layout: 2 rows × 2 cols.
    # Row 1: panel (a) — left = raw, right = standardized.
    # Row 2: panel (b) — left = PZ (overlay), right = TZ (overlay).
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    plt.subplots_adjust(wspace=0.45, hspace=0.35)

    x = np.arange(N_BINS)
    xlabels = [f"{d:g}" for d in DIFFUSIVITIES]
    bw = 0.36

    # ====================================================================
    # Panel (a): per-bin LR coefficients, PZ vs TZ grouped, raw + std.
    # ====================================================================
    for ax, key, ylab, title in [
        (axes[0, 0], "raw", r"LR coefficient $w_{\mathrm{raw}}$",
         r"(a) Per-bin LR coefficient — raw"),
        (axes[0, 1], "std", r"LR coefficient $w_{\mathrm{std}}$ (per-SD)",
         r"(a) Per-bin LR coefficient — standardized"),
    ]:
        for i, zone in enumerate(ZONE_ORDER):
            off = (i - 0.5) * bw
            vals = lr[zone][key]
            ax.bar(
                x + off, vals, bw,
                color=ZONE_COLOR[zone], edgecolor="black", linewidth=0.5,
                label=ZONE_LABEL[zone],
            )
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=FS_TICK, rotation=0)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.tick_params(axis="x", labelsize=FS_TICK)
        ax.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)", fontsize=FS_AXIS)
        ax.set_ylabel(ylab, fontsize=FS_AXIS)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.grid(axis="y", alpha=0.25)
        # Highlight outer bins (D=0.25 and D=3.0) with a soft tumor / normal tint.
        ax.axvspan(-0.5, 0.5, color="#d62728", alpha=0.06, zorder=0)  # D=0.25 (tumor-dir)
        ax.axvspan(5.5, 6.5, color="#1f77b4", alpha=0.06, zorder=0)   # D=3.0  (normal-dir)
        # Legend outside.
        ax.legend(
            fontsize=FS_LEGEND, frameon=True,
            loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
        )

    # ====================================================================
    # Panel (b): LR w_std vs ADC sensitivity overlay, per zone.
    # ====================================================================
    for ax, zone in zip([axes[1, 0], axes[1, 1]], ZONE_ORDER):
        s = sens[zone]
        w = s["w_std"]
        # Sign convention: plot +(-sens) so positive = tumor-direction; this
        # makes the anti-correlation visually obvious (bars align if perfectly
        # anti-correlated).
        sens_plot = -s["sens_tumor"]
        r_p = s["r_pearson_tumor"]
        r_s = s["r_spearman_tumor"]

        ax_l = ax
        ax_r = ax.twinx()

        ax_l.bar(
            x - bw / 2, w, bw,
            color=ZONE_COLOR[zone], edgecolor="black", linewidth=0.4,
            label=r"NUTS LR $w_{\mathrm{std}}$",
        )
        ax_r.bar(
            x + bw / 2, sens_plot, bw,
            color=SENS_COLOR, edgecolor="black", linewidth=0.4, alpha=0.85,
            label=r"$-\,\partial\mathrm{ADC}/\partial R_j$ (tumor op.)",
        )
        ax_l.axhline(0, color="k", lw=0.7)
        ax_l.set_xticks(x)
        ax_l.set_xticklabels(xlabels, fontsize=FS_TICK)
        ax_l.tick_params(axis="y", labelsize=FS_TICK, labelcolor=ZONE_COLOR[zone])
        ax_r.tick_params(axis="y", labelsize=FS_TICK, labelcolor=SENS_COLOR)
        ax_l.set_xlabel(r"Diffusivity $D$  ($\mu$m$^2$/ms)", fontsize=FS_AXIS)
        ax_l.set_ylabel(r"LR $w_{\mathrm{std}}$ (NUTS)",
                        fontsize=FS_AXIS, color=ZONE_COLOR[zone])
        ax_r.set_ylabel(r"$-\,\partial\mathrm{ADC}/\partial R_j$",
                        fontsize=FS_AXIS, color=SENS_COLOR)
        ax_l.set_title(
            f"(b) {zone.upper()}  —  Pearson r = {r_p:+.2f},  "
            f"Spearman $\\rho$ = {r_s:+.2f}",
            fontsize=FS_TITLE,
        )
        ax_l.grid(axis="y", alpha=0.25)
        ax_l.axvspan(-0.5, 0.5, color="#d62728", alpha=0.06, zorder=0)
        ax_l.axvspan(5.5, 6.5, color="#1f77b4", alpha=0.06, zorder=0)

        # Combined legend outside, to the right of the right-axis labels.
        h1, l1 = ax_l.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax_r.legend(
            h1 + h2, l1 + l2,
            fontsize=FS_LEGEND, frameon=True,
            loc="upper left", bbox_to_anchor=(1.15, 1.0), borderaxespad=0.0,
        )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")

    # -------- Console table for the report --------
    print("\nPer-bin LR coefficients (NUTS features, tumor-vs-normal):")
    header = "  D (um^2/ms) | " + " | ".join(f"{d:>6g}" for d in DIFFUSIVITIES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for zone in ZONE_ORDER:
        r = lr[zone]["raw"]
        s = lr[zone]["std"]
        print(f"  {zone.upper()} w_raw    | " + " | ".join(f"{v:+6.2f}" for v in r))
        print(f"  {zone.upper()} w_std    | " + " | ".join(f"{v:+6.2f}" for v in s))
    print()
    for zone in ["pz", "tz"]:
        p = sens[zone]
        print(
            f"  {zone.upper()}: Pearson(w_std, -sens_tumor) = "
            f"{p['r_pearson_tumor']:+.3f},  "
            f"Spearman = {p['r_spearman_tumor']:+.3f}"
        )


if __name__ == "__main__":
    main()
