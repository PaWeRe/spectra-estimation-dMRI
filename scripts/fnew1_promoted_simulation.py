"""
F-new-1 (promoted figure for MRM submission)
============================================

Single 2-panel figure summarising the central methodology-honesty finding:
  At MAP λ ≈ 1e-4 — 1e-3 (the BWH-tuned range) MAP recovers REALISTIC
  prostate-like (log-normal) spectra essentially as well as NUTS, while
  NUTS retains a modest ~10–15 pp advantage on concentrated (δ, bimodal)
  ground truths.

  This justifies the manuscript's choice to use MAP at tuned λ as the
  point estimator, with NUTS reserved for the contribution it uniquely
  provides — per-bin posterior uncertainty σ_j.

Panels
------
  (a) Fraction of true mass recovered at the dominant true bin vs λ.
  (b) Total 8-bin MSE vs λ (log y).

Both panels at SNR = 400 (canonical). One line per ground truth shape,
coloured by GT category (δ grey, bimodal red/blue, log-normal green
emphasised). Each GT has a horizontal dashed line at the NUTS reference
value (NUTS does not depend on λ in the same way; it is a single point
per GT).

A shaded vertical band marks the BWH-tuned λ ∈ [1e-4, 1e-3] range.

Inputs
------
  results/simulation/map_lambda_sweep_summary.csv
  results/simulation/sim_summary.csv  (NUTS reference, λ=0.1)

Outputs
-------
  paper/figures/fnew1_simulation.png  (300 dpi)
  paper/figures/fnew1_simulation.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SUMMARY_CSV = "results/simulation/map_lambda_sweep_summary.csv"
NUTS_CSV = "results/simulation/sim_summary.csv"
OUT_PNG = "paper/figures/fnew1_simulation.png"
OUT_PDF = "paper/figures/fnew1_simulation.pdf"

SNR = 400
TUNED_LAMBDA_BAND = (1e-4, 1e-3)

# Ground truth ordering, labels, and style.
# δ-spectra → grey (unrealistic for prostate)
# bimodal → red / blue (intermediate realism)
# log-normal → green, emphasized (most prostate-realistic)
GT_STYLE = {
    "GT-A_d0.25":      dict(label=r"$\delta$ @ D=0.25",
                            color="#666666", category="delta",
                            ls="-",  lw=1.4, marker="s", ms=5),
    "GT-D_d3.00":      dict(label=r"$\delta$ @ D=3.00",
                            color="#aaaaaa", category="delta",
                            ls="-",  lw=1.4, marker="D", ms=5),
    "GT-E_bi-tumor":   dict(label="bimodal {0.25:0.7, 3.0:0.3} (tumour-like)",
                            color="#c0392b", category="bimodal",
                            ls="-",  lw=1.6, marker="^", ms=6),
    "GT-F_bi-norm":    dict(label="bimodal {0.25:0.3, 3.0:0.7} (normal-like)",
                            color="#2874a6", category="bimodal",
                            ls="-",  lw=1.6, marker="v", ms=6),
    "GT-H_lognorm0.5": dict(label=r"log-normal $\mu$=0.5 (realistic)",
                            color="#1e8449", category="lognormal",
                            ls="-",  lw=2.8, marker="o", ms=9),
    "GT-I_lognorm1.5": dict(label=r"log-normal $\mu$=1.5 (realistic)",
                            color="#52be80", category="lognormal",
                            ls="-",  lw=2.8, marker="o", ms=9),
}

GT_ORDER = list(GT_STYLE.keys())


def nuts_reference(gt: str, snr: int, df_nuts: pd.DataFrame):
    """Return (fraction_recovered_at_true_peaks, total_mse_8bins) for NUTS."""
    sub = df_nuts[(df_nuts["gt"] == gt) & (df_nuts["snr"] == snr)
                  & (df_nuts["estimator"] == "NUTS")]
    if len(sub) == 0:
        return None, None
    peak_mask = sub["R_true"] > 0.1
    frac = (sub[peak_mask]["R_hat_mean"].sum()
            / sub[peak_mask]["R_true"].sum())
    mse = sub["mse"].sum()
    return float(frac), float(mse)


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    map_df = pd.read_csv(SUMMARY_CSV)
    nuts_df = pd.read_csv(NUTS_CSV)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax_frac, ax_mse) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # Plot every GT on each panel
    for gt in GT_ORDER:
        sub = map_df[(map_df["gt"] == gt) & (map_df["snr"] == SNR)].sort_values("lambda")
        if sub.empty:
            continue
        x = sub["lambda"].values
        y_frac = sub["fraction_recovered"].values
        y_mse = sub["total_mse_8bins"].values
        style = GT_STYLE[gt]

        ax_frac.plot(x, y_frac, color=style["color"], lw=style["lw"],
                     ls=style["ls"], marker=style["marker"], ms=style["ms"],
                     label=style["label"], zorder=3 if style["category"] == "lognormal" else 2)
        ax_mse.plot(x, y_mse, color=style["color"], lw=style["lw"],
                    ls=style["ls"], marker=style["marker"], ms=style["ms"],
                    label=style["label"], zorder=3 if style["category"] == "lognormal" else 2)

        # NUTS reference horizontal line per GT (drawn only over the
        # right-hand sliver so it doesn't visually overlap all of the
        # MAP traces — the line indicates the NUTS asymptote per GT.)
        nuts_frac, nuts_mse = nuts_reference(gt, SNR, nuts_df)
        if nuts_frac is not None:
            ax_frac.axhline(nuts_frac, color=style["color"],
                            ls=":", lw=1.0, alpha=0.55, zorder=1)
            ax_mse.axhline(nuts_mse, color=style["color"],
                           ls=":", lw=1.0, alpha=0.55, zorder=1)
            # Mark NUTS reference with a star at λ=0.1 (NUTS sim was @ λ=0.1).
            ax_frac.plot(0.1, nuts_frac, marker="*", color=style["color"],
                         ms=14, mec="black", mew=0.8, zorder=5)
            ax_mse.plot(0.1, nuts_mse, marker="*", color=style["color"],
                        ms=14, mec="black", mew=0.8, zorder=5)

    # Tuned-λ band (BWH-optimum range), and manuscript-λ marker
    for ax in (ax_frac, ax_mse):
        ax.axvspan(TUNED_LAMBDA_BAND[0], TUNED_LAMBDA_BAND[1],
                   color="#f1c40f", alpha=0.18, zorder=0)
        ax.axvline(0.1, color="#d35400", ls="--", lw=1.1, alpha=0.7, zorder=0)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel(r"ridge $\lambda$")

    ax_frac.set_ylim(0.0, 1.18)
    ax_frac.set_ylabel("fraction of true mass recovered at dominant bin(s)")
    ax_frac.set_title("(a) Mass recovery", loc="left", fontweight="bold")

    ax_mse.set_yscale("log")
    ax_mse.set_ylabel("total 8-bin MSE (lower is better)")
    ax_mse.set_title("(b) Mean-squared error", loc="left", fontweight="bold")

    # Annotate the bands on the left panel
    band_mid_log = np.sqrt(TUNED_LAMBDA_BAND[0] * TUNED_LAMBDA_BAND[1])
    ax_frac.annotate(
        "BWH-tuned\n" + r"$\lambda \in [10^{-4}, 10^{-3}]$",
        xy=(band_mid_log, 1.16),
        ha="center", va="top",
        fontsize=9, color="#b7950b", fontweight="bold",
    )
    ax_frac.annotate(
        r"manuscript $\lambda$=0.1",
        xy=(0.11, 0.02),
        ha="left", va="bottom",
        fontsize=8.5, color="#d35400",
        rotation=90,
    )

    # Custom legend: combine the GT lines plus auxiliary entries
    gt_handles = [
        Line2D([0], [0], color=GT_STYLE[gt]["color"], lw=GT_STYLE[gt]["lw"],
               marker=GT_STYLE[gt]["marker"], ms=GT_STYLE[gt]["ms"],
               label=GT_STYLE[gt]["label"])
        for gt in GT_ORDER
    ]
    aux_handles = [
        Line2D([0], [0], color="black", marker="*", ms=13, lw=0,
               mec="black", mew=0.8,
               label="NUTS (per-GT)"),
        Patch(facecolor="#f1c40f", alpha=0.35,
              label=r"BWH-tuned $\lambda$ band"),
        Line2D([0], [0], color="#d35400", ls="--", lw=1.1,
               label=r"manuscript $\lambda$=0.1"),
    ]

    fig.legend(handles=gt_handles + aux_handles,
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle(
        f"MAP at tuned $\\lambda$ recovers realistic spectra as well as NUTS  "
        f"(simulation, SNR={SNR})",
        fontsize=13, fontweight="bold", y=1.00,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"  Wrote {OUT_PNG}")
    print(f"  Wrote {OUT_PDF}")
    plt.close(fig)

    # Diagnostic dump for the meeting
    print("\n=== Per-GT numbers at SNR=400 ===")
    print(f"{'GT':22s} | {'MAP best λ':>12s} | "
          f"{'MAP frac':>9s} {'NUTS frac':>9s} | "
          f"{'MAP MSE':>9s} {'NUTS MSE':>9s}")
    for gt in GT_ORDER:
        sub = map_df[(map_df["gt"] == gt) & (map_df["snr"] == SNR)]
        if sub.empty:
            continue
        best_row = sub.loc[sub["total_mse_8bins"].idxmin()]
        nuts_frac, nuts_mse = nuts_reference(gt, SNR, nuts_df)
        print(f"{gt:22s} | λ={best_row['lambda']:.0e}  | "
              f"{best_row['fraction_recovered']:9.3f} "
              f"{(nuts_frac if nuts_frac is not None else float('nan')):9.3f} | "
              f"{best_row['total_mse_8bins']:9.4f} "
              f"{(nuts_mse if nuts_mse is not None else float('nan')):9.4f}")


if __name__ == "__main__":
    main()
