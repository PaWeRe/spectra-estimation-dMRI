"""
Figure 8 (new) — Simulation + CRLB diagnostic
==============================================

Replaces the previous Figure 8 (classical-CRLB vs NUTS bar plot at SNR=150),
whose ~factor-2000 gap was actually an apples-to-oranges comparison.

Two-panel main figure:

  (a) MAP-vs-NUTS spectrum recovery as a function of ridge λ on simulated
      ground truths at SNR=400. Re-styled from F-new-1: prostate-realistic
      log-normal GTs are emphasized; δ and bimodal GTs as background traces.
      Per-GT NUTS reference at λ=0.1 marked with a star. BWH-tuned-λ band
      [1e-4, 1e-3] shaded.

  (b) Three-bar comparison per D-component at cohort-median SNR=303:
        1. Classical unconstrained CRLB (no prior)
        2. Bayesian CRLB / van Trees lower bound (HalfNormal-as-Gaussian, λ=0.1)
        3. Empirical NUTS posterior std
      Log-y axis. Bayesian/NUTS gap ratio annotated above each bin.
      Honestly shows where the achieved precision comes from
      (data + prior + non-negativity).

Inputs
------
  results/simulation/map_lambda_sweep_summary.csv
  results/simulation/sim_summary.csv

Outputs
-------
  paper/figures/fig8_v1.png  (300 dpi)
  paper/figures/fig8_v1.pdf

Style
-----
  Axis labels 17 pt, ticks 17 pt, legend 15 pt, titles 15 pt.
  Legends placed outside the plotting axes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ---------------------------------------------------------------- paths
SUMMARY_CSV = "results/simulation/map_lambda_sweep_summary.csv"
NUTS_CSV = "results/simulation/sim_summary.csv"
OUT_PNG = "paper/figures/fig8_v1.png"
OUT_PDF = "paper/figures/fig8_v1.pdf"

# ---------------------------------------------------------------- params
PANEL_A_SNR = 400          # canonical simulation SNR for panel (a)
PANEL_B_SNR = 303          # cohort-median SNR for panel (b) (replaces old 150)
LAMBDA_PRIOR = 0.1         # HalfNormal prior precision (matches manuscript)
TUNED_LAMBDA_BAND = (1e-4, 1e-3)

# diffusivity bins (ms/μm² scaling on b)
B_VALUES = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750,
                     2000, 2250, 2500, 2750, 3000, 3250, 3500]) / 1000.0
D_BINS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
D_LABELS = [f"{d:g}" for d in D_BINS]


# ---------------------------------------------------------------- panel (a) styles
# δ-spectra → grey (unrealistic for prostate)
# bimodal → red / blue (intermediate realism)
# log-normal → green, emphasized (most prostate-realistic)
GT_STYLE = {
    "GT-A_d0.25":      dict(label=r"$\delta$ @ D=0.25",
                            color="#888888", category="delta",
                            ls="-",  lw=1.2, marker="s", ms=4, alpha=0.55),
    "GT-D_d3.00":      dict(label=r"$\delta$ @ D=3.00",
                            color="#b0b0b0", category="delta",
                            ls="-",  lw=1.2, marker="D", ms=4, alpha=0.55),
    "GT-E_bi-tumor":   dict(label="bimodal (tumour-like)",
                            color="#c0392b", category="bimodal",
                            ls="--", lw=1.4, marker="^", ms=5, alpha=0.75),
    "GT-F_bi-norm":    dict(label="bimodal (normal-like)",
                            color="#2874a6", category="bimodal",
                            ls="--", lw=1.4, marker="v", ms=5, alpha=0.75),
    "GT-H_lognorm0.5": dict(label=r"log-normal $\mu$=0.5 (realistic)",
                            color="#1e8449", category="lognormal",
                            ls="-",  lw=3.0, marker="o", ms=9, alpha=1.0),
    "GT-I_lognorm1.5": dict(label=r"log-normal $\mu$=1.5 (realistic)",
                            color="#58d68d", category="lognormal",
                            ls="-",  lw=3.0, marker="o", ms=9, alpha=1.0),
}
GT_ORDER = list(GT_STYLE.keys())


def nuts_reference(gt: str, snr: int, df_nuts: pd.DataFrame):
    """Per-GT NUTS reference: fraction of mass at true peaks, total 8-bin MSE."""
    sub = df_nuts[(df_nuts["gt"] == gt) & (df_nuts["snr"] == snr)
                  & (df_nuts["estimator"] == "NUTS")]
    if len(sub) == 0:
        return None, None
    peak_mask = sub["R_true"] > 0.1
    frac = (sub[peak_mask]["R_hat_mean"].sum()
            / sub[peak_mask]["R_true"].sum())
    mse = sub["mse"].sum()
    return float(frac), float(mse)


# ---------------------------------------------------------------- panel (b) data
def compute_crlb_bars(snr: float, lam: float):
    """Compute (unconstrained CRLB, Bayesian CRLB, NUTS empirical std) per D bin."""
    U = np.exp(-np.outer(B_VALUES, D_BINS))           # (15, 8) design matrix
    F_data = (snr ** 2) * (U.T @ U)
    F_post = F_data + lam * np.eye(len(D_BINS))
    crlb_unc = np.sqrt(np.diag(np.linalg.inv(F_data)))
    crlb_bay = np.sqrt(np.diag(np.linalg.inv(F_post)))

    # NUTS empirical std (matches notes/CRLB_NOTE_FOR_SANDY.md table)
    nuts_std = np.array([0.014, 0.030, 0.032, 0.041,
                         0.083, 0.121, 0.078, 0.016])
    return crlb_unc, crlb_bay, nuts_std


# ---------------------------------------------------------------- main
def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    map_df = pd.read_csv(SUMMARY_CSV)
    nuts_df = pd.read_csv(NUTS_CSV)

    # Per-spec style: ticks 17, labels 17, legend 15, title 15
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 17,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(20, 8.6))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.0, 1.05],
        wspace=0.28,
        left=0.06, right=0.78, top=0.88, bottom=0.12,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ============================================================ Panel (a)
    for gt in GT_ORDER:
        sub = map_df[(map_df["gt"] == gt) & (map_df["snr"] == PANEL_A_SNR)].sort_values("lambda")
        if sub.empty:
            continue
        x = sub["lambda"].values
        y = sub["fraction_recovered"].values
        st = GT_STYLE[gt]
        z = 4 if st["category"] == "lognormal" else 2
        ax_a.plot(x, y, color=st["color"], lw=st["lw"], ls=st["ls"],
                  marker=st["marker"], ms=st["ms"], alpha=st["alpha"],
                  label=st["label"], zorder=z)

        # NUTS reference: horizontal line + star at λ=0.1
        nuts_frac, _ = nuts_reference(gt, PANEL_A_SNR, nuts_df)
        if nuts_frac is not None:
            ax_a.axhline(nuts_frac, color=st["color"], ls=":", lw=0.9,
                         alpha=0.45 if st["category"] != "lognormal" else 0.75,
                         zorder=1)
            ax_a.plot(0.1, nuts_frac, marker="*", color=st["color"],
                      ms=15 if st["category"] == "lognormal" else 11,
                      mec="black", mew=0.8, zorder=6)

    # Tuned-λ band + manuscript-λ marker
    ax_a.axvspan(TUNED_LAMBDA_BAND[0], TUNED_LAMBDA_BAND[1],
                 color="#f1c40f", alpha=0.20, zorder=0)
    ax_a.axvline(0.1, color="#d35400", ls="--", lw=1.2, alpha=0.75, zorder=0)

    ax_a.set_xscale("log")
    ax_a.set_xlim(5e-7, 5)
    ax_a.set_ylim(0.0, 1.12)
    ax_a.set_xlabel(r"ridge $\lambda$")
    ax_a.set_ylabel("fraction of true mass at dominant bin(s)")
    ax_a.set_title("(a)  MAP vs NUTS spectrum recovery vs $\\lambda$"
                   f"  (sim, SNR={PANEL_A_SNR})", loc="left",
                   fontweight="bold", pad=14)
    ax_a.grid(True, which="both", alpha=0.25)

    # Annotate the band
    band_mid = np.sqrt(TUNED_LAMBDA_BAND[0] * TUNED_LAMBDA_BAND[1])
    ax_a.annotate("BWH-tuned\n" + r"$\lambda \in [10^{-4}, 10^{-3}]$",
                  xy=(band_mid, 1.10), ha="center", va="top",
                  fontsize=13, color="#b7950b", fontweight="bold")
    ax_a.annotate(r"manuscript $\lambda$=0.1",
                  xy=(0.115, 0.03), ha="left", va="bottom",
                  fontsize=12, color="#d35400", rotation=90)

    # Panel (a) legend OUTSIDE — upper right of figure
    gt_handles = [
        Line2D([0], [0], color=GT_STYLE[gt]["color"], lw=GT_STYLE[gt]["lw"],
               ls=GT_STYLE[gt]["ls"], marker=GT_STYLE[gt]["marker"],
               ms=GT_STYLE[gt]["ms"], label=GT_STYLE[gt]["label"])
        for gt in GT_ORDER
    ]
    aux_handles = [
        Line2D([0], [0], color="black", marker="*", ms=14, lw=0,
               mec="black", mew=0.8, label="NUTS reference (per GT)"),
        Patch(facecolor="#f1c40f", alpha=0.35,
              label=r"BWH-tuned $\lambda$ band"),
        Line2D([0], [0], color="#d35400", ls="--", lw=1.2,
               label=r"manuscript $\lambda$=0.1"),
    ]
    leg_a = fig.legend(handles=gt_handles + aux_handles,
                       title="Panel (a)",
                       loc="upper left", bbox_to_anchor=(0.795, 0.93),
                       frameon=False, fontsize=13, title_fontsize=14,
                       borderaxespad=0.0)
    leg_a._legend_box.align = "left"

    # ============================================================ Panel (b)
    crlb_unc, crlb_bay, nuts_std = compute_crlb_bars(PANEL_B_SNR, LAMBDA_PRIOR)

    x_idx = np.arange(len(D_BINS))
    width = 0.27

    # Grayscale-friendly palette: light → mid → dark
    color_unc = "#cccccc"
    color_bay = "#7f7f7f"
    color_nuts = "#1a1a1a"

    b1 = ax_b.bar(x_idx - width, crlb_unc, width, color=color_unc,
                  edgecolor="black", linewidth=0.7,
                  label="Unconstrained CRLB (no prior)")
    b2 = ax_b.bar(x_idx,         crlb_bay, width, color=color_bay,
                  edgecolor="black", linewidth=0.7,
                  label="Bayesian CRLB (van Trees, $\\lambda$=0.1)")
    b3 = ax_b.bar(x_idx + width, nuts_std, width, color=color_nuts,
                  edgecolor="black", linewidth=0.7,
                  label="NUTS empirical posterior std")

    ax_b.set_yscale("log")
    ax_b.set_ylim(5e-3, 1e3)
    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(D_LABELS)
    ax_b.set_xlabel(r"diffusivity bin D ($\mu m^2 / ms$)")
    ax_b.set_ylabel("posterior / CRLB std (signal-normalized)")
    ax_b.set_title(
        f"(b)  Where the precision comes from  "
        f"(SNR={PANEL_B_SNR}, cohort median)",
        loc="left", fontweight="bold", pad=24,
    )
    ax_b.grid(True, which="major", axis="y", alpha=0.30)
    ax_b.grid(True, which="minor", axis="y", alpha=0.10)

    # Annotate Bayesian/NUTS gap ratio above each bin group
    y_anno = 3.5e2
    for i in range(len(D_BINS)):
        ratio = crlb_bay[i] / nuts_std[i]
        ax_b.text(x_idx[i], y_anno,
                  f"{ratio:.1f}$\\times$",
                  ha="center", va="center",
                  fontsize=12, color="#222",
                  fontweight="bold")
    # Left-side caption for the row of ratios — placed between title and bars
    ax_b.text(0.0, 1.015, "Bayesian-CRLB / NUTS gap:",
              transform=ax_b.transAxes,
              ha="left", va="bottom",
              fontsize=12, color="#222", fontweight="bold", style="italic")

    # Panel (b) legend OUTSIDE — lower right of figure
    leg_b = fig.legend(
        handles=[b1, b2, b3],
        title="Panel (b)",
        loc="upper left", bbox_to_anchor=(0.795, 0.45),
        frameon=False, fontsize=13, title_fontsize=14,
        borderaxespad=0.0,
    )
    leg_b._legend_box.align = "left"

    # ============================================================ supt
    fig.suptitle(
        "Figure 8.  Simulation-based λ recovery (a) and a CRLB-decomposed "
        "view of where NUTS precision originates (b)",
        fontsize=16, fontweight="bold", y=0.985,
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"  Wrote {OUT_PNG}")
    print(f"  Wrote {OUT_PDF}")
    plt.close(fig)

    # Diagnostic dump
    print()
    print(f"=== Panel (b) numbers at SNR={PANEL_B_SNR}, λ={LAMBDA_PRIOR} ===")
    print(f"{'D':>6s} | {'unc CRLB':>10s} {'Bayes CRLB':>11s} {'NUTS std':>10s}"
          f" | {'unc/NUTS':>10s} {'Bayes/NUTS':>11s}")
    for i, d in enumerate(D_BINS):
        print(f"{d:6.2f} | {crlb_unc[i]:10.4f} {crlb_bay[i]:11.4f} "
              f"{nuts_std[i]:10.4f} | "
              f"{crlb_unc[i]/nuts_std[i]:9.1f}× {crlb_bay[i]/nuts_std[i]:10.1f}×")


if __name__ == "__main__":
    main()
