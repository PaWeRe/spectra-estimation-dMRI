"""
Visualisation of the MAP λ-sweep result with NUTS reference.

Two plots:
  results/simulation/map_lambda_sweep_fraction.png
  results/simulation/map_lambda_sweep_mse.png

Each shows, per GT × SNR=400, the fraction of mass recovered at the true
peak (or total 8-bin MSE) as a function of λ. NUTS @ λ=0.1 is overlaid
as a horizontal reference.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

map_df = pd.read_csv("results/simulation/map_lambda_sweep_summary.csv")
nuts_raw = pd.read_csv("results/simulation/sim_summary.csv")

GTS = ["GT-A_d0.25", "GT-D_d3.00", "GT-E_bi-tumor",
       "GT-F_bi-norm", "GT-H_lognorm0.5", "GT-I_lognorm1.5"]
GT_LABELS = {
    "GT-A_d0.25":      "δ @ D=0.25",
    "GT-D_d3.00":      "δ @ D=3.00",
    "GT-E_bi-tumor":   "bimodal {0.25:0.7, 3.0:0.3} (tumour-like)",
    "GT-F_bi-norm":    "bimodal {0.25:0.3, 3.0:0.7} (normal-like)",
    "GT-H_lognorm0.5": "log-normal μ=0.5",
    "GT-I_lognorm1.5": "log-normal μ=1.5",
}

SNR = 400


def nuts_summary(gt, snr):
    sub = nuts_raw[(nuts_raw["gt"] == gt) & (nuts_raw["snr"] == snr)
                   & (nuts_raw["estimator"] == "NUTS")]
    if len(sub) == 0:
        return None, None
    peak_mask = sub["R_true"] > 0.1
    frac = (sub[peak_mask]["R_hat_mean"].sum()
            / sub[peak_mask]["R_true"].sum())
    mse = sub["mse"].sum()
    return frac, mse


def make_plot(metric: str, title_suffix: str, fname: str,
              ylim=None, log_y=False):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axes = axes.flatten()
    for ax, gt in zip(axes, GTS):
        sub = map_df[(map_df["gt"] == gt) & (map_df["snr"] == SNR)]
        x = sub["lambda"].values
        if metric == "frac":
            y = sub["fraction_recovered"].values
        else:
            y = sub["total_mse_8bins"].values
        ax.plot(x, y, "o-", color="tab:green", label="MAP")
        nuts_frac, nuts_mse = nuts_summary(gt, SNR)
        ref = nuts_frac if metric == "frac" else nuts_mse
        if ref is not None:
            ax.axhline(ref, color="tab:orange", linestyle="--",
                       label="NUTS (λ=0.1)")
        # Manuscript baseline marker
        baseline_lambda = 0.1
        baseline_row = sub[sub["lambda"] == baseline_lambda]
        if len(baseline_row):
            ax.scatter(
                [baseline_lambda],
                [baseline_row["fraction_recovered"].iloc[0] if metric == "frac"
                 else baseline_row["total_mse_8bins"].iloc[0]],
                s=120, facecolors="none", edgecolors="red", linewidths=2,
                zorder=5, label="manuscript λ=0.1")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(GT_LABELS[gt], fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes[3:]:
        ax.set_xlabel("ridge λ")
    if metric == "frac":
        for ax in (axes[0], axes[3]):
            ax.set_ylabel("fraction of true mass recovered")
    else:
        for ax in (axes[0], axes[3]):
            ax.set_ylabel("total 8-bin MSE")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"MAP λ sweep at SNR={SNR}: {title_suffix}", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    print(f"  Wrote {fname}")
    plt.close(fig)


make_plot("frac", "fraction of true peak mass recovered",
          "results/simulation/map_lambda_sweep_fraction.png",
          ylim=(0, 1.05))
make_plot("mse", "total 8-bin MSE (lower is better)",
          "results/simulation/map_lambda_sweep_mse.png",
          log_y=True)
