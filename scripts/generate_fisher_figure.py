"""
Generate Fisher information analysis figure for MRM paper (now Fig 1, the lead
"why estimating the spectrum is hard" figure).

2+1 layout (Patrick 2026-06-07): panels (a) and (b) on the top row, panel (c)
centered on the bottom row, so each panel is larger and fonts match the other
figures.

  Panel (a) — Parameter correlation matrix from the Fisher information matrix
              (kept: it is the cleanest visual for inter-bin indistinguishability).
  Panel (b) — THREE-bar comparison per diffusivity bin at the cohort-median
              SNR = 303: (1) unconstrained classical CRLB, (2) Bayesian /
              van-Trees CRLB (HalfNormal-as-Gaussian, lambda=0.1), (3) empirical
              NUTS posterior std. Improvement factors are drawn ON the bars for
              BOTH steps: gray = prior gain (unconstrained->Bayesian), orange =
              constraint gain (Bayesian->NUTS).
  Panel (c) — Component decay curves vs SNR noise floors, with the SNR
              identities labelled inline (not in a separate legend).

Outputs paper/figures/fig_fisher_v2.{pdf,png}.

CAVEAT: the van-Trees Bayesian-CRLB derivation (panel b) is PENDING SANDY'S
VALIDATION (see notes/CRLB_NOTE_FOR_SANDY.txt, PROJECT_STATE.md F10).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from spectra_estimation_dmri.visualization.paper_style import (
    apply_style, COLORS, DIFFUSIVITIES, DLABELS, DIFF_AXIS_LABEL,
)

# ── Style ─────────────────────────────────────────────────────────────────────
apply_style("grid")
# Larger panels in the 2+1 layout -> fonts match the other figures more closely.
plt.rcParams.update({
    "axes.labelsize": 17,
    "axes.titlesize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

# ── Parameters ──────────────────────────────────────────────────────────────
b_values = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750,
                     2000, 2250, 2500, 2750, 3000, 3250, 3500]) / 1000.0  # ms/um²
D = DIFFUSIVITIES                       # 8 diffusivity bins (shared contract)
D_labels = DLABELS
SNR = 303                               # cohort median (IQR 176-478)
LAMBDA_PRIOR = 0.1                      # HalfNormal prior precision (lambda)

# ── Design matrix ───────────────────────────────────────────────────────────
U = np.exp(-np.outer(b_values, D))      # shape (15, 8)

# ── Fisher information / CRLB ─────────────────────────────────────────────────
# Data Fisher information at SNR: F_data = (1/sigma^2) U^T U = SNR^2 U^T U
F_data = (SNR ** 2) * (U.T @ U)
# Bayesian posterior information (van Trees): add prior precision lambda*I
F_post = F_data + LAMBDA_PRIOR * np.eye(len(D))

crlb_unc = np.sqrt(np.diag(np.linalg.inv(F_data)))   # unconstrained CRLB
crlb_bay = np.sqrt(np.diag(np.linalg.inv(F_post)))   # Bayesian van-Trees CRLB

# Parameter correlation matrix: normalize the (unconstrained) covariance F_inv
F_inv = np.linalg.inv(F_data)
diag_inv = np.sqrt(np.diag(F_inv))
C = F_inv / np.outer(diag_inv, diag_inv)

# ── Observed NUTS posterior std (cohort, from identifiability.csv) ────────────
project_root = Path(__file__).resolve().parent.parent
ident_path = project_root / "results" / "biomarkers" / "identifiability.csv"
df = pd.read_csv(ident_path)
nuts_std = df["mean_posterior_std"].values

# ── Figure: 2+1 layout (a, b on top; c centered below) ───────────────────────
fig = plt.figure(figsize=(15, 11))
# wspace widened so panel (a)'s colorbar no longer crowds panel (b)'s y-axis;
# top lowered to make room for the unified two-row legend above the panels.
gs = fig.add_gridspec(2, 4, hspace=0.80, wspace=0.95,
                      left=0.07, right=0.95, top=0.85, bottom=0.08)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[1, 1:3])   # centered between the two upper panels

# ─── Panel A: Correlation matrix heatmap ─────────────────────────────────────
im = ax_a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_a.set_xticks(range(len(D)))
ax_a.set_xticklabels(D_labels)
ax_a.set_yticks(range(len(D)))
ax_a.set_yticklabels(D_labels)
ax_a.set_xlabel(DIFF_AXIS_LABEL)
ax_a.set_ylabel(DIFF_AXIS_LABEL)
ax_a.set_title("(a) Fisher Correlation Matrix", fontweight="bold")
for i in range(len(D)):
    for j in range(len(D)):
        val = C[i, j]
        color = "white" if abs(val) > 0.7 else "black"
        ax_a.text(j, i, f"{val:.2f}", ha="center", va="center",
                  fontsize=8.5, color=color,
                  fontweight="bold" if i == j else "normal")
cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
cb.set_label("Correlation", fontsize=12)
cb.ax.tick_params(labelsize=11)

# ─── Panel B: 3-bar CRLB comparison with factors ON the bars ─────────────────
x = np.arange(len(D))
width = 0.27
b1 = ax_b.bar(x - width, crlb_unc, width, label="unconstrained CRLB",
              color=COLORS["crlb"], edgecolor="black", linewidth=0.5)
b2 = ax_b.bar(x, crlb_bay, width, label="Bayesian CRLB (van Trees)",
              color=COLORS["crlb_bayes"], edgecolor="black", linewidth=0.5)
b3 = ax_b.bar(x + width, nuts_std, width, label="NUTS posterior std",
              color=COLORS["nuts"], edgecolor="black", linewidth=0.5)
ax_b.set_yscale("log")
ax_b.set_ylim(8e-3, 5e3)

# Per-bin improvement factors (unconstrained -> Bayesian -> NUTS) are NOT drawn
# on the bars (Stephan 2026-06-09: the log axis already conveys the 1-2 orders
# of magnitude separation between the three uncertainty estimates). They remain
# in the stdout summary below for reference.
ax_b.set_xticks(x)
ax_b.set_xticklabels(D_labels)
ax_b.set_xlabel(DIFF_AXIS_LABEL)
ax_b.set_ylabel("Std of fraction (log scale)")
ax_b.set_title("(b) Per-Component Estimation Uncertainty at SNR 303",
               fontweight="bold")

# ─── Panel C: Component decay curves vs noise floors (SNR labelled inline) ────
b_fine = np.linspace(0, 3.5, 300)
# Palette deliberately AVOIDS orange (#ff7f0e = NUTS bar) and grey
# (#8c8c8c = unconstrained CRLB bar) so the unified legend has no colour clash.
colors = ["#1f77b4", "#17becf", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]
dcurve_handles = []
for j, (d, col) in enumerate(zip(D, colors)):
    curve = np.exp(-b_fine * d) / len(D)
    lw = 2.2 if d in [0.25, 1.0, 3.0, 20.0] else 1.4
    line, = ax_c.plot(b_fine, curve, color=col, linewidth=lw,
                      label=f"$D$ = {d:g}", alpha=0.9)
    dcurve_handles.append(line)
# Noise-floor lines per SNR; described in the top legend above panel (c)
# (Stephan 2026-06-10) instead of inline text labels.
snr_handles = []
for snr_val, ls in zip([50, 100, 303], [":", "--", "-"]):
    noise_floor = 1.0 / snr_val
    line = ax_c.axhline(noise_floor, color="black", linestyle=ls, linewidth=1.9,
                        alpha=0.95, label=f"SNR {snr_val} noise floor")
    snr_handles.append(line)
for bv in b_values:
    ax_c.plot(bv, 0.18, marker="|", color="black", markersize=5, alpha=0.4)
ax_c.set_xlabel(r"$b$-value (ms/$\mu$m$^2$)")
ax_c.set_ylabel("Relative signal contribution")
ax_c.set_title("(c) Component Decay Curves vs Noise Floor", fontweight="bold")
ax_c.set_yscale("log")
ax_c.set_ylim(5e-4, 0.2)
ax_c.set_xlim(0, 3.6)

# ─── Two legends, each placed above its own panel (Stephan 2026-06-09) ───────
# The previous single top legend mixed two unrelated keys; split so the CRLB-bar
# key sits above panel (b) and the diffusivity-component colour key above panel
# (c). Figure-fraction coordinates keep each legend centred over its panel.
fig.legend([b1, b2, b3],
           ["unconstrained CRLB", "Bayesian CRLB (van Trees)",
            "NUTS posterior std"],
           loc="center", bbox_to_anchor=(0.73, 0.935), ncol=1,
           frameon=True, framealpha=0.95, fontsize=15)
# One merged legend for panel (c): diffusivity-component colours + SNR
# noise-floor line styles, in TWO rows (ncol=6) so it stays narrow
# (Stephan/Patrick 2026-06-10). Column-major fill keeps the 8 diffusivity
# buckets in the left columns and the 3 SNR floors in the right columns.
c_handles = dcurve_handles + snr_handles
c_labels = [f"$D$ = {d:g}" for d in D] + [f"SNR {s}" for s in [50, 100, 303]]
fig.legend(c_handles, c_labels,
           loc="center", bbox_to_anchor=(0.51, 0.46), ncol=6,
           frameon=True, framealpha=0.95, fontsize=13)

# ── Save ──────────────────────────────────────────────────────────────────
out_dir = project_root / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_fisher_v2.pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "fig_fisher_v2.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved to {out_dir / 'fig_fisher_v2.pdf'}")
print(f"Saved to {out_dir / 'fig_fisher_v2.png'}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n── Fisher / CRLB analysis (Fig 1) ──")
print(f"cond(U) = {np.linalg.cond(U):.3e}, cond(F_data) = {np.linalg.cond(F_data):.3e}")
print(f"SNR = {SNR} (cohort median), lambda = {LAMBDA_PRIOR}")
print()
print(f"{'D':>6} {'unc CRLB':>10} {'Bayes CRLB':>11} {'NUTS std':>10} "
      f"{'unc/bay':>8} {'bay/NUTS':>9}")
print("-" * 60)
for j in range(len(D)):
    print(f"{D[j]:6.2f} {crlb_unc[j]:10.4f} {crlb_bay[j]:11.4f} {nuts_std[j]:10.4f} "
          f"{crlb_unc[j]/crlb_bay[j]:7.0f}x {crlb_bay[j]/nuts_std[j]:8.1f}x")
print()
print("CAVEAT: van-Trees Bayesian-CRLB derivation (panel b) PENDING SANDY'S VALIDATION.")
