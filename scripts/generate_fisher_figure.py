"""
Generate Fisher information analysis figure for MRM paper (Fig 2 in the
3-block scheme; \\label{fig:fisher}).

Layout (Patrick 2026-06-06): 2-over-1 -- panels (a) and (b) on the top row,
panel (c) centered below them -- so each panel is larger and the fonts match
the other figures.

  Panel (a) — Parameter correlation matrix from the Fisher information matrix
              (KEPT, per Patrick + Stefan).
  Panel (b) — THREE-bar comparison per diffusivity bin at the cohort-median
              SNR = 303: unconstrained classical CRLB, Bayesian/van-Trees CRLB
              (HalfNormal-as-Gaussian, lambda=0.1), empirical NUTS posterior std.
              BOTH improvement factors are printed IN the panel, colour-coded:
              the prior gain (unconstrained -> Bayesian) above the Bayesian bar
              in the Bayesian colour, and the constraint gain (Bayesian -> NUTS)
              above the NUTS bar in the NUTS colour.
  Panel (c) — Component decay curves vs SNR noise floors, with the SNR labels
              placed INSIDE the panel (not in the legend).

Outputs paper/figures/fig_fisher_v2.{pdf,png}; leaves fig_fisher.* intact.

CAVEAT: the van-Trees Bayesian-CRLB derivation is PENDING SANDY'S VALIDATION
(see notes/CRLB_NOTE_FOR_SANDY.txt, PROJECT_STATE.md F10).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from spectra_estimation_dmri.visualization.paper_style import (
    apply_style, COLORS, DIFFUSIVITIES, DLABELS,
)

# ── Style ─────────────────────────────────────────────────────────────────────
apply_style("grid")
# Panels are now half-width (2-over-1), so fonts can match the other grid
# figures (Patrick: match axis/title/legend sizes across figures).
plt.rcParams.update({
    "axes.labelsize": 18,
    "axes.titlesize": 17,
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
F_data = (SNR ** 2) * (U.T @ U)                  # data Fisher info at SNR
F_post = F_data + LAMBDA_PRIOR * np.eye(len(D))  # van-Trees posterior info

crlb_unc = np.sqrt(np.diag(np.linalg.inv(F_data)))   # unconstrained CRLB
crlb_bay = np.sqrt(np.diag(np.linalg.inv(F_post)))   # Bayesian van-Trees CRLB

F_inv = np.linalg.inv(F_data)
diag_inv = np.sqrt(np.diag(F_inv))
C = F_inv / np.outer(diag_inv, diag_inv)             # parameter correlation matrix

# ── Observed NUTS posterior std (cohort, from identifiability.csv) ────────────
project_root = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_root / "results" / "biomarkers" / "identifiability.csv")
nuts_std = df["mean_posterior_std"].values

# Improvement factors printed in panel (b)
f_prior = crlb_unc / crlb_bay          # unconstrained -> Bayesian (prior gain)
f_constr = crlb_bay / nuts_std         # Bayesian -> NUTS (constraint gain)

# ── Figure: 2-over-1 (a,b top; c centered below) ──────────────────────────────
fig = plt.figure(figsize=(14, 11.5))
gs = GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.85,
              left=0.07, right=0.965, top=0.93, bottom=0.07)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[1, 1:3])     # centered below the upper two

# ─── Panel A: Correlation matrix heatmap ─────────────────────────────────────
ax = ax_a
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
ax.set_xticks(range(len(D))); ax.set_xticklabels(D_labels)
ax.set_yticks(range(len(D))); ax.set_yticklabels(D_labels)
ax.set_xlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
ax.set_ylabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
ax.set_title("(a) parameter correlation matrix", fontweight="bold")
for i in range(len(D)):
    for j in range(len(D)):
        val = C[i, j]
        color = "white" if abs(val) > 0.7 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8.5, color=color,
                fontweight="bold" if i == j else "normal")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("correlation", fontsize=14)
cb.ax.tick_params(labelsize=12)

# ─── Panel B: 3-bar CRLB comparison at SNR=303, factors in-panel ─────────────
ax = ax_b
x = np.arange(len(D))
width = 0.27
b1 = ax.bar(x - width, crlb_unc, width, color=COLORS["crlb"],
            edgecolor="black", linewidth=0.5, label="unconstrained CRLB")
b2 = ax.bar(x, crlb_bay, width, color=COLORS["crlb_bayes"],
            edgecolor="black", linewidth=0.5, label="Bayesian CRLB (van Trees)")
b3 = ax.bar(x + width, nuts_std, width, color=COLORS["nuts"],
            edgecolor="black", linewidth=0.5, label="NUTS posterior std")
ax.set_xticks(x); ax.set_xticklabels(D_labels)
ax.set_xlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
ax.set_ylabel(r"standard deviation of fraction $R_j$")
ax.set_title(f"(b) estimation uncertainty at SNR {SNR}", fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(8e-3, 4e3)

# Both improvement factors printed in-panel, colour-matched to the target bar:
#   prior gain (unc->Bayes) above the Bayesian bar; constraint gain
#   (Bayes->NUTS) above the NUTS bar. Rotated 90 so they fit over thin bars.
for j in range(len(D)):
    ax.text(x[j], crlb_bay[j] * 1.6, rf"{f_prior[j]:.0f}$\times$",
            ha="center", va="bottom", fontsize=8.5, rotation=90,
            color=COLORS["crlb_bayes"], fontweight="bold")
    ax.text(x[j] + width, nuts_std[j] * 1.6, rf"{f_constr[j]:.0f}$\times$",
            ha="center", va="bottom", fontsize=8.5, rotation=90,
            color=COLORS["nuts"], fontweight="bold")
# Compact in-panel legend (3 bar identities). The factor colour coding (gray =
# prior gain unc->Bayes; orange = constraint gain Bayes->NUTS) is explained in
# the caption, so no in-panel key is needed.
ax.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.95,
          fontsize=11.5, handlelength=1.4)

# ─── Panel C: Component decay curves vs noise floors (SNR labels in-panel) ────
ax = ax_c
b_fine = np.linspace(0, 3.5, 300)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
for j, (d, col) in enumerate(zip(D, colors)):
    curve = np.exp(-b_fine * d) / len(D)
    lw = 2.4 if d in [0.25, 1.0, 3.0, 20.0] else 1.5
    ax.plot(b_fine, curve, color=col, linewidth=lw, label=f"$D$ = {d:g}", alpha=0.9)

# Noise-floor lines with SNR labels placed INSIDE the panel (not the legend).
for snr_val, ls in zip([50, 100, 303], [":", "--", "-"]):
    nf = 1.0 / snr_val
    ax.axhline(nf, color="gray", linestyle=ls, linewidth=1.5, alpha=0.85)
    ax.text(3.5, nf * 1.06, f"SNR {snr_val}", ha="right", va="bottom",
            fontsize=11, color="gray", style="italic")
for bv in b_values:
    ax.plot(bv, 0.18, marker="|", color="black", markersize=5, alpha=0.4)

ax.set_xlabel(r"$b$-value (ms/$\mu$m$^2$)")
ax.set_ylabel("signal contribution per component")
ax.set_title("(c) component decay curves vs noise floor", fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(5e-4, 0.22)
ax.set_xlim(0, 3.6)
ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95,
          fontsize=11, handlelength=1.6, columnspacing=1.0)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = project_root / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_fisher_v2.pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "fig_fisher_v2.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {out_dir / 'fig_fisher_v2.pdf'}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n── Fisher / CRLB analysis (Fig 2 v2, 2-over-1) ──")
print(f"SNR = {SNR}, lambda = {LAMBDA_PRIOR}")
print(f"{'D':>6} {'uncCRLB':>9} {'BayCRLB':>9} {'NUTS':>8} "
      f"{'prior x':>8} {'constr x':>9}")
print("-" * 56)
for j in range(len(D)):
    print(f"{D[j]:6.2f} {crlb_unc[j]:9.3f} {crlb_bay[j]:9.3f} {nuts_std[j]:8.3f} "
          f"{f_prior[j]:7.0f}x {f_constr[j]:8.0f}x")
print("\nCAVEAT: van-Trees Bayesian-CRLB derivation PENDING SANDY'S VALIDATION.")
