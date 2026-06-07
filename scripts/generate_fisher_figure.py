"""
Generate Fisher information analysis figure for MRM paper (Fig 7).

Promoted from supplementary to a MAIN figure (Stefan 2026-06-03). 1x3 panels:

  Panel (a) — Parameter correlation matrix derived from the Fisher information
              matrix (Stefan explicitly wants the matrix kept).
  Panel (b) — THREE-bar comparison per diffusivity bin at the cohort-median
              SNR = 303 (NOT the old hard-coded 150):
                1. Unconstrained classical CRLB (no prior; data-only floor)
                2. Bayesian / van-Trees CRLB (HalfNormal-as-Gaussian, lambda=0.1)
                3. Empirical NUTS posterior std (from identifiability.csv)
              Improvement-factor labels are Bayesian-CRLB / NUTS (the gap
              relative to the proper Bayesian bound). Legend in a free area.
  Panel (c) — Component decay curves vs SNR noise floors. Non-angled x ticks,
              larger SNR labels.

Outputs paper/figures/fig_fisher_v2.{pdf,png}; leaves fig_fisher.* intact.

CAVEAT: the van-Trees Bayesian-CRLB derivation is PENDING SANDY'S VALIDATION
(see notes/CRLB_NOTE_FOR_SANDY.txt, PROJECT_STATE.md F10).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from spectra_estimation_dmri.visualization.paper_style import (
    apply_style, COLORS, DIFFUSIVITIES, DLABELS,
)

# ── Style ─────────────────────────────────────────────────────────────────────
apply_style("grid")
# The 3-in-a-row panels are physically small; bump labels/ticks for readability
# (Stefan). Keep within the grid preset family but a touch larger.
plt.rcParams.update({
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

# ── Parameters ──────────────────────────────────────────────────────────────
b_values = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750,
                     2000, 2250, 2500, 2750, 3000, 3250, 3500]) / 1000.0  # ms/um²
D = DIFFUSIVITIES                       # 8 diffusivity bins (shared contract)
D_labels = DLABELS
SNR = 303                               # cohort median (IQR 176-478); was 150
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

# ── Figure ──────────────────────────────────────────────────────────────────
# Larger canvas so the three panels breathe; extra headroom for a single
# figure-level legend on top (consolidates panel (b) bars + panel (c) decay
# lines + SNR noise-floor identities). Panel (c) gets the most room.
fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.4),
                         gridspec_kw={"wspace": 0.34, "left": 0.045,
                                      "right": 0.975})

# ─── Panel A: Correlation matrix heatmap ─────────────────────────────────────
ax = axes[0]
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
ax.set_xticks(range(len(D)))
ax.set_xticklabels(D_labels)          # no rotation (shared contract)
ax.set_yticks(range(len(D)))
ax.set_yticklabels(D_labels)
ax.set_xlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
ax.set_ylabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
ax.set_title("(a) parameter correlation matrix", fontweight="bold")

for i in range(len(D)):
    for j in range(len(D)):
        val = C[i, j]
        color = "white" if abs(val) > 0.7 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8.5, color=color, fontweight="bold" if i == j else "normal")

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("correlation", fontsize=13)
cb.ax.tick_params(labelsize=12)

# ─── Panel B: 3-bar CRLB comparison at SNR=303 ───────────────────────────────
ax = axes[1]
x = np.arange(len(D))
width = 0.27

b1 = ax.bar(x - width, crlb_unc, width, label="unconstrained CRLB (no prior)",
            color=COLORS["crlb"], edgecolor="black", linewidth=0.5)
b2 = ax.bar(x, crlb_bay, width, label=r"Bayesian CRLB (van Trees, $\lambda$=0.1)",
            color=COLORS["crlb_bayes"], edgecolor="black", linewidth=0.5)
b3 = ax.bar(x + width, nuts_std, width, label="NUTS posterior std",
            color=COLORS["nuts"], edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(D_labels)          # no rotation
ax.set_xlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
ax.set_ylabel("std of fraction (signal-normalized)")
ax.set_title(f"(b) CRLB bounds vs NUTS std (SNR = {SNR})", fontweight="bold",
             pad=8)
ax.set_yscale("log")
ax.tick_params(axis="both", which="both")

# Keep the "Bayesian-CRLB / NUTS gap" text label at the top of the panel,
# placed ABOVE the title so the two do not collide. The per-bar improvement-
# factor numbers are REMOVED from the plot (they go in the figure caption).
ax.text(0.5, 1.075, "Bayesian-CRLB / NUTS gap", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=12, color="#222222",
        fontweight="bold", style="italic")

ax.set_ylim(8e-3, 1.5e3)

# (Bar legend moves to the single figure-level legend on top.)

# ─── Panel C: Component decay curves vs noise floors ─────────────────────────
ax = axes[2]
b_fine = np.linspace(0, 3.5, 300)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

for j, (d, col) in enumerate(zip(D, colors)):
    curve = np.exp(-b_fine * d) / len(D)   # equal-fraction component contribution
    lw = 2.2 if d in [0.25, 1.0, 3.0, 20.0] else 1.4
    ax.plot(b_fine, curve, color=col, linewidth=lw, label=f"$D$ = {d:g}", alpha=0.9)

# Noise-floor lines for several SNR levels (cohort median 303 emphasized solid).
# Labelled so the SNR identities live in the single top legend (no in-axes
# text -> nothing overflows panel (c) anymore).
snr_levels = [50, 100, 303]
line_styles = [":", "--", "-"]
snr_handles = []
for snr_val, ls in zip(snr_levels, line_styles):
    noise_floor = 1.0 / snr_val
    h = ax.axhline(noise_floor, color="gray", linestyle=ls, linewidth=1.4,
                   alpha=0.8, label=f"noise floor (SNR={snr_val})")
    snr_handles.append(h)

# Actual b-value sampling marks along the top.
for bv in b_values:
    ax.plot(bv, 0.18, marker="|", color="black", markersize=5, alpha=0.4)

ax.set_xlabel(r"$b$-value (ms/$\mu$m$^2$)")
ax.set_ylabel("signal contribution (per component)")
ax.set_title("(c) component decay curves vs noise floor", fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(5e-4, 0.2)
ax.set_xlim(0, 3.6)
ax.tick_params(axis="both", which="both")

# ── Single figure-level legend on TOP ────────────────────────────────────────
# Consolidates: panel (b) 3-bar identities, panel (c) 8 component decay lines,
# and the 3 SNR noise-floor identities. Nothing lives inside the panels now.
decay_handles, decay_labels = ax.get_legend_handles_labels()
# get_legend_handles_labels returns the 8 decay lines + 3 SNR axhlines in
# plotting order; split them out explicitly.
line_handles = decay_handles[:len(D)]
line_labels = decay_labels[:len(D)]

bar_handles = [b1, b2, b3]
bar_labels = ["unconstrained CRLB (no prior)",
              r"Bayesian CRLB (van Trees, $\lambda$=0.1)",
              "NUTS posterior std"]

all_handles = bar_handles + list(line_handles) + snr_handles
all_labels = bar_labels + list(line_labels) + \
    [f"noise floor (SNR={s})" for s in snr_levels]

leg = fig.legend(all_handles, all_labels, loc="lower center",
                 bbox_to_anchor=(0.5, 1.005), ncol=7, frameon=True,
                 framealpha=0.95, fontsize=12, columnspacing=1.3,
                 handlelength=1.8, borderaxespad=0.4)

# ── Layout and save ─────────────────────────────────────────────────────────
# Leave a clear band at the top for the legend so it never collides with the
# (b) title or its "Bayesian-CRLB / NUTS gap" annotation. The legend is
# anchored ABOVE the axes (lower edge at y=1.005) and passed to savefig as an
# extra artist so the tight bbox keeps it.
fig.subplots_adjust(bottom=0.12, top=0.86)

out_dir = project_root / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_fisher_v2.pdf", dpi=300, bbox_inches="tight",
            bbox_extra_artists=(leg,))
fig.savefig(out_dir / "fig_fisher_v2.png", dpi=300, bbox_inches="tight",
            bbox_extra_artists=(leg,))
plt.close()

print(f"Saved to {out_dir / 'fig_fisher_v2.pdf'}")
print(f"Saved to {out_dir / 'fig_fisher_v2.png'}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n── Fisher / CRLB analysis (Fig 7 v2) ──")
print(f"cond(U) = {np.linalg.cond(U):.3e}, cond(F_data) = {np.linalg.cond(F_data):.3e}")
print(f"SNR = {SNR} (cohort median), lambda = {LAMBDA_PRIOR}")
print()
print(f"{'D':>6} {'unc CRLB':>10} {'Bayes CRLB':>11} {'NUTS std':>10} "
      f"{'unc/NUTS':>9} {'Bay/NUTS':>9}")
print("-" * 62)
for j in range(len(D)):
    print(f"{D[j]:6.2f} {crlb_unc[j]:10.4f} {crlb_bay[j]:11.4f} {nuts_std[j]:10.4f} "
          f"{crlb_unc[j]/nuts_std[j]:8.0f}x {crlb_bay[j]/nuts_std[j]:8.1f}x")
print()
print(f"unconstrained CRLB range : {crlb_unc.min():.3f} – {crlb_unc.max():.3f}")
print(f"Bayesian van-Trees range : {crlb_bay.min():.3f} – {crlb_bay.max():.3f}")
print(f"NUTS posterior std range : {nuts_std.min():.3f} – {nuts_std.max():.3f}")
print()
print("CAVEAT: van-Trees Bayesian-CRLB derivation PENDING SANDY'S VALIDATION.")
