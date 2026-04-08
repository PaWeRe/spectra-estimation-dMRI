"""
Generate Fisher information analysis figure for MRM paper.

Panel A: Correlation matrix derived from Fisher information matrix
Panel B: CRLB vs observed NUTS posterior std per component (log scale)
Panel C: Individual decay curves with SNR noise floors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Parameters ──────────────────────────────────────────────────────────────
b_values = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5])  # ms/um²
D = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])  # um²/ms
SNR = 150  # typical for our data
sigma = 1.0 / SNR

# Labels for components
D_labels = [f"{d}" for d in D]

# ── Design matrix ───────────────────────────────────────────────────────────
U = np.exp(-np.outer(b_values, D))  # shape (15, 8)

# ── Fisher information matrix ───────────────────────────────────────────────
F = (1.0 / sigma**2) * (U.T @ U)  # shape (8, 8)

# ── CRLB ────────────────────────────────────────────────────────────────────
F_inv = np.linalg.inv(F)
crlb_std = np.sqrt(np.diag(F_inv))  # theoretical minimum std (unconstrained)

# Parameter correlation matrix: normalize F^{-1} (covariance), not F
diag_inv = np.sqrt(np.diag(F_inv))
C = F_inv / np.outer(diag_inv, diag_inv)

# ── Load observed posterior std from NUTS ────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
ident_path = project_root / "results" / "biomarkers" / "identifiability.csv"
df = pd.read_csv(ident_path)
nuts_std = df["mean_posterior_std"].values

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                         gridspec_kw={"wspace": 0.38, "left": 0.05, "right": 0.97})

# ─── Panel A: Correlation matrix heatmap ─────────────────────────────────────
ax = axes[0]
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
ax.set_xticks(range(len(D)))
ax.set_xticklabels(D_labels, fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(len(D)))
ax.set_yticklabels(D_labels, fontsize=9)
ax.set_xlabel(r"Diffusivity $D$ ($\mu$m$^2$/ms)", fontsize=11)
ax.set_ylabel(r"Diffusivity $D$ ($\mu$m$^2$/ms)", fontsize=11)
ax.set_title("(a) Parameter Correlation\nMatrix (from CRLB)", fontsize=12, fontweight="bold")

# Add text annotations
for i in range(len(D)):
    for j in range(len(D)):
        val = C[i, j]
        color = "white" if abs(val) > 0.7 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold" if i == j else "normal")

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Correlation", fontsize=10)
cb.ax.tick_params(labelsize=8)

# ─── Panel B: CRLB vs NUTS posterior std (log scale) ────────────────────────
ax = axes[1]
x = np.arange(len(D))
width = 0.35

bars_crlb = ax.bar(x - width / 2, crlb_std, width, label="CRLB (unconstrained)",
                    color="#888888", edgecolor="black", linewidth=0.5, alpha=0.85)
bars_nuts = ax.bar(x + width / 2, nuts_std, width, label="NUTS posterior std",
                   color="#ff7f0e", edgecolor="black", linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(D_labels, fontsize=9, rotation=45, ha="right")
ax.set_xlabel(r"Diffusivity $D$ ($\mu$m$^2$/ms)", fontsize=11)
ax.set_ylabel("Standard Deviation of Fraction", fontsize=11)
ax.set_title(f"(b) Theoretical CRLB vs Observed\nPosterior Std (SNR = {SNR})", fontsize=12, fontweight="bold")
ax.set_yscale("log")
ax.legend(fontsize=8.5, loc="upper left")
ax.tick_params(axis="both", labelsize=9)

# Add improvement factor annotations
for i in range(len(D)):
    if crlb_std[i] > 0 and nuts_std[i] > 0:
        ratio = crlb_std[i] / nuts_std[i]
        y_pos = max(crlb_std[i], nuts_std[i]) * 1.5
        ax.text(x[i], y_pos, f"{ratio:.0f}x", ha="center", va="bottom",
                fontsize=7, color="#333333", fontweight="bold")

ax.set_ylim(5e-3, ax.get_ylim()[1] * 3)

# Add reference line at fraction = 1 (max possible)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.text(7.5, 1.05, "fraction = 1", fontsize=7, color="gray", ha="right", va="bottom")

# ─── Panel C: Individual decay curves with noise floors ──────────────────────
ax = axes[2]
b_fine = np.linspace(0, 3.5, 300)

# Colors for each component — use a well-separated qualitative palette
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

for j, (d, col) in enumerate(zip(D, colors)):
    # Each component contributes 1/8 of the total signal (equal fractions)
    curve = np.exp(-b_fine * d) / len(D)
    label = f"$D$ = {d}"
    lw = 2.0 if d in [0.25, 1.0, 3.0, 20.0] else 1.3
    ax.plot(b_fine, curve, color=col, linewidth=lw, label=label, alpha=0.9)

# Noise floor lines for different SNR levels
snr_levels = [50, 100, 150, 300]
line_styles = [":", "-.", "--", "-"]
for snr_val, ls in zip(snr_levels, line_styles):
    noise_floor = 1.0 / snr_val
    ax.axhline(noise_floor, color="gray", linestyle=ls, linewidth=0.9, alpha=0.6)
    ax.text(3.58, noise_floor, f"SNR={snr_val}", fontsize=6.5, va="center",
            color="#555555", clip_on=False)

# Mark actual b-values as subtle tick marks at the top
for bv in b_values:
    ax.plot(bv, 0.18, marker="|", color="black", markersize=4, alpha=0.4)

ax.set_xlabel(r"$b$-value (ms/$\mu$m$^2$)", fontsize=11)
ax.set_ylabel("Signal Contribution (per component)", fontsize=11)
ax.set_title("(c) Component Decay Curves\nvs Noise Floor", fontsize=12, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(5e-4, 0.2)
ax.set_xlim(0, 4.1)
ax.tick_params(axis="both", labelsize=9)
ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.32),
          ncol=1, framealpha=0.9, borderaxespad=0.5)

# ── Layout and save ─────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.18, top=0.88)

out_dir = project_root / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(out_dir / "fig_fisher.pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "fig_fisher.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to {out_dir / 'fig_fisher.pdf'}")
print(f"Saved to {out_dir / 'fig_fisher.png'}")

# ── Print summary statistics ────────────────────────────────────────────────
print("\n── Fisher Information Analysis Summary ──")
print(f"Design matrix U: shape {U.shape}, condition number = {np.linalg.cond(U):.1f}")
print(f"Fisher matrix F: condition number = {np.linalg.cond(F):.1f}")
print(f"SNR = {SNR}, sigma = {sigma:.6f}")
print()
print(f"{'D (um²/ms)':<12} {'CRLB std':<14} {'NUTS std':<14} {'Improvement':<14} {'Max off-diag corr'}")
print("-" * 70)
for j in range(len(D)):
    # Find max off-diagonal correlation for this component
    row = np.abs(C[j, :].copy())
    row[j] = 0
    max_corr = np.max(row)
    max_corr_idx = np.argmax(row)
    ratio = crlb_std[j] / nuts_std[j] if nuts_std[j] > 0 else float("inf")
    print(f"{D[j]:<12.2f} {crlb_std[j]:<14.4f} {nuts_std[j]:<14.6f} {ratio:<14.0f}x {max_corr:.3f} (D={D[max_corr_idx]})")

print()
print("Key insight: The unconstrained CRLB is 100-10000x larger than the observed")
print("NUTS posterior std, because NUTS benefits from non-negativity constraints and")
print("regularizing priors that break the collinearity between adjacent components.")
