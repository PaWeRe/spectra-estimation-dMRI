"""
Quick comparison script to visualize different discretization strategies.

Usage:
    uv run python scripts/compare_discretizations.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define discretizations
discretizations = {
    "4 bins\n(Well-calibrated)": [0.35, 0.7, 1.2, 2.0],
    "5 bins\n(Clinical)": [0.3, 0.5, 0.8, 1.3, 2.0],
    "7 bins (OLD)\n(Poor identif.)": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "7 bins (NEW)\n✅ OPTIMAL": [0.25, 0.52, 0.78, 1.05, 1.32, 2.0, 3.0],
    "10 bins (OLD)\n(Very poor)": [
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
        20.0,
    ],
    "10 bins (NEW)\n(Challenging)": [
        0.25,
        0.40,
        0.55,
        0.71,
        0.86,
        1.01,
        1.16,
        1.32,
        2.0,
        3.0,
    ],
}

# B-values
b_max = 3.5
b_values = np.linspace(0, b_max, 15)

# Identifiability threshold
threshold = 0.01
D_cutoff = -np.log(threshold) / b_max

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, D_bins) in enumerate(discretizations.items()):
    ax = axes[idx]

    # Compute signal contribution
    D_fine = np.linspace(0, 3.5, 500)

    # Plot identifiable region
    ax.axvspan(
        0,
        D_cutoff,
        alpha=0.2,
        color="green",
        label=f"Identifiable\n(D < {D_cutoff:.2f})",
    )
    ax.axvline(
        D_cutoff, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Cutoff"
    )

    # Plot bins
    for i, D in enumerate(D_bins):
        signal_at_bmax = np.exp(-b_max * D)
        color = (
            "green" if D < D_cutoff else "orange" if signal_at_bmax > 0.005 else "red"
        )

        ax.scatter(
            D,
            signal_at_bmax,
            s=200,
            c=color,
            edgecolors="black",
            linewidths=2,
            zorder=10,
            alpha=0.8,
        )
        ax.text(D, signal_at_bmax + 0.03, f"{D:.2f}", ha="center", fontsize=8)

    # Plot signal decay curve
    signal_curve = np.exp(-b_max * D_fine)
    ax.plot(D_fine, signal_curve, "k-", alpha=0.3, linewidth=2, label="Signal at b=3.5")
    ax.axhline(
        threshold,
        color="red",
        linestyle=":",
        alpha=0.5,
        label=f"Threshold ({threshold})",
    )

    ax.set_xlabel("Diffusivity (μm²/ms)", fontsize=11)
    ax.set_ylabel("Signal: exp(-b_max·D)", fontsize=11)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-0.2, 3.5)
    ax.grid(alpha=0.3)

    # Add legend only for first subplot
    if idx == 0:
        ax.legend(fontsize=8, loc="upper right")

    # Count identifiable
    n_ident = sum(D < D_cutoff for D in D_bins)
    n_weak = sum((D >= D_cutoff) and (np.exp(-b_max * D) > 0.005) for D in D_bins)
    n_poor = sum(np.exp(-b_max * D) <= 0.005 for D in D_bins)

    stats_text = f"Strong: {n_ident} | Weak: {n_weak} | Poor: {n_poor}"
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

plt.tight_layout()
output_path = Path("results/discretization_analysis/comparison_overview.pdf")
plt.savefig(output_path, dpi=150)
print(f"✓ Saved comparison plot: {output_path}")
plt.close()

# Print summary table
print("\n" + "=" * 80)
print("DISCRETIZATION COMPARISON SUMMARY")
print("=" * 80 + "\n")
print(f"{'Configuration':<25} {'# Bins':<8} {'Strong':<8} {'Weak':<8} {'Poor':<8}")
print("-" * 80)

for name, D_bins in discretizations.items():
    name_clean = name.replace("\n", " ")
    n_ident = sum(D < D_cutoff for D in D_bins)
    n_weak = sum((D >= D_cutoff) and (np.exp(-b_max * D) > 0.005) for D in D_bins)
    n_poor = sum(np.exp(-b_max * D) <= 0.005 for D in D_bins)

    print(f"{name_clean:<25} {len(D_bins):<8} {n_ident:<8} {n_weak:<8} {n_poor:<8}")

print("\nLegend:")
print("  Strong: D < 1.32 (exp(-b_max·D) > 0.01) - Well identified")
print("  Weak:   D ≥ 1.32, signal > 0.005 - Moderately identified")
print("  Poor:   Signal ≤ 0.005 - Poorly identified")
print("\n" + "=" * 80)
print("\n✅ RECOMMENDATION: Use '7 bins (NEW) OPTIMAL' with ridge strength=0.5\n")
