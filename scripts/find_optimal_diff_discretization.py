"""
Minimalistic script to find optimal diffusivity discretization based on signal identifiability.

Usage:
    uv run python scripts/find_optimal_diff_discretization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
B_VALUES = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
)
N_BINS_TO_TEST = [7, 8, 9, 10]
SIGNAL_THRESHOLD = 0.01  # Minimum signal contribution to be considered identifiable
OUTPUT_DIR = Path("results/discretization_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def signal_contribution(b_values, D):
    """Calculate signal contribution exp(-b*D) for a given diffusivity."""
    return np.exp(-np.outer(b_values, D))


def identifiability_score(b_values, D_bins):
    """
    Score how well each diffusivity bin is identified by the b-values.

    Returns:
        scores: array of length len(D_bins) with identifiability scores
        min_signal: minimum signal contribution for each bin
    """
    U = signal_contribution(b_values, D_bins)

    # For each D bin, find:
    # 1. Maximum signal (at b=0): exp(0) = 1.0
    # 2. Minimum detectable signal (at b_max): exp(-b_max * D)
    min_signal = np.min(U, axis=0)  # Minimum across all b-values
    max_signal = np.max(U, axis=0)  # Maximum across all b-values

    # Signal range: larger is better
    signal_range = max_signal - min_signal

    # Identifiability score: how much unique signal this bin contributes
    # Penalize bins with min_signal < threshold
    scores = signal_range * (min_signal > SIGNAL_THRESHOLD)

    return scores, min_signal


def compute_condition_number(b_values, D_bins):
    """Compute condition number of design matrix U."""
    U = signal_contribution(b_values, D_bins)
    return np.linalg.cond(U)


def propose_discretizations(b_values, n_bins):
    """
    Propose multiple discretization strategies for n_bins.

    Returns:
        dict of {name: diff_values}
    """
    b_max = np.max(b_values)

    # Strategy 1: Signal-aware (only where exp(-b_max*D) > threshold)
    D_cutoff = -np.log(SIGNAL_THRESHOLD) / b_max
    D_signal_aware = np.linspace(0.25, min(D_cutoff, 1.5), n_bins - 2)
    # Add bins for higher D (less identifiable but clinically relevant)
    D_signal_aware = np.append(D_signal_aware, [2.0, 3.0])

    # Strategy 2: Geometric spacing
    D_geometric = np.geomspace(0.3, 3.0, n_bins)

    # Strategy 3: Log-spacing
    D_log = np.logspace(np.log10(0.25), np.log10(3.0), n_bins)

    # Strategy 4: Tumor-weighted (dense at low D)
    if n_bins == 7:
        D_tumor = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    elif n_bins == 8:
        D_tumor = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    elif n_bins == 9:
        D_tumor = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    elif n_bins == 10:
        D_tumor = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 20.0])
    else:
        D_tumor = np.linspace(0.25, 3.0, n_bins)

    # Strategy 5: Hybrid (optimal balance)
    # Dense in tumor range (0.25-1.0), sparse in normal (1.0-3.0)
    if n_bins >= 7:
        n_tumor = max(3, n_bins // 2)
        n_normal = n_bins - n_tumor
        D_hybrid = np.concatenate(
            [np.linspace(0.25, 1.0, n_tumor), np.linspace(1.2, 3.0, n_normal)]
        )
    else:
        D_hybrid = np.linspace(0.25, 3.0, n_bins)

    return {
        "signal_aware": D_signal_aware[:n_bins],
        "geometric": D_geometric,
        "log_spacing": D_log,
        "tumor_weighted": D_tumor,
        "hybrid": D_hybrid,
    }


def analyze_discretization(b_values, D_bins, name):
    """Analyze a single discretization."""
    scores, min_signal = identifiability_score(b_values, D_bins)
    cond_num = compute_condition_number(b_values, D_bins)

    # Count identifiable bins
    n_identifiable = np.sum(min_signal > SIGNAL_THRESHOLD)

    return {
        "name": name,
        "D_bins": D_bins,
        "scores": scores,
        "min_signal": min_signal,
        "cond_number": cond_num,
        "n_identifiable": n_identifiable,
        "mean_score": np.mean(scores[scores > 0]),  # Average of identifiable bins
    }


def plot_analysis(results, n_bins):
    """Create visualization of discretization analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Signal contribution heatmap for best discretization
    ax = axes[0, 0]
    best = max(results, key=lambda x: x["mean_score"])
    U = signal_contribution(B_VALUES, best["D_bins"])
    im = ax.imshow(U.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("b-value index", fontsize=12)
    ax.set_ylabel("Diffusivity bin", fontsize=12)
    ax.set_title(
        f"Best: {best['name']} (κ={best['cond_number']:.1e})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yticks(range(len(best["D_bins"])))
    ax.set_yticklabels([f"{d:.2f}" for d in best["D_bins"]])
    plt.colorbar(im, ax=ax, label="Signal: exp(-b·D)")

    # Plot 2: Identifiability scores
    ax = axes[0, 1]
    for r in results:
        ax.plot(r["D_bins"], r["scores"], "o-", label=r["name"], alpha=0.7)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="Non-identifiable")
    ax.set_xlabel("Diffusivity (μm²/ms)", fontsize=12)
    ax.set_ylabel("Identifiability Score", fontsize=12)
    ax.set_title("Identifiability by Discretization", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 3: Minimum signal per bin
    ax = axes[1, 0]
    for r in results:
        ax.semilogy(r["D_bins"], r["min_signal"], "o-", label=r["name"], alpha=0.7)
    ax.axhline(
        SIGNAL_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Threshold ({SIGNAL_THRESHOLD})",
    )
    ax.set_xlabel("Diffusivity (μm²/ms)", fontsize=12)
    ax.set_ylabel("Min Signal (log scale)", fontsize=12)
    ax.set_title("Minimum Signal Contribution", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 4: Summary metrics
    ax = axes[1, 1]
    names = [r["name"] for r in results]
    cond_nums = [r["cond_number"] for r in results]
    n_ident = [r["n_identifiable"] for r in results]
    mean_scores = [
        r["mean_score"] if not np.isnan(r["mean_score"]) else 0 for r in results
    ]

    x = np.arange(len(names))
    width = 0.25

    ax2 = ax.twinx()
    bars1 = ax.bar(
        x - width, n_ident, width, label="# Identifiable", color="green", alpha=0.7
    )
    bars2 = ax2.bar(x, mean_scores, width, label="Mean Score", color="blue", alpha=0.7)
    bars3 = ax2.bar(
        x + width,
        np.log10(cond_nums),
        width,
        label="log₁₀(κ)",
        color="orange",
        alpha=0.7,
    )

    ax.set_ylabel("# Identifiable Bins", fontsize=12, color="green")
    ax2.set_ylabel("Score / log₁₀(κ)", fontsize=12, color="blue")
    ax.set_xlabel("Discretization", fontsize=12)
    ax.set_title("Summary Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelcolor="green")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax.axhline(n_bins, color="green", linestyle="--", alpha=0.5, label="All bins")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"discretization_analysis_{n_bins}bins.pdf", dpi=150)
    print(f"Saved plot: {OUTPUT_DIR / f'discretization_analysis_{n_bins}bins.pdf'}")


def print_results(results, n_bins):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS FOR {n_bins} BINS")
    print(f"{'='*80}\n")

    # Sort by mean score (best first)
    results_sorted = sorted(results, key=lambda x: x["mean_score"], reverse=True)

    for i, r in enumerate(results_sorted, 1):
        print(
            f"{i}. {r['name'].upper():20s} | κ={r['cond_number']:8.1e} | "
            f"Identifiable: {r['n_identifiable']}/{n_bins} | "
            f"Score: {r['mean_score']:.3f}"
        )
        print(f"   D bins: {[f'{d:.2f}' for d in r['D_bins']]}")
        print()


def generate_yaml_config(D_bins, name, n_bins):
    """Generate YAML config snippet for the discretization."""
    yaml_str = f"""
  optimal_{name}_{n_bins}bins:
    # Auto-generated optimal discretization
    # Strategy: {name}
    # Identifiability threshold: exp(-b_max*D) > {SIGNAL_THRESHOLD}
    diff_values: {[round(d, 2) for d in D_bins]}
    true_spectrum: {[round(1.0/len(D_bins), 4) for _ in D_bins]}  # Uniform (update with real data)
"""
    return yaml_str


def main():
    """Main analysis loop."""
    print(f"Diffusivity Discretization Optimizer")
    print(f"{'='*80}")
    print(f"B-values: {B_VALUES}")
    print(f"Signal threshold: {SIGNAL_THRESHOLD}")
    print(f"Max b-value: {np.max(B_VALUES)}")
    print(
        f"Identifiable D range: 0.25 to {-np.log(SIGNAL_THRESHOLD)/np.max(B_VALUES):.2f} μm²/ms"
    )
    print(f"{'='*80}\n")

    all_configs = []

    for n_bins in N_BINS_TO_TEST:
        print(f"\nAnalyzing {n_bins}-bin discretizations...")

        # Generate proposals
        proposals = propose_discretizations(B_VALUES, n_bins)

        # Analyze each
        results = []
        for name, D_bins in proposals.items():
            result = analyze_discretization(B_VALUES, D_bins, name)
            results.append(result)

        # Print and plot
        print_results(results, n_bins)
        plot_analysis(results, n_bins)

        # Save best config
        best = max(results, key=lambda x: x["mean_score"])
        yaml_config = generate_yaml_config(best["D_bins"], best["name"], n_bins)
        all_configs.append((n_bins, best["name"], yaml_config))

    # Save YAML configs
    yaml_output = OUTPUT_DIR / "optimal_discretizations.yaml"
    with open(yaml_output, "w") as f:
        f.write("# Optimal Diffusivity Discretizations\n")
        f.write("# Generated by find_optimal_diff_discretization.py\n\n")
        f.write("spectrum_pairs:\n")
        for n_bins, name, config in all_configs:
            f.write(config)

    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"✓ Plots saved to: {OUTPUT_DIR}")
    print(f"✓ YAML configs saved to: {yaml_output}")
    print(f"{'='*80}\n")

    print("RECOMMENDATION:")
    print("1. Review plots to see identifiability vs condition number trade-offs")
    print(
        "2. Copy optimal configs from optimal_discretizations.yaml to configs/dataset/simulated.yaml"
    )
    print("3. Test with stronger regularization (strength=0.1-0.5)")
    print("\nNext command:")
    print("  uv run python src/spectra_estimation_dmri/main.py \\")
    print("    dataset=simulated \\")
    print("    dataset.spectrum_pair=optimal_<name>_<N>bins \\")
    print("    prior.strength=0.1,0.5 \\")
    print("    inference=nuts \\")
    print("    dataset.snr=1000")


if __name__ == "__main__":
    main()
