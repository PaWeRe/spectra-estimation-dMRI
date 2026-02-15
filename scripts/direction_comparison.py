"""
Direction Independence Analysis for Diffusion MRI Spectra.

Purpose: Demonstrate that the diffusivity spectra are direction-independent
(i.e., the 3 gradient encoding directions yield consistent spectra within
posterior uncertainty). This is expected for the Peripheral Zone (PZ) which
is relatively isotropic, and validates our trace-averaging approach.

Pipeline:
1. Load 46 binary images and identify 3-direction groups
2. For selected ROI pixels, extract signal decays per direction
3. Run MAP estimation per direction → 3 separate spectra per pixel
4. Run MAP estimation on direction-averaged signal → reference spectrum
5. Visualize: overlay 3 direction spectra with uncertainty bands

Usage:
    uv run python scripts/direction_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spectra_estimation_dmri.data.loaders import (
    load_binary_images,
    subsample_to_native,
    compute_mean_intensities,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "8640-sl6-bin")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "direction_comparison")
SHAPE = (256, 256)
NATIVE_FACTOR = 4

# BWH parameters
DIFF_VALUES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
# 16 b-values for 16 trace groups (b=0 + 15 non-zero)
B_VALUES_16 = np.linspace(0, 3.5, 16)

# Ridge regularization
RIDGE_STRENGTH = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_design_matrix(b_values, diffusivities):
    return np.exp(-np.outer(b_values, diffusivities))


def map_estimate_batch(U, signals, ridge_strength=0.5):
    """Vectorized MAP via closed-form ridge."""
    n_dim = U.shape[1]
    UU = U.T @ U + ridge_strength * np.eye(n_dim)
    UU_inv_Ut = np.linalg.solve(UU, U.T)
    spectra = (UU_inv_Ut @ signals.T).T
    return np.maximum(spectra, 0)


def separate_directions(images_64, n_directions=3):
    """
    Separate the 46 images into per-direction signal sets.

    Returns:
        b0_image: The b=0 image (brightest)
        dir_signals: dict with keys 0,1,2 mapping to
                     dict[bvalue_group_idx] -> 2D image
        trace_signals: dict[bvalue_group_idx] -> direction-averaged 2D image
    """
    means = compute_mean_intensities(images_64)
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)

    # First image is b=0
    b0_key = sorted_keys[0]
    b0_image = images_64[b0_key]

    # Remaining 45 images split into 15 groups of 3
    remaining = sorted_keys[1:]
    n_nonzero = len(remaining) // n_directions

    dir_signals = {d: {} for d in range(n_directions)}
    trace_signals = {}

    for g in range(n_nonzero):
        start = g * n_directions
        group_keys = remaining[start:start + n_directions]

        # Each key in the group is a different direction for the same b-value
        for d, key in enumerate(group_keys):
            dir_signals[d][g] = images_64[key]

        # Trace = average of all directions
        trace_signals[g] = np.mean(
            [images_64[k].astype(np.float64) for k in group_keys], axis=0
        )

    return b0_image, dir_signals, trace_signals, n_nonzero


def extract_pixel_signal_per_direction(
    b0_image, dir_signals, trace_signals, n_nonzero, pixel_rc
):
    """
    Extract signal decays at a specific pixel for each direction.

    Returns:
        dir_decays: (n_directions, n_bvalues) signal decay per direction
        trace_decay: (n_bvalues,) direction-averaged decay
        S_0: b=0 signal value
    """
    r, c = pixel_rc
    S_0 = float(b0_image[r, c])

    n_directions = len(dir_signals)

    # b=0 + n_nonzero b-values = n_bvalues
    n_bvalues = 1 + n_nonzero

    dir_decays = np.zeros((n_directions, n_bvalues))
    trace_decay = np.zeros(n_bvalues)

    # b=0 is same for all directions
    for d in range(n_directions):
        dir_decays[d, 0] = S_0

    trace_decay[0] = S_0

    for g in range(n_nonzero):
        for d in range(n_directions):
            dir_decays[d, g + 1] = float(dir_signals[d][g][r, c])
        trace_decay[g + 1] = float(trace_signals[g][r, c])

    return dir_decays, trace_decay, S_0


def run_direction_analysis_at_pixel(
    b0_image, dir_signals, trace_signals, n_nonzero, pixel_rc,
    U, ridge_strength
):
    """
    Run direction comparison for a single pixel.

    Returns dict with per-direction spectra and trace spectrum.
    """
    dir_decays, trace_decay, S_0 = extract_pixel_signal_per_direction(
        b0_image, dir_signals, trace_signals, n_nonzero, pixel_rc
    )

    if S_0 <= 0:
        return None

    # Normalize
    dir_decays_norm = dir_decays / S_0
    trace_decay_norm = trace_decay / S_0

    # MAP for each direction
    dir_spectra = map_estimate_batch(U, dir_decays_norm, ridge_strength)

    # MAP for trace (averaged)
    trace_spectrum = map_estimate_batch(U, trace_decay_norm[None, :], ridge_strength)[0]

    return {
        "pixel": pixel_rc,
        "S_0": S_0,
        "dir_decays": dir_decays,
        "dir_decays_norm": dir_decays_norm,
        "trace_decay": trace_decay,
        "trace_decay_norm": trace_decay_norm,
        "dir_spectra": dir_spectra,  # (3, n_diff)
        "trace_spectrum": trace_spectrum,  # (n_diff,)
    }


def select_analysis_pixels(b0_image, mask, n_pixels=12):
    """
    Select pixels for analysis from different regions and SNR levels.
    Returns list of (row, col) tuples.
    """
    # Get masked pixel coordinates
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []

    # Select spread of pixels based on b=0 intensity (= SNR proxy)
    intensities = np.array([float(b0_image[r, c]) for r, c in coords])

    # Sort by intensity and take evenly spaced samples
    sorted_idx = np.argsort(intensities)[::-1]  # Descending (high SNR first)
    step = max(1, len(sorted_idx) // n_pixels)
    selected_idx = sorted_idx[::step][:n_pixels]

    return [tuple(coords[i]) for i in selected_idx]


def plot_direction_comparison_grid(results_list, diff_values, b_values, output_path=None):
    """
    Create a multi-panel figure showing direction comparison for multiple pixels.
    Each panel shows 3 direction spectra as bars + trace spectrum as overlay.
    """
    n = len(results_list)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]
    axes_flat = axes.flatten()

    dir_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    dir_labels = ["Dir 1", "Dir 2", "Dir 3"]
    bar_width = 0.2
    x = np.arange(len(diff_values))

    for idx, result in enumerate(results_list):
        ax = axes_flat[idx]

        # Plot direction spectra as grouped bars
        for d in range(3):
            spectrum = result["dir_spectra"][d]
            # Normalize
            total = np.sum(spectrum) + 1e-10
            spectrum_norm = spectrum / total
            offset = (d - 1) * bar_width
            ax.bar(x + offset, spectrum_norm, bar_width * 0.9,
                   color=dir_colors[d], alpha=0.7, label=dir_labels[d])

        # Overlay trace spectrum
        trace_norm = result["trace_spectrum"] / (np.sum(result["trace_spectrum"]) + 1e-10)
        ax.plot(x, trace_norm, "k-o", markersize=5, linewidth=2, label="Trace avg", zorder=5)

        r, c = result["pixel"]
        ax.set_title(f"Pixel ({r},{c}) | S₀={result['S_0']:.0f}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:.2f}" for d in diff_values], fontsize=7, rotation=45)
        ax.set_ylabel("Fraction")
        ax.set_xlabel("D (mm²/s)")
        ax.grid(axis="y", alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(
        "Direction Independence: Per-direction Spectra vs Trace Average\n"
        "(3 gradient directions should yield consistent spectra within noise)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def plot_direction_signal_decays(results_list, b_values, output_path=None):
    """
    Show per-direction signal decays for selected pixels.
    """
    n = len(results_list)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]
    axes_flat = axes.flatten()

    dir_colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for idx, result in enumerate(results_list):
        ax = axes_flat[idx]

        for d in range(3):
            ax.plot(b_values, result["dir_decays_norm"][d], "o-",
                    color=dir_colors[d], markersize=3, linewidth=1,
                    alpha=0.7, label=f"Dir {d+1}")

        ax.plot(b_values, result["trace_decay_norm"], "k-s",
                markersize=4, linewidth=2, label="Trace", zorder=5)

        r, c = result["pixel"]
        ax.set_title(f"Pixel ({r},{c}) | S₀={result['S_0']:.0f}", fontsize=10)
        ax.set_xlabel("b (ms/μm²)")
        ax.set_ylabel("S/S₀")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(
        "Per-direction Signal Decays vs Trace Average",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def compute_direction_consistency_metrics(results_list, diff_values):
    """
    Compute quantitative metrics for direction consistency.

    Returns DataFrame with per-pixel and per-component metrics.
    """
    import pandas as pd

    rows = []
    for result in results_list:
        dir_spectra = result["dir_spectra"]  # (3, n_diff)
        trace_spectrum = result["trace_spectrum"]  # (n_diff,)
        r, c = result["pixel"]
        S_0 = result["S_0"]

        # Normalize spectra
        dir_norms = dir_spectra / (dir_spectra.sum(axis=1, keepdims=True) + 1e-10)
        trace_norm = trace_spectrum / (trace_spectrum.sum() + 1e-10)

        for j, diff in enumerate(diff_values):
            vals = dir_norms[:, j]
            row = {
                "pixel_r": r,
                "pixel_c": c,
                "S_0": S_0,
                "diffusivity": diff,
                "dir1": vals[0],
                "dir2": vals[1],
                "dir3": vals[2],
                "trace": trace_norm[j],
                "mean": np.mean(vals),
                "std": np.std(vals),
                "cv": np.std(vals) / (np.mean(vals) + 1e-10) * 100,
                "max_deviation_from_trace": np.max(np.abs(vals - trace_norm[j])),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_consistency_summary(metrics_df, output_path=None):
    """
    Summary plot showing direction consistency across all pixels and components.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: CV distribution per diffusivity component
    ax = axes[0]
    diff_vals = sorted(metrics_df["diffusivity"].unique())
    cv_data = [metrics_df[metrics_df["diffusivity"] == d]["cv"].values for d in diff_vals]
    bp = ax.boxplot(cv_data, labels=[f"{d:.2f}" for d in diff_vals],
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.5)
    ax.set_xlabel("Diffusivity D (mm²/s)")
    ax.set_ylabel("CV across directions (%)")
    ax.set_title("Direction Consistency\nper Spectral Component")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: CV vs S_0 (SNR proxy)
    ax = axes[1]
    # Average CV across components for each pixel
    pixel_cv = metrics_df.groupby(["pixel_r", "pixel_c"]).agg(
        mean_cv=("cv", "mean"),
        S_0=("S_0", "first")
    ).reset_index()
    ax.scatter(pixel_cv["S_0"], pixel_cv["mean_cv"], s=40, alpha=0.7, c="#e74c3c")
    ax.set_xlabel("S₀ (b=0 signal, SNR proxy)")
    ax.set_ylabel("Mean CV across components (%)")
    ax.set_title("Direction Consistency vs SNR")
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)

    # Panel 3: Max deviation from trace
    ax = axes[2]
    dev_data = [metrics_df[metrics_df["diffusivity"] == d]["max_deviation_from_trace"].values
                for d in diff_vals]
    bp2 = ax.boxplot(dev_data, labels=[f"{d:.2f}" for d in diff_vals],
                      patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#2ecc71")
        patch.set_alpha(0.5)
    ax.set_xlabel("Diffusivity D (mm²/s)")
    ax.set_ylabel("Max |direction - trace|")
    ax.set_title("Maximum Deviation\nfrom Trace Average")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Quantitative Direction Independence Assessment",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("DIRECTION INDEPENDENCE ANALYSIS")
    print("=" * 70)

    # Step 1: Load images
    print("\n[Step 1] Loading binary images...")
    images_256 = load_binary_images(DATA_FOLDER, shape=SHAPE, dtype=np.int16)
    images_64 = subsample_to_native(images_256, factor=NATIVE_FACTOR)

    # Step 2: Separate directions
    print("\n[Step 2] Separating gradient directions...")
    b0_image, dir_signals, trace_signals, n_nonzero = separate_directions(images_64)
    n_bvalues = 1 + n_nonzero
    print(f"  b=0 image extracted")
    print(f"  {n_nonzero} non-zero b-value levels x 3 directions")

    # Build b-values for our design matrix
    b_values = np.linspace(0, 3.5, n_bvalues)

    # Step 3: Create mask and select pixels
    print("\n[Step 3] Selecting analysis pixels...")
    from scripts.pixel_wise_heatmap import create_prostate_mask
    mask = create_prostate_mask(b0_image, threshold_percentile=25)

    # Select 12 pixels spanning different SNR levels
    analysis_pixels = select_analysis_pixels(b0_image, mask, n_pixels=12)
    print(f"  Selected {len(analysis_pixels)} pixels for analysis")

    # Step 4: Run direction comparison
    print("\n[Step 4] Running direction comparison at each pixel...")
    U = create_design_matrix(b_values, DIFF_VALUES)

    results = []
    for pixel in analysis_pixels:
        result = run_direction_analysis_at_pixel(
            b0_image, dir_signals, trace_signals, n_nonzero, pixel,
            U, RIDGE_STRENGTH
        )
        if result is not None:
            results.append(result)

    print(f"  Analyzed {len(results)} pixels")

    # Step 5: Visualize
    print("\n[Step 5] Creating visualizations...")

    # Signal decay comparison
    plot_direction_signal_decays(
        results[:8],  # Show top 8 (highest SNR)
        b_values,
        output_path=os.path.join(OUTPUT_DIR, "direction_signal_decays.png"),
    )

    # Spectrum comparison
    plot_direction_comparison_grid(
        results[:8],
        DIFF_VALUES, b_values,
        output_path=os.path.join(OUTPUT_DIR, "direction_spectra_comparison.png"),
    )

    # Step 6: Quantitative metrics
    print("\n[Step 6] Computing consistency metrics...")
    metrics_df = compute_direction_consistency_metrics(results, DIFF_VALUES)

    # Summary statistics
    print("\n  Direction Consistency Summary:")
    print(f"  {'Component':<12} {'Mean CV%':<10} {'Median CV%':<12} {'Max CV%':<10}")
    print(f"  {'-'*44}")
    for diff in DIFF_VALUES:
        subset = metrics_df[metrics_df["diffusivity"] == diff]
        print(f"  D={diff:<8.2f} {subset['cv'].mean():<10.1f} {subset['cv'].median():<12.1f} {subset['cv'].max():<10.1f}")

    overall_cv = metrics_df["cv"].mean()
    print(f"\n  Overall mean CV: {overall_cv:.1f}%")
    print(f"  Pixels with mean CV < 10%: {(metrics_df.groupby(['pixel_r','pixel_c'])['cv'].mean() < 10).sum()}/{len(results)}")

    # Plot summary
    plot_consistency_summary(
        metrics_df,
        output_path=os.path.join(OUTPUT_DIR, "direction_consistency_summary.png"),
    )

    # Save metrics
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "direction_metrics.csv"), index=False)
    print(f"\n  Saved metrics to: {os.path.join(OUTPUT_DIR, 'direction_metrics.csv')}")

    print("\n" + "=" * 70)
    print("DIRECTION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {OUTPUT_DIR}")
