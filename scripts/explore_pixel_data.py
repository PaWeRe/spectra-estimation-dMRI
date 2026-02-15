"""
Exploration script for the 8640-sl6-bin binary image data.

This script:
1. Reads all 46 binary images (256x256, int16)
2. Visualizes them in a grid sorted by mean intensity
3. Analyzes mean intensities to identify b-value groupings
4. Extracts and plots signal decays at sample pixel locations
5. Attempts to group images by b-value (assuming multiple gradient directions)

Usage:
    uv run python scripts/explore_pixel_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spectra_estimation_dmri.data.loaders import (
    load_binary_images,
    subsample_to_native,
    compute_mean_intensities,
    group_images_by_bvalue,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "8640-sl6-bin")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pixel_exploration")
SHAPE = (256, 256)
NATIVE_FACTOR = 4  # 256 / 64 = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_image_grid(images, title="All Images", output_path=None):
    """Plot all images in a grid, sorted by mean intensity (descending)."""
    means = {k: float(np.mean(img[img > 0])) for k, img in images.items()}
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)

    n = len(sorted_keys)
    ncols = 8
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 2.5))
    axes = axes.flatten()

    # Use consistent color scale across all images
    vmin = min(img.min() for img in images.values())
    vmax = max(img.max() for img in images.values())

    for i, key in enumerate(sorted_keys):
        ax = axes[i]
        im = ax.imshow(images[key], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{key:06d}\n(mean={means[key]:.0f})", fontsize=7)
        ax.axis("off")

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def plot_mean_intensity_analysis(images, output_path=None):
    """
    Plot mean intensities to identify b-value groups.

    Images at the same b-value (different gradient directions) should have
    similar mean intensities. Look for clusters/plateaus in the sorted plot.
    """
    means = compute_mean_intensities(images)
    sorted_items = sorted(means.items(), key=lambda x: x[1], reverse=True)
    file_nums = [item[0] for item in sorted_items]
    mean_vals = [item[1] for item in sorted_items]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Mean intensity sorted by value
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(file_nums)))
    bars = ax1.bar(range(len(file_nums)), mean_vals, color=colors, edgecolor="none")
    ax1.set_xticks(range(len(file_nums)))
    ax1.set_xticklabels([f"{fn:06d}" for fn in file_nums], rotation=90, fontsize=6)
    ax1.set_ylabel("Mean Intensity (nonzero pixels)")
    ax1.set_title("Mean Intensity per Image (sorted descending -- b=0 is brightest)")
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Differences between consecutive sorted means (to find group boundaries)
    ax2 = axes[1]
    diffs = np.diff(mean_vals)
    ax2.bar(range(len(diffs)), np.abs(diffs), color="coral", edgecolor="none")
    ax2.set_xticks(range(len(diffs)))
    ax2.set_xticklabels(
        [f"{file_nums[i]:06d}-{file_nums[i+1]:06d}" for i in range(len(diffs))],
        rotation=90,
        fontsize=5,
    )
    ax2.set_ylabel("|Difference in Mean Intensity|")
    ax2.set_title("Gaps Between Consecutive Mean Intensities (large gaps = b-value boundary)")
    ax2.grid(axis="y", alpha=0.3)

    # Mark the largest gaps (likely b-value boundaries)
    abs_diffs = np.abs(diffs)
    threshold = np.mean(abs_diffs) + 1.5 * np.std(abs_diffs)
    for i, d in enumerate(abs_diffs):
        if d > threshold:
            ax2.axvline(i, color="red", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def plot_signal_decays_at_pixels(images, pixel_coords, output_path=None):
    """
    Extract and plot signal decay at specific pixel coordinates.

    Each pixel's signal across all 46 images gives us the full
    multi-b-value decay curve (unsorted -- we sort by mean intensity
    as a proxy for b-value ordering).
    """
    means = compute_mean_intensities(images)
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)
    sorted_means = [means[k] for k in sorted_keys]

    n_pixels = len(pixel_coords)
    fig, axes = plt.subplots(n_pixels, 1, figsize=(14, 4 * n_pixels), squeeze=False)

    for idx, (row, col) in enumerate(pixel_coords):
        ax = axes[idx, 0]
        signal_values = [float(images[k][row, col]) for k in sorted_keys]

        ax.plot(range(len(signal_values)), signal_values, "o-", markersize=3, linewidth=1)
        ax.set_xlabel("Image index (sorted by descending mean intensity)")
        ax.set_ylabel("Signal Intensity")
        ax.set_title(f"Signal Decay at pixel ({row}, {col})")
        ax.grid(alpha=0.3)

        # Add secondary x-axis with file numbers
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_positions = list(range(0, len(sorted_keys), 3))
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f"{sorted_keys[i]:06d}" for i in tick_positions], fontsize=6)
        ax2.set_xlabel("File number", fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def plot_bvalue_groups(images, n_bvalues_options=(15, 16, 46), output_path=None):
    """
    Try different numbers of b-value groups and show the averaged images.
    """
    for n_bvalues in n_bvalues_options:
        if n_bvalues > len(images):
            continue

        groups, averaged = group_images_by_bvalue(images, n_bvalues=n_bvalues)

        n = len(averaged)
        ncols = min(8, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 2.5))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i in range(n):
            ax = axes[i]
            ax.imshow(averaged[i], cmap="gray")
            group_files = groups[i]
            ax.set_title(
                f"Group {i}\n({len(group_files)} imgs)\n{group_files}",
                fontsize=6,
            )
            ax.axis("off")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        fig.suptitle(
            f"Direction-averaged images ({n_bvalues} b-value groups from {len(images)} files)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_path:
            path = output_path.replace(".png", f"_n{n_bvalues}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved: {path}")
        plt.close(fig)


def plot_native_vs_interpolated(images_256, images_64, output_path=None):
    """Side-by-side comparison of first image at 256x256 vs 64x64."""
    key = list(images_256.keys())[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(images_256[key], cmap="gray")
    axes[0].set_title(f"Interpolated 256x256 (file {key:06d})")

    axes[1].imshow(images_64[key], cmap="gray")
    axes[1].set_title(f"Native 64x64 (subsampled, file {key:06d})")

    # Show the difference (should be smooth interpolation artifacts)
    from scipy.ndimage import zoom

    upsampled = zoom(images_64[key].astype(float), 4, order=1)
    diff = images_256[key].astype(float) - upsampled
    im = axes[2].imshow(diff, cmap="RdBu", vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axes[2].set_title("Difference (256x256 - upsampled 64x64)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def create_prostate_mask(b0_image, threshold_percentile=30):
    """
    Create a rough prostate mask from the b=0 image using intensity thresholding.

    Args:
        b0_image: 2D array of the b=0 image (brightest)
        threshold_percentile: Percentile of nonzero pixels to use as threshold

    Returns:
        Boolean mask (True = prostate tissue)
    """
    # Use only nonzero pixels for threshold computation
    nonzero = b0_image[b0_image > 0]
    if len(nonzero) == 0:
        return np.zeros_like(b0_image, dtype=bool)

    threshold = np.percentile(nonzero, threshold_percentile)
    mask = b0_image > threshold
    return mask


def print_data_summary(images):
    """Print a summary of the loaded data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Number of files:   {len(images)}")

    keys = sorted(images.keys())
    print(f"File numbers:      {keys[0]:06d} to {keys[-1]:06d}")
    print(f"File increment:    {keys[1] - keys[0] if len(keys) > 1 else 'N/A'}")

    first = images[keys[0]]
    print(f"Image shape:       {first.shape}")
    print(f"Image dtype:       {first.dtype}")
    print(f"Value range:       [{first.min()}, {first.max()}]")

    # Summary statistics
    all_means = [float(np.mean(img[img > 0])) for img in images.values()]
    print(f"Mean intensity:    min={min(all_means):.1f}, max={max(all_means):.1f}")
    print(f"Mean ratio max/min: {max(all_means)/min(all_means):.1f}x")

    # Check divisibility for b-value grouping
    n = len(images)
    print(f"\nDivisibility check for {n} files:")
    for d in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 23, 46]:
        if n % d == 0:
            print(f"  {n} / {d} = {n // d} groups")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PIXEL DATA EXPLORATION")
    print("=" * 60)

    # Step 1: Load all binary images
    print("\n[Step 1] Loading binary images...")
    images_256 = load_binary_images(DATA_FOLDER, shape=SHAPE, dtype=np.int16)
    print_data_summary(images_256)

    # Step 2: Subsample to native 64x64 resolution
    print("\n[Step 2] Subsampling to native 64x64 resolution...")
    images_64 = subsample_to_native(images_256, factor=NATIVE_FACTOR)
    first_key = sorted(images_64.keys())[0]
    print(f"  Native shape: {images_64[first_key].shape}")

    # Step 3: Visualize all 46 images in a grid
    print("\n[Step 3] Creating image grid visualization...")
    plot_image_grid(
        images_256,
        title="All 46 Images (256x256, sorted by mean intensity descending)",
        output_path=os.path.join(OUTPUT_DIR, "image_grid_256.png"),
    )
    plot_image_grid(
        images_64,
        title="All 46 Images (64x64 native, sorted by mean intensity descending)",
        output_path=os.path.join(OUTPUT_DIR, "image_grid_64.png"),
    )

    # Step 4: Mean intensity analysis
    print("\n[Step 4] Analyzing mean intensities for b-value grouping...")
    plot_mean_intensity_analysis(
        images_256,
        output_path=os.path.join(OUTPUT_DIR, "mean_intensity_analysis.png"),
    )

    # Step 5: Native vs interpolated comparison
    print("\n[Step 5] Comparing native vs interpolated resolution...")
    plot_native_vs_interpolated(
        images_256,
        images_64,
        output_path=os.path.join(OUTPUT_DIR, "native_vs_interpolated.png"),
    )

    # Step 6: Signal decay at sample pixels
    print("\n[Step 6] Extracting signal decays at sample pixels...")
    # Pick pixels: center, and 4 quadrant samples
    # (Using 256x256 coords -- roughly center and off-center locations)
    sample_pixels = [
        (128, 128),  # Center
        (100, 100),  # Upper-left quadrant
        (100, 156),  # Upper-right quadrant
        (156, 100),  # Lower-left quadrant
        (156, 156),  # Lower-right quadrant
    ]
    plot_signal_decays_at_pixels(
        images_256,
        sample_pixels,
        output_path=os.path.join(OUTPUT_DIR, "signal_decays_sample_pixels.png"),
    )

    # Step 7: Try different b-value groupings
    print("\n[Step 7] Testing b-value groupings...")
    # The most likely groupings: 46 = 46x1, 23x2, 15*3+1, or 16*2+14
    # We try several options so the user can visually evaluate
    plot_bvalue_groups(
        images_256,
        n_bvalues_options=[46, 23, 16, 15],
        output_path=os.path.join(OUTPUT_DIR, "bvalue_groups.png"),
    )

    # Step 8: Create rough prostate mask from brightest image (b=0 proxy)
    print("\n[Step 8] Creating rough prostate mask...")
    means = compute_mean_intensities(images_256)
    b0_key = max(means, key=means.get)  # Brightest image is likely b=0
    b0_image = images_256[b0_key]
    mask = create_prostate_mask(b0_image, threshold_percentile=30)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(b0_image, cmap="gray")
    axes[0].set_title(f"b=0 proxy (file {b0_key:06d})")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Prostate mask (thresh={30}th percentile)")
    axes[2].imshow(b0_image, cmap="gray")
    axes[2].imshow(mask, cmap="Reds", alpha=0.3)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    mask_path = os.path.join(OUTPUT_DIR, "prostate_mask.png")
    fig.savefig(mask_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {mask_path}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Review the image grid and mean intensity analysis")
    print("  2. Determine the b-value mapping (check supervisor or group by intensity)")
    print("  3. If gradient directions present, verify grouping in bvalue_groups_*.png")
    print("  4. Proceed to pixel-wise spectrum estimation (Phase 2)")
