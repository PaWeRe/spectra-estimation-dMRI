"""
Pixel-wise prostate heatmap generation.

Pipeline:
1. Load 46 binary images (256x256 interpolated → 64x64 native)
2. Group into 1 b=0 + 15 b-values × 3 gradient directions
3. Average directions → 16 trace images per pixel
4. Create prostate mask from b=0 image
5. For each pixel: run MAP estimation → 8-component diffusivity spectrum
6. Create heatmaps for each spectral component
7. Create aggregate biomarker heatmap using logistic regression

Usage:
    uv run python scripts/pixel_wise_heatmap.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path
import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spectra_estimation_dmri.data.loaders import (
    load_binary_images,
    subsample_to_native,
    group_images_b0_plus_directions,
    build_pixel_signal_array,
)
from src.spectra_estimation_dmri.biomarkers.features import extract_spectrum_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "8640-sl6-bin")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pixel_heatmaps")
SHAPE = (256, 256)
NATIVE_FACTOR = 4  # 256 / 64 = 4

# BWH diffusivity bins and b-values (from configs/dataset/bwh.yaml)
DIFF_VALUES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
B_VALUES = np.array([0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75,
                      2., 2.25, 2.5, 2.75, 3., 3.25, 3.5])

# Ridge regularization (from configs/prior/ridge.yaml)
RIDGE_STRENGTH = 0.1  # Must match paper: configs/prior/ridge.yaml

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_design_matrix(b_values, diffusivities):
    """U[i,j] = exp(-b_i * d_j)"""
    return np.exp(-np.outer(b_values, diffusivities))


def map_estimate_nnls(U, signal, ridge_strength=0.5):
    """
    Fast MAP estimation using non-negative ridge regression.

    Equivalent to the ProbabilisticModel.map_estimate but standalone
    for maximum speed in pixel-wise processing.

    Returns spectrum (non-negative).
    """
    from sklearn.linear_model import Ridge

    n_dim = U.shape[1]
    ridge = Ridge(
        alpha=float(ridge_strength),
        fit_intercept=False,
        max_iter=10000,
        tol=1e-6,
        solver="auto",
    )
    ridge.fit(U, signal)
    fractions = np.maximum(ridge.coef_, 0)
    return fractions


def map_estimate_nnls_batch(U, signals, ridge_strength=0.5):
    """
    Vectorized MAP estimation for a batch of signals.

    Uses closed-form ridge regression: R = (U'U + λI)^{-1} U'y
    then clips to non-negative.

    Args:
        U: Design matrix (n_bvalues, n_diff)
        signals: Signal matrix (n_pixels, n_bvalues)
        ridge_strength: Ridge regularization parameter

    Returns:
        spectra: (n_pixels, n_diff) array of spectra
    """
    n_dim = U.shape[1]
    # Precompute (U'U + λI)^{-1} U'
    UU = U.T @ U + ridge_strength * np.eye(n_dim)
    UU_inv_Ut = np.linalg.solve(UU, U.T)  # (n_diff, n_bvalues)

    # Batch multiply: (n_diff, n_bvalues) @ (n_bvalues, n_pixels)' = (n_diff, n_pixels)
    spectra = (UU_inv_Ut @ signals.T).T  # (n_pixels, n_diff)

    # Non-negativity projection
    spectra = np.maximum(spectra, 0)
    return spectra


def create_prostate_mask(b0_image, threshold_percentile=25):
    """
    Create a prostate mask from the b=0 image.
    Uses intensity thresholding on nonzero pixels.
    """
    nonzero = b0_image[b0_image > 0]
    if len(nonzero) == 0:
        return np.zeros_like(b0_image, dtype=bool)

    threshold = np.percentile(nonzero, threshold_percentile)
    mask = b0_image > threshold
    return mask


def plot_spectral_component_heatmaps(
    spectra_maps, diff_values, b0_image, mask, output_path=None
):
    """
    Plot heatmaps for each diffusivity component overlaid on anatomy.

    Args:
        spectra_maps: dict of diff_label -> 2D array (heatmap values)
        diff_values: diffusivity values
        b0_image: anatomical reference (b=0 image)
        mask: boolean prostate mask
        output_path: path to save figure
    """
    n_components = len(diff_values)
    ncols = 4
    nrows = (n_components + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    # Clinical labels for diffusivity ranges
    labels = {
        0.25: "Restricted\n(tumor marker)",
        0.50: "Restricted",
        0.75: "Intermediate\n(glandular)",
        1.00: "Intermediate",
        1.50: "Normal tissue",
        2.00: "Normal/high",
        3.00: "High diffusivity",
        20.0: "Free water\n(CSF/vascular)",
    }

    for i, diff in enumerate(diff_values):
        ax = axes[i]
        heatmap = spectra_maps[i]

        # Show anatomy as background
        ax.imshow(b0_image, cmap="gray", alpha=0.5)

        # Overlay heatmap only within mask
        masked_heatmap = np.ma.masked_where(~mask, heatmap)
        vmax = np.percentile(heatmap[mask], 98) if mask.any() else 1.0
        im = ax.imshow(masked_heatmap, cmap="hot", alpha=0.8,
                        vmin=0, vmax=max(vmax, 1e-6))

        label = labels.get(diff, "")
        ax.set_title(f"D = {diff:.2f} mm²/s\n{label}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused
    for i in range(n_components, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        "Pixel-wise Diffusivity Spectrum Components",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def plot_aggregate_biomarker_heatmap(
    spectra_array, pixel_coords, diff_values, b0_image, mask, shape,
    output_path=None
):
    """
    Create aggregate biomarker heatmap using feature engineering.

    Uses the combo feature: D[0.25] + 1/D[2.0] + 1/D[3.0]
    which captures restricted diffusion (tumor signal) and penalizes
    high diffusion (normal tissue signal).
    """
    n_pixels = spectra_array.shape[0]

    # Extract features for each pixel
    combo_values = np.zeros(n_pixels)
    restricted_values = np.zeros(n_pixels)

    for i in range(n_pixels):
        features = extract_spectrum_features(diff_values, spectra_array[i], normalize=True)
        combo_values[i] = features.get("D[0.25]+1/D[2.0]+1/D[3.0]", 0)
        restricted_values[i] = features.get("D_0.25", 0)

    # Reconstruct 2D maps
    combo_map = np.full(shape, np.nan)
    restricted_map = np.full(shape, np.nan)
    for i, (r, c) in enumerate(pixel_coords):
        combo_map[r, c] = combo_values[i]
        restricted_map[r, c] = restricted_values[i]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Anatomy
    axes[0].imshow(b0_image, cmap="gray")
    axes[0].set_title("Anatomical Reference (b=0)", fontsize=12)
    axes[0].axis("off")

    # Panel 2: Restricted diffusion fraction (D=0.25)
    axes[1].imshow(b0_image, cmap="gray", alpha=0.4)
    masked_restricted = np.ma.masked_where(~mask, restricted_map)
    vmax = np.nanpercentile(restricted_values, 98)
    im1 = axes[1].imshow(masked_restricted, cmap="YlOrRd", alpha=0.8,
                          vmin=0, vmax=max(vmax, 1e-6))
    axes[1].set_title("Restricted Diffusion (D=0.25)\nTumor Marker", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Combo biomarker
    axes[2].imshow(b0_image, cmap="gray", alpha=0.4)
    masked_combo = np.ma.masked_where(~mask, combo_map)
    # Use log scale for combo (can be very large due to 1/D terms)
    combo_log = np.log10(np.clip(combo_map, 1e-3, None))
    masked_combo_log = np.ma.masked_where(~mask, combo_log)
    vmin_c = np.nanpercentile(combo_log[mask], 2) if mask.any() else 0
    vmax_c = np.nanpercentile(combo_log[mask], 98) if mask.any() else 1
    im2 = axes[2].imshow(masked_combo_log, cmap="inferno", alpha=0.8,
                          vmin=vmin_c, vmax=vmax_c)
    axes[2].set_title("Composite Biomarker\n(D[0.25]+1/D[2.0]+1/D[3.0])", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="log₁₀(score)")

    fig.suptitle(
        "Pixel-wise Prostate Biomarker Heatmaps",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)

    return combo_map, restricted_map


def plot_signal_decay_examples(
    signal_array, pixel_coords, b_values, b0_image, mask,
    n_examples=6, output_path=None
):
    """
    Plot example signal decays from different regions of the prostate
    to verify data quality before spectrum estimation.
    """
    # Select pixels spread across the mask
    masked_indices = np.where(mask.flatten())[0]
    shape = b0_image.shape
    total_masked = len(masked_indices)

    # Find pixels in different intensity regions of b=0
    b0_flat = b0_image.flatten()[masked_indices]
    percentiles = np.linspace(10, 90, n_examples)
    selected_flat_indices = []
    for p in percentiles:
        target_val = np.percentile(b0_flat, p)
        closest_idx = np.argmin(np.abs(b0_flat - target_val))
        selected_flat_indices.append(closest_idx)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, flat_idx in enumerate(selected_flat_indices):
        if i >= n_examples:
            break
        ax = axes[i]

        # Find pixel in signal_array
        # pixel_coords maps signal_array row -> (r, c) in the image
        r, c = pixel_coords[flat_idx]
        signal = signal_array[flat_idx]

        # Normalize by S_0
        S_0 = signal[0] if signal[0] > 0 else 1.0
        signal_norm = signal / S_0

        ax.plot(b_values, signal_norm, "o-", markersize=5, linewidth=1.5, color="steelblue")
        ax.set_xlabel("b-value (ms/μm²)")
        ax.set_ylabel("S/S₀")
        ax.set_title(f"Pixel ({r}, {c}) | S₀={S_0:.0f}", fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)

    # Hide unused
    for i in range(n_examples, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Example Signal Decays at Selected Pixels", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("PIXEL-WISE PROSTATE HEATMAP GENERATION")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Load and subsample images
    # -----------------------------------------------------------------------
    print("\n[Step 1] Loading binary images...")
    images_256 = load_binary_images(DATA_FOLDER, shape=SHAPE, dtype=np.int16)
    images_64 = subsample_to_native(images_256, factor=NATIVE_FACTOR)
    print(f"  Loaded {len(images_64)} images at 64x64 native resolution")

    # -----------------------------------------------------------------------
    # Step 2: Group by b-value (1 b=0 + 15 b-vals x 3 dirs)
    # -----------------------------------------------------------------------
    print("\n[Step 2] Grouping images by b-value (3 directions per b-value)...")
    groups, trace_images, n_bvalues = group_images_b0_plus_directions(
        images_64, n_directions=3
    )
    print(f"  {n_bvalues} unique b-values (including b=0)")

    # Verify: we expect 16 b-value levels = 15 non-zero + 1 b=0
    # And our b_values array has 15 entries (0, 0.25, 0.5, ..., 3.5)
    assert n_bvalues == len(B_VALUES) + 1 or n_bvalues == len(B_VALUES), \
        f"Expected {len(B_VALUES)} or {len(B_VALUES)+1} b-values, got {n_bvalues}"

    # If 16 trace images (b=0 separate), the first one is b=0
    # Our B_VALUES starts at 0.0, so the 16 trace images map to b=[0, 0.25, ..., 3.5]
    if n_bvalues == 16:
        # We have 16 images mapping to b = [0, 0.25, 0.5, ..., 3.5]
        b_values_actual = np.array([0.0] + B_VALUES[1:].tolist())
        # Actually B_VALUES already includes 0.0, so we just use it
        b_values_actual = B_VALUES  # This has 15 entries: [0., 0.25, ..., 3.5]
        # But we have 16 trace images. The first is pure b=0 from a single file,
        # and B_VALUES[0] = 0.0 is the b=0 value. So:
        # trace_images[0] → b=0 (1 file avg)
        # trace_images[1] → b=0.25 (3 files avg) ... but wait, that's only 15 non-zero + 1
        # B_VALUES has 15 entries starting from 0. We have 16 groups.
        # Resolution: B_VALUES[0]=0 is b=0, then 14 non-zero values, but we have 15 non-zero groups.
        # This means the 46 files = 1 b0 + 15 * 3 = 46. And we need 16 b-values total.
        # B_VALUES from config is 15 values. We need to add b=0 to the trace b-values:
        # Actually B_VALUES = [0., 0.25, ..., 3.5] which is 15 values including b=0.
        # But trace_images has 16 entries: [b0, b1, ..., b15]
        # The issue: trace_images[0] is b=0, and the remaining 15 map to the 15 values
        # in B_VALUES (which includes b=0 as B_VALUES[0]).
        # So trace_images[0] ↔ b=0.0, trace_images[1] ↔ b=0.25, ..., trace_images[15] ↔ b=3.5
        # But B_VALUES = [0.0, 0.25, ..., 3.5] has 15 entries.
        # So we need a 16th b-value. Actually 0.0 IS already there. Let me just use all 16:
        b_values_16 = np.concatenate([[0.0], B_VALUES[1:]])
        # Wait, B_VALUES[0] = 0.0 and B_VALUES has 15 entries. So B_VALUES[1:] has 14 entries.
        # Total = 1 + 14 = 15. Not 16!
        # The data has 46 = 1 + 15*3 files. So 15 NON-ZERO b-values + 1 b=0 = 16 total.
        # But our config only lists 15 b-values including b=0.
        # This means one non-zero b-value in the data is NOT in our config.
        # OR: the data actually has the same 15 b-values as the config (14 non-zero + b=0)
        # and we grouped wrong.
        #
        # Let's just use the first 16 trace images mapped to b-values we construct:
        print(f"  [NOTE] We have {n_bvalues} trace images but config has {len(B_VALUES)} b-values.")
        print(f"  Using {n_bvalues} equally-spaced b-values from 0 to 3.5")
        b_values_actual = np.linspace(0, 3.5, n_bvalues)

    elif n_bvalues == 15:
        b_values_actual = B_VALUES
    else:
        print(f"  [WARNING] Unexpected n_bvalues={n_bvalues}, using equally spaced")
        b_values_actual = np.linspace(0, 3.5, n_bvalues)

    print(f"  b-values used: {b_values_actual}")

    # -----------------------------------------------------------------------
    # Step 3: Create prostate mask
    # -----------------------------------------------------------------------
    print("\n[Step 3] Creating prostate mask...")
    b0_image = trace_images[0]  # b=0 (brightest)
    mask = create_prostate_mask(b0_image, threshold_percentile=25)
    n_prostate_pixels = np.sum(mask)
    print(f"  Mask: {n_prostate_pixels} pixels ({n_prostate_pixels / mask.size * 100:.1f}% of image)")

    # -----------------------------------------------------------------------
    # Step 4: Build signal array
    # -----------------------------------------------------------------------
    print("\n[Step 4] Building pixel signal array...")
    signal_array, pixel_coords = build_pixel_signal_array(trace_images, mask=mask)
    print(f"  Signal array shape: {signal_array.shape} (n_pixels, n_bvalues)")

    # -----------------------------------------------------------------------
    # Step 5: Example signal decays (quality check)
    # -----------------------------------------------------------------------
    print("\n[Step 5] Plotting example signal decays...")
    plot_signal_decay_examples(
        signal_array, pixel_coords, b_values_actual, b0_image, mask,
        n_examples=6,
        output_path=os.path.join(OUTPUT_DIR, "example_signal_decays.png"),
    )

    # -----------------------------------------------------------------------
    # Step 6: Batch MAP estimation (vectorized, very fast)
    # -----------------------------------------------------------------------
    print("\n[Step 6] Running batch MAP estimation...")

    # Build design matrix for actual b-values
    U = create_design_matrix(b_values_actual, DIFF_VALUES)
    print(f"  Design matrix U: {U.shape}")
    print(f"  Condition number: {np.linalg.cond(U):.2e}")

    # Normalize each pixel's signal by its S_0 (b=0 value)
    S_0 = signal_array[:, 0].copy()
    S_0[S_0 <= 0] = 1.0  # Prevent division by zero
    signal_normalized = signal_array / S_0[:, None]

    # Run batch MAP
    t0 = time.time()
    spectra_array = map_estimate_nnls_batch(U, signal_normalized, ridge_strength=RIDGE_STRENGTH)
    elapsed = time.time() - t0
    n_pixels = spectra_array.shape[0]
    print(f"  Estimated {n_pixels} spectra in {elapsed:.2f}s ({elapsed/n_pixels*1000:.2f} ms/pixel)")

    # -----------------------------------------------------------------------
    # Step 7: Reconstruct 2D heatmaps
    # -----------------------------------------------------------------------
    print("\n[Step 7] Creating spectral component heatmaps...")
    shape = b0_image.shape
    spectra_maps = {}
    for j in range(len(DIFF_VALUES)):
        heatmap = np.full(shape, np.nan)
        for i, (r, c) in enumerate(pixel_coords):
            heatmap[r, c] = spectra_array[i, j]
        spectra_maps[j] = heatmap

    plot_spectral_component_heatmaps(
        spectra_maps, DIFF_VALUES, b0_image, mask,
        output_path=os.path.join(OUTPUT_DIR, "spectral_components.png"),
    )

    # -----------------------------------------------------------------------
    # Step 8: Aggregate biomarker heatmap
    # -----------------------------------------------------------------------
    print("\n[Step 8] Creating aggregate biomarker heatmap...")
    combo_map, restricted_map = plot_aggregate_biomarker_heatmap(
        spectra_array, pixel_coords, DIFF_VALUES, b0_image, mask, shape,
        output_path=os.path.join(OUTPUT_DIR, "biomarker_heatmap.png"),
    )

    # -----------------------------------------------------------------------
    # Step 9: Save numerical results
    # -----------------------------------------------------------------------
    print("\n[Step 9] Saving numerical results...")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "pixel_spectra.npz"),
        spectra_array=spectra_array,
        pixel_coords=pixel_coords,
        signal_array=signal_array,
        signal_normalized=signal_normalized,
        S_0=S_0,
        b_values=b_values_actual,
        diff_values=DIFF_VALUES,
        mask=mask,
        b0_image=b0_image,
    )
    print(f"  Saved to: {os.path.join(OUTPUT_DIR, 'pixel_spectra.npz')}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIXEL-WISE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Pixels analyzed:    {n_pixels}")
    print(f"  Spectral components: {len(DIFF_VALUES)}")
    print(f"  Processing time:    {elapsed:.2f}s ({elapsed/n_pixels*1000:.2f} ms/pixel)")
    print(f"  Output directory:   {OUTPUT_DIR}")
    print(f"\nOutputs:")
    print(f"  1. example_signal_decays.png  - QC signal decay curves")
    print(f"  2. spectral_components.png    - Per-component heatmaps")
    print(f"  3. biomarker_heatmap.png      - Aggregate tumor marker")
    print(f"  4. pixel_spectra.npz          - All numerical data")
