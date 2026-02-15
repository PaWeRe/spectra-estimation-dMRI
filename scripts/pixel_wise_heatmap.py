"""
Pixel-wise tumor probability heatmap from binary dMRI image data.

This script implements the full pipeline:
  Phase 1b: Group images by b-value (1 b=0 + 15 x 3 directions = 46 files)
  Phase 2:  Pixel-wise MAP spectrum estimation (ridge regression)
  Phase 3:  Apply ROI-trained logistic regression -> tumor probability heatmap

Usage:
    uv run python scripts/pixel_wise_heatmap.py

Assumptions (update when supervisor provides exact protocol):
  - 46 binary files = 1 b=0 + 15 non-zero b-values x 3 gradient directions
  - b-values (in s/mm^2): [0, 250, 500, 750, 1000, 1250, 1500, 1750,
                            2000, 2250, 2500, 2750, 3000, 3250, 3500]
  - Images are 256x256 (interpolated from native 64x64)
  - Data type: 2-byte signed short integers (int16)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from pathlib import Path
from scipy.ndimage import median_filter, gaussian_filter
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import sys
import os
import json
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spectra_estimation_dmri.data.loaders import (
    load_binary_images,
    subsample_to_native,
    compute_mean_intensities,
    group_images_b0_plus_directions,
    build_pixel_signal_array,
)
from src.spectra_estimation_dmri.biomarkers.features import extract_spectrum_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "8640-sl6-bin")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pixel_heatmap")
BWH_JSON = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "spectra_estimation_dmri",
    "data",
    "bwh",
    "signal_decays.json",
)
BWH_META = os.path.join(
    os.path.dirname(__file__),
    "..",
    "src",
    "spectra_estimation_dmri",
    "data",
    "bwh",
    "metadata.csv",
)

# Image parameters
IMAGE_SHAPE = (256, 256)
NATIVE_FACTOR = 4  # 256/64
N_DIRECTIONS = 3  # Gradient directions per non-zero b-value

# B-values for the BWH protocol (in units used by the model: x10^-3 s/mm^2 = ms/um^2)
# The BWH config stores them as [0, 0.25, 0.5, ...] which are in units of x1000 s/mm^2
# 46 files = 1 b=0 + 15 non-zero x 3 directions -> 16 unique b-values
# The 16th b-value (b=3750 s/mm^2 = 3.75) is assumed from the 250 s/mm^2 increment pattern.
# UPDATE THIS when the exact protocol is confirmed by your supervisor.
B_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]

# Diffusivity grid (must match the ROI-level analysis)
DIFF_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]

# Ridge regularization strength for MAP estimation (stronger than ROI-level due to lower SNR)
RIDGE_ALPHA = 1.0

# Prostate mask threshold (percentile of b=0 nonzero pixels)
# Higher value = tighter mask (more selective, fewer false inclusions)
MASK_THRESHOLD_PERCENTILE = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# Phase 1b: B-value grouping and trace image computation
# ===========================================================================
def phase_1b_bvalue_grouping(images_native):
    """
    Group 46 images into 16 b-value levels (1 b=0 + 15 x 3 directions).
    Average gradient directions to produce isotropic trace images.
    """
    print("\n" + "=" * 60)
    print("PHASE 1b: B-VALUE GROUPING")
    print("=" * 60)

    n_expected_bvalues = len(B_VALUES)
    groups, trace_images, n_bvalues = group_images_b0_plus_directions(
        images_native, n_directions=N_DIRECTIONS
    )

    # Verify we got the expected number
    if n_bvalues != n_expected_bvalues:
        print(
            f"\n[WARNING] Got {n_bvalues} b-value groups, expected {n_expected_bvalues}."
        )
        print("  Will use min(available, expected) b-values for spectrum estimation.")
    else:
        print(f"\n[OK] {n_bvalues} b-value groups match expected {n_expected_bvalues}.")

    # Visualize the trace images
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    axes = axes.flatten()
    for i in range(min(n_bvalues, 16)):
        ax = axes[i]
        ax.imshow(trace_images[i], cmap="gray")
        b_label = f"b={B_VALUES[i]:.2f}" if i < len(B_VALUES) else f"Group {i}"
        ax.set_title(f"{b_label}\n({len(groups[i])} imgs)", fontsize=8)
        ax.axis("off")
    for i in range(n_bvalues, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Trace Images (direction-averaged) per b-value", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "trace_images.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved trace images visualization")

    return groups, trace_images, n_bvalues


# ===========================================================================
# Phase 2: Pixel-wise MAP spectrum estimation
# ===========================================================================
def create_prostate_mask(b0_image, threshold_percentile=MASK_THRESHOLD_PERCENTILE):
    """Create a prostate mask from the b=0 image."""
    nonzero = b0_image[b0_image > 0]
    if len(nonzero) == 0:
        return np.zeros_like(b0_image, dtype=bool)
    threshold = np.percentile(nonzero, threshold_percentile)
    mask = b0_image > threshold
    return mask


def build_design_matrix(b_values, diff_values):
    """Build design matrix U where U[i,j] = exp(-b_i * d_j)."""
    b = np.array(b_values)
    d = np.array(diff_values)
    return np.exp(-np.outer(b, d))


def map_estimate_ridge(signal, U, alpha=RIDGE_ALPHA):
    """
    Compute MAP spectrum estimate using ridge regression with non-negativity.

    Args:
        signal: Signal decay vector (n_bvalues,)
        U: Design matrix (n_bvalues, n_diffusivities)
        alpha: Ridge regularization strength

    Returns:
        Spectrum vector (n_diffusivities,)
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    ridge.fit(U, signal)
    fractions = np.maximum(ridge.coef_, 0)  # Non-negativity projection
    return fractions


def phase_2_spectrum_estimation(trace_images, n_bvalues):
    """
    Estimate diffusivity spectrum at every pixel using MAP (ridge regression).
    """
    print("\n" + "=" * 60)
    print("PHASE 2: PIXEL-WISE SPECTRUM ESTIMATION")
    print("=" * 60)

    # Create prostate mask from b=0 image
    b0_image = trace_images[0]
    mask = create_prostate_mask(b0_image)
    n_masked = np.sum(mask)
    total_pixels = mask.size
    print(f"  Prostate mask: {n_masked} pixels ({100*n_masked/total_pixels:.1f}% of FOV)")

    # Use only the b-values we have trace images for
    n_bvals_available = min(n_bvalues, len(B_VALUES))
    b_values_used = B_VALUES[:n_bvals_available]
    print(f"  Using {n_bvals_available} b-values: {b_values_used}")
    print(f"  Diffusivity grid: {DIFF_VALUES}")

    # Build design matrix
    U = build_design_matrix(b_values_used, DIFF_VALUES)
    cond_num = np.linalg.cond(U)
    print(f"  Design matrix condition number: {cond_num:.2e}")

    # Build pixel signal array
    signal_array, pixel_coords = build_pixel_signal_array(trace_images, mask)
    # Only use the b-values we have
    signal_array = signal_array[:, :n_bvals_available]
    print(f"  Signal array shape: {signal_array.shape}")

    # Normalize each pixel's signal by its b=0 value
    s0 = signal_array[:, 0].copy()
    s0[s0 <= 0] = 1.0  # Avoid division by zero
    signal_normalized = signal_array / s0[:, np.newaxis]

    # MAP estimation for each pixel
    print(f"\n  Running MAP (ridge, alpha={RIDGE_ALPHA}) on {n_masked} pixels...")
    start_time = time.time()

    n_diff = len(DIFF_VALUES)
    spectra_array = np.zeros((n_masked, n_diff), dtype=np.float64)

    for i in range(n_masked):
        spectra_array[i] = map_estimate_ridge(signal_normalized[i], U, alpha=RIDGE_ALPHA)

        if (i + 1) % 500 == 0 or i == n_masked - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_masked - i - 1) / rate if rate > 0 else 0
            print(f"    Processed {i+1}/{n_masked} pixels ({rate:.0f} px/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"  Done! {n_masked} pixels in {elapsed:.1f}s ({n_masked/elapsed:.0f} px/s)")

    # Reconstruct 2D spectrum maps
    img_shape = trace_images[0].shape
    spectrum_maps = np.zeros((*img_shape, n_diff), dtype=np.float64)
    for i, (row, col) in enumerate(pixel_coords):
        spectrum_maps[row, col, :] = spectra_array[i]

    # Visualize individual diffusivity bin maps
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, d in enumerate(DIFF_VALUES):
        ax = axes[i]
        bin_map = spectrum_maps[:, :, i]
        # Mask background
        bin_map_masked = np.ma.masked_where(~mask, bin_map)
        im = ax.imshow(bin_map_masked, cmap="hot")
        ax.set_title(f"D = {d:.2f} μm²/ms", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Diffusivity Spectrum Maps (per-bin fraction)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "spectrum_maps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved spectrum maps")

    # Also compute a simple ADC map for comparison
    adc_map = np.zeros(img_shape, dtype=np.float64)
    for i, (row, col) in enumerate(pixel_coords):
        sig = signal_normalized[i]
        # Simple monoexponential fit using b=0 to b=1.0 (first 5 b-values)
        b_fit = np.array(b_values_used[:5])
        s_fit = sig[:5]
        valid = s_fit > 0
        if np.sum(valid) >= 2:
            log_s = np.log(s_fit[valid])
            slope, _ = np.polyfit(b_fit[valid], log_s, 1)
            adc_map[row, col] = -slope
        else:
            adc_map[row, col] = 0.0

    return spectrum_maps, mask, pixel_coords, spectra_array, adc_map, signal_normalized


# ===========================================================================
# Phase 3: Tumor probability heatmap
# ===========================================================================
def train_classifier_on_roi_data():
    """
    Train a logistic regression classifier on the full ROI dataset.

    Returns the trained model, scaler, feature names, and training data summary.
    """
    print("\n  Training classifier on ROI data...")

    # Load ROI data
    from src.spectra_estimation_dmri.data.loaders import load_bwh_signal_decays
    from src.spectra_estimation_dmri.models.prob_model import ProbabilisticModel

    signal_decays = load_bwh_signal_decays(BWH_JSON, BWH_META)
    print(f"    Loaded {len(signal_decays.samples)} ROI signal decays")

    # Estimate spectra for each ROI using MAP (fast)
    # Use the ROI's own b-values (BWH 15 b-values), NOT the pixel B_VALUES
    from omegaconf import OmegaConf

    prior_config = OmegaConf.create({"type": "ridge", "strength": RIDGE_ALPHA})

    # Get b-values from the first ROI (they're all the same BWH protocol)
    # ROI b-values are in s/mm^2 (e.g., 0, 250, 500, ...),
    # but the model uses units of x1000 s/mm^2 (e.g., 0, 0.25, 0.5, ...)
    roi_b_values_raw = signal_decays.samples[0].b_values
    roi_b_values = [b / 1000.0 if b > 10 else b for b in roi_b_values_raw]
    U = build_design_matrix(roi_b_values, DIFF_VALUES)
    print(f"    ROI b-values ({len(roi_b_values)}): {roi_b_values[:5]}... (converted to model units)")

    roi_features = []
    roi_labels = []
    roi_info = []

    for sd in signal_decays.samples:
        # Normalize signal
        signal = np.array(sd.signal_values)
        s0 = signal[0] if signal[0] > 0 else 1.0
        signal_norm = signal / s0

        # MAP estimate
        spectrum = map_estimate_ridge(signal_norm, U, alpha=RIDGE_ALPHA)

        # Extract features
        features = extract_spectrum_features(
            np.array(DIFF_VALUES), spectrum, normalize=True
        )

        # Get individual bin features (same order as DIFF_VALUES)
        feature_vec = [features[f"D_{d:.2f}"] for d in DIFF_VALUES]

        roi_features.append(feature_vec)

        # Label: 1 = tumor, 0 = normal
        if sd.is_tumor:
            roi_labels.append(1)
        else:
            roi_labels.append(0)

        roi_info.append(f"{sd.patient}_{sd.a_region}_{'tumor' if sd.is_tumor else 'normal'}")

    X_roi = np.array(roi_features)
    y_roi = np.array(roi_labels)

    print(f"    ROI features: {X_roi.shape}")
    print(f"    Labels: {np.sum(y_roi==0)} normal, {np.sum(y_roi==1)} tumor")

    # Train logistic regression (L2 regularized)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_roi)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(X_scaled, y_roi)

    # Report training accuracy (not LOOCV, just sanity check)
    train_acc = clf.score(X_scaled, y_roi)
    print(f"    Training accuracy: {train_acc:.3f}")
    print(f"    Feature names: {[f'D_{d:.2f}' for d in DIFF_VALUES]}")
    print(f"    Coefficients: {clf.coef_[0]}")

    feature_names = [f"D_{d:.2f}" for d in DIFF_VALUES]
    return clf, scaler, feature_names, X_roi, y_roi


def phase_3_heatmap(spectrum_maps, mask, pixel_coords, spectra_array, adc_map, b0_image):
    """
    Apply trained classifier pixel-wise to generate tumor probability heatmap.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: TUMOR PROBABILITY HEATMAP")
    print("=" * 60)

    # Train classifier on ROI data
    clf, scaler, feature_names, X_roi, y_roi = train_classifier_on_roi_data()

    # Extract features for each pixel
    print("\n  Extracting pixel-wise features...")
    n_pixels = len(pixel_coords)
    n_diff = len(DIFF_VALUES)

    pixel_features = np.zeros((n_pixels, n_diff), dtype=np.float64)
    for i in range(n_pixels):
        spectrum = spectra_array[i]
        features = extract_spectrum_features(
            np.array(DIFF_VALUES), spectrum, normalize=True
        )
        pixel_features[i] = [features[f"D_{d:.2f}"] for d in DIFF_VALUES]

    # Predict tumor probabilities
    print("  Predicting tumor probabilities...")
    pixel_features_scaled = scaler.transform(pixel_features)
    tumor_proba = clf.predict_proba(pixel_features_scaled)[:, 1]

    # Build probability map
    img_shape = b0_image.shape
    proba_map = np.full(img_shape, np.nan, dtype=np.float64)
    for i, (row, col) in enumerate(pixel_coords):
        proba_map[row, col] = tumor_proba[i]

    # Apply spatial smoothing (median filter to reduce noise)
    proba_map_smoothed = proba_map.copy()
    valid = ~np.isnan(proba_map)
    proba_filled = np.where(valid, proba_map, 0)
    proba_filled_smooth = median_filter(proba_filled, size=3)
    proba_map_smoothed[valid] = proba_filled_smooth[valid]

    # Also do Gaussian smoothing
    proba_gaussian = proba_map.copy()
    proba_filled2 = np.where(valid, proba_map, 0)
    proba_gaussian_smooth = gaussian_filter(proba_filled2, sigma=1.0)
    # Re-normalize within mask
    count_map = gaussian_filter(valid.astype(float), sigma=1.0)
    count_map[count_map == 0] = 1  # Avoid division by zero
    proba_gaussian_smooth = proba_gaussian_smooth / count_map
    proba_gaussian[valid] = proba_gaussian_smooth[valid]

    # =======================================================================
    # Visualization
    # =======================================================================
    print("\n  Creating visualizations...")

    # --- Figure 1: Main heatmap panel ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # b=0 anatomical image
    ax = axes[0, 0]
    ax.imshow(b0_image, cmap="gray")
    ax.set_title("b=0 Anatomical Image", fontsize=12)
    ax.axis("off")

    # ADC map
    ax = axes[0, 1]
    adc_masked = np.ma.masked_where(~mask, adc_map)
    im = ax.imshow(adc_masked, cmap="viridis", vmin=0, vmax=3.0)
    ax.set_title("ADC Map (0-1000 s/mm²)", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="ADC (×10⁻³ mm²/s)")

    # Raw tumor probability
    ax = axes[0, 2]
    proba_masked = np.ma.masked_where(~mask, proba_map)
    im = ax.imshow(proba_masked, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_title("Tumor Probability (raw)", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="P(tumor)")

    # Smoothed tumor probability (median)
    ax = axes[1, 0]
    proba_sm_masked = np.ma.masked_where(~mask, proba_map_smoothed)
    im = ax.imshow(proba_sm_masked, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_title("Tumor Probability (median filtered)", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="P(tumor)")

    # Smoothed tumor probability (Gaussian)
    ax = axes[1, 1]
    proba_g_masked = np.ma.masked_where(~mask, proba_gaussian)
    im = ax.imshow(proba_g_masked, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_title("Tumor Probability (Gaussian smoothed)", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="P(tumor)")

    # Overlay: probability on anatomy
    ax = axes[1, 2]
    ax.imshow(b0_image, cmap="gray")
    overlay = np.ma.masked_where(~mask, proba_gaussian)
    im = ax.imshow(overlay, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    ax.set_title("Probability Overlay on Anatomy", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="P(tumor)")

    fig.suptitle(
        "Pixel-Wise Tumor Probability Heatmap (MAP + Logistic Regression)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "tumor_heatmap_panel.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved tumor heatmap panel")

    # --- Figure 2: High-quality overlay for presentation ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(b0_image, cmap="gray")
    overlay = np.ma.masked_where(~mask, proba_gaussian)
    im = ax.imshow(overlay, cmap="hot", alpha=0.55, vmin=0, vmax=1)
    ax.set_title("Tumor Probability Heatmap", fontsize=14, fontweight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("P(tumor)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "tumor_heatmap_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved high-quality overlay")

    # --- Figure 3: Histogram of tumor probabilities ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    valid_proba = tumor_proba[~np.isnan(tumor_proba)]
    n_bins = min(50, max(10, int(np.sqrt(len(valid_proba)))))
    ax.hist(valid_proba, bins=n_bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Decision boundary")
    ax.set_xlabel("P(tumor)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Distribution of Tumor Probabilities")
    ax.legend()
    n_high = np.sum(valid_proba > 0.5)
    n_low = np.sum(valid_proba <= 0.5)
    ax.text(
        0.95, 0.95,
        f"P>0.5: {n_high} ({100*n_high/len(valid_proba):.1f}%)\nP≤0.5: {n_low} ({100*n_low/len(valid_proba):.1f}%)",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ADC vs tumor probability scatter
    ax = axes[1]
    valid_adc = adc_map[mask]
    valid_prob = proba_map[mask]
    ax.scatter(valid_adc, valid_prob, s=1, alpha=0.3, c="steelblue")
    ax.set_xlabel("ADC (×10⁻³ mm²/s)")
    ax.set_ylabel("P(tumor)")
    ax.set_title("ADC vs Tumor Probability")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "probability_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved probability analysis")

    # --- Figure 4: Sample pixel spectra comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Find pixels with high, medium, and low tumor probability
    sorted_proba_idx = np.argsort(tumor_proba)
    example_indices = [
        sorted_proba_idx[len(sorted_proba_idx) // 10],       # Low P(tumor)
        sorted_proba_idx[len(sorted_proba_idx) // 2],        # Medium P(tumor)
        sorted_proba_idx[int(len(sorted_proba_idx) * 0.9)],  # High P(tumor)
    ]
    labels = ["Low P(tumor)", "Medium P(tumor)", "High P(tumor)"]

    for ax, idx, label in zip(axes, example_indices, labels):
        spectrum = spectra_array[idx]
        # Normalize
        total = np.sum(spectrum) + 1e-10
        spectrum_norm = spectrum / total

        ax.bar(range(len(DIFF_VALUES)), spectrum_norm, color="steelblue", edgecolor="white")
        ax.set_xticks(range(len(DIFF_VALUES)))
        ax.set_xticklabels([f"{d:.2f}" for d in DIFF_VALUES], fontsize=8)
        ax.set_xlabel("Diffusivity (μm²/ms)")
        ax.set_ylabel("Fraction")
        row, col = pixel_coords[idx]
        ax.set_title(f"{label}\nPixel ({row},{col}), P={tumor_proba[idx]:.3f}")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Example Pixel Spectra at Different Tumor Probability Levels", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "example_spectra.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved example spectra")

    return proba_map, proba_map_smoothed, proba_gaussian


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PIXEL-WISE TUMOR PROBABILITY HEATMAP")
    print("=" * 60)
    overall_start = time.time()

    # Step 1: Load images
    print("\n[Step 1] Loading binary images...")
    images_256 = load_binary_images(DATA_FOLDER, shape=IMAGE_SHAPE, dtype=np.int16)

    # Step 2: Subsample to native 64x64
    print("\n[Step 2] Subsampling to native 64x64 resolution...")
    images_64 = subsample_to_native(images_256, factor=NATIVE_FACTOR)

    # Phase 1b: Group by b-value
    groups, trace_images, n_bvalues = phase_1b_bvalue_grouping(images_64)

    # Phase 2: Pixel-wise spectrum estimation
    b0_image = trace_images[0]
    spectrum_maps, mask, pixel_coords, spectra_array, adc_map, signal_norm = (
        phase_2_spectrum_estimation(trace_images, n_bvalues)
    )

    # Phase 3: Tumor probability heatmap
    proba_map, proba_smoothed, proba_gaussian = phase_3_heatmap(
        spectrum_maps, mask, pixel_coords, spectra_array, adc_map, b0_image
    )

    # Summary
    elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    for f in sorted(Path(OUTPUT_DIR).glob("*.png")):
        print(f"  {f.name}")

    print("\n[NOTE] The b-value mapping is currently assumed based on the BWH protocol.")
    print("Please verify with your supervisor that the mapping is correct.")
    print("If the b-values differ, update B_VALUES at the top of this script.")
