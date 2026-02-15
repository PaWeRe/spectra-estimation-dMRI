"""
Flagship Figure: ADC vs Spectral Biomarker vs Uncertainty Map.

Creates the money figure for the MRM paper:
  Panel A: Anatomical reference (b=0)
  Panel B: ADC map (monoexponential, traditional)
  Panel C: Spectral biomarker map (learned LR probability)
  Panel D: Uncertainty map (Laplace approximation of D=0.25 posterior std)

Pipeline:
1. Load pixel data and compute direction-averaged trace images
2. Per-pixel MAP spectrum estimation (vectorized, instant)
3. Per-pixel ADC computation (monoexponential fit to low-b)
4. Train logistic regression on ROI data (PZ tumor vs normal)
5. Apply LR weights to pixel spectra → p(tumor) map
6. Laplace approximation for per-pixel spectral uncertainty
7. Composite figure

Usage:
    uv run python scripts/flagship_figure.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
import time
import json

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
ROI_JSON = os.path.join(os.path.dirname(__file__), "..",
                         "src", "spectra_estimation_dmri", "data", "bwh", "signal_decays.json")
ROI_META = os.path.join(os.path.dirname(__file__), "..",
                         "src", "spectra_estimation_dmri", "data", "bwh", "metadata.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "flagship_figure")
SHAPE = (256, 256)
NATIVE_FACTOR = 4

# BWH parameters
DIFF_VALUES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
B_VALUES_CONFIG = np.array([0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75,
                             2., 2.25, 2.5, 2.75, 3., 3.25, 3.5])
RIDGE_STRENGTH = 0.5

# ADC: fit mono-exponential to b-values in range [0, 1.0] ms/um²
ADC_B_MAX = 1.0  # Use b-values up to 1000 s/mm²

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


def compute_adc_batch(signals, b_values, b_max=1.0):
    """
    Compute ADC per pixel using monoexponential fit S = S_0 * exp(-b * ADC).

    Uses linear regression on log(S/S_0) vs b for b <= b_max.
    ADC = -slope of log(S/S_0) vs b.

    Args:
        signals: (n_pixels, n_bvalues) normalized signal array (S/S_0)
        b_values: (n_bvalues,) b-value array
        b_max: Maximum b-value to include in fit (ms/um²)

    Returns:
        adc: (n_pixels,) ADC values in mm²/s
    """
    # Select b-values for fit
    mask_b = b_values <= b_max
    b_fit = b_values[mask_b]
    s_fit = signals[:, mask_b]

    # Clip to avoid log(0)
    s_fit = np.clip(s_fit, 1e-6, None)

    # Linear regression: log(S/S_0) = -ADC * b
    # y = log(S), x = b
    # Slope = -ADC
    log_s = np.log(s_fit)

    # Vectorized least squares: ADC = -Σ(b * log(S)) / Σ(b²)
    # Since S is already normalized (S/S_0), intercept should be ~0
    # Use proper OLS: y = a + bx, slope = (n*Σxy - ΣxΣy) / (n*Σx² - (Σx)²)
    n = len(b_fit)
    sum_b = np.sum(b_fit)
    sum_b2 = np.sum(b_fit ** 2)
    sum_logs = np.sum(log_s, axis=1)  # (n_pixels,)
    sum_b_logs = np.sum(b_fit[None, :] * log_s, axis=1)  # (n_pixels,)

    slope = (n * sum_b_logs - sum_b * sum_logs) / (n * sum_b2 - sum_b ** 2)
    adc = -slope  # ADC = -slope

    # Clip to reasonable range [0, 5] mm²/s
    adc = np.clip(adc, 0, 5.0)

    return adc


def laplace_uncertainty_batch(U, signals_norm, spectra_map, ridge_strength=0.5):
    """
    Compute Laplace approximation of posterior uncertainty per pixel.

    At the MAP estimate R*, the Hessian of the neg-log-posterior gives:
        Σ_post = (1/σ² U'U + λI)^{-1}
    where σ is estimated from the residuals.

    Args:
        U: Design matrix (n_bvalues, n_diff)
        signals_norm: (n_pixels, n_bvalues) normalized signal
        spectra_map: (n_pixels, n_diff) MAP spectra
        ridge_strength: Regularization parameter λ

    Returns:
        uncertainty: (n_pixels, n_diff) posterior std per component
    """
    n_dim = U.shape[1]
    n_bvalues = U.shape[0]

    # Estimate noise sigma per pixel from residuals
    residuals = signals_norm - spectra_map @ U.T  # (n_pixels, n_bvalues)
    sigma2 = np.sum(residuals ** 2, axis=1) / max(n_bvalues - n_dim, 1)  # (n_pixels,)

    # For each unique sigma, compute the posterior covariance
    # Since UU is shared, we precompute U'U
    UU = U.T @ U  # (n_diff, n_diff)
    lam_I = ridge_strength * np.eye(n_dim)

    # Per-pixel: Sigma_post = (1/sigma^2 * U'U + lambda * I)^{-1}
    # Diagonal of Sigma_post gives per-component variance
    uncertainty = np.zeros_like(spectra_map)

    for i in range(len(sigma2)):
        if sigma2[i] > 0:
            precision = (1.0 / sigma2[i]) * UU + lam_I
            cov = np.linalg.inv(precision)
            uncertainty[i] = np.sqrt(np.maximum(np.diag(cov), 0))
        else:
            uncertainty[i] = 0.0

    return uncertainty


def train_lr_on_roi_data(roi_json_path, metadata_path, diff_values, b_values, ridge_strength):
    """
    Train a logistic regression classifier on the ROI signal decay data.

    Returns the trained scaler, model, and feature names.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import csv

    # Load ROI data
    with open(roi_json_path, "r") as f:
        signal_data = json.load(f)
    metadata = {}
    with open(metadata_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata[row["patient_id"]] = row

    # Build design matrix
    U = create_design_matrix(np.array(b_values), diff_values)

    # Extract features from each ROI
    features_list = []
    labels = []
    regions = []

    for patient_id, rois in signal_data.items():
        for roi_name, roi in rois.items():
            anatomical_region = roi["anatomical_region"]
            is_tumor = "tumor" in anatomical_region
            is_pz = "pz" in anatomical_region

            # Focus on PZ for the primary classifier
            if not is_pz:
                continue

            signal = np.array(roi["signal_values"])
            b_vals = np.array(roi["b_values"])

            # Normalize by S_0
            S_0 = signal[0] if signal[0] > 0 else 1.0
            signal_norm = signal / S_0

            # MAP estimate
            spectrum = map_estimate_batch(U, signal_norm[None, :], ridge_strength)[0]

            # Extract features
            feats = extract_spectrum_features(diff_values, spectrum, normalize=True)
            features_list.append(feats)
            labels.append(1 if is_tumor else 0)
            regions.append(anatomical_region)

    # Build feature matrix
    feature_names = [f"D_{d:.2f}" for d in diff_values] + ["D[0.25]+1/D[2.0]+1/D[3.0]"]
    X = np.array([[f[fn] for fn in feature_names] for f in features_list])
    y = np.array(labels)

    print(f"  ROI data: {len(y)} samples ({np.sum(y)} tumor, {np.sum(1-y)} normal)")

    # Train LR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(X_scaled, y)

    print(f"  LR coefficients:")
    for fn, c in zip(feature_names, clf.coef_[0]):
        print(f"    {fn}: {c:+.3f}")

    return scaler, clf, feature_names


def apply_lr_to_pixels(spectra_array, diff_values, scaler, clf, feature_names):
    """
    Apply the trained LR to each pixel's spectrum.

    Returns p(tumor) per pixel.
    """
    n_pixels = spectra_array.shape[0]
    features = np.zeros((n_pixels, len(feature_names)))

    for i in range(n_pixels):
        feats = extract_spectrum_features(diff_values, spectra_array[i], normalize=True)
        for j, fn in enumerate(feature_names):
            features[i, j] = feats.get(fn, 0)

    features_scaled = scaler.transform(features)
    proba = clf.predict_proba(features_scaled)[:, 1]
    return proba


def create_prostate_mask(b0_image, threshold_percentile=25):
    nonzero = b0_image[b0_image > 0]
    if len(nonzero) == 0:
        return np.zeros_like(b0_image, dtype=bool)
    threshold = np.percentile(nonzero, threshold_percentile)
    return b0_image > threshold


def plot_flagship_figure(b0_image, mask, pixel_coords, shape,
                          adc_values, proba_values, uncertainty_d025,
                          output_path=None):
    """
    Create the 4-panel flagship figure.
    """
    fig = plt.figure(figsize=(20, 5.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.05)

    # Reconstruct 2D maps
    adc_map = np.full(shape, np.nan)
    proba_map = np.full(shape, np.nan)
    unc_map = np.full(shape, np.nan)

    for i, (r, c) in enumerate(pixel_coords):
        adc_map[r, c] = adc_values[i]
        proba_map[r, c] = proba_values[i]
        unc_map[r, c] = uncertainty_d025[i]

    panels = [
        {"data": b0_image, "title": "(A) Anatomical Reference\n(b = 0 s/mm²)",
         "cmap": "gray", "is_overlay": False, "label": "Signal Intensity"},
        {"data": adc_map, "title": "(B) ADC Map\n(monoexponential, b ≤ 1000 s/mm²)",
         "cmap": "RdYlBu", "is_overlay": True, "label": "ADC (mm²/s)",
         "vmin": 0.3, "vmax": 2.5},
        {"data": proba_map, "title": "(C) Spectral Biomarker\n(LR tumor probability)",
         "cmap": "RdYlGn_r", "is_overlay": True, "label": "p(tumor)",
         "vmin": 0, "vmax": 1},
        {"data": unc_map, "title": "(D) Uncertainty Map\n(posterior σ at D = 0.25 mm²/s)",
         "cmap": "magma", "is_overlay": True, "label": "σ(D₀.₂₅)"},
    ]

    for idx, panel in enumerate(panels):
        ax = fig.add_subplot(gs[idx])

        if not panel["is_overlay"]:
            # Pure anatomy
            im = ax.imshow(panel["data"], cmap=panel["cmap"])
        else:
            # Show anatomy as background
            ax.imshow(b0_image, cmap="gray", alpha=0.4)
            # Overlay heatmap within mask
            masked_data = np.ma.masked_where(~mask, panel["data"])
            vmin = panel.get("vmin", np.nanpercentile(panel["data"][mask], 2))
            vmax = panel.get("vmax", np.nanpercentile(panel["data"][mask], 98))
            im = ax.imshow(masked_data, cmap=panel["cmap"], alpha=0.85,
                           vmin=vmin, vmax=vmax)

        ax.set_title(panel["title"], fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(panel["label"], fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Pixel-wise Prostate Characterization: Traditional ADC vs Bayesian Spectral Biomarker",
        fontsize=13, fontweight="bold", y=1.02
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def plot_adc_vs_biomarker_scatter(adc_values, proba_values, uncertainty_d025,
                                   output_path=None):
    """
    Scatter plot comparing ADC vs spectral biomarker, colored by uncertainty.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: ADC vs p(tumor)
    ax = axes[0]
    sc = ax.scatter(adc_values, proba_values, c=uncertainty_d025,
                     cmap="magma", s=3, alpha=0.5, rasterized=True)
    ax.set_xlabel("ADC (mm²/s)")
    ax.set_ylabel("p(tumor) from spectral LR")
    ax.set_title("ADC vs Spectral Biomarker\n(colored by uncertainty)")
    plt.colorbar(sc, ax=ax, label="σ(D₀.₂₅)")
    ax.grid(alpha=0.3)

    # Panel 2: ADC distribution
    ax = axes[1]
    ax.hist(adc_values, bins=50, color="#3498db", alpha=0.7, edgecolor="none")
    ax.set_xlabel("ADC (mm²/s)")
    ax.set_ylabel("Pixel count")
    ax.set_title("ADC Distribution")
    ax.axvline(x=1.0, color="red", linestyle="--", label="ADC=1.0 (typical threshold)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: p(tumor) distribution
    ax = axes[2]
    ax.hist(proba_values, bins=50, color="#e74c3c", alpha=0.7, edgecolor="none")
    ax.set_xlabel("p(tumor)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Spectral Biomarker Distribution")
    ax.axvline(x=0.5, color="black", linestyle="--", label="Decision boundary")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

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
    print("FLAGSHIP FIGURE: ADC vs Spectral Biomarker vs Uncertainty")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Load pixel data
    # -----------------------------------------------------------------------
    print("\n[Step 1] Loading pixel data...")
    images_256 = load_binary_images(DATA_FOLDER, shape=SHAPE, dtype=np.int16)
    images_64 = subsample_to_native(images_256, factor=NATIVE_FACTOR)
    groups, trace_images, n_bvalues = group_images_b0_plus_directions(images_64)

    b0_image = trace_images[0]
    mask = create_prostate_mask(b0_image, threshold_percentile=25)
    signal_array, pixel_coords = build_pixel_signal_array(trace_images, mask=mask)

    # b-values for pixel data (16 equally spaced as approximation)
    b_values_pixel = np.linspace(0, 3.5, n_bvalues)

    # Normalize signals
    S_0 = signal_array[:, 0].copy()
    S_0[S_0 <= 0] = 1.0
    signal_normalized = signal_array / S_0[:, None]

    n_pixels = signal_normalized.shape[0]
    print(f"  {n_pixels} prostate pixels loaded")

    # -----------------------------------------------------------------------
    # Step 2: MAP spectrum estimation
    # -----------------------------------------------------------------------
    print("\n[Step 2] MAP spectrum estimation...")
    U_pixel = create_design_matrix(b_values_pixel, DIFF_VALUES)
    t0 = time.time()
    spectra_array = map_estimate_batch(U_pixel, signal_normalized, RIDGE_STRENGTH)
    print(f"  {n_pixels} spectra in {time.time()-t0:.3f}s")

    # -----------------------------------------------------------------------
    # Step 3: ADC computation
    # -----------------------------------------------------------------------
    print("\n[Step 3] Computing ADC per pixel...")
    adc_values = compute_adc_batch(signal_normalized, b_values_pixel, b_max=ADC_B_MAX)
    print(f"  ADC range: [{adc_values.min():.3f}, {adc_values.max():.3f}] mm²/s")
    print(f"  ADC median: {np.median(adc_values):.3f} mm²/s")

    # -----------------------------------------------------------------------
    # Step 4: Train LR on ROI data
    # -----------------------------------------------------------------------
    print("\n[Step 4] Training LR classifier on ROI data...")
    scaler, clf, feature_names = train_lr_on_roi_data(
        ROI_JSON, ROI_META, DIFF_VALUES, B_VALUES_CONFIG.tolist(), RIDGE_STRENGTH
    )

    # -----------------------------------------------------------------------
    # Step 5: Apply LR to pixel spectra
    # -----------------------------------------------------------------------
    print("\n[Step 5] Computing p(tumor) per pixel...")
    proba_values = apply_lr_to_pixels(spectra_array, DIFF_VALUES, scaler, clf, feature_names)
    print(f"  p(tumor) range: [{proba_values.min():.3f}, {proba_values.max():.3f}]")
    print(f"  p(tumor) > 0.5: {np.sum(proba_values > 0.5)} pixels "
          f"({np.sum(proba_values > 0.5)/n_pixels*100:.1f}%)")

    # -----------------------------------------------------------------------
    # Step 6: Laplace uncertainty
    # -----------------------------------------------------------------------
    print("\n[Step 6] Computing Laplace uncertainty...")
    t0 = time.time()
    uncertainty = laplace_uncertainty_batch(
        U_pixel, signal_normalized, spectra_array, RIDGE_STRENGTH
    )
    uncertainty_d025 = uncertainty[:, 0]  # Uncertainty in D=0.25 component
    print(f"  Computed in {time.time()-t0:.2f}s")
    print(f"  σ(D=0.25) range: [{uncertainty_d025.min():.4f}, {uncertainty_d025.max():.4f}]")

    # -----------------------------------------------------------------------
    # Step 7: Create figures
    # -----------------------------------------------------------------------
    print("\n[Step 7] Creating figures...")

    shape = b0_image.shape

    # Main flagship figure
    plot_flagship_figure(
        b0_image, mask, pixel_coords, shape,
        adc_values, proba_values, uncertainty_d025,
        output_path=os.path.join(OUTPUT_DIR, "flagship_4panel.png"),
    )

    # Supplementary scatter plots
    plot_adc_vs_biomarker_scatter(
        adc_values, proba_values, uncertainty_d025,
        output_path=os.path.join(OUTPUT_DIR, "adc_vs_biomarker_scatter.png"),
    )

    # -----------------------------------------------------------------------
    # Step 8: Save all data
    # -----------------------------------------------------------------------
    print("\n[Step 8] Saving numerical data...")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "flagship_data.npz"),
        spectra_array=spectra_array,
        adc_values=adc_values,
        proba_values=proba_values,
        uncertainty=uncertainty,
        uncertainty_d025=uncertainty_d025,
        pixel_coords=pixel_coords,
        signal_normalized=signal_normalized,
        S_0=S_0,
        mask=mask,
        b0_image=b0_image,
        b_values_pixel=b_values_pixel,
        diff_values=DIFF_VALUES,
        lr_coef=clf.coef_[0],
        lr_intercept=clf.intercept_,
        feature_names=np.array(feature_names),
    )
    print(f"  Saved to: {os.path.join(OUTPUT_DIR, 'flagship_data.npz')}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FLAGSHIP FIGURE COMPLETE")
    print("=" * 70)
    print(f"  Pixels: {n_pixels}")
    print(f"  ADC median: {np.median(adc_values):.3f} mm²/s")
    print(f"  p(tumor) > 0.5: {np.sum(proba_values > 0.5)} pixels")
    print(f"  Output: {OUTPUT_DIR}")
