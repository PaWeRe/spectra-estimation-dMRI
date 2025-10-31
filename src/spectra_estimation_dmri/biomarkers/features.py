"""
Feature extraction from diffusivity spectra with Monte Carlo uncertainty propagation.

Features:
- Individual diffusivity bin fractions
- Combo feature: D[0.25] + 1/D[2.0] + 1/D[3.0]
- MC-based feature distributions for uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def extract_spectrum_features(
    diffusivities: np.ndarray,
    spectrum_vector: np.ndarray,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Extract features from a single spectrum (point estimate or single sample).

    Args:
        diffusivities: Array of diffusivity values
        spectrum_vector: Spectrum values (same length as diffusivities)
        normalize: Whether to normalize spectrum to sum to 1

    Returns:
        Dictionary of features
    """
    # Normalize if requested
    if normalize:
        spectrum_norm = spectrum_vector / (np.sum(spectrum_vector) + 1e-10)
    else:
        spectrum_norm = spectrum_vector

    # Create feature dictionary with diffusivity values as keys
    features = {}

    # Individual diffusivity features
    for i, diff in enumerate(diffusivities):
        features[f"D_{diff:.2f}"] = spectrum_norm[i]

    # Combo feature: D[0.25] + 1/D[2.0] + 1/D[3.0]
    # Find indices for required diffusivities
    idx_025 = np.argmin(np.abs(diffusivities - 0.25))
    idx_200 = np.argmin(np.abs(diffusivities - 2.0))
    idx_300 = np.argmin(np.abs(diffusivities - 3.0))

    d_025 = spectrum_norm[idx_025]
    d_200 = spectrum_norm[idx_200]
    d_300 = spectrum_norm[idx_300]

    # Compute combo feature
    combo = d_025 + 1.0 / (d_200 + 1e-10) + 1.0 / (d_300 + 1e-10)
    features["D[0.25]+1/D[2.0]+1/D[3.0]"] = combo

    return features


def extract_mc_features(
    diffusivities: np.ndarray,
    spectrum_samples: np.ndarray,
    n_mc_samples: int = 200,
    random_seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
    """
    Extract features with Monte Carlo uncertainty propagation.

    Args:
        diffusivities: Array of diffusivity values
        spectrum_samples: Posterior samples, shape (n_samples, n_diffusivities)
        n_mc_samples: Number of MC samples to use (randomly subsample if more available)
        random_seed: Random seed for reproducible subsampling

    Returns:
        (mean_features, std_features, feature_samples):
            - mean_features: Mean feature values across MC samples
            - std_features: Std of feature values (uncertainty)
            - feature_samples: List of feature dicts (one per MC sample)
    """
    n_available = spectrum_samples.shape[0]

    # Subsample if we have more than needed
    if n_available > n_mc_samples:
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(n_available, n_mc_samples, replace=False)
        samples = spectrum_samples[indices]
    else:
        samples = spectrum_samples

    # Extract features for each MC sample
    feature_samples = []
    for sample in samples:
        features = extract_spectrum_features(diffusivities, sample, normalize=True)
        feature_samples.append(features)

    # Compute mean and std across samples
    feature_keys = feature_samples[0].keys()
    mean_features = {}
    std_features = {}

    for key in feature_keys:
        values = [fs[key] for fs in feature_samples]
        mean_features[key] = np.mean(values)
        std_features[key] = np.std(values)

    return mean_features, std_features, feature_samples


def extract_features_from_dataset(
    spectra_dataset,
    n_mc_samples: int = 200,
    include_metadata: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features from all spectra in dataset.

    Args:
        spectra_dataset: DiffusivitySpectraDataset object
        n_mc_samples: Number of MC samples per spectrum
        include_metadata: Whether to include patient/region metadata
        random_seed: Random seed for reproducible MC sampling

    Returns:
        (features_df, uncertainty_df):
            - features_df: Mean features + metadata
            - uncertainty_df: Feature uncertainties (std)
    """
    feature_rows = []
    uncertainty_rows = []

    for i, spectrum in enumerate(spectra_dataset.spectra):
        arrays = spectrum.as_numpy()
        diffusivities = arrays["diffusivities"]
        samples = arrays["spectrum_samples"]

        if samples is None:
            # Fallback to point estimate if no samples available
            spectrum_vec = arrays["spectrum_vector"]
            features = extract_spectrum_features(diffusivities, spectrum_vec)
            uncertainties = {k: 0.0 for k in features.keys()}
        else:
            # MC feature extraction with unique seed per spectrum for reproducibility
            features, uncertainties, _ = extract_mc_features(
                diffusivities, samples, n_mc_samples, random_seed=random_seed + i
            )

        # Add metadata if requested
        if include_metadata:
            patient_id = getattr(spectrum.signal_decay, "patient", None)
            region = getattr(spectrum.signal_decay, "a_region", None)
            is_tumor = getattr(spectrum.signal_decay, "is_tumor", False)

            # Create unique ROI identifier (patient_region_tumor_status)
            # CRITICAL: Include tumor status to distinguish tumor_pz from normal_pz!
            if patient_id and region:
                roi_id = f"{patient_id}_{region}_{'tumor' if is_tumor else 'normal'}"
            else:
                roi_id = None

            features["roi_id"] = roi_id
            features["patient_id"] = patient_id
            features["region"] = region
            features["ggg"] = getattr(spectrum.signal_decay, "ggg", None)
            features["gs"] = getattr(spectrum.signal_decay, "gs", None)
            features["is_tumor"] = is_tumor

            uncertainties["roi_id"] = roi_id
            uncertainties["patient_id"] = patient_id
            uncertainties["region"] = region

        feature_rows.append(features)
        uncertainty_rows.append(uncertainties)

    features_df = pd.DataFrame(feature_rows)
    uncertainty_df = pd.DataFrame(uncertainty_rows)

    return features_df, uncertainty_df


def get_feature_names(
    diffusivities: List[float], include_combo: bool = True
) -> List[str]:
    """
    Get list of feature names for a given diffusivity grid.

    Args:
        diffusivities: List of diffusivity values
        include_combo: Whether to include combo feature

    Returns:
        List of feature column names
    """
    feature_names = [f"D_{d:.2f}" for d in diffusivities]

    if include_combo:
        feature_names.append("D[0.25]+1/D[2.0]+1/D[3.0]")

    return feature_names
