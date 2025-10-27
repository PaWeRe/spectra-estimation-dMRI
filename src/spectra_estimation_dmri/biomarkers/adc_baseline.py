"""
ADC (Apparent Diffusion Coefficient) computation as baseline comparison.

Computes zone-specific ADC values for comparison with spectrum-based biomarkers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def compute_adc(
    signal_values: np.ndarray,
    b_values: np.ndarray,
    b_range: Tuple[float, float] = (0.0, 1000.0),
) -> float:
    """
    Calculate ADC using monoexponential model.

    Signal model: S(b) = S_0 * exp(-b * ADC)

    Args:
        signal_values: Measured signal intensities
        b_values: Corresponding b-values (in s/mm²)
        b_range: (b_min, b_max) range to use for fitting

    Returns:
        adc: Apparent Diffusion Coefficient (×10^-3 mm²/s)
    """
    b_min, b_max = b_range

    # Select b-values in range
    mask = (b_values >= b_min) & (b_values <= b_max)
    valid_mask = mask & (signal_values > 0)  # Exclude non-positive signals

    if not np.any(valid_mask):
        return np.nan

    # Fit log-linear model: log(S) = log(S_0) - b * ADC
    log_signal = np.log(signal_values[valid_mask])
    b_valid = b_values[valid_mask]

    # Linear regression
    slope, intercept = np.polyfit(b_valid, log_signal, 1)
    adc = -slope  # ADC = -slope of log(S) vs b

    return adc


def compute_adc_from_signal_decay(
    signal_decay,
    b_range: Tuple[float, float] = (0.0, 1.0),  # Note: in ms units
) -> float:
    """
    Compute ADC from SignalDecay object.

    Args:
        signal_decay: SignalDecay object with signal_values and b_values
        b_range: (b_min, b_max) in ms units (will convert to s/mm² internally)

    Returns:
        adc: ADC value
    """
    # Convert to numpy
    signal_values = np.array(signal_decay.signal_values)
    b_values = np.array(signal_decay.b_values)

    # Convert b_range from ms to s/mm² (multiply by 1000)
    b_range_smm2 = (b_range[0] * 1000, b_range[1] * 1000)

    return compute_adc(signal_values, b_values, b_range_smm2)


def extract_adc_features(
    spectra_dataset,
    b_range: Tuple[float, float] = (0.0, 1.0),  # in ms units
    zone_specific: bool = True,
) -> pd.DataFrame:
    """
    Extract ADC features from all spectra in dataset.

    Args:
        spectra_dataset: DiffusivitySpectraDataset object
        b_range: (b_min, b_max) for ADC computation (in ms units)
        zone_specific: If True, compute separate ADC for PZ and TZ

    Returns:
        DataFrame with ADC values and metadata
    """
    rows = []

    for spectrum in spectra_dataset.spectra:
        signal_decay = spectrum.signal_decay

        # Compute ADC
        adc = compute_adc_from_signal_decay(signal_decay, b_range)

        # Extract metadata
        patient_id = getattr(signal_decay, "patient", None)
        region = getattr(signal_decay, "a_region", None)
        ggg = getattr(signal_decay, "ggg", None)
        gs = getattr(signal_decay, "gs", None)
        is_tumor = getattr(signal_decay, "is_tumor", None)

        # Parse zone
        if region is not None:
            region_lower = region.lower()
            if "pz" in region_lower:
                zone = "pz"
            elif "tz" in region_lower:
                zone = "tz"
            else:
                zone = "unknown"
        else:
            zone = "unknown"

        rows.append(
            {
                "patient_id": patient_id,
                "region": region,
                "zone": zone,
                "ggg": ggg,
                "gs": gs,
                "is_tumor": is_tumor,
                "adc": adc,
                "adc_b_range": f"{b_range[0]:.0f}-{b_range[1]:.0f}",
            }
        )

    df = pd.DataFrame(rows)
    return df


def create_adc_feature_vector(
    adc_df: pd.DataFrame,
    zone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create feature vector from ADC dataframe for classification.

    Args:
        adc_df: DataFrame from extract_adc_features()
        zone: If specified, filter to specific zone ('pz' or 'tz')

    Returns:
        DataFrame with 'adc' as feature column + metadata
    """
    if zone is not None:
        df_filtered = adc_df[adc_df["zone"] == zone].copy()
    else:
        df_filtered = adc_df.copy()

    # Rename for consistency with spectrum features
    df_filtered = df_filtered.rename(columns={"adc": "ADC"})

    return df_filtered
