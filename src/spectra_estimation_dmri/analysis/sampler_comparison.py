"""
Sampler comparison metrics and analysis tools.

This module provides functions to extract and log key metrics for comparing
Gibbs and NUTS samplers across different configurations.
"""

import os
import csv
import time
import numpy as np
import arviz as az
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class SamplerComparisonMetrics:
    """
    Container for sampler comparison metrics.

    Attributes:
        # Identifiers
        spectra_id: str
        inference_method: str
        spectrum_pair: str
        snr: float
        prior_type: str
        prior_strength: float
        n_chains: int
        n_iterations: int

        # Convergence metrics
        max_rhat: float
        min_ess_bulk: float
        min_ess_tail: float
        mean_ess_bulk: float
        mean_ess_tail: float
        convergence_status: str  # "converged", "marginal", "not_converged"

        # Accuracy metrics
        reconstruction_error_l2: float
        reconstruction_error_l1: float
        reconstruction_error_max: float

        # Uncertainty calibration metrics
        mean_interval_width: float
        median_interval_width: float
        interval_sharpness: float  # Ratio of width to reconstruction error

        # Efficiency metrics
        sampling_time_seconds: Optional[float]
        samples_per_second: Optional[float]
        ess_per_second: Optional[float]  # ESS_bulk / time

        # Additional info
        n_diffusivities: int
        condition_number: float
    """

    # Identifiers
    spectra_id: str
    inference_method: str
    spectrum_pair: str
    snr: float
    prior_type: str
    prior_strength: float
    n_chains: int
    n_iterations: int

    # Convergence
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    mean_ess_bulk: float
    mean_ess_tail: float
    convergence_status: str

    # Accuracy
    reconstruction_error_l2: float
    reconstruction_error_l1: float
    reconstruction_error_max: float

    # Uncertainty
    mean_interval_width: float
    median_interval_width: float
    interval_sharpness: float

    # Efficiency
    sampling_time_seconds: Optional[float] = None
    samples_per_second: Optional[float] = None
    ess_per_second: Optional[float] = None

    # Additional
    n_diffusivities: int = 0
    condition_number: float = 0.0


def extract_metrics_from_spectrum(
    spectrum, exp_config=None, sampling_time: Optional[float] = None
) -> SamplerComparisonMetrics:
    """
    Extract comparison metrics from a DiffusivitySpectrum object.

    Args:
        spectrum: DiffusivitySpectrum object
        exp_config: Experiment configuration (for additional context)
        sampling_time: Optional sampling time in seconds

    Returns:
        SamplerComparisonMetrics object
    """
    # Extract basic identifiers
    spectra_id = spectrum.spectra_id or "unknown"
    inference_method = spectrum.inference_method
    prior_type = getattr(spectrum, "prior_type", "unknown")
    prior_strength = getattr(spectrum, "prior_strength", 0.0)
    snr = getattr(spectrum, "data_snr", 0.0)

    # Get spectrum pair from config
    spectrum_pair = "unknown"
    if exp_config and hasattr(exp_config, "dataset"):
        if hasattr(exp_config.dataset, "spectrum_pair"):
            spectrum_pair = exp_config.dataset.spectrum_pair
        elif hasattr(exp_config.dataset, "name"):
            spectrum_pair = exp_config.dataset.name

    # Get number of diffusivities
    n_diffusivities = len(spectrum.diffusivities) if spectrum.diffusivities else 0

    # Compute condition number
    U = (
        np.array(spectrum.design_matrix_U)
        if spectrum.design_matrix_U
        else np.array([[]])
    )
    condition_number = np.linalg.cond(U) if U.size > 0 else 0.0

    # Initialize convergence metrics
    max_rhat = np.nan
    min_ess_bulk = np.nan
    min_ess_tail = np.nan
    mean_ess_bulk = np.nan
    mean_ess_tail = np.nan
    convergence_status = "unknown"
    n_chains = 1
    n_iterations = 0

    # Extract convergence metrics from inference data if available
    if spectrum.inference_data and os.path.exists(spectrum.inference_data):
        try:
            idata = az.from_netcdf(spectrum.inference_data)

            # Get chain info
            if hasattr(idata, "posterior"):
                n_chains = idata.posterior.sizes.get("chain", 1)
                n_iterations = idata.posterior.sizes.get("draw", 0)

            # Only compute R-hat and ESS for multi-chain runs
            if n_chains > 1:
                # Compute R-hat
                rhat = az.rhat(idata)
                rhat_values = []
                for var_name in rhat.data_vars:
                    rhat_values.append(float(rhat[var_name].values))
                if rhat_values:
                    max_rhat = np.max(rhat_values)

                # Compute ESS
                ess = az.ess(idata)
                ess_bulk_values = []
                ess_tail_values = []
                for var_name in ess.data_vars:
                    ess_bulk_values.append(float(ess[var_name].values))

                # Also get tail ESS
                ess_tail = az.ess(idata, method="tail")
                for var_name in ess_tail.data_vars:
                    ess_tail_values.append(float(ess_tail[var_name].values))

                if ess_bulk_values:
                    min_ess_bulk = np.min(ess_bulk_values)
                    mean_ess_bulk = np.mean(ess_bulk_values)
                if ess_tail_values:
                    min_ess_tail = np.min(ess_tail_values)
                    mean_ess_tail = np.mean(ess_tail_values)

                # Determine convergence status
                if not np.isnan(max_rhat) and not np.isnan(min_ess_bulk):
                    if max_rhat < 1.01 and min_ess_bulk > 400:
                        convergence_status = "converged"
                    elif max_rhat < 1.05 and min_ess_bulk > 100:
                        convergence_status = "marginal"
                    else:
                        convergence_status = "not_converged"
        except Exception as e:
            print(f"[WARNING] Could not extract convergence metrics: {e}")

    # Compute accuracy metrics
    reconstruction_error_l2 = np.nan
    reconstruction_error_l1 = np.nan
    reconstruction_error_max = np.nan

    if spectrum.spectrum_vector and spectrum.true_spectrum:
        estimated = np.array(spectrum.spectrum_vector)
        true = np.array(spectrum.true_spectrum)

        if len(estimated) == len(true):
            diff = estimated - true
            reconstruction_error_l2 = np.sqrt(np.mean(diff**2))  # RMSE
            reconstruction_error_l1 = np.mean(np.abs(diff))  # MAE
            reconstruction_error_max = np.max(np.abs(diff))  # Max abs error

    # Compute uncertainty metrics
    mean_interval_width = np.nan
    median_interval_width = np.nan
    interval_sharpness = np.nan

    if spectrum.spectrum_std:
        std_values = np.array(spectrum.spectrum_std)
        # 95% credible interval width ≈ 3.92 * std (for normal distribution)
        interval_widths = 3.92 * std_values
        mean_interval_width = np.mean(interval_widths)
        median_interval_width = np.median(interval_widths)

        # Sharpness: how tight are intervals relative to accuracy?
        if not np.isnan(reconstruction_error_l2) and reconstruction_error_l2 > 0:
            interval_sharpness = mean_interval_width / reconstruction_error_l2

    # Compute efficiency metrics
    samples_per_second = None
    ess_per_second = None
    if sampling_time and sampling_time > 0:
        total_samples = n_chains * n_iterations
        samples_per_second = total_samples / sampling_time
        if not np.isnan(mean_ess_bulk):
            ess_per_second = mean_ess_bulk / sampling_time

    # Create metrics object
    metrics = SamplerComparisonMetrics(
        spectra_id=spectra_id,
        inference_method=inference_method,
        spectrum_pair=spectrum_pair,
        snr=snr,
        prior_type=prior_type,
        prior_strength=prior_strength,
        n_chains=n_chains,
        n_iterations=n_iterations,
        max_rhat=max_rhat,
        min_ess_bulk=min_ess_bulk,
        min_ess_tail=min_ess_tail,
        mean_ess_bulk=mean_ess_bulk,
        mean_ess_tail=mean_ess_tail,
        convergence_status=convergence_status,
        reconstruction_error_l2=reconstruction_error_l2,
        reconstruction_error_l1=reconstruction_error_l1,
        reconstruction_error_max=reconstruction_error_max,
        mean_interval_width=mean_interval_width,
        median_interval_width=median_interval_width,
        interval_sharpness=interval_sharpness,
        sampling_time_seconds=sampling_time,
        samples_per_second=samples_per_second,
        ess_per_second=ess_per_second,
        n_diffusivities=n_diffusivities,
        condition_number=condition_number,
    )

    return metrics


def log_metrics_to_wandb(
    metrics: SamplerComparisonMetrics, prefix: str = "sampler_comparison"
):
    """
    Log metrics to Weights & Biases.

    Args:
        metrics: SamplerComparisonMetrics object
        prefix: Prefix for metric names (default: "sampler_comparison")
    """
    try:
        import wandb

        # Convert to dict and log
        metrics_dict = asdict(metrics)
        wandb_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
        wandb.log(wandb_dict)

    except ImportError:
        print("[WARNING] wandb not available, skipping logging")
    except Exception as e:
        print(f"[WARNING] Failed to log metrics to wandb: {e}")


def save_metrics_to_csv(
    metrics: SamplerComparisonMetrics, csv_path: str, append: bool = True
):
    """
    Save metrics to CSV file.

    Args:
        metrics: SamplerComparisonMetrics object
        csv_path: Path to CSV file
        append: If True, append to existing file; if False, overwrite
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Convert to dict
    metrics_dict = asdict(metrics)

    # Check if file exists
    file_exists = os.path.exists(csv_path)
    mode = "a" if (append and file_exists) else "w"

    # Write to CSV
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())

        # Write header if new file or overwriting
        if not file_exists or not append:
            writer.writeheader()

        # Write data
        writer.writerow(metrics_dict)

    print(f"[INFO] Saved metrics to: {csv_path}")


def print_metrics_summary(metrics: SamplerComparisonMetrics):
    """
    Print a formatted summary of the metrics.

    Args:
        metrics: SamplerComparisonMetrics object
    """
    print("\n" + "=" * 60)
    print("SAMPLER COMPARISON METRICS")
    print("=" * 60)
    print(f"Spectrum ID: {metrics.spectra_id}")
    print(f"Inference: {metrics.inference_method} | Spectrum: {metrics.spectrum_pair}")
    print(
        f"SNR: {metrics.snr} | Prior: {metrics.prior_type} ({metrics.prior_strength})"
    )
    print(f"Chains: {metrics.n_chains} | Iterations: {metrics.n_iterations}")
    print(
        f"Diffusivities: {metrics.n_diffusivities} | κ: {metrics.condition_number:.2e}"
    )

    print("\n--- Convergence ---")
    print(f"Status: {metrics.convergence_status}")
    print(f"Max R-hat: {metrics.max_rhat:.4f}")
    print(f"Min ESS (bulk): {metrics.min_ess_bulk:.0f}")
    print(f"Min ESS (tail): {metrics.min_ess_tail:.0f}")

    print("\n--- Accuracy ---")
    print(f"L2 Error (RMSE): {metrics.reconstruction_error_l2:.6f}")
    print(f"L1 Error (MAE): {metrics.reconstruction_error_l1:.6f}")
    print(f"Max Error: {metrics.reconstruction_error_max:.6f}")

    print("\n--- Uncertainty ---")
    print(f"Mean Interval Width: {metrics.mean_interval_width:.6f}")
    print(f"Median Interval Width: {metrics.median_interval_width:.6f}")
    print(f"Interval Sharpness: {metrics.interval_sharpness:.2f}")

    if metrics.sampling_time_seconds:
        print("\n--- Efficiency ---")
        print(f"Sampling Time: {metrics.sampling_time_seconds:.2f}s")
        print(f"Samples/sec: {metrics.samples_per_second:.0f}")
        if metrics.ess_per_second:
            print(f"ESS/sec: {metrics.ess_per_second:.0f}")

    print("=" * 60 + "\n")
