# TODO: Implement gibbs_snr = min(data_snr, some_threshold=600),
# TODO: Why MCMC and not regularization (uncertainty estimates...) ... understand Sandy's implementation fully!
# TODO: Mean > Mode for disease prediction (metric: closeness to true spectrum and stability)
# TODO: Is there a problem in the implementation because it appears gSNR influences mode calcualtion?
# TODO: Consider multiple noise realization (according to AI indeed a different kind of uncertainty)
# TODO: Create more pltos to check convergence (see proposals from AI, e.g. correlation of samples!!!!)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
import os
import yaml
import json
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy.random as npr
from dataclasses import dataclass
from typing import List, Dict, Any
import matplotlib.patches as patches
from spectra_estimation_dmri.models.samplers.gibbs_sampler import GibbsSampler
from spectra_estimation_dmri.models.samplers.base import (
    d_spectrum,
    d_spectra_sample,
    signal_data,
)
from spectra_estimation_dmri.utils import plotting
import importlib.resources

sys.path.append(os.path.join(os.getcwd() + "/src/models/"))


@dataclass
class SimConfig:
    """Configuration class for simulation parameters"""

    name: str
    data_snr: float  # SNR for generating the noisy data
    gibbs_snr: float  # SNR used in Gibbs sampling
    b_values: np.ndarray
    iterations: int
    l2_lambda: float
    burn_in: int  # Number of initial samples to discard
    n_runs: int  # Number of noise realizations

    @classmethod
    def create_configs(cls) -> List["SimConfig"]:
        """Create a list of simulation configurations"""

        # Helper function to create non-uniform b-value sampling
        def create_nonuniform_bvals(
            max_b: float, n_points: int, focus_range: tuple
        ) -> np.ndarray:
            """Create non-uniform b-value sampling with more points in focus_range"""
            # Split points between focus range and rest
            n_focus = int(0.7 * n_points)  # 70% of points in focus range
            n_rest = n_points - n_focus

            # Create focused sampling in specified range
            focus_points = np.linspace(focus_range[0], focus_range[1], n_focus)

            # Create sparse sampling for rest of range
            rest_points = np.concatenate(
                [
                    np.linspace(0, focus_range[0], n_rest // 2, endpoint=False),
                    np.linspace(focus_range[1], max_b, n_rest // 2 + n_rest % 2),
                ]
            )

            return np.sort(np.concatenate([focus_points, rest_points])) / 1000

        configs = []

        # Base parameters
        base_iterations = 100000
        base_l2_lambda = 1e-5
        base_burn_in = 10000
        base_n_runs = 1  # Default number of noise realizations

        # 1. Ratiometric SNR variations
        uniform_bvals_3500 = np.linspace(0, 3500, 15) / 1000
        configs.extend(
            [
                # cls(
                #     "Match",
                #     100,
                #     100,
                #     uniform_bvals_3500,
                #     base_iterations,
                #     base_l2_lambda,
                #     base_burn_in,
                #     base_n_runs,
                # ),
                # cls(
                #     "Match",
                #     300,
                #     300,
                #     uniform_bvals_3500,
                #     base_iterations,
                #     base_l2_lambda,
                #     base_burn_in,
                #     base_n_runs,
                # ),
                cls(
                    "Match",
                    600,
                    600,
                    uniform_bvals_3500,
                    base_iterations,
                    base_l2_lambda,
                    base_burn_in,
                    base_n_runs,
                ),
                # cls(
                #     "Match",
                #     1200,
                #     1200,
                #     uniform_bvals_3500,
                #     base_iterations,
                #     base_l2_lambda,
                #     base_burn_in,
                #     base_n_runs,
                # ),
            ]
        )

        return configs


def get_averaged_spectra(gibbs_json_path, gibbs_hdf5_path):
    """Fetch and compute averaged spectra for each zone."""
    avg_dict = {}

    # Use importlib.resources for JSON and HDF5 files in package data
    with importlib.resources.files("spectra_estimation_dmri.data").joinpath(
        gibbs_json_path
    ).open("r") as f:
        rois = json.load(f)

    with importlib.resources.files("spectra_estimation_dmri.data").joinpath(
        gibbs_hdf5_path
    ).open("rb") as h5file:
        with h5py.File(h5file, "r") as hf:
            for zone_key, zone_list in rois.items():
                if len(zone_list) == 0 or zone_key == "Neglected!":
                    continue

                diffusivities = np.array(zone_list[0]["diffusivities"])
                avg_samples = np.mean(
                    [hf[d["hdf5_dataset"]][()] for d in zone_list], axis=0
                )
                avg_fractions = np.mean(avg_samples, axis=0)
                avg_dict[zone_key] = d_spectrum(avg_fractions, diffusivities)

    return avg_dict


def predict_signals(s_0, b_values, diffusivities, fractions):
    signals = 0
    for i in range(len(diffusivities)):
        signals += s_0 * fractions[i] * np.exp(-b_values * diffusivities[i])
    return signals


# def generate_signals(s_0, b_values, d_spectrum, sigma):
#     predictions = predict_signals(
#         s_0, b_values, d_spectrum.diffusivities, d_spectrum.fractions
#     )
#     if sigma > 0.0:
#         predictions += npr.normal(0, sigma, len(predictions))
#     return gs.signal_data(predictions, b_values)


def generate_signals(s_0, b_values, d_spectrum, sigma, plot=False):
    """Generate diffusion signals with optional plotting.

    Args:
        s_0: Initial signal intensity
        b_values: Array of b-values
        d_spectrum: Diffusion spectrum object containing diffusivities and fractions
        sigma: Noise standard deviation
        plot: Boolean flag to control plotting (default: False)

    Returns:
        gs.signal_data object containing the generated signals
    """
    predictions = predict_signals(
        s_0, b_values, d_spectrum.diffusivities, d_spectrum.fractions
    )

    if sigma > 0.0:
        noise = npr.normal(0, sigma, len(predictions))
        noisy_predictions = predictions + noise
    else:
        noisy_predictions = predictions
        noise = np.zeros_like(predictions)

    if plot:
        plotting.plot_signal_data(
            signal_data(noisy_predictions, b_values),
            title="Generated Diffusion Signals",
        )

    return signal_data(noisy_predictions, b_values)


def run_simulation(
    true_spectrum: d_spectrum, config: SimConfig, sampler_class=GibbsSampler
) -> Dict[str, Any]:
    """Run multiple simulations with given parameters to assess stability.

    Args:
        true_spectrum: The true diffusion spectrum
        config: Configuration parameters containing n_runs for noise realizations
        sampler_class: The sampler class to use (must implement BaseSampler interface)
    """
    # Lists to store results across runs
    all_modes = []  # Store initial R (mode) from each run
    all_means = []  # Store mean from each run's Gibbs samples

    # Run simulation multiple times
    for _ in range(config.n_runs):
        # Generate noisy data using data_snr
        noisy_signal_normalized = generate_signals(
            1.0, config.b_values, true_spectrum, 1 / config.data_snr
        )

        # Use the pluggable sampler interface
        sampler = sampler_class(
            noisy_signal_normalized,
            true_spectrum.diffusivities,
            1 / config.gibbs_snr,  # Use gibbs_snr for sampling
            l2_lambda=config.l2_lambda,
        )
        sample = sampler.sample(config.iterations)
        sample.sample = sample.sample[config.burn_in :]  # Discard burn-in

        # Store initial R (mode) and mean of Gibbs samples
        all_modes.append(sample.initial_R)
        all_means.append(np.mean(sample.sample, axis=0))

    # Convert to numpy arrays for easier computation
    all_modes = np.array(all_modes)
    all_means = np.array(all_means)

    # Calculate variances across runs for each diffusivity component
    mode_variances = np.var(all_modes, axis=0)
    mean_variances = np.var(all_means, axis=0)

    # Store the last sample for other plots
    return {
        "sample": sample,
        "config": config,
        "mode_variances": mode_variances,
        "mean_variances": mean_variances,
        "all_modes": all_modes,
        "all_means": all_means,
    }


def plot_true_spectrum(ax, true_spectrum, title):
    """Plot the true spectrum on the given axis"""
    plotting.plot_d_spectrum(true_spectrum, title=title)


def generate_true_spectrum(diffusivities, complexity_level: int = 1) -> d_spectrum:
    """Generate a true spectrum with varying complexity levels.

    Args:
        diffusivities: Array of diffusion coefficients
        complexity_level: Integer from 1-5 indicating spectrum complexity:
            1: Two peaks (simple bimodal)
            2: Three peaks (trimodal)
            3: Four peaks with varying heights
            4: Multiple peaks with overlap
            5: Complex distribution similar to real data

    Returns:
        gs.d_spectrum object containing the generated spectrum
    """
    fractions = np.zeros_like(diffusivities)
    n_components = len(diffusivities)

    if complexity_level == 0:
        # Simple bimodal: Two clear peaks
        fractions[5] = 0.6  # Strong peak at low diffusivity

    elif complexity_level == 1:
        # Simple bimodal: Two clear peaks
        fractions[1] = 0.6  # Strong peak at low diffusivity
        fractions[-2] = 0.4  # Weaker peak at high diffusivity

    elif complexity_level == 2:
        # Trimodal: Three distinct peaks
        fractions[1] = 0.4  # Low diffusivity
        fractions[4] = 0.35  # Medium diffusivity
        fractions[-2] = 0.25  # High diffusivity

    elif complexity_level == 3:
        # Four peaks with varying heights
        fractions[1] = 0.35  # Low diffusivity
        fractions[3] = 0.25  # Medium-low diffusivity
        fractions[6] = 0.15  # Medium-high diffusivity
        fractions[-2] = 0.25  # High diffusivity

    elif complexity_level == 4:
        # Multiple peaks with overlap
        fractions[0] = 0.15  # Very low diffusivity
        fractions[1] = 0.25  # Low diffusivity
        fractions[2] = 0.15  # Low-medium diffusivity
        fractions[5] = 0.20  # Medium diffusivity
        fractions[-3] = 0.15  # High diffusivity
        fractions[-1] = 0.10  # Very high diffusivity

    elif complexity_level == 5:
        # Complex distribution similar to real data
        # Multiple peaks with varying widths and overlaps
        fractions[0] = 0.10  # Very low diffusivity
        fractions[1] = 0.20  # Low diffusivity
        fractions[2] = 0.15  # Low-medium diffusivity
        fractions[3] = 0.10  # Medium-low diffusivity
        fractions[5] = 0.15  # Medium diffusivity
        fractions[7] = 0.10  # Medium-high diffusivity
        fractions[-3] = 0.10  # High diffusivity
        fractions[-2] = 0.05  # Very high diffusivity
        fractions[-1] = 0.05  # Highest diffusivity

    else:
        raise ValueError("Complexity level must be between 1 and 5")

    # Ensure fractions sum to 1
    fractions /= np.sum(fractions)

    return d_spectrum(fractions, diffusivities)


def plot_results(
    true_spectra: Dict[str, d_spectrum],
    all_results: List[Dict[str, Any]],
    output_pdf_path: str,
    output_csv_path: str,
):
    """Plot results with one zone per page, true spectrum always top-left"""

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    n_configs = len(all_results) // len(true_spectra)
    rows = int(np.ceil((n_configs) / 2))
    cols = 2

    with PdfPages(output_pdf_path) as pdf:
        for zone_name, true_spectrum in true_spectra.items():
            fig = plt.figure(figsize=(15, 5 * rows))
            plt.suptitle(f"{zone_name}", fontsize=16)

            gs = plt.GridSpec(rows, cols, figure=fig)

            zone_results = [r for r in all_results if r["zone"] == zone_name]
            for i, result in enumerate(zone_results):
                row = (i + 0) // cols
                col = (i + 0) % cols
                ax = fig.add_subplot(gs[row, col])

                config = result["config"]
                # Calculate b-value range
                b_min, b_max = min(config.b_values), max(config.b_values)

                # Plot Gibbs sampling distribution
                plotting.plot_d_spectra_sample_boxplot(
                    result["sample"],
                    ax=ax,
                    title=f"{config.name}\n"
                    f"Data SNR: {config.data_snr}, Gibbs SNR: {config.gibbs_snr}\n"
                    f"L2 λ: {config.l2_lambda:.1e}, Iterations: {config.iterations} (burn-in: {config.burn_in})\n"
                    f"b-values: {b_min:.1f}-{b_max:.1f} ms/µm²",
                )

                # Use the correct x positions for the true spectrum
                x_positions = np.arange(1, len(true_spectrum.diffusivities) + 1)

                # Overlay true spectrum at the same x positions
                ax.vlines(
                    x_positions,
                    0,
                    true_spectrum.fractions,
                    colors="b",
                    alpha=0.8,
                    linewidth=2,
                    label="True Spectrum",
                )

                # Add legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)

                str_diffs = [""] + list(map(str, true_spectrum.diffusivities)) + [""]
                ax.set_xticks(
                    np.arange(len(str_diffs)), labels=str_diffs, fontsize=8, rotation=90
                )

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()


def modified_plot_results(true_spectra, all_results, output_pdf_path, output_csv_path):
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    n_configs = len(all_results)
    rows = int(np.ceil((n_configs) / 2))
    cols = 2

    with PdfPages(output_pdf_path) as pdf:
        # First page: Distribution plots
        fig = plt.figure(figsize=(15, 5 * rows))
        plt.suptitle("Complexity - Distribution", fontsize=16)

        gs = plt.GridSpec(rows, cols, figure=fig)

        for i, result in enumerate(all_results):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            config = result["config"]
            # Calculate b-value range
            b_min, b_max = min(config.b_values), max(config.b_values)

            # Plot Gibbs sampling distribution
            plotting.plot_d_spectra_sample_boxplot(
                result["sample"],
                ax=ax,
                title=f"{config.name}\n"
                f"Data SNR: {config.data_snr}, Gibbs SNR: {config.gibbs_snr}\n"
                f"L2 λ: {config.l2_lambda:.1e}, Iterations: {config.iterations} (burn-in: {config.burn_in})\n"
                f"b-values: {b_min:.1f}-{b_max:.1f} ms/µm²",
            )

            # Get the appropriate true spectrum
            if "true_spectrum" in result:
                # For test data
                true_spectrum = result["true_spectrum"]
            else:
                # For real data
                true_spectrum = true_spectra[result["zone"]]

            x_positions = np.arange(1, len(true_spectrum.diffusivities) + 1)

            # Overlay true spectrum
            ax.vlines(
                x_positions,
                0,
                true_spectrum.fractions,
                colors="b",
                alpha=0.8,
                linewidth=2,
                label="True Spectrum",
            )

            # Add legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

            str_diffs = [""] + list(map(str, true_spectrum.diffusivities)) + [""]
            ax.set_xticks(
                np.arange(len(str_diffs)), labels=str_diffs, fontsize=8, rotation=90
            )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Second page: Autocorrelation plots
        fig = plt.figure(figsize=(15, 5 * rows))
        plt.suptitle("Complexity - Autocorrelation", fontsize=16)

        gs = plt.GridSpec(rows, cols, figure=fig)

        for i, result in enumerate(all_results):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            config = result["config"]

            # Plot autocorrelation
            plotting.plot_d_spectra_sample_autocorrelation(result["sample"], ax=ax)

            # Add zone information to title for real data
            title = f"{config.name}\n"
            if "zone" in result:
                title += f"Zone: {result['zone']}\n"
            title += f"Data SNR: {config.data_snr}, Gibbs SNR: {config.gibbs_snr}"

            ax.set_title(title)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Third page: Mode vs Mean Stability Analysis with Boxplots
        fig = plt.figure(figsize=(15, 5 * rows))
        plt.suptitle("Mode vs Mean Stability Analysis", fontsize=16)

        gs = plt.GridSpec(rows, cols, figure=fig)

        for i, result in enumerate(all_results):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            config = result["config"]

            # Get the appropriate true spectrum
            if "true_spectrum" in result:
                true_spectrum = result["true_spectrum"]
            else:
                true_spectrum = true_spectra[result["zone"]]

            # Prepare data for boxplots
            n_diffusivities = len(true_spectrum.diffusivities)
            positions_mode = np.arange(1, n_diffusivities + 1) - 0.2
            positions_mean = np.arange(1, n_diffusivities + 1) + 0.2

            # Create boxplots
            bp_mode = ax.boxplot(
                result["all_modes"],
                positions=positions_mode,
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="blue"),
                whiskerprops=dict(color="blue"),
                capprops=dict(color="blue"),
                flierprops=dict(color="blue", markeredgecolor="blue"),
            )

            bp_mean = ax.boxplot(
                result["all_means"],
                positions=positions_mean,
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="lightgreen", color="green"),
                medianprops=dict(color="green"),
                whiskerprops=dict(color="green"),
                capprops=dict(color="green"),
                flierprops=dict(color="green", markeredgecolor="green"),
            )

            # Overlay true spectrum
            x_positions = np.arange(1, len(true_spectrum.diffusivities) + 1)
            true_spec_lines = ax.vlines(
                x_positions,
                0,
                true_spectrum.fractions,
                colors="red",
                alpha=0.8,
                linewidth=2,
                label="True Spectrum",
            )

            # Customize plot
            ax.set_ylabel("Relative Fraction")
            title = f"{config.name}"
            if "zone" in result:
                title += f"\nZone: {result['zone']}\nData SNR: {config.data_snr}, Gibbs SNR: {config.gibbs_snr}"
            title += f"\nN={config.n_runs} noise realizations"
            ax.set_title(title)

            # Set x-ticks to show diffusivity values
            ax.set_xticks(x_positions)
            ax.set_xticklabels(true_spectrum.diffusivities, rotation=90)

            # Add legend with proper handles
            ax.legend(
                [bp_mode["boxes"][0], bp_mean["boxes"][0], true_spec_lines],
                ["Initial R (Mode)", "Final R (Mean)", "True Spectrum"],
                loc="upper right",
            )

            # Add grid for better readability
            ax.grid(True, alpha=0.3)

            # Set y-axis limits with some padding
            max_val = max(
                np.max(result["all_modes"]),
                np.max(result["all_means"]),
                np.max(true_spectrum.fractions),
            )
            ax.set_ylim(0, max_val * 1.1)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Fourth page: Trace Plots
        fig = plt.figure(figsize=(15, 5 * rows))
        plt.suptitle("MCMC Trace Plots", fontsize=16)

        gs = plt.GridSpec(rows, cols, figure=fig)

        for i, result in enumerate(all_results):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            config = result["config"]

            # Get sample data
            sample_array = np.array(result["sample"].sample)
            n_iterations = len(sample_array)

            # Plot trace for each diffusivity component
            for j, diff in enumerate(result["sample"].diffusivities):
                ax.plot(
                    np.arange(n_iterations),
                    sample_array[:, j],
                    label=f"D={diff}",
                    alpha=0.7,
                    linewidth=1,
                )

            # Customize plot
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fraction")
            title = f"{config.name}"
            if "zone" in result:
                title += f"\nZone: {result['zone']}"
            title += f"\nData SNR: {config.data_snr}, Gibbs SNR: {config.gibbs_snr}"
            ax.set_title(title)

            # Add legend outside the plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

            # Add grid for better readability
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Fifth page: Gibbs Sampler Trajectory Plots (Mixing Visualization)
        fig = plt.figure(figsize=(15, 5 * rows))
        plt.suptitle("Gibbs Sampler Trajectories (Mixing Visualization)", fontsize=16)

        gs = plt.GridSpec(rows, cols, figure=fig)

        for i, result in enumerate(all_results):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            config = result["config"]
            sample_array = np.array(result["sample"].sample)
            n_steps = min(
                50, len(sample_array)
            )  # Show first 50 steps or less if not enough

            # Select two components with highest variance (where mixing is worst)
            variances = np.var(sample_array, axis=0)
            idx_sorted = np.argsort(variances)[::-1]
            x_idx, y_idx = idx_sorted[0], idx_sorted[1]
            x = sample_array[:n_steps, x_idx]
            y = sample_array[:n_steps, y_idx]

            # Plot trajectory with arrows
            ax.plot(x, y, marker="o", linestyle="-", color="k", alpha=0.7)
            for j in range(n_steps - 1):
                ax.annotate(
                    "",
                    xy=(x[j + 1], y[j + 1]),
                    xytext=(x[j], y[j]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1, alpha=0.7),
                    size=8,
                )

            # Optionally, plot the mean as a red star
            ax.plot(np.mean(x), np.mean(y), "r*", markersize=12, label="Mean")

            # Overlay 1σ and 2σ covariance ellipses (contours, mahalanobis distance assuming gaussian distribution)
            xy = np.vstack([x, y])
            mean = np.mean(xy, axis=1)
            cov = np.cov(xy)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            for nsig in [1, 2]:
                width, height = 2 * nsig * np.sqrt(eigvals)
                ellip = patches.Ellipse(
                    xy=mean,
                    width=width,
                    height=height,
                    angle=angle,
                    edgecolor="C1",
                    fc="None",
                    lw=2,
                    ls="--",
                    label=f"{nsig}σ contour" if nsig == 1 else None,
                )
                ax.add_patch(ellip)

            # Axis labels and title
            ax.set_xlabel(f"Fraction D={result['sample'].diffusivities[x_idx]}")
            ax.set_ylabel(f"Fraction D={result['sample'].diffusivities[y_idx]}")
            title = f"{config.name}"
            if "zone" in result:
                title += f"\nZone: {result['zone']}"
            title += f"\nFirst {n_steps} steps, Data SNR: {config.data_snr}, Gibbs SNR: {config.gibbs_snr}"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main(configs: dict) -> None:
    if configs["USE_TEST_DATA"]:
        diffusivities = np.array(
            [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
        )

        # Create a single zone with multiple configurations for different complexity levels
        true_spectra = {
            "Complexity": generate_true_spectrum(diffusivities, 1)  # Base spectrum
        }

        # Get simulation configurations
        sim_configs = SimConfig.create_configs()

        # Run simulations
        all_results = []

        # Add both complexity levels to results
        for i in range(0, 5):  # Complexity levels 1 and 2
            test_spectrum = generate_true_spectrum(diffusivities, i)
            # For first config, use the SimConfig from sim_configs
            config = sim_configs[0]
            # Create a modified copy of the config for this complexity level
            modified_config = SimConfig(
                name=f"Complexity Level {i}",
                data_snr=config.data_snr,
                gibbs_snr=config.gibbs_snr,
                b_values=config.b_values,
                iterations=config.iterations,
                l2_lambda=config.l2_lambda,
                burn_in=config.burn_in,
            )
            result = run_simulation(
                test_spectrum, modified_config, sampler_class=GibbsSampler
            )
            result["zone"] = "Complexity"
            result["true_spectrum"] = test_spectrum
            all_results.append(result)

        # Use our modified plotting function instead of the original
        modified_plot_results(
            true_spectra, all_results, configs["SIM_PDF"], configs["SIM_CSV"]
        )

    else:
        # Get averaged spectra from real data
        true_spectra = get_averaged_spectra(
            configs["SIM_JSON_PATH"], configs["SIM_HDF5_PATH"]
        )

        # Get simulation configurations
        sim_configs = SimConfig.create_configs()

        # Run all simulations
        all_results = []
        for zone_name, true_spectrum in true_spectra.items():
            for config in sim_configs:
                result = run_simulation(
                    true_spectrum, config, sampler_class=GibbsSampler
                )
                result["zone"] = zone_name
                all_results.append(result)

        # Plot and save results
        modified_plot_results(
            true_spectra, all_results, configs["SIM_PDF"], configs["SIM_CSV"]
        )


if __name__ == "__main__":
    configs = {}
    # Use importlib.resources to load configs.yaml from the package
    with importlib.resources.files("spectra_estimation_dmri").joinpath(
        "configs.yaml"
    ).open("r") as file:
        configs.update(yaml.safe_load(file))
    main(configs)
