from typing import Optional, Literal, List
from pydantic import BaseModel, computed_field, field_validator

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import hashlib
import json
import ast


class SignalDecay(BaseModel):
    patient: str
    signal_values: list[float]
    b_values: list[float]
    voxel_count: Optional[int] = None  # alias for v_count in JSON
    a_region: Literal["pz", "tz", "sim"]
    is_tumor: bool = False
    ggg: Optional[int] = None
    gs: Optional[str] = None
    true_spectrum: Optional[list[float]] = None
    snr: Optional[float] = None

    @field_validator("ggg")
    @classmethod
    def validate_ggg(cls, v):
        if v is not None and not (0 <= v <= 5):
            raise ValueError("GGG must be between 1 and 5.")
        return v

    def fit_adc(self, b_range="0-1250", plot=True):
        """
        Calculate ADC using a monoexponential model.

        Parameters:
        b_range : str, optional
            The range of b-values to use. Either '0-1000', '0-1250' or '250-1250'
        plot : bool, optional
            If True, plot the signal decay and fitted line

        Returns:
        adc : float
            The calculated Apparent Diffusion Coefficient
        """
        # convert attributes to numpy arrays
        signal_values, b_values = self.as_numpy()

        if b_range == "0-1000":
            mask = b_values <= 1000
        elif b_range == "0-1250":
            mask = b_values <= 1250
        elif b_range == "250-1000":
            mask = (b_values >= 250) & (b_values <= 1000)
        elif b_range == "250-1250":
            mask = (b_values >= 250) & (b_values <= 1250)
        else:
            raise ValueError(
                "Invalid b_range. Use '0-1000', '0-1250', '250-1000' or '250-1250'"
            )

        valid_mask = (signal_values > 0) & mask  # Exclude non-positive signal values

        if not np.any(valid_mask):
            raise ValueError("No valid signal values available for ADC calculation.")

        log_signal = np.log(signal_values[valid_mask])
        valid_b_values = b_values[valid_mask]

        slope, intercept = np.polyfit(valid_b_values, log_signal, 1)
        adc = -slope

        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(b_values, signal_values, label="Original data")
            plt.scatter(valid_b_values, np.exp(log_signal), label="Used for fitting")
            fit_b_values = np.linspace(min(valid_b_values), max(valid_b_values), 100)
            fit_signal = np.exp(intercept - adc * fit_b_values)
            plt.plot(fit_b_values, fit_signal, "r-", label="Fitted line")
            plt.xlabel("b-value (s/mm²)")
            plt.ylabel("Signal intensity")
            plt.title(
                f"Signal Decay and ADC Fit (ADC = {adc:.4f} mm²/s)|{self.a_region}|{b_range}"
            )
            plt.legend()
            plt.yscale("log")
            plt.grid(True)
            # plt.savefig("adc_fit_plot.png")
            plt.show()

        return adc

    def as_numpy(self):
        """Return signal_values and b_values as numpy arrays."""
        return np.array(self.signal_values), np.array(self.b_values)


class DiffusivitySpectrum(BaseModel):
    """
    Stores the result of spectrum inference for a single SignalDecay object and inference method.
    - spectrum_init: starting point of sampling process (important for convergence assessment)
    - spectrum_vector: main estimate (MAP, posterior mean, etc.)
    - spectrum_samples: optional, posterior samples (for Bayesian methods)
    - spectrum_std: optional, posterior std (for Bayesian methods)
    - true_spectrum: optional, for simulation/benchmarking
    - inference_data: path to .nc or similar file
    - inference_method: e.g., 'map', 'gibbs', 'vb'
    - spectra_id: unique hash for retrieval and grouping
    """

    inference_method: str
    signal_decay: SignalDecay
    diffusivities: list[float]
    design_matrix_U: list[list[float]]
    spectrum_init: Optional[list[float]]
    spectrum_vector: list[float]
    spectrum_samples: Optional[list[list[float]]] = None
    spectrum_std: Optional[list[float]] = None
    true_spectrum: Optional[list[float]] = None
    inference_data: str
    spectra_id: Optional[str] = None
    init_method: Optional[str] = None
    prior_type: Optional[str] = None
    prior_strength: Optional[float] = None
    data_snr: Optional[float] = None  # SNR used for data generation
    sampler_snr: Optional[float] = None  # SNR used by the sampler

    def as_numpy(self):
        """Return all list fields as numpy arrays."""
        return {
            "diffusivities": np.array(self.diffusivities),
            "design_matrix_U": np.array(self.design_matrix_U),
            "spectrum_vector": np.array(self.spectrum_vector),
            "spectrum_samples": (
                np.array(self.spectrum_samples)
                if self.spectrum_samples is not None
                else None
            ),
            "spectrum_std": (
                np.array(self.spectrum_std) if self.spectrum_std is not None else None
            ),
            "true_spectrum": (
                np.array(self.true_spectrum) if self.true_spectrum is not None else None
            ),
        }

    def plot(self, config_info=None, save_dir=None, show=True):
        """
        Plot the spectrum (MAP and posterior mean) for this object.
        config_info: str or dict to include in the title/filename.
        save_dir: directory to save the plot (optional).
        show: whether to display the plot.
        """
        # Example implementation (update as needed):
        import matplotlib.pyplot as plt
        import numpy as np

        diffusivities = np.array(self.diffusivities)
        spectrum = np.array(self.spectrum_vector)
        plt.figure(figsize=(8, 5))
        plt.plot(diffusivities, spectrum, label="Estimate")
        if self.true_spectrum is not None:
            plt.plot(diffusivities, self.true_spectrum, label="True", linestyle="--")
        plt.xlabel("Diffusivity")
        plt.ylabel("Fraction")
        title = (
            f"Spectrum | Data SNR: {self.data_snr} | Sampler SNR: {self.sampler_snr}"
        )
        if config_info is not None:
            title += f" | {config_info}"
        plt.title(title)
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"spectrum_{self.spectra_id}.png"))
        if show:
            plt.show()
        plt.close()


class SignalDecayDataset(BaseModel):
    samples: list[SignalDecay]  # Forward reference if SignalDecay is defined later

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_patient(self, pid: str) -> Optional["SignalDecay"]:
        matches = [m for m in self.samples if m.patient == pid]
        return matches if matches else None

    def filter_tumor(self, tumor: bool = True) -> "SignalDecayDataset":
        return SignalDecayDataset(
            samples=[s for s in self.samples if s.is_tumor == tumor]
        )

    def filter_by_region(self, region: Literal["pz", "tz"]) -> "SignalDecayDataset":
        return SignalDecayDataset(
            samples=[s for s in self.samples if s.a_region == region]
        )

    def filter_by_ggg(self, ggg: int) -> "SignalDecayDataset":
        return SignalDecayDataset(samples=[s for s in self.samples if s.ggg == ggg])

    def to_numpy_matrix(self):
        return np.stack([np.array(s.signal_values) for s in self.samples])

    def to_b_matrix(self):
        return np.stack([np.array(s.b_values) for s in self.samples])

    def stratify_by_ggg(self) -> dict[int, "SignalDecayDataset"]:
        stratified = {}
        for g in range(1, 6):
            subset = [s for s in self.samples if s.ggg == g]
            if subset:
                stratified[g] = SignalDecayDataset(samples=subset)
        return stratified

    def summary(self):
        from collections import Counter

        print("Total samples:", len(self.samples))
        print("Tumor samples:", sum(s.is_tumor for s in self.samples))
        print("Regions:", Counter(s.a_region for s in self.samples))
        print(
            "GGG distribution:",
            Counter(s.ggg for s in self.samples if s.ggg is not None),
        )

    def as_numpy(self):
        """Return all signal_values and b_values as numpy arrays for all samples."""
        return [s.as_numpy() for s in self.samples]


class DiffusivitySpectraDataset(BaseModel):
    spectra: list[DiffusivitySpectrum]

    def __len__(self):
        return len(self.spectra)

    def __item__(self, idx):
        return self.spectra[idx]

    def as_numpy(self):
        return [s.as_numpy() for s in self.spectra]

    @staticmethod
    def save_index(index_dict, index_path):
        """Save an index (dict) mapping config hashes and sample IDs to file paths as JSON."""
        with open(index_path, "w") as f:
            json.dump(index_dict, f, indent=2)

    @staticmethod
    def load_index(index_path):
        """Load an index (dict) from JSON file."""
        with open(index_path, "r") as f:
            return json.load(f)

    def run_diagnostics(self, exp_config=None, local=True):
        """
        Run diagnostics and generate three types of plots:
        1. Distribution/box plots for Gibbs sampling
        2. Stability analysis comparing MAP and Gibbs point estimates
        3. Trace plots for Gibbs sampling

        Groups spectra by group_id to analyze different noise realizations together.
        """
        import wandb
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict
        from spectra_estimation_dmri.utils.spectra_id import get_group_id
        from spectra_estimation_dmri.utils import plotting

        if exp_config is None:
            raise ValueError("exp_config must be provided to group spectra correctly.")

        # Group spectra by group_id (excludes noise realizations)
        groups = defaultdict(list)
        for spectrum in self.spectra:
            group_id = get_group_id(spectrum.signal_decay, exp_config)
            groups[group_id].append(spectrum)

        # Process each group
        for group_id, spectra_list in groups.items():
            # Get group info from first spectrum
            first_spectrum = spectra_list[0]
            sd = first_spectrum.signal_decay

            # Calculate design matrix condition number
            U = np.array(first_spectrum.design_matrix_U)
            kappa = np.linalg.cond(U) if U.size > 0 else np.nan

            # Log structured metrics for parameter analysis
            self._log_parameter_metrics(
                group_id, spectra_list, kappa, exp_config, local
            )

            # Separate spectra by inference method
            map_spectra = [s for s in spectra_list if s.inference_method == "map"]
            gibbs_spectra = [s for s in spectra_list if s.inference_method == "gibbs"]

            # Get true spectrum if available
            true_spectra = [
                np.array(s.true_spectrum)
                for s in spectra_list
                if s.true_spectrum is not None
            ]
            has_true = len(true_spectra) > 0
            mean_true = np.mean(true_spectra, axis=0) if has_true else None

            # Gibbs Plotting
            if len(gibbs_spectra) > 0:
                self._plot_gibbs_distributions(
                    group_id, gibbs_spectra, kappa, mean_true, sd, local
                )
                self._plot_stability_analysis(
                    group_id, gibbs_spectra, kappa, mean_true, sd, local
                )
                self._plot_trace_plots(group_id, gibbs_spectra, kappa, sd, local)
                self._plot_autocorrelation_plots(
                    group_id, gibbs_spectra, kappa, sd, local
                )
            # MAP Plotting
            if len(map_spectra) > 0:
                self._plot_stability_analysis(
                    group_id, map_spectra, kappa, mean_true, sd, local
                )

    def run_cross_config_diagnostics(self, exp_config=None):
        """
        Run cross-configuration diagnostics comparing different priors across SNRs.
        Creates stability analysis plots similar to the third page of modified_plot_results.
        Groups spectra by SNR and signal properties (excluding prior/regularization).
        """
        import wandb
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict

        if exp_config is None:
            raise ValueError("exp_config must be provided for cross-config analysis.")

        # Group spectra by base parameters (excluding prior-specific info)
        groups = defaultdict(list)

        for spectrum in self.spectra:
            sd = spectrum.signal_decay
            # Create a base group key excluding prior and regularization info
            base_key = f"snr_{sd.snr}_region_{sd.a_region}_spectrum_{getattr(sd, 'true_spectrum_name', 'unknown')}"
            groups[base_key].append(spectrum)

        # Process each group
        for group_key, spectra_list in groups.items():
            self._plot_cross_config_stability_analysis(
                group_key, spectra_list, exp_config
            )

    def _plot_cross_config_stability_analysis(
        self, group_key, spectra_list, exp_config, local
    ):
        """
        Plot stability analysis comparing different priors for the same base configuration.
        Creates boxplots for each prior type side-by-side for each diffusivity bin.
        """
        import wandb
        import matplotlib.pyplot as plt
        import numpy as np

        if not spectra_list:
            return

        # Group spectra by prior type
        prior_groups = defaultdict(list)
        for spectrum in spectra_list:
            # Extract prior info from the spectrum or config
            prior_type = self._extract_prior_type(spectrum)
            prior_groups[prior_type].append(spectrum)

        # Get diffusivities and true spectrum from first spectrum
        first_spectrum = spectra_list[0]
        diffusivities = np.array(first_spectrum.diffusivities)
        n_diffusivities = len(diffusivities)

        # Get true spectrum if available
        true_spectrum = None
        if first_spectrum.true_spectrum is not None:
            true_spectrum = np.array(first_spectrum.true_spectrum)

        # Create plot
        fig, ax = plt.subplots(figsize=(15, 8))

        # Set up positions for boxplots
        n_priors = len(prior_groups)
        width = 0.8 / n_priors  # Total width divided by number of priors
        positions_base = np.arange(1, n_diffusivities + 1)

        colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow"]
        legend_handles = []

        for i, (prior_type, prior_spectra) in enumerate(prior_groups.items()):
            if not prior_spectra:
                continue

            # Collect estimates for this prior
            estimates = []
            for spectrum in prior_spectra:
                if spectrum.spectrum_vector is not None:
                    estimates.append(np.array(spectrum.spectrum_vector))

            if not estimates:
                continue

            # Create boxplot positions
            positions = positions_base + (i - n_priors / 2 + 0.5) * width

            if len(estimates) > 1:
                # Multiple estimates - create boxplot
                estimates_matrix = np.stack(estimates)
                bp = ax.boxplot(
                    estimates_matrix,
                    positions=positions,
                    widths=width * 0.8,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors[i % len(colors)], color="black"),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"),
                    flierprops=dict(color="black", markeredgecolor="black"),
                    showfliers=False,
                )
                legend_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=colors[i % len(colors)],
                        lw=4,
                        label=f"{prior_type.upper()} (N={len(estimates)})",
                    )
                )
            else:
                # Single estimate - scatter plot
                ax.scatter(
                    positions,
                    estimates[0],
                    color=colors[i % len(colors)],
                    s=50,
                    alpha=0.8,
                    label=f"{prior_type.upper()} (N=1)",
                )

        # Plot true spectrum if available
        if true_spectrum is not None:
            ax.vlines(
                positions_base,
                0,
                true_spectrum,
                colors="red",
                linewidth=2,
                alpha=0.8,
                label="True Spectrum",
            )
            legend_handles.append(
                plt.Line2D([0], [0], color="red", lw=2, label="True Spectrum")
            )

        # Customize plot
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f"{d:.1f}" for d in diffusivities], rotation=45)
        ax.set_ylabel("Relative Fraction")
        ax.set_xlabel("Diffusivity Value (μm²/ms)")

        # Extract SNR and other info for title
        first_sd = first_spectrum.signal_decay
        first_data_snr = getattr(first_spectrum, "data_snr", None)
        first_sampler_snr = getattr(first_spectrum, "sampler_snr", None)
        title = f"Prior Comparison | {group_key}\n"
        title += f"Data SNR: {first_data_snr} | Sampler SNR: {first_sampler_snr} | Region: {first_sd.a_region}"
        if hasattr(first_sd, "true_spectrum_name"):
            title += (
                f' | Spectrum: {getattr(first_sd, "true_spectrum_name", "unknown")}'
            )
        ax.set_title(title)

        # Add legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Set y-axis limits
        if true_spectrum is not None:
            max_val = max(
                np.max(
                    [
                        np.max(spectrum.spectrum_vector)
                        for spectrum in spectra_list
                        if spectrum.spectrum_vector is not None
                    ]
                ),
                np.max(true_spectrum),
            )
        else:
            max_val = np.max(
                [
                    np.max(spectrum.spectrum_vector)
                    for spectrum in spectra_list
                    if spectrum.spectrum_vector is not None
                ]
            )
        ax.set_ylim(0, max_val * 1.1)

        plt.tight_layout()
        if not local:
            wandb.log({f"cross_config_stability_{group_key}": wandb.Image(fig)})
        plt.close(fig)

    def _extract_prior_type(self, spectrum):
        """Extract prior type from spectrum object."""
        if hasattr(spectrum, "prior_type") and spectrum.prior_type is not None:
            return spectrum.prior_type
        else:
            return "unknown"

    def _log_parameter_metrics(self, group_id, spectra_list, kappa, exp_config, local):
        """Log structured metrics for parameter analysis across runs."""
        import wandb
        import numpy as np

        if not spectra_list:
            return

        # Get representative info from first spectrum
        first_spectrum = spectra_list[0]
        sd = first_spectrum.signal_decay

        # Extract config parameters for logging
        inference_method = first_spectrum.inference_method
        diff_values = first_spectrum.diffusivities
        n_diff_bins = len(diff_values)
        diff_range = max(diff_values) - min(diff_values)

        # Calculate metrics for each spectrum in the group
        recon_errors = []
        for spectrum in spectra_list:
            if spectrum.true_spectrum and spectrum.spectrum_vector:
                true_spec = np.array(spectrum.true_spectrum)
                est_spec = np.array(spectrum.spectrum_vector)
                recon_error = np.linalg.norm(est_spec - true_spec)
                recon_errors.append(recon_error)

        # Aggregate metrics
        mean_recon_error = np.mean(recon_errors) if recon_errors else np.nan
        std_recon_error = np.std(recon_errors) if len(recon_errors) > 1 else np.nan

        # Log structured metrics with parameter tags for aggregation
        base_metrics = {
            # Core metrics
            "condition_number": float(kappa),
            "recon_error_mean": float(mean_recon_error),
            "recon_error_std": float(std_recon_error),
            "n_realizations": len(spectra_list),
            # Parameter space info for aggregation
            "snr": sd.snr,
            "a_region": sd.a_region,
            "inference_method": inference_method,
            "n_diff_bins": n_diff_bins,
            "diff_range": diff_range,
            "diff_min": min(diff_values),
            "diff_max": max(diff_values),
            # Config parameters
            "prior": exp_config.prior.type,
            "likelihood": exp_config.likelihood.type,
        }

        # Add inference-specific parameters for aggregation
        if hasattr(exp_config.inference, "n_iter"):
            base_metrics["gibbs_n_iter"] = exp_config.inference.n_iter
        if hasattr(exp_config.inference, "burn_in"):
            base_metrics["gibbs_burn_in"] = exp_config.inference.burn_in
        if hasattr(exp_config.inference, "l2_lambda"):
            base_metrics["map_l2_lambda"] = exp_config.inference.l2_lambda

        # Add stability metrics for uncertainty quantification assessment
        if len(recon_errors) > 1:
            base_metrics["recon_error_cv"] = (
                float(std_recon_error / mean_recon_error)
                if mean_recon_error > 0
                else np.nan
            )
            base_metrics["recon_error_range"] = float(
                max(recon_errors) - min(recon_errors)
            )

        # Log group_id for easier filtering
        base_metrics["group_id"] = group_id

        # Add diffusivity discretization info
        diff_spacing = np.diff(diff_values)
        base_metrics.update(
            {
                "diff_spacing_mean": float(np.mean(diff_spacing)),
                "diff_spacing_std": float(np.std(diff_spacing)),
                "diff_spacing_uniform": bool(
                    np.allclose(diff_spacing, diff_spacing[0], rtol=1e-3)
                ),
            }
        )

        # Log metrics with prefixes for easy filtering in wandb
        if not local:
            wandb.log({f"param_analysis/{k}": v for k, v in base_metrics.items()})

            # Also log with method-specific prefix for easy method comparison
            wandb.log(
                {
                    f"param_analysis_{inference_method}/{k}": v
                    for k, v in base_metrics.items()
                }
            )
        print({f"param_analysis/{k}": v for k, v in base_metrics.items()})
        print(
            {
                f"param_analysis_{inference_method}/{k}": v
                for k, v in base_metrics.items()
            }
        )

    def _plot_gibbs_distributions(
        self, group_id, gibbs_spectra, kappa, mean_true, signal_decay, local
    ):
        """Plot distribution/box plots for Gibbs sampling results."""
        if not gibbs_spectra:
            return

        import wandb
        import matplotlib.pyplot as plt
        import numpy as np

        for idx, spectrum in enumerate(gibbs_spectra):
            if spectrum.spectrum_samples is None:
                continue

            samples = np.array(spectrum.spectrum_samples)
            diffusivities = np.array(spectrum.diffusivities)

            fig, ax = plt.subplots(figsize=(12, 8))

            # Create boxplot
            ax.boxplot(
                samples,
                showfliers=False,
                showmeans=True,
                meanline=True,
                labels=[f"{d:.1f}" for d in diffusivities],
            )

            # Add MAP initialization if available
            if spectrum.spectrum_init is not None:
                x_positions = np.arange(1, len(diffusivities) + 1)
                ax.plot(
                    x_positions,
                    spectrum.spectrum_init,
                    "r*",
                    label="Initial R (MAP)",
                    markersize=8,
                )

            # Add true spectrum if available
            if mean_true is not None:
                x_positions = np.arange(1, len(diffusivities) + 1)
                ax.vlines(
                    x_positions,
                    0,
                    mean_true,
                    colors="blue",
                    linewidth=2,
                    label="True Spectrum",
                )

            # Customize plot
            ax.set_xlabel(r"Diffusivity Value ($\mu$m$^2$/ms)")
            ax.set_ylabel("Relative Fraction")
            ax.set_xticklabels([f"{d:.1f}" for d in diffusivities], rotation=45)

            title = (
                f"Gibbs Distribution | Group: {group_id[:8]} | Realization: {idx+1}\n"
            )
            title += (
                f"Zone: {signal_decay.a_region} | Data SNR: {getattr(spectrum, 'data_snr', None)} | "
                f"Sampler SNR: {getattr(spectrum, 'sampler_snr', None)} | "
                f"Prior: {getattr(spectrum, 'prior_type', None)} | Strength: {getattr(spectrum, 'prior_strength', None)} | "
                f"κ={kappa:.2e}"
            )
            ax.set_title(title)

            if spectrum.spectrum_vector is not None or mean_true is not None:
                ax.legend()

            plt.tight_layout()
            if not local:
                wandb.log({f"distribution_{group_id}_real{idx+1}": wandb.Image(fig)})
            if local:
                output_pdf_path = "/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI/results/plots/plot/distributions.pdf"
                os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
                with PdfPages(output_pdf_path) as pdf:
                    pdf.savefig(fig)
            plt.close(fig)

    def _plot_stability_analysis(
        self, group_id, spectra, kappa, mean_true, signal_decay, local
    ):
        """Plot stability analysis of either MAP or Gibbs point estimates across realizations."""
        import wandb
        import matplotlib.pyplot as plt
        import numpy as np

        # Collect estimates
        init_point_estimates = []
        point_estimates = []

        if spectra[0].inference_method == "map":
            for s in spectra:
                if s.spectrum_vector is not None:
                    point_estimates.append(np.array(s.spectrum_vector))

        if spectra[0].inference_method == "gibbs":
            for s in spectra:
                if s.spectrum_samples is not None:
                    point_estimates.append(
                        np.mean(np.array(s.spectrum_samples), axis=0)
                    )
                    init_point_estimates.append(np.array(s.spectrum_init))

        if not point_estimates:
            return

        # Determine number of diffusivity bins
        if point_estimates:
            n_bins = len(point_estimates[0])
            diffusivities = np.array(spectra[0].diffusivities)
        positions = np.arange(1, n_bins + 1)

        fig, ax = plt.subplots(figsize=(12, 8))
        legend_handles = []

        # Plot MAP estimates
        if (
            spectra[0].inference_method == "map"
            and point_estimates
            and len(point_estimates) > 1
        ):
            map_matrix = np.stack(point_estimates)
            bp_map = ax.boxplot(
                map_matrix,
                positions=positions - 0.2,
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="lightgreen", color="green"),
                medianprops=dict(color="green"),
            )
            legend_handles.append(
                plt.Line2D([0], [0], color="green", lw=2, label="MAP (mode)")
            )
        elif spectra[0].inference_method == "map" and point_estimates:
            ax.scatter(
                positions - 0.2,
                point_estimates[0],
                color="green",
                label="MAP (mode)",
                s=50,
            )

        # Plot Gibbs means
        if (
            spectra[0].inference_method == "gibbs"
            and point_estimates
            and len(point_estimates) > 1
        ):
            gibbs_matrix = np.stack(point_estimates)
            init_matrix = np.stack(init_point_estimates)
            n_diffusivities = len(spectra[0].diffusivities)
            positions_init = np.arange(1, n_diffusivities + 1) - 0.2
            positions_gibbs = np.arange(1, n_diffusivities + 1) + 0.2
            bp_gibbs = ax.boxplot(
                gibbs_matrix,
                positions=positions_gibbs,
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="blue"),
            )
            bp_init = ax.boxplot(
                init_matrix,
                positions=positions_init,
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="lightgreen", color="green"),
                medianprops=dict(color="green"),
            )
            legend_handles.append(
                plt.Line2D([0], [0], color="blue", lw=2, label="Gibbs (mean)")
            )

        elif spectra[0].inference_method == "gibbs" and point_estimates:
            ax.scatter(
                positions + 0.2,
                point_estimates[0],
                color="blue",
                label="Gibbs (mean)",
                s=50,
            )

        # Plot true spectrum
        if mean_true is not None:
            ax.vlines(
                positions,
                0,
                mean_true,
                colors="red",
                linewidth=2,
                label="True Spectrum",
            )
            legend_handles.append(
                plt.Line2D([0], [0], color="red", lw=2, label="True Spectrum")
            )

            # Calculate and log reconstruction errors
            recon_errors = []
            for est in point_estimates:
                recon_errors.append(np.linalg.norm(est - mean_true))

            if recon_errors:
                recon_err_mean = float(np.mean(recon_errors))
                recon_err_std = float(np.std(recon_errors))
                print({f"recon_err_mean": recon_err_mean})
                print({f"recon_err_std": recon_err_std})
                if not local:
                    wandb.log(
                        {
                            f"recon_err_mean": recon_err_mean,
                            f"recon_err_std": recon_err_std,
                        }
                    )
                subtitle = f"\nRecon err: {recon_err_mean:.3g}±{recon_err_std:.2g}"
            else:
                subtitle = ""
        else:
            subtitle = ""

        # Customize plot
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{d:.1f}" for d in diffusivities], rotation=90)
        ax.set_ylabel("Relative Fraction")
        ax.set_xlabel("Diffusivity Value")

        title = f"Stability Analysis | Inference: {spectra[0].inference_method} | Prior: {spectra[0].prior_type} | Strength: {spectra[0].prior_strength} \n"
        title += f"Zone: {signal_decay.a_region} | Data SNR: {getattr(spectra[0], 'data_snr', None)} | Sampler SNR: {getattr(spectra[0], 'sampler_snr', None)} | κ={kappa:.2e}{subtitle}"
        title += f"\nN={len(point_estimates)} realizations | Group: {group_id[:8]}"
        ax.set_title(title)

        if legend_handles or mean_true is not None:
            if spectra[0].inference_method == "gibbs":
                # ax.legend()
                ax.legend(
                    [bp_init["boxes"][0], bp_gibbs["boxes"][0], mean_true],
                    ["MAP (Posterior Mode)", "Posterior Mean", "True Spectrum"],
                    loc="upper right",
                )
            if spectra[0].inference_method == "map":
                ax.legend(
                    [bp_map["boxes"][0], mean_true],
                    ["MAP (Posterior Mode)", "True Spectrum"],
                    loc="upper right",
                )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if not local:
            wandb.log({f"stability": wandb.Image(fig)})
        if local:
            output_pdf_path = "/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI/results/plots/plot/stability.pdf"
            os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
            with PdfPages(output_pdf_path) as pdf:
                pdf.savefig(fig)
        plt.close(fig)

    def _plot_trace_plots(self, group_id, gibbs_spectra, kappa, signal_decay, local):
        """Plot MCMC trace plots for Gibbs sampling with improved scaling options."""
        import wandb
        import matplotlib.pyplot as plt
        import numpy as np

        for idx, spectrum in enumerate(gibbs_spectra):
            if spectrum.spectrum_samples is None:
                continue

            samples = np.array(spectrum.spectrum_samples)
            diffusivities = np.array(spectrum.diffusivities)
            init_method = getattr(spectrum, "init_method", "map")

            # Create multiple plot variants for better visualization
            plot_variants = [
                self._plot_trace_subplots,
                self._plot_trace_offset,
            ]

            for variant_idx, plot_func in enumerate(plot_variants):
                fig = plot_func(
                    samples,
                    diffusivities,
                    group_id,
                    idx,
                    signal_decay,
                    spectrum,
                    kappa,
                    init_method,
                )

                if not local:
                    variant_name = plot_func.__name__.replace("_plot_trace_", "")
                    wandb.log(
                        {
                            f"trace_{group_id}_real{idx+1}_{variant_name}": wandb.Image(
                                fig
                            )
                        }
                    )
                if local:
                    variant_name = plot_func.__name__.replace("_plot_trace_", "")
                    output_pdf_path = f"/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI/results/plots/plot/traceplot_{variant_name}.pdf"
                    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
                    with PdfPages(output_pdf_path) as pdf:
                        pdf.savefig(fig)
                plt.close(fig)

    def _plot_autocorrelation_plots(
        self, group_id, gibbs_spectra, kappa, signal_decay, local
    ):
        """Plot autocorrelation plots for Gibbs sampling to assess mixing."""
        import wandb
        import matplotlib.pyplot as plt
        import numpy as np

        last_fig = None
        last_joint_fig = None

        for idx, spectrum in enumerate(gibbs_spectra):
            if spectrum.spectrum_samples is None:
                continue

            samples = np.array(spectrum.spectrum_samples)
            diffusivities = np.array(spectrum.diffusivities)
            init_method = getattr(spectrum, "init_method", "map")

            # Create separate autocorrelation plot
            fig = self._plot_autocorrelation(
                samples,
                diffusivities,
                group_id,
                idx,
                signal_decay,
                spectrum,
                kappa,
                init_method,
            )

            if not local:
                wandb.log({f"autocorrelation_{group_id}_real{idx+1}": wandb.Image(fig)})

            last_fig = fig
            plt.close(fig)

            # Create joint autocorrelation plot
            joint_fig = self._plot_joint_autocorrelation(
                samples,
                diffusivities,
                group_id,
                idx,
                signal_decay,
                spectrum,
                kappa,
                init_method,
            )

            if not local:
                wandb.log(
                    {
                        f"joint_autocorrelation_{group_id}_real{idx+1}": wandb.Image(
                            joint_fig
                        )
                    }
                )

            last_joint_fig = joint_fig
            plt.close(joint_fig)

        # Save figures to PDF
        if local:
            if last_fig is not None:
                output_pdf_path = f"/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI/results/plots/plot/autocorrelation.pdf"
                os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
                with PdfPages(output_pdf_path) as pdf:
                    pdf.savefig(last_fig)
                plt.close(last_fig)

            if last_joint_fig is not None:
                output_pdf_path = f"/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI/results/plots/plot/joint_autocorrelation.pdf"
                os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
                with PdfPages(output_pdf_path) as pdf:
                    pdf.savefig(last_joint_fig)
                plt.close(last_joint_fig)

    def _plot_trace_subplots(
        self,
        samples,
        diffusivities,
        group_id,
        idx,
        signal_decay,
        spectrum,
        kappa,
        init_method,
    ):
        """Plot each D value in separate subplots for clear separation."""
        import matplotlib.pyplot as plt
        import numpy as np

        n_diffs = len(diffusivities)
        n_cols = min(3, n_diffs)
        n_rows = (n_diffs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for j, diff in enumerate(diffusivities):
            row = j // n_cols
            col = j % n_cols
            ax = axes[row, col]

            ax.plot(
                np.arange(len(samples)),
                samples[:, j],
                label=f"D={diff:.1f}",
                alpha=0.8,
                linewidth=1,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fraction")
            ax.set_title(f"D = {diff:.1f}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide empty subplots
        for j in range(n_diffs, n_rows * n_cols):
            row = j // n_cols
            col = j % n_cols
            axes[row, col].set_visible(False)

        # Add overall title
        title = f"MCMC Trace Plots (Subplots) | Group: {group_id[:8]} | Realization: {idx+1}\n"
        title += (
            f"Zone: {signal_decay.a_region} | Data SNR: {getattr(spectrum, 'data_snr', None)} | "
            f"Sampler SNR: {getattr(spectrum, 'sampler_snr', None)} | "
            f"Prior: {getattr(spectrum, 'prior_type', None)} | Strength: {getattr(spectrum, 'prior_strength', None)} | "
            f"Init: {init_method}\n"
            f"Iterations: {len(samples)} | κ={kappa:.2e}"
        )
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        return fig

    def _plot_autocorrelation(
        self,
        samples,
        diffusivities,
        group_id,
        idx,
        signal_decay,
        spectrum,
        kappa,
        init_method,
        max_lag=None,
    ):
        """Plot autocorrelation for each diffusivity dimension to assess mixing."""
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        if max_lag is None:
            max_lag = min(
                100, len(samples) // 10
            )  # Default to 100 or 10% of chain length

        n_diffs = len(diffusivities)
        n_cols = min(3, n_diffs)
        n_rows = (n_diffs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for j, diff in enumerate(diffusivities):
            row = j // n_cols
            col = j % n_cols
            ax = axes[row, col]

            # Calculate autocorrelation
            trace = samples[:, j]
            autocorr = []
            for lag in range(1, max_lag + 1):
                if lag < len(trace):
                    # Calculate autocorrelation coefficient
                    corr = np.corrcoef(trace[:-lag], trace[lag:])[0, 1]
                    autocorr.append(corr)
                else:
                    break

            lags = np.arange(1, len(autocorr) + 1)
            ax.plot(lags, autocorr, "b-", linewidth=1.5, alpha=0.8)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            ax.axhline(
                y=0.1, color="r", linestyle=":", alpha=0.7, label="0.1 threshold"
            )
            ax.axhline(y=-0.1, color="r", linestyle=":", alpha=0.7)

            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorrelation")
            ax.set_title(f"D = {diff:.1f}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Remove empty subplots
        for j in range(n_diffs, n_rows * n_cols):
            row = j // n_cols
            col = j % n_cols
            fig.delaxes(axes[row, col])

        title = f"Autocorrelation Plot | Group: {group_id[:8]} | Realization: {idx+1}\n"
        title += (
            f"Zone: {signal_decay.a_region} | Data SNR: {getattr(spectrum, 'data_snr', None)} | "
            f"Sampler SNR: {getattr(spectrum, 'sampler_snr', None)} | "
            f"Prior: {getattr(spectrum, 'prior_type', None)} | Strength: {getattr(spectrum, 'prior_strength', None)} | "
            f"Init: {init_method}\n"
            f"Iterations: {len(samples)} | κ={kappa:.2e} | Max Lag: {max_lag}"
        )
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        return fig

    def _plot_joint_autocorrelation(
        self,
        samples,
        diffusivities,
        group_id,
        idx,
        signal_decay,
        spectrum,
        kappa,
        init_method,
        max_lag=None,
    ):
        """Plot joint autocorrelation for all diffusivity dimensions in a single plot."""
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        if max_lag is None:
            max_lag = min(
                100, len(samples) // 10
            )  # Default to 100 or 10% of chain length

        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate autocorrelation for each dimension
        for j, diff in enumerate(diffusivities):
            trace = samples[:, j]
            autocorr = []
            for lag in range(1, max_lag + 1):
                if lag < len(trace):
                    # Calculate autocorrelation coefficient
                    corr = np.corrcoef(trace[:-lag], trace[lag:])[0, 1]
                    autocorr.append(corr)
                else:
                    break

            lags = np.arange(1, len(autocorr) + 1)
            ax.plot(lags, autocorr, label=f"D = {diff:.1f}", linewidth=1.5, alpha=0.8)

        # Add reference lines
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(y=0.1, color="r", linestyle=":", alpha=0.7, label="0.1 threshold")
        ax.axhline(y=-0.1, color="r", linestyle=":", alpha=0.7)

        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

        title = f"Joint Autocorrelation Plot | Group: {group_id[:8]} | Realization: {idx+1}\n"
        title += (
            f"Zone: {signal_decay.a_region} | Data SNR: {getattr(spectrum, 'data_snr', None)} | "
            f"Sampler SNR: {getattr(spectrum, 'sampler_snr', None)} | "
            f"Prior: {getattr(spectrum, 'prior_type', None)} | Strength: {getattr(spectrum, 'prior_strength', None)} | "
            f"Init: {init_method}\n"
            f"Iterations: {len(samples)} | κ={kappa:.2e} | Max Lag: {max_lag}"
        )
        ax.set_title(title)
        plt.tight_layout()
        return fig

    def _plot_trace_offset(
        self,
        samples,
        diffusivities,
        group_id,
        idx,
        signal_decay,
        spectrum,
        kappa,
        init_method,
    ):
        """Plot traces with vertical offsets to separate them."""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(14, 8))

        # Calculate offsets based on the range of each trace
        offsets = np.zeros(len(diffusivities))
        for j in range(len(diffusivities)):
            trace_range = np.max(samples[:, j]) - np.min(samples[:, j])
            if j > 0:
                offsets[j] = offsets[j - 1] + trace_range * 0.5

        # Plot traces with offsets
        for j, diff in enumerate(diffusivities):
            ax.plot(
                np.arange(len(samples)),
                samples[:, j] + offsets[j],
                label=f"D={diff:.1f}",
                alpha=0.7,
                linewidth=1,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fraction (with offsets)")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        title = (
            f"MCMC Trace Plot (Offset) | Group: {group_id[:8]} | Realization: {idx+1}\n"
        )
        title += (
            f"Zone: {signal_decay.a_region} | Data SNR: {getattr(spectrum, 'data_snr', None)} | "
            f"Sampler SNR: {getattr(spectrum, 'sampler_snr', None)} | "
            f"Prior: {getattr(spectrum, 'prior_type', None)} | Strength: {getattr(spectrum, 'prior_strength', None)} | "
            f"Init: {init_method}\n"
            f"Iterations: {len(samples)} | κ={kappa:.2e}"
        )
        ax.set_title(title)
        plt.tight_layout()
        return fig
