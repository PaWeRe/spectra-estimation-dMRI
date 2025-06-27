from typing import Optional, Literal, List
from pydantic import BaseModel, computed_field, field_validator
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import os


class SignalDecay(BaseModel):
    patient: str
    signal_values: list[float]
    b_values: list[float]
    voxel_count: int  # alias for v_count in JSON
    a_region: Literal["pz", "tz", "sim"]
    is_tumor: bool = False
    ggg: Optional[int] = None
    gs: Optional[str] = None

    @computed_field
    @property
    def snr(self) -> float:
        return float(np.sqrt(self.voxel_count / 16) * 150)

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
    - spectrum_vector: main estimate (MAP, posterior mean, etc.)
    - spectrum_samples: optional, posterior samples (for Bayesian methods)
    - spectrum_std: optional, posterior std (for Bayesian methods)
    - ground_truth_spectrum: optional, for simulation/benchmarking
    - inference_data: path to .nc or similar file
    - inference_method: e.g., 'map', 'gibbs', 'vb'
    - config_hash/config_tag: for retrieval and grouping
    """

    inference_method: str
    signal_decay: SignalDecay
    diffusivities: list[float]
    design_matrix_U: list[list[float]]
    spectrum_vector: list[float]
    spectrum_samples: Optional[list[list[float]]] = None
    spectrum_std: Optional[list[float]] = None
    ground_truth_spectrum: Optional[list[float]] = None
    inference_data: str
    config_hash: Optional[str] = None
    config_tag: Optional[str] = None

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
            "ground_truth_spectrum": (
                np.array(self.ground_truth_spectrum)
                if self.ground_truth_spectrum is not None
                else None
            ),
        }

    def plot(self, config_info=None, save_dir=None, show=True):
        """
        Plot the spectrum (MAP and posterior mean) for this object.
        config_info: str or dict to include in the title/filename.
        save_dir: directory to save the plot (optional).
        show: whether to display the plot.
        """
        # TODO: should be implemented or deleted?
        pass


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
        """Save an index (dict) mapping config hashes/tags and sample IDs to file paths as JSON."""
        import json

        with open(index_path, "w") as f:
            json.dump(index_dict, f, indent=2)

    @staticmethod
    def load_index(index_path):
        """Load an index (dict) from JSON file."""
        import json

        with open(index_path, "r") as f:
            return json.load(f)

    def plot_group_boxplot(
        self, group_key=None, save_dir=None, config_info=None, show=True
    ):
        """
        Plot a boxplot comparison of spectra grouped by a key (e.g., ground truth, SNR, etc.).
        - Aggregates spectrum_samples (if available) or spectrum_vector for each inference method.
        - Plots as boxplots (one per inference method), with ground truth as vertical lines if available.
        - Uses config info in the title/filename and filename.
        - Follows the style of plot_d_spectra_sample_boxplot from utils/plotting.py.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from collections import defaultdict

        # Group spectra by inference method
        method_to_samples = defaultdict(list)
        method_to_vectors = defaultdict(list)
        ground_truth = None
        d = None
        for spec in self.spectra:
            if d is None:
                d = np.array(spec.diffusivities)
            if spec.spectrum_samples is not None:
                method_to_samples[spec.inference_method].append(
                    np.array(spec.spectrum_samples)
                )
            else:
                method_to_vectors[spec.inference_method].append(
                    np.array(spec.spectrum_vector)
                )
            if spec.ground_truth_spectrum is not None:
                ground_truth = np.array(spec.ground_truth_spectrum)

        fig, ax = plt.subplots(figsize=(10, 6))
        boxplot_positions = np.arange(1, len(d) + 1)
        colors = ["blue", "red", "orange", "purple", "cyan", "magenta"]
        for idx, (method, samples_list) in enumerate(method_to_samples.items()):
            # Concatenate all samples for this method
            all_samples = np.concatenate(
                samples_list, axis=0
            )  # shape [n_total_samples, n_diffusivities]
            bp = ax.boxplot(
                all_samples,
                positions=boxplot_positions + 0.15 * idx,
                widths=0.12,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                boxprops=dict(facecolor=colors[idx % len(colors)], alpha=0.3),
                medianprops=dict(color=colors[idx % len(colors)]),
                meanprops=dict(color=colors[idx % len(colors)], linewidth=2),
                flierprops=dict(
                    markerfacecolor=colors[idx % len(colors)], marker="o", alpha=0.2
                ),
                manage_ticks=False,
            )
            # Add legend entry
            bp["boxes"][0].set_label(f"{method} (samples)")
        for idx, (method, vectors_list) in enumerate(method_to_vectors.items()):
            # Each vector is a single spectrum estimate
            all_vectors = np.stack(
                vectors_list, axis=0
            )  # shape [n_vectors, n_diffusivities]
            bp = ax.boxplot(
                all_vectors,
                positions=boxplot_positions + 0.15 * (len(method_to_samples) + idx),
                widths=0.12,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                boxprops=dict(
                    facecolor=colors[(len(method_to_samples) + idx) % len(colors)],
                    alpha=0.3,
                ),
                medianprops=dict(
                    color=colors[(len(method_to_samples) + idx) % len(colors)]
                ),
                meanprops=dict(
                    color=colors[(len(method_to_samples) + idx) % len(colors)],
                    linewidth=2,
                ),
                flierprops=dict(
                    markerfacecolor=colors[
                        (len(method_to_samples) + idx) % len(colors)
                    ],
                    marker="o",
                    alpha=0.2,
                ),
                manage_ticks=False,
            )
            bp["boxes"][0].set_label(f"{method} (point)")
        # Plot ground truth if available
        if ground_truth is not None:
            ax.vlines(
                boxplot_positions,
                0,
                ground_truth,
                colors="green",
                alpha=0.8,
                linewidth=2,
                label="Ground Truth",
            )
        ax.set_xticks(boxplot_positions)
        ax.set_xticklabels([f"{val:.2g}" for val in d], rotation=45)
        ax.set_xlabel("Diffusivity")
        ax.set_ylabel("Fraction")
        title = "Spectra Boxplot Comparison"
        if config_info:
            if isinstance(config_info, dict):
                title += " | " + ", ".join(f"{k}={v}" for k, v in config_info.items())
            else:
                title += f" | {config_info}"
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        fname = "spectra_boxplot_comparison"
        if self.spectra and self.spectra[0].config_hash:
            fname += f"_{self.spectra[0].config_hash}"
        if self.spectra and self.spectra[0].config_tag:
            fname += f"_{self.spectra[0].config_tag}"
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fpath = os.path.join(save_dir, fname + ".png")
            plt.savefig(fpath)
        if show:
            plt.show()
        plt.close()
