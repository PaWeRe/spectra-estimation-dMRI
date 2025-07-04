from typing import Optional, Literal, List
from pydantic import BaseModel, computed_field, field_validator, root_validator
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
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
    spectrum_vector: list[float]
    spectrum_samples: Optional[list[list[float]]] = None
    spectrum_std: Optional[list[float]] = None
    true_spectrum: Optional[list[float]] = None
    inference_data: str
    spectra_id: Optional[str] = None

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
        """Save an index (dict) mapping config hashes and sample IDs to file paths as JSON."""
        with open(index_path, "w") as f:
            json.dump(index_dict, f, indent=2)

    @staticmethod
    def load_index(index_path):
        """Load an index (dict) from JSON file."""
        with open(index_path, "r") as f:
            return json.load(f)

    def plot_group_boxplot(
        self, group_key=None, save_dir=None, config_info=None, group_by=None, show=True
    ):
        """
        Plot spectra grouped by user-specified attributes (e.g., SNR, region, true_spectrum, noise_realizations).
        For each group (tuple of group_by values), aggregate all spectra and plot them together, with a detailed title and filename.
        The plot title and filename will include all group_by parameters for reproducibility.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from collections import defaultdict

        # # --- 1. Parse group_by as a list ---
        # if group_by is None:
        #     group_by = group_key
        # if group_by is None:
        #     raise ValueError(
        #         "You must specify a group_by attribute for grouping plots."
        #     )
        # # Robustly handle both string and list input
        # if isinstance(group_by, str):
        #     # Remove brackets if present and split by comma
        #     group_by = group_by.strip().strip("[]")
        #     group_by = [
        #         g.strip().strip("'\"") for g in group_by.split(",") if g.strip()
        #     ]
        # if not isinstance(group_by, (list, tuple)):
        #     raise ValueError("group_by must be a list or tuple of attribute names.")

        # --- 2. Group spectra by tuple of group_by values ---
        group_dict = defaultdict(list)
        for spec in self.spectra:
            group_vals = []
            for key in group_by:
                # Try to get the attribute from signal_decay, then from spec, then from config_info
                value = None
                if hasattr(spec.signal_decay, key):
                    value = getattr(spec.signal_decay, key)
                elif hasattr(spec, key):
                    value = getattr(spec, key)
                elif config_info and key in config_info:
                    value = config_info[key]
                group_vals.append(value)
            group_dict[tuple(group_vals)].append(spec)

        # --- 3. For each group, plot all inference methods side by side ---
        for group_vals, group_spectra in group_dict.items():
            # Collect all unique inference methods
            methods = sorted(set(spec.inference_method for spec in group_spectra))
            d = np.array(group_spectra[0].diffusivities)
            n_bins = len(d)
            # For each method, collect all spectra (one per noise realization)
            method_to_vectors = defaultdict(list)
            method_to_samples = defaultdict(list)
            ground_truths = []
            for spec in group_spectra:
                if spec.spectrum_samples is not None:
                    method_to_samples[spec.inference_method].append(
                        np.array(spec.spectrum_samples)
                    )
                else:
                    method_to_vectors[spec.inference_method].append(
                        np.array(spec.spectrum_vector)
                    )
                if spec.true_spectrum is not None:
                    ground_truths.append(np.array(spec.true_spectrum))

            # --- 4. Plotting ---
            fig, ax = plt.subplots(figsize=(12, 6))
            boxplot_positions = np.arange(n_bins)
            width = 0.18  # width of each boxplot group
            colors = ["#4F81BD", "#C0504D", "#9BBB59", "#8064A2", "#F79646", "#2C4D75"]
            method_labels = {"map": "Initial R (Mode)", "gibbs": "Final R (Mean)"}
            # Plot boxplots for each method, offset horizontally
            n_methods = len(methods)
            for idx, method in enumerate(methods):
                color = colors[idx % len(colors)]
                offset = (idx - (n_methods - 1) / 2) * width
                # Gather all realizations for this method
                if method in method_to_samples:
                    # Stack all samples from all realizations
                    all_samples = np.concatenate(method_to_samples[method], axis=0)
                    bp = ax.boxplot(
                        all_samples,
                        positions=boxplot_positions + offset,
                        widths=width * 0.9,
                        patch_artist=True,
                        showmeans=True,
                        meanline=True,
                        boxprops=dict(facecolor=color, alpha=0.3),
                        medianprops=dict(color=color),
                        meanprops=dict(color=color, linewidth=2),
                        flierprops=dict(markerfacecolor=color, marker="o", alpha=0.2),
                        manage_ticks=False,
                    )
                elif method in method_to_vectors:
                    # Stack all vectors (one per realization)
                    all_vectors = np.stack(method_to_vectors[method], axis=0)
                    bp = ax.boxplot(
                        all_vectors,
                        positions=boxplot_positions + offset,
                        widths=width * 0.9,
                        patch_artist=True,
                        showmeans=True,
                        meanline=True,
                        boxprops=dict(facecolor=color, alpha=0.3),
                        medianprops=dict(color=color),
                        meanprops=dict(color=color, linewidth=2),
                        flierprops=dict(markerfacecolor=color, marker="o", alpha=0.2),
                        manage_ticks=False,
                    )
                # Add legend entry
                label = method_labels.get(method, method.capitalize())
                bp["boxes"][0].set_label(label)
            # Overlay the true spectrum as a red bar (as in your example)
            if ground_truths:
                # Use the first true spectrum (should be the same for all in group)
                true_spec = ground_truths[0]
                ax.bar(
                    boxplot_positions,
                    true_spec,
                    width=width * n_methods * 0.9,
                    color="red",
                    alpha=0.3,
                    label="True Spectrum",
                    zorder=0,
                )
            # X labels and ticks
            ax.set_xticks(boxplot_positions)
            ax.set_xticklabels([f"{val:.2g}" for val in d], rotation=45)
            ax.set_xlabel("Diffusivity")
            ax.set_ylabel("Relative Fraction")
            # Build a detailed title and filename
            title_parts = []
            fname_parts = []
            for key, val in zip(group_by, group_vals):
                title_parts.append(f"{key}: {val}")
                fname_parts.append(f"{key}{val}")
            title = " | ".join(title_parts)
            ax.set_title(title)
            ax.legend()
            plt.tight_layout()
            # Build filename
            fname = "_".join(fname_parts)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fpath = os.path.join(save_dir, f"{fname}.pdf")
                plt.savefig(fpath)
            if show:
                plt.show()
            plt.close()
