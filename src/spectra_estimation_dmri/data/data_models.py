from typing import Optional, Literal
from pydantic import BaseModel, computed_field, field_validator
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


class SignalDecay(BaseModel):
    patient: str  # TODO: add validator with unique patient ids
    signal_values: list[float]
    b_values: list[float]
    voxel_count: int  # alias for v_count in JSON
    a_region: Literal["pz", "tz"]
    is_tumor: bool = False
    ggg: Optional[int] = None
    gs: Optional[str] = None  # TODO: add validator for strings e.g. 3+4 or 3+4+5

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

    # TODO: implement
    def plot(self):
        pass


class DiffusivitySpectrum(BaseModel):
    inference_method: str
    signal_decay: SignalDecay
    # TODO: think of usage for finding opt diff discr for design matrix U (List of Lists?)
    diffusivities: list[float]
    design_matrix_U: list[list[float]]
    # TODO: think of usage during simulation (List of Lists for stability analysis?)
    fractions_mode: list[float]  # posterior map estimate from reg. NNLS
    fractions_mean: list[float]  # bayesian posterior mean from approx. inf.
    fractions_variance: list[float]  # uncertainty / spread of posterior
    estimated_snr: Optional[
        float
    ]  # TODO: check typing to work for simulation (float) and on real data (None)
    true_fractions: Optional[list[float]]  # TODO: for simulation purposes optional
    noise_realizations: int = 1
    hdf5_dataset: str
    # TODO: do I still need the raw samples for downstream analysis? If so add "hdf5_dataset" attribute!

    def as_numpy(self):
        """Return all list fields as numpy arrays."""
        return {
            "diffusivities": np.array(self.diffusivities),
            "design_matrix_U": np.array(self.design_matrix_U),
            "fractions_mode": np.array(self.fractions_mode),
            "fractions_mean": np.array(self.fractions_mean),
            "fractions_variance": np.array(self.fractions_variance),
            "true_fractions": (
                np.array(self.true_fractions)
                if self.true_fractions is not None
                else None
            ),
        }

    # TODO: implement
    def plot(self):
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

    def plot_avg_by_region(self):
        pass

    def as_numpy(self):
        return [s.as_numpy() for s in self.spectra]
