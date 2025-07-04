import numpy as np
from scipy.optimize import nnls
from typing import Optional
from sklearn.linear_model import Lasso
from spectra_estimation_dmri.data.data_models import SignalDecay, SignalDecayDataset


class SpectrumModel:
    def __init__(
        self,
        diffusivities,
        b_values,
        snr: Optional[float] = None,
        sigma: Optional[float] = None,
        prior_type: Optional[str] = None,
        prior_params: Optional[dict] = None,
        true_spectrum: Optional[float] = None,
    ):
        self.diffusivities = np.array(diffusivities)
        self.b_values = np.array(b_values)
        self.snr = snr
        self.sigma = (
            sigma if sigma is not None else (1.0 / snr if snr is not None else None)
        )
        self.prior_type = prior_type
        self.prior_params = prior_params
        self.true_spectrum = true_spectrum

    def U_matrix(self, b_values=None, diffusivities=None):
        b = np.array(b_values) if b_values is not None else self.b_values
        d = np.array(diffusivities) if diffusivities is not None else self.diffusivities
        return np.exp(-np.outer(b, d))

    def simulate_signal(self):
        U = self.U_matrix(b_values=self.b_values)
        signal = U @ self.true_spectrum
        # TODO: check conversion of sigma and SNR again (before took sigma = 1/SNR but seems to sqrt?)
        sigma = 1 / np.sqrt(self.snr)
        noisy_signal = signal + np.random.normal(0, sigma, size=signal.shape)
        return noisy_signal

    def log_likelihood(self, signal, spectrum, snr=None, sigma=None, b_values=None):
        U = self.U_matrix(b_values=b_values)
        mu = U @ spectrum
        sigma_ = (
            sigma
            if sigma is not None
            else (1.0 / snr if snr is not None else self.sigma)
        )
        return -0.5 * np.sum(((signal - mu) / sigma_) ** 2)

    def log_prior(self, spectrum, prior_type="uniform", prior_params=None):
        if prior_type == "uniform":
            # Uniform prior: constant if all elements >= 0
            return 0.0 if np.all(spectrum >= 0) else -np.inf
        elif prior_type == "l2":
            # Gaussian prior: -0.5 * lambda * ||spectrum||^2
            l2_lambda = (
                prior_params["l2_lambda"]
                if prior_params and "l2_lambda" in prior_params
                else 1.0
            )
            return -0.5 * l2_lambda * np.sum(spectrum**2)
        elif prior_type == "l1":
            # Laplace prior: -lambda * ||spectrum||_1
            l1_lambda = (
                prior_params["l1_lambda"]
                if prior_params and "l1_lambda" in prior_params
                else 1.0
            )
            return -l1_lambda * np.sum(np.abs(spectrum))
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")

    def map_estimate(
        self,
        signal,
        prior_type="uniform",
        regularization=0.0,
        b_values=None,
        prior_params=None,
    ):
        """
        Compute the MAP estimate for the given prior.
        prior_type: 'uniform' (NNLS), 'l2' (ridge), 'l1' (lasso)
        regularization: lambda for l2 or l1
        Returns: estimated spectrum (fractions)
        """
        U = self.U_matrix(b_values=b_values)
        if prior_type == "uniform":
            # NNLS (non-negative)
            fractions, _ = nnls(U, signal)
        elif prior_type == "l2":
            # Ridge regression with non-negativity (NNLS with L2)
            n = U.shape[1]
            A = np.vstack([U, np.sqrt(regularization) * np.eye(n)])
            b = np.concatenate([signal, np.zeros(n)])
            fractions, _ = nnls(A, b)
        elif prior_type == "l1":
            # Lasso regression (non-negative)
            lasso = Lasso(
                alpha=regularization, positive=True, fit_intercept=False, max_iter=10000
            )
            lasso.fit(U, signal)
            fractions = lasso.coef_
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")
        return fractions

    # TODO: modify function to simulate_signal_decay_dataset to return SignalDecayDataset (for consistency in main.py)
    def simulate_signal_decay_dataset(self):
        """
        Simulate a SignalDecayDataset for a given true spectrum.
        Returns a SignalDecayDataset with one SignalDecay object.
        """
        noisy_signal = self.simulate_signal()
        signal_decay = SignalDecay(
            patient="simulated",
            signal_values=noisy_signal.tolist(),
            b_values=self.b_values.tolist(),
            snr=self.snr,
            voxel_count=None,
            a_region="sim",
            is_tumor=False,
            ggg=None,
            gs=None,
            true_spectrum=self.true_spectrum,
        )
        return SignalDecayDataset(samples=[signal_decay])
