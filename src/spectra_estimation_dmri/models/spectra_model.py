import numpy as np
from scipy.optimize import nnls
from typing import Optional


class SpectrumModel:
    def __init__(
        self,
        diffusivities,
        b_values,
        snr: Optional[float] = None,
        sigma: Optional[float] = None,
        prior_type: Optional[str] = None,
        prior_params: Optional[dict] = None,
    ):
        self.diffusivities = np.array(diffusivities)
        self.b_values = np.array(b_values)
        self.snr = snr
        self.sigma = (
            sigma if sigma is not None else (1.0 / snr if snr is not None else None)
        )
        self.prior_type = prior_type
        self.prior_params = prior_params

    def U_matrix(self, b_values=None, diffusivities=None):
        b = np.array(b_values) if b_values is not None else self.b_values
        d = np.array(diffusivities) if diffusivities is not None else self.diffusivities
        return np.exp(-np.outer(b, d))

    def simulate_signal(self, true_spectrum, snr=None, sigma=None, b_values=None):
        U = self.U_matrix(b_values=b_values)
        signal = U @ true_spectrum
        sigma_ = (
            sigma
            if sigma is not None
            else (1.0 / snr if snr is not None else self.sigma)
        )
        noisy_signal = signal + np.random.normal(0, sigma_, size=signal.shape)
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

    def map_estimate(self, signal, regularization=0.0, b_values=None):
        """
        Compute the MAP/NNLS estimate (optionally L2-regularized).
        Returns: estimated spectrum (fractions)
        """
        U = self.U_matrix(b_values=b_values)
        if regularization > 0:
            # Solve (U^T U + lambda*I) x = U^T s, x >= 0
            n = U.shape[1]
            A = np.vstack([U, np.sqrt(regularization) * np.eye(n)])
            b = np.concatenate([signal, np.zeros(n)])
            fractions, _ = nnls(A, b)
        else:
            fractions, _ = nnls(U, signal)
        return fractions

    def to_diffusivity_spectrum(
        self,
        inference_method: str,
        signal_decay,
        fractions_mode,
        fractions_mean=None,
        fractions_variance=None,
        estimated_snr=None,
        true_fractions=None,
        noise_realizations=1,
        hdf5_dataset="",
    ):
        from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum

        return DiffusivitySpectrum(
            inference_method=inference_method,
            signal_decay=signal_decay,
            diffusivities=self.diffusivities.tolist(),
            design_matrix_U=self.U_matrix().tolist(),
            fractions_mode=fractions_mode.tolist(),
            fractions_mean=(
                fractions_mean.tolist() if fractions_mean is not None else []
            ),
            fractions_variance=(
                fractions_variance.tolist() if fractions_variance is not None else []
            ),
            estimated_snr=estimated_snr,
            true_fractions=(
                true_fractions.tolist() if true_fractions is not None else None
            ),
            noise_realizations=noise_realizations,
            hdf5_dataset=hdf5_dataset,
        )

    def compute_U(self):
        return -np.exp(np.outer(self.signal_decay.b_values, self.diffusivities))

    def compute_R_mode(self, regularizer):
        pass
