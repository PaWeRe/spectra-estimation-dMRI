# prob_model.py
import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Lasso, ElasticNet
from typing import Optional
from sklearn.linear_model import Lasso
from spectra_estimation_dmri.data.data_models import SignalDecay, SignalDecayDataset


class ProbabilisticModel:
    """
    Probabilistic model for dMRI spectrum estimation.

    Supports different priors:
    - 'uniform': Improper uniform prior over nonnegative orthant
    - 'ridge': Gaussian prior (L2 regularization)
    - 'lasso': Laplace prior (L1 regularization) - MAP only
    """

    def __init__(
        self,
        snr: Optional[float] = None,
        likelihood_config: Optional[dict] = None,
        prior_config: Optional[dict] = None,
        true_spectrum: Optional[np.ndarray] = None,
        b_values: Optional[list] = None,
        diffusivities: Optional[list] = None,
    ):
        self.snr = snr
        self.sigma = 1 / np.sqrt(self.snr) if snr is not None else 1.0
        self.likelihood_config = likelihood_config or {}
        self.prior_config = prior_config
        self.true_spectrum = true_spectrum
        self.b_values = np.array(b_values) if b_values is not None else None
        self.diffusivities = (
            np.array(diffusivities) if diffusivities is not None else None
        )

    def U_matrix(self):
        """Design matrix U where U[i,j] = exp(-b_i * d_j)"""
        if self.b_values is None or self.diffusivities is None:
            raise ValueError("b_values and diffusivities must be set")
        return np.exp(-np.outer(self.b_values, self.diffusivities))

    def simulate_signal(self):
        """Simulate noisy signal from true spectrum"""
        if self.true_spectrum is None:
            raise ValueError("true_spectrum must be set for simulation")

        U = self.U_matrix()
        signal = U @ self.true_spectrum
        noisy_signal = signal + np.random.normal(0, self.sigma, size=signal.shape)
        return noisy_signal

    def log_likelihood(self, signal, spectrum):
        """Log likelihood p(signal | spectrum)"""
        U = self.U_matrix()
        mu = U @ spectrum
        return -0.5 * np.sum(((signal - mu) / self.sigma) ** 2)

    def log_prior(self, spectrum):
        """Log prior p(spectrum)"""
        if self.prior_config.type == "uniform":
            return 0.0 if np.all(spectrum >= 0) else -np.inf
        elif self.prior_config.type == "ridge":
            strength = self.prior_config.get("strength", 1.0)
            return -0.5 * strength * np.sum(spectrum**2)
        elif self.prior_config.type == "lasso":
            strength = self.prior_config.get("strength", 1.0)
            return -strength * np.sum(np.abs(spectrum))
        else:
            raise ValueError(f"Unknown prior type: {self.prior_config['type']}")

    def log_posterior(self, signal, spectrum):
        """Log posterior p(spectrum | signal)"""
        return self.log_likelihood(signal, spectrum) + self.log_prior(spectrum)

    def map_estimate(self, signal):
        """
        Compute MAP estimate for different priors.
        """
        U = self.U_matrix()
        n_dim = U.shape[1]

        if self.prior_config.type == "uniform":
            # MAP = NNLS for uniform prior
            fractions, _ = nnls(U, signal)

        elif self.prior_config["type"] == "ridge":
            # Use ElasticNet with l1_ratio=0 for pure ridge + non-negativity
            # This is more principled than Ridge + projection
            strength = self.prior_config.get("strength", 1.0)

            # ElasticNet with l1_ratio=0 is equivalent to non-negative ridge
            elastic = ElasticNet(
                alpha=strength / (2 * len(signal)),  # Match sklearn scaling
                l1_ratio=0.0,  # Pure L2 penalty (ridge)
                positive=True,  # Non-negativity constraint
                fit_intercept=False,
                max_iter=10000,
                tol=1e-6,
            )
            elastic.fit(U, signal)

            fractions = elastic.coef_

        elif self.prior_config.type == "lasso":
            # MAP for Laplace prior = LASSO
            strength = self.prior_config.get("strength", 1.0)
            lasso = Lasso(
                alpha=strength / (2 * len(signal)),  # sklearn scaling
                positive=True,
                fit_intercept=False,
                max_iter=10000,
                tol=1e-6,
            )
            lasso.fit(U, signal)
            fractions = lasso.coef_

        else:
            raise ValueError(f"Unknown prior type: {self.prior_config['type']}")

        return fractions

    def get_posterior_params(self, signal):
        """
        Get posterior mean and precision matrix for Gibbs sampling.
        Only works for uniform and ridge priors (conjugate cases).
        Returns:
            mean: Posterior mean vector
            precision: Posterior precision matrix (NOT covariance)
        """
        if self.prior_config.type not in ["uniform", "ridge"]:
            raise ValueError(
                f"Posterior parameters not available for {self.prior_config.type} prior"
            )

        U = self.U_matrix()
        n_dim = U.shape[1]

        # Precision matrix and mean vector
        precision = (1.0 / self.sigma**2) * (U.T @ U)  # Sigma inverted
        mean_vec = (1.0 / self.sigma**2) * (U.T @ signal)

        if self.prior_config.type == "ridge":
            # Add prior precision
            strength = self.prior_config.get("strength", 1.0)
            precision += strength * np.eye(n_dim)
            # Prior mean is zero, so no change to mean_vec

        # Diagnostic: print condition number and sigma
        print(f"[DEBUG] self.sigma: {self.sigma}")
        print(f"[DEBUG] Precision matrix condition number: {np.linalg.cond(precision)}")
        print(
            f"[DEBUG] Min Eigenvalue precision matrix: {np.linalg.eigvalsh(precision)}"
        )

        mean = np.linalg.solve(precision, mean_vec)
        return mean, precision

    def supports_gibbs_sampling(self):
        """Check if Gibbs sampling is supported for this prior"""
        return self.prior_config.type in ["uniform", "ridge"]

    def simulate_signal_decay_dataset(self):
        """Simulate a SignalDecayDataset for a given true spectrum."""
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
            true_spectrum=(
                self.true_spectrum.tolist() if self.true_spectrum is not None else None
            ),
        )
        return SignalDecayDataset(samples=[signal_decay])

    def pre_inference_diagnostics(self, signal_decay, cfg, wandb_run=None):
        """
        Compute and log condition numbers of U and precision matrix before inference.
        Returns True if both are below thresholds in cfg.diagnostics, else False.
        """
        # Use get_posterior_params to get mean and precision
        try:
            mean, precision = self.get_posterior_params(signal_decay.signal_values)
        except Exception as e:
            print(f"[ERROR] Could not compute posterior params: {e}")
            return False
        U = self.U_matrix()
        cond_U = np.linalg.cond(U)
        cond_precision = np.linalg.cond(precision)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "cond_U": cond_U,
                    "cond_precision": cond_precision,
                }
            )
        cond_ok = (
            cond_U < cfg.diagnostics.max_cond_U
            and cond_precision < cfg.diagnostics.max_cond_precision
        )
        if not cond_ok:
            print(
                f"[WARNING] Ill-conditioned: cond_U={cond_U}, cond_precision={cond_precision}"
            )
        return cond_ok
