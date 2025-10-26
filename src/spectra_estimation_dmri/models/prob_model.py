# prob_model.py
import numpy as np
import numpy.random as npr
from scipy.optimize import nnls
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, Ridge, RidgeCV
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
        data_snr: Optional[float] = None,
        likelihood_config: Optional[dict] = None,
        prior_config: Optional[dict] = None,
        true_spectrum: Optional[np.ndarray] = None,
        b_values: Optional[list] = None,
        diffusivities: Optional[list] = None,
        no_noise: bool = False,
    ):
        self.data_snr = data_snr
        # For real data (BWH), data_snr is None (unknown)
        self.data_sigma = 1 / self.data_snr if self.data_snr is not None else None
        self.likelihood_config = likelihood_config or {}
        self.prior_config = prior_config
        self.true_spectrum = true_spectrum
        self.b_values = np.array(b_values) if b_values is not None else None
        self.diffusivities = (
            np.array(diffusivities) if diffusivities is not None else None
        )
        self.no_noise = no_noise

    def U_matrix(self):
        """Design matrix U where U[i,j] = exp(-b_i * d_j)"""
        if self.b_values is None or self.diffusivities is None:
            raise ValueError("b_values and diffusivities must be set")
        return np.exp(-np.outer(self.b_values, self.diffusivities))

    def simulate_signal(self):
        """Simulate signal from true spectrum (with or without noise)"""
        if self.true_spectrum is None:
            raise ValueError("true_spectrum must be set for simulation")

        U = self.U_matrix()
        signal = U @ self.true_spectrum

        if self.no_noise:
            return signal  # Return noiseless signal
        else:
            noisy_signal = signal + npr.normal(0, self.data_sigma, size=signal.shape)
            return noisy_signal

    def log_likelihood(self, signal, spectrum):
        """Log likelihood p(signal | spectrum)"""
        U = self.U_matrix()
        mu = U @ spectrum
        return -0.5 * np.sum(((signal - mu) / self.data_sigma) ** 2)

    def log_prior(self, spectrum):
        """Log prior p(spectrum)"""
        if self.prior_config.type == "uniform":
            return 0.0 if np.all(spectrum >= 0) else -np.inf
        elif self.prior_config.type == "ridge":
            strength = self.prior_config.strength
            return -0.5 * strength * np.sum(spectrum**2)
        elif self.prior_config.type == "lasso":
            strength = self.prior_config.strength
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
            # fractions, _ = nnls(U, signal)
            reg_nnls = LinearRegression(positive=True)
            fit_model = reg_nnls.fit(U, signal)
            fractions = fit_model.coef_

        elif self.prior_config["type"] == "ridge":
            # Use sklearn's Ridge class for better convergence
            strength = self.prior_config.strength

            # Check if we should use RidgeCV for automatic hyperparameter selection
            if strength == 0:
                # Use RidgeCV to automatically select best alpha
                alphas = np.logspace(-10, 2, 20)  # Log-spaced alpha values
                ridge_cv = RidgeCV(
                    alphas=alphas,
                    fit_intercept=False,
                    cv=3,  # 3-fold cross-validation
                    scoring="neg_mean_squared_error",
                )
                ridge_cv.fit(U, signal)
                fractions = ridge_cv.coef_
                self.prior_config.strength = str(ridge_cv.alpha_)
                print(f"RidgeCV selected alpha: {ridge_cv.alpha_}")
            else:
                print(f"Ridge strength: {strength}")
                # Use fixed Ridge with specified strength
                ridge = Ridge(
                    alpha=float(strength),  # Direct strength parameter
                    fit_intercept=False,
                    max_iter=10000,
                    tol=1e-6,  # More reasonable tolerance
                    solver="auto",  # Let sklearn choose best solver
                )
                ridge.fit(U, signal)
                fractions = ridge.coef_

            # Apply non-negativity constraint by projection
            fractions = np.maximum(fractions, 0)

        elif self.prior_config.type == "lasso":
            # MAP for Laplace prior = LASSO
            strength = self.prior_config.strength
            lasso = Lasso(
                alpha=strength / (2 * len(signal)),  # sklearn scaling
                positive=True,
                fit_intercept=False,
                max_iter=10000,
                tol=1e-6,
            )
            lasso.fit(U, signal)
            fractions = lasso.coef_

        elif self.prior_config.type == "cvxopt":
            pass
            # REF: sandy way of getting mode (involves sampler_snr, the other ones I have don't...why?)
            # TODO: adapt code and see if it outperforms other modes
            # from cvxopt import matrix, solvers
            # M_count = signal_data.signal_values.shape[0]
            # N = diffusivities.shape[0]
            # u_vec_tuple = ()
            # for i in range(M_count):
            #     u_vec_tuple += (np.exp((-signal_data.b_values[i] * diffusivities)),)
            # U_vecs = np.vstack(u_vec_tuple)
            # U_outer_prods = np.zeros((N, N))
            # for i in range(M_count):
            #     U_outer_prods += np.outer(U_vecs[i], U_vecs[i])
            # Sigma_inverse = (1.0 / (sigma * sigma)) * U_outer_prods
            # if L2_lambda > 0.0:
            #     inverse_prior_covariance = L2_lambda * np.eye(N)
            # if inverse_prior_covariance is not None:
            #     Sigma_inverse += inverse_prior_covariance
            # weighted_U_vecs = np.zeros(N)
            # for i in range(M_count):
            #     weighted_U_vecs += signal_data.signal_values[i] * U_vecs[i]
            # One_vec = np.ones(N)
            # Sigma_inverse_M = (
            #     1.0 / (sigma * sigma) * weighted_U_vecs
            # ) - L1_lambda * One_vec
            # M = np.linalg.solve(Sigma_inverse, Sigma_inverse_M)
            # P = matrix(Sigma_inverse)
            # Q = matrix(-Sigma_inverse_M)
            # G = matrix(-np.identity(N), tc="d")
            # H = matrix(np.zeros(N))
            # sol = solvers.qp(P, Q, G, H)
            # mode = np.array(sol["x"]).T[0]

        else:
            raise ValueError(f"Unknown prior type: {self.prior_config['type']}")

        return fractions

    def get_posterior_params(self, sampler_sigma, signal):
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
        precision = (1.0 / sampler_sigma**2) * (U.T @ U)  # Sigma inverted
        mean_vec = (1.0 / sampler_sigma**2) * (U.T @ signal)

        if self.prior_config.type == "ridge":
            # Add prior precision
            strength = self.prior_config.get("strength", 1.0)
            print(f"strength:{strength}")
            precision += float(strength) * np.eye(n_dim)
            # Prior mean is zero, so no change to mean_vec

        # Diagnostic: print condition number and sigma
        print(f"[DEBUG] sampler_sigma: {sampler_sigma}")
        print(f"[DEBUG] data_sigma: {self.data_sigma}")
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
            snr=self.data_snr,
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
        U = self.U_matrix()
        cond_U = np.linalg.cond(U)

        # For conjugate priors (uniform, ridge), also check precision matrix
        if self.prior_config.type in ["uniform", "ridge"]:
            try:
                # Get sampler_snr from inference config, fall back to dataset snr if not available
                sampler_snr = getattr(cfg.inference, "sampler_snr", None)
                if sampler_snr is None:
                    sampler_snr = getattr(cfg.dataset, "snr", None)

                if sampler_snr is None:
                    print(
                        "[WARNING] No SNR available for diagnostics, skipping precision matrix check"
                    )
                    cond_ok = cond_U < cfg.diagnostics.max_cond_U
                    if not cond_ok:
                        print(f"[WARNING] Ill-conditioned: cond_U={cond_U}")
                    return cond_ok

                sampler_sigma = 1 / sampler_snr
                mean, precision = self.get_posterior_params(
                    sampler_sigma, signal_decay.signal_values
                )
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

            except Exception as e:
                print(f"[ERROR] Could not compute posterior params: {e}")
                return False

        else:
            # For non-conjugate priors (lasso), only check U matrix condition
            print(
                f"[INFO] Non-conjugate prior ({self.prior_config.type}): only checking U matrix condition"
            )

            if wandb_run is not None:
                wandb_run.log({"cond_U": cond_U})

            cond_ok = cond_U < cfg.diagnostics.max_cond_U
            if not cond_ok:
                print(f"[WARNING] Ill-conditioned: cond_U={cond_U}")

        return cond_ok
