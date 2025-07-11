# gibbs_sampler.py
import numpy as np
import arviz as az
from scipy.stats import truncnorm
from tqdm import tqdm
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum
import os


class GibbsSampler:
    """
    Gibbs sampler for spectrum estimation.

    Supports:
    - uniform prior: Truncated multivariate normal posterior
    - ridge prior: Truncated multivariate normal posterior
    - lasso prior: Not supported (non-conjugate)
    """

    def __init__(self, model, signal_decay, config):
        self.model = model
        self.signal_decay = signal_decay
        self.config = config

        # Check if Gibbs sampling is supported
        if not self.model.supports_gibbs_sampling():
            raise ValueError(
                f"Gibbs sampling not supported for prior type: {self.model.prior_config['type']}"
            )

    def _sample_truncated_normal(self, mu, sigma, lower=0.0, upper=np.inf):
        """Sample from truncated normal distribution"""
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
            print(f"[ERROR] Bad mu or sigma: mu={mu}, sigma={sigma}")
            raise ValueError("Non-finite or non-positive sigma in truncated normal")
        return truncnorm.rvs(a, b, loc=mu, scale=sigma)

    def _sample_normal_non_neg(self, mu, sigma):
        import numpy.random as npr

        if mu >= 0:
            while True:
                print("1")
                u = npr.normal(mu, sigma)
                if u > 0.0:
                    return u
        else:
            sigma_minus = -mu / sigma
            alpha_star = (sigma_minus + np.sqrt(sigma_minus**2 + 4)) / 2.0
            while True:
                print("2")
                x = npr.exponential(1.0 / alpha_star)
                z = sigma_minus + x
                eta = np.exp(-((z - alpha_star) ** 2) / 2.0)
                u = npr.uniform()
                if u <= eta:
                    return mu + sigma * z

    def _gibbs_step(self, R, signal, posterior_mean, posterior_precision, prior_type):
        """Perform one Gibbs step using the precision matrix only"""
        n_dim = len(R)

        for i in range(n_dim):
            # Conditional variance and mean from precision matrix
            precision_ii = posterior_precision[i, i]
            if precision_ii <= 0 or not np.isfinite(precision_ii):
                print(f"[ERROR] Bad precision_ii: {precision_ii} at index {i}")
                raise ValueError(
                    f"Non-finite or non-positive precision_ii at index {i}"
                )
            sigma_i = 1.0 / np.sqrt(precision_ii)

            # Off-diagonal elements for conditional mean (remove i-th element)
            precision_i_rest = np.delete(posterior_precision[i, :], i)
            M_slash_i = np.delete(posterior_mean, i)
            R_slash_i = np.delete(R, i)
            dot_prod = np.dot(precision_i_rest / precision_ii, (M_slash_i - R_slash_i))
            mu_i = posterior_mean[i] + dot_prod
            # Sample from conditional distribution
            if prior_type in ["uniform", "ridge"]:
                R[i] = self._sample_normal_non_neg(mu_i, sigma_i)
                # R[i] = self._sample_truncated_normal(mu_i, sigma_i, lower=0.0)
            else:
                raise ValueError(f"Unsupported prior type: {prior_type}")

        return R

    def _gibbs_step_elementwise(
        self, R, signal, U, sigma, prior_type, prior_strength=0.0
    ):
        """
        More efficient element-wise Gibbs sampling.
        Avoids matrix inversion at each step.
        """
        n_dim = len(R)
        precision = 1.0 / (sigma**2)

        for i in range(n_dim):
            # Residual excluding component i
            residual = signal - U @ R + U[:, i] * R[i]

            # Conditional precision and mean
            if prior_type == "uniform":
                prec_i = precision * np.sum(U[:, i] ** 2)
                mu_i = (precision * U[:, i] @ residual) / prec_i
            elif prior_type == "ridge":
                prec_i = precision * np.sum(U[:, i] ** 2) + prior_strength
                mu_i = (precision * U[:, i] @ residual) / prec_i
            else:
                raise ValueError(f"Unsupported prior type: {prior_type}")

            sigma_i = 1.0 / np.sqrt(prec_i)

            # Sample from truncated normal
            R[i] = self._sample_truncated_normal(mu_i, sigma_i, lower=0.0)

        return R

    def run(
        self,
        return_idata=True,
        show_progress=False,
        save_dir=None,
        true_spectrum=None,
        unique_hash=None,
    ):
        """
        Run Gibbs sampling for spectrum estimation.

        Returns:
            DiffusivitySpectrum object
        """
        signal = np.array(self.signal_decay.signal_values)
        U = self.model.U_matrix()
        prior_type = self.model.prior_config.type
        prior_strength = self.model.prior_config.get("strength", 0.0)

        n_iter = self.config.inference.n_iter
        burn_in = self.config.inference.burn_in
        n_dim = U.shape[1]

        # Initialize R
        init_method = self.config.inference.get("init", "map")
        if init_method == "map":
            R = self.model.map_estimate(signal).copy()
        elif init_method == "random":
            R = np.abs(np.random.randn(n_dim))
        elif init_method == "zeros":
            R = np.ones(n_dim) * 0.1  # Small positive values
        else:
            raise ValueError(f"Unknown init method: {init_method}")

        # Get posterior parameters (for full precision approach)
        try:
            posterior_mean, posterior_precision = self.model.get_posterior_params(
                signal
            )
            use_full_precision = True
        except:
            use_full_precision = False

        # Sample
        samples = []
        iterator = range(n_iter)
        if show_progress:
            iterator = tqdm(iterator, desc="Gibbs Sampling", unit="iter")

        for it in iterator:
            # Choose sampling method
            if (
                use_full_precision and n_dim < 100
            ):  # Use full precision for small problems
                R = self._gibbs_step(
                    R, signal, posterior_mean, posterior_precision, prior_type
                )
            else:  # Use element-wise updates for efficiency
                R = self._gibbs_step_elementwise(
                    R, signal, U, self.model.sigma, prior_type, prior_strength
                )

            # Store sample after burn-in
            if it >= burn_in:
                samples.append(R.copy())

        samples = np.array(samples)

        # Create inference data
        idata = None
        inference_data_path = None
        if return_idata:
            idata = az.from_dict(
                posterior={"R": samples[None, :, :]}  # Add chain dimension
            )
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{unique_hash}.nc"
                inference_data_path = os.path.join(save_dir, fname)
                idata.to_netcdf(inference_data_path)

        # Summary statistics
        spectrum_vector = np.mean(samples, axis=0)
        spectrum_std = np.std(samples, axis=0)

        # Create result object
        pair = self.config.dataset.spectrum_pair
        diffusvities = self.config.dataset.spectrum_pairs[pair].diff_values
        true_spectrum = self.config.dataset.spectrum_pairs[pair].true_spectrum
        spectrum = DiffusivitySpectrum(
            inference_method="gibbs",
            signal_decay=self.signal_decay,
            diffusivities=diffusvities,
            design_matrix_U=U,
            spectrum_vector=spectrum_vector.tolist(),
            spectrum_samples=samples.tolist(),
            spectrum_std=spectrum_std.tolist(),
            true_spectrum=true_spectrum,
            inference_data=inference_data_path,
            spectra_id=unique_hash,
            init_method=init_method,
            prior_type=prior_type,
            prior_strength=self.model.prior_config.strength,
        )

        return spectrum
