"""
Clean, simple Gibbs sampler implementation.
ONE method, mathematically verified, no legacy code.
"""

import numpy as np
import arviz as az
from scipy.stats import truncnorm
from tqdm import tqdm
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum
import os


class GibbsSamplerClean:
    """
    Clean Gibbs sampler for spectrum estimation.

    Uses the mathematically derived conditional distributions:
        residual_i = y - U·R + u_i·R_i
        precision_i = (1/σ²)||u_i||² + λ
        mean_i = (1/σ²) u_i·residual_i / precision_i
        std_i = 1/√(precision_i)
        R_i ~ TruncatedNormal(mean_i, std_i, lower=0)
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
            raise ValueError(f"Invalid parameters: mu={mu}, sigma={sigma}")
        return truncnorm.rvs(a, b, loc=mu, scale=sigma)

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
        """
        # Get data
        signal = np.array(self.signal_decay.signal_values)
        U = self.model.U_matrix()
        n_dim = U.shape[1]

        # Get configuration
        n_iter = self.config.inference.n_iter
        burn_in = self.config.inference.burn_in
        prior_type = self.model.prior_config.type
        prior_strength = self.model.prior_config.get("strength", 0.0)

        # Determine SNR and sigma
        sampler_snr = getattr(self.config.inference, "sampler_snr", None)
        if sampler_snr is None:
            sampler_snr = getattr(self.config.dataset, "snr", None)

        if sampler_snr is not None:
            sigma = 1.0 / np.sqrt(sampler_snr)  # Standard deviation
        else:
            sigma = 1.0

        precision = 1.0 / (sigma**2)  # Precision (1/variance)

        print(
            f"[Gibbs Clean] SNR={sampler_snr}, σ={sigma:.6f}, prior={prior_type}, λ={prior_strength}"
        )
        print(f"[Gibbs Clean] n_dim={n_dim}, n_iter={n_iter}, burn_in={burn_in}")

        # Initialize R
        init_method = self.config.inference.get("init", "map")
        if init_method == "map":
            R = self.model.map_estimate(signal).copy()
        elif init_method == "random":
            R = np.abs(np.random.randn(n_dim))
        elif init_method == "zeros":
            R = np.ones(n_dim) * 0.1
        else:
            raise ValueError(f"Unknown init method: {init_method}")

        initial_R = R.copy()
        print(f"[Gibbs Clean] Initial R: {R}")

        # Sample
        samples = []
        iterator = range(n_iter)
        if show_progress:
            iterator = tqdm(iterator, desc="Gibbs Sampling", unit="iter")

        for it in iterator:
            # One Gibbs sweep: update each component
            for i in range(n_dim):
                # 1. Compute residual excluding component i
                residual = signal - U @ R + U[:, i] * R[i]

                # 2. Compute conditional precision
                if prior_type == "uniform":
                    prec_i = precision * np.sum(U[:, i] ** 2)
                elif prior_type == "ridge":
                    prec_i = precision * np.sum(U[:, i] ** 2) + prior_strength
                else:
                    raise ValueError(f"Unsupported prior: {prior_type}")

                # 3. Compute conditional mean
                mu_i = (precision * U[:, i] @ residual) / prec_i

                # 4. Compute conditional std
                sigma_i = 1.0 / np.sqrt(prec_i)

                # 5. Sample from truncated normal
                R[i] = self._sample_truncated_normal(mu_i, sigma_i, lower=0.0)

            # Store sample after burn-in
            if it >= burn_in:
                samples.append(R.copy())

        samples = np.array(samples)
        print(f"[Gibbs Clean] Collected {len(samples)} samples after burn-in")
        print(f"[Gibbs Clean] Final R mean: {np.mean(samples, axis=0)}")

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
        diffusivities = self.config.dataset.spectrum_pairs[pair].diff_values
        true_spectrum = self.config.dataset.spectrum_pairs[pair].true_spectrum

        spectrum = DiffusivitySpectrum(
            inference_method="gibbs",
            signal_decay=self.signal_decay,
            diffusivities=diffusivities,
            design_matrix_U=U,
            spectrum_init=initial_R.tolist(),
            spectrum_vector=spectrum_vector.tolist(),
            spectrum_samples=samples.tolist(),
            spectrum_std=spectrum_std.tolist(),
            true_spectrum=true_spectrum,
            inference_data=inference_data_path,
            spectra_id=unique_hash,
            init_method=init_method,
            prior_type=prior_type,
            prior_strength=prior_strength,
            data_snr=getattr(self.config.dataset, "snr", None),
            sampler_snr=sampler_snr,
        )

        return spectrum
