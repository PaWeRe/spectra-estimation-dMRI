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

        # Ensure burn_in doesn't exceed n_iter
        if burn_in >= n_iter:
            original_burn_in = burn_in
            burn_in = max(0, n_iter // 5)  # Use 20% of iterations as burn-in
            print(
                f"[WARNING] burn_in ({original_burn_in}) >= n_iter ({n_iter}). "
                f"Setting burn_in to {burn_in} (20% of n_iter)"
            )

        n_chains = getattr(self.config.inference, "n_chains", 4)  # Default: 4 chains
        prior_type = self.model.prior_config.type
        prior_strength = self.model.prior_config.get("strength", 0.0)
        random_seed = getattr(self.config, "seed", 42)

        # Determine SNR and sigma
        sampler_snr = getattr(self.config.inference, "sampler_snr", None)
        if sampler_snr is None:
            sampler_snr = getattr(self.config.dataset, "snr", None)

        if sampler_snr is not None:
            sigma = (
                1.0 / sampler_snr
            )  # Standard deviation (consistent with data generation)
        else:
            sigma = 1.0

        precision = 1.0 / (sigma**2)  # Precision (1/variance)

        print(
            f"[Gibbs Clean] SNR={sampler_snr}, σ={sigma:.6f}, prior={prior_type}, λ={prior_strength}"
        )
        print(
            f"[Gibbs Clean] n_dim={n_dim}, n_iter={n_iter}, burn_in={burn_in}, n_chains={n_chains}"
        )

        # Get initial R (shared across chains)
        init_method = self.config.inference.get("init", "map")
        if init_method == "map":
            R_init = self.model.map_estimate(signal).copy()
        elif init_method == "random":
            R_init = np.abs(np.random.randn(n_dim))
        elif init_method == "zeros":
            R_init = np.ones(n_dim) * 0.1
        else:
            raise ValueError(f"Unknown init method: {init_method}")

        initial_R = R_init.copy()
        print(f"[Gibbs Clean] Initial R: {R_init}")

        # Run multiple chains
        all_chains = []
        for chain_id in range(n_chains):
            # Set random seed for reproducibility (different for each chain)
            np.random.seed(random_seed + chain_id)

            # Initialize this chain
            R = R_init.copy()

            # Add small random perturbation to break symmetry between chains
            if chain_id > 0:  # Keep chain 0 at exact MAP
                R += np.random.normal(0, 0.01, size=n_dim)
                R = np.maximum(R, 0)  # Ensure non-negative
                R = R / np.sum(R) if np.sum(R) > 0 else R_init

            # Run Gibbs sampling for this chain
            samples = []

            # Create fresh iterator for this chain
            if show_progress and chain_id == 0:
                iterator = tqdm(
                    range(n_iter),
                    desc=f"Gibbs Chain {chain_id+1}/{n_chains}",
                    unit="iter",
                )
            else:
                iterator = range(n_iter)

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

            all_chains.append(np.array(samples))
            if not show_progress or chain_id > 0:
                print(
                    f"[Gibbs Clean] Chain {chain_id+1}/{n_chains}: {len(samples)} samples collected"
                )

        # Stack chains: (n_chains, n_iterations, n_dim)
        all_chains = np.stack(all_chains, axis=0)
        print(
            f"[Gibbs Clean] Combined shape: {all_chains.shape} (chains, iterations, dimensions)"
        )
        print(
            f"[Gibbs Clean] Final R mean (across all chains): {np.mean(all_chains, axis=(0,1))}"
        )

        # Create inference data with proper chain structure
        # Create separate variables for each diffusivity (matches NUTS format)
        idata = None
        inference_data_path = None
        if return_idata:
            # all_chains shape: (n_chains, n_iterations, n_dim)
            # Create variable names for each diffusivity
            pair = self.config.dataset.spectrum_pair
            diffusivities = self.config.dataset.spectrum_pairs[pair].diff_values
            var_names = [f"diff_{diff:.2f}" for diff in diffusivities]

            # Create posterior dict with separate variables
            posterior_data = {}
            for i, var_name in enumerate(var_names):
                # Extract data for this diffusivity across all chains
                posterior_data[var_name] = all_chains[
                    :, :, i
                ]  # (n_chains, n_iterations)

            idata = az.from_dict(posterior=posterior_data)

            # Print convergence diagnostics
            print("\n[Gibbs Clean] Convergence Diagnostics:")
            summary = az.summary(idata)
            max_rhat = summary["r_hat"].max()
            min_ess_bulk = summary["ess_bulk"].min()
            min_ess_tail = summary["ess_tail"].min()
            print(f"  Max R-hat: {max_rhat:.4f}")
            print(f"  Min ESS_bulk: {min_ess_bulk:.0f}")
            print(f"  Min ESS_tail: {min_ess_tail:.0f}")

            if max_rhat < 1.05:
                print("  ✓ CONVERGED (R-hat < 1.05)")
            elif max_rhat < 1.1:
                print("  ⚠ CHECK CONVERGENCE (1.05 ≤ R-hat < 1.1)")
            else:
                print("  ✗ NOT CONVERGED (R-hat ≥ 1.1)")

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{unique_hash}.nc"
                inference_data_path = os.path.join(save_dir, fname)
                idata.to_netcdf(inference_data_path)

        # Summary statistics (flatten across all chains for compatibility)
        # For downstream analysis, we combine chains
        samples_flat = all_chains.reshape(-1, n_dim)  # (n_chains * n_iterations, n_dim)
        spectrum_vector = np.mean(samples_flat, axis=0)
        spectrum_std = np.std(samples_flat, axis=0)

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
            spectrum_samples=samples_flat.tolist(),  # Flattened across chains for compatibility
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
