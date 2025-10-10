"""
NUTS sampler for spectrum estimation using PyMC.

NUTS (No-U-Turn Sampler) is a variant of Hamiltonian Monte Carlo that
handles truncated distributions and complex geometries much better than Gibbs sampling.
"""

import numpy as np
import arviz as az
import os
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum

try:
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


class NUTSSampler:
    """
    NUTS sampler for spectrum estimation.

    Supports:
    - uniform prior: Truncated normal likelihood with flat prior
    - ridge prior: Truncated normal likelihood with Gaussian prior
    - lasso prior: Truncated normal likelihood with Laplace prior

    Advantages over Gibbs:
    - Better exploration of truncated spaces
    - Handles correlations more efficiently
    - Typically faster convergence
    """

    def __init__(self, model, signal_decay, config):
        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC is required for NUTS sampling. Install with: uv add pymc"
            )

        self.model = model
        self.signal_decay = signal_decay
        self.config = config

    def run(
        self,
        return_idata=True,
        show_progress=True,
        save_dir=None,
        true_spectrum=None,
        unique_hash=None,
    ):
        """
        Run NUTS sampling for spectrum estimation.

        Returns:
            DiffusivitySpectrum object
        """
        # Get configuration
        signal = np.array(self.signal_decay.signal_values)
        U = self.model.U_matrix()
        n_dim = U.shape[1]

        # Get inference parameters
        n_iter = self.config.inference.n_iter
        n_chains = getattr(self.config.inference, "n_chains", 4)
        tune_steps = getattr(self.config.inference, "tune", 1000)
        target_accept = getattr(self.config.inference, "target_accept", 0.95)

        # Get sampler SNR (falls back to data SNR)
        sampler_snr = getattr(self.config.inference, "sampler_snr", None)
        if sampler_snr is None:
            sampler_snr = getattr(self.config.dataset, "snr", None)
        if sampler_snr is not None:
            sampler_sigma = 1.0 / sampler_snr
        else:
            sampler_sigma = 1.0

        # Get prior configuration
        prior_type = self.model.prior_config.type
        prior_strength = self.model.prior_config.get("strength", 0.0)

        # Get spectrum pair info
        pair = self.config.dataset.spectrum_pair
        diffusivities = self.config.dataset.spectrum_pairs[pair].diff_values
        true_spectrum = self.config.dataset.spectrum_pairs[pair].true_spectrum

        print(f"[NUTS] Starting sampling with {n_chains} chains, {n_iter} iterations")
        print(f"[NUTS] Prior: {prior_type}, strength: {prior_strength}")
        print(f"[NUTS] Sampler σ: {sampler_sigma:.4f} (SNR: {sampler_snr})")
        print(f"[NUTS] Target acceptance: {target_accept}")

        # Build PyMC model
        with pm.Model() as pymc_model:
            # Define prior for spectrum R (with truncation R >= 0)
            if prior_type == "uniform":
                # Flat prior (improper but works with truncation)
                R = pm.TruncatedNormal(
                    "R",
                    mu=0.5,  # Arbitrary, will be dominated by likelihood
                    sigma=100.0,  # Very wide
                    lower=0.0,
                    shape=n_dim,
                )

            elif prior_type == "ridge":
                # Gaussian (L2) prior - naturally handled by PyMC
                prior_sigma = (
                    1.0 / np.sqrt(prior_strength) if prior_strength > 0 else 100.0
                )
                R = pm.TruncatedNormal(
                    "R",
                    mu=0.0,
                    sigma=prior_sigma,
                    lower=0.0,
                    shape=n_dim,
                )

            elif prior_type == "lasso":
                # Laplace (L1) prior
                prior_b = 1.0 / prior_strength if prior_strength > 0 else 100.0
                # Use Half-Laplace since we have R >= 0 constraint
                R = pm.Laplace(
                    "R_unconstrained",
                    mu=0.0,
                    b=prior_b,
                    shape=n_dim,
                )
                R = pm.Deterministic("R", pm.math.maximum(R, 0.0))

            else:
                raise ValueError(f"Unknown prior type: {prior_type}")

            # Likelihood: signal = U @ R + noise
            mu = pm.math.dot(U, R)
            obs = pm.Normal(
                "obs",
                mu=mu,
                sigma=sampler_sigma,
                observed=signal,
            )

            # Sample using NUTS
            print("[NUTS] Compiling model and starting sampling...")
            idata = pm.sample(
                draws=n_iter,
                tune=tune_steps,
                chains=n_chains,
                target_accept=target_accept,
                return_inferencedata=True,
                progressbar=show_progress,
                random_seed=getattr(self.config, "seed", 42),
            )

        print("[NUTS] Sampling complete!")

        # Extract samples (shape: chains × iterations × n_dim)
        samples = idata.posterior["R"].values

        # For compatibility with existing code, flatten across chains
        # (chains, iterations, n_dim) -> (chains * iterations, n_dim)
        n_chains_actual, n_iters, n_dim = samples.shape
        samples_flat = samples.reshape(n_chains_actual * n_iters, n_dim)

        # Compute summary statistics
        spectrum_vector = np.mean(samples_flat, axis=0)
        spectrum_std = np.std(samples_flat, axis=0)

        # Get initial estimate (MAP from model)
        init_method = getattr(self.config.inference, "init", "map")
        if init_method == "map":
            initial_R = self.model.map_estimate(signal).copy()
        else:
            initial_R = np.ones(n_dim) * 0.1

        # Reformat inference data to match Gibbs format (with diff_X.X variable names)
        # This is needed for compatibility with diagnostic plotting functions
        var_names = [f"diff_{diff:.1f}" for diff in diffusivities]
        posterior_data = {}
        for i, var_name in enumerate(var_names):
            # Extract data for this diffusivity across all chains
            posterior_data[var_name] = samples[:, :, i]  # (n_chains, n_iterations)

        # Create new InferenceData with properly named variables
        idata_formatted = az.from_dict(posterior=posterior_data)

        # Save inference data
        inference_data_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{unique_hash}.nc"
            inference_data_path = os.path.join(save_dir, fname)
            idata_formatted.to_netcdf(inference_data_path)
            print(f"[NUTS] Saved inference data to: {inference_data_path}")

        # Print convergence diagnostics
        print("\n[NUTS] Convergence Diagnostics:")
        summary = az.summary(idata_formatted)
        max_rhat = summary["r_hat"].max()
        min_ess_bulk = summary["ess_bulk"].min()
        min_ess_tail = summary["ess_tail"].min()
        print(f"  Max R-hat: {max_rhat:.4f}")
        print(f"  Min ESS_bulk: {min_ess_bulk:.0f}")
        print(f"  Min ESS_tail: {min_ess_tail:.0f}")

        if max_rhat < 1.05:
            print("  ✓ CONVERGED (R-hat < 1.05)")
        elif max_rhat < 1.10:
            print("  ⚠️  Nearly converged (R-hat < 1.10)")
        else:
            print("  ❌ NOT CONVERGED (R-hat >= 1.10)")

        # Create result object (same format as Gibbs)
        spectrum = DiffusivitySpectrum(
            inference_method="nuts",
            signal_decay=self.signal_decay,
            diffusivities=diffusivities,
            design_matrix_U=U,
            spectrum_init=initial_R.tolist(),
            spectrum_vector=spectrum_vector.tolist(),
            spectrum_samples=samples_flat.tolist(),
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
