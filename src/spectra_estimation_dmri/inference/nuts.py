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
    NUTS sampler for spectrum estimation with HalfNormal prior.

    Uses:
    - HalfNormal prior for R (naturally enforces R >= 0 with smooth gradients)
    - Inferred sigma with weakly informative prior (improves uncertainty calibration)
    - Ridge regularization via prior_strength parameter

    Advantages over Gibbs:
    - Better exploration of positive-constrained spaces
    - Smooth gradient geometry (no boundary discontinuities)
    - Handles correlations more efficiently
    - Adaptive noise level estimation
    - Robust to prior misspecification
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
        infer_sigma = getattr(
            self.config.inference, "infer_sigma", True
        )  # Default: infer
        sigma_prior = getattr(
            self.config.inference, "sigma_prior", "halfcauchy"
        )  # halfnormal or halfcauchy

        # Get prior configuration (simplified - only using ridge)
        prior_type = self.model.prior_config.type
        prior_strength = self.model.prior_config.get("strength", 0.01)

        # Estimate data noise level from SNR (if available)
        # Model: signal = U @ R + noise, where noise ~ Normal(0, σ)
        # If signal amplitude ~ 1 and SNR is given, then σ ≈ 1/SNR
        sampler_snr = getattr(self.config.inference, "sampler_snr", None)
        if sampler_snr is None:
            sampler_snr = getattr(self.config.dataset, "snr", None)

        # For HalfCauchy: SNR is optional (robust to no prior knowledge)
        # For HalfNormal: SNR helps center the prior (less robust)
        if sampler_snr is not None:
            sigma_expected = 1.0 / sampler_snr
        else:
            # Generic weakly informative prior for dMRI noise
            # Covers typical range: σ ∈ [0.0001, 0.1] for normalized signal
            if sigma_prior == "halfcauchy":
                sigma_expected = 0.01  # HalfCauchy beta (median)
            else:
                sigma_expected = 0.015  # HalfNormal scale (~ mean 0.012)

        # Get spectrum pair info
        pair = self.config.dataset.spectrum_pair
        diffusivities = self.config.dataset.spectrum_pairs[pair].diff_values
        true_spectrum = self.config.dataset.spectrum_pairs[pair].true_spectrum

        print(f"[NUTS] Starting sampling with {n_chains} chains, {n_iter} iterations")
        print(
            f"[NUTS] Ridge regularization: λ = {prior_strength} → σ_R = {1.0/np.sqrt(prior_strength):.2f}"
        )
        if infer_sigma:
            snr_info = f" (from SNR={sampler_snr:.0f})" if sampler_snr else " (generic)"
            print(f"[NUTS] Noise: σ will be INFERRED using {sigma_prior.upper()} prior")
            print(f"[NUTS]   Prior scale: {sigma_expected:.4f}{snr_info}")
        else:
            print(f"[NUTS] Noise: σ FIXED at {sigma_expected:.4f}")
        print(f"[NUTS] Target acceptance: {target_accept}")

        # Build PyMC model with clear mathematical priors
        with pm.Model() as pymc_model:
            # ═══════════════════════════════════════════════════════════════
            # PRIOR 1: Spectrum R (the quantity we care about)
            # ═══════════════════════════════════════════════════════════════
            # Ridge regularization: minimize ||y - U@R||² + λ||R||²
            # Bayesian equivalent: R ~ Normal(0, σ_R²) where σ_R² = 1/λ
            # Since R ≥ 0, use HalfNormal with same variance
            #
            # Math: prior_strength = λ → σ_R = 1/√λ
            sigma_R = 1.0 / np.sqrt(prior_strength) if prior_strength > 0 else 10.0
            R = pm.HalfNormal(
                "R",
                sigma=sigma_R,
                shape=n_dim,
            )

            # ═══════════════════════════════════════════════════════════════
            # PRIOR 2: Noise level σ (can be inferred or fixed)
            # ═══════════════════════════════════════════════════════════════
            if infer_sigma:
                if sigma_prior == "halfcauchy":
                    # HALFCAUCHY: Robust to prior misspecification (Gelman recommendation)
                    # HalfCauchy(beta=b) has:
                    #   - Median = b (50% probability σ < b)
                    #   - Heavy tails → robust when SNR estimate is very wrong
                    #   - 95% mass approximately in [0.025×b, 40×b] (very wide!)
                    #
                    # Best for: Real data with uncertain noise levels
                    sigma = pm.HalfCauchy(
                        "sigma",
                        beta=sigma_expected,  # Median at expected value
                    )
                else:  # halfnormal
                    # HALFNORMAL: Faster sampling, needs better SNR estimate
                    # HalfNormal(scale=s) has:
                    #   - Mean ≈ 0.8s, mode at ≈ 0.7s
                    #   - Light tails → less robust when SNR estimate is wrong
                    #   - 95% mass approximately in [0, 3.3×s]
                    #
                    # Best for: Simulations or data with reliable SNR estimate
                    sigma_prior_scale = 2.0 * sigma_expected  # Allows flexibility
                    sigma = pm.HalfNormal(
                        "sigma",
                        sigma=sigma_prior_scale,
                    )
            else:
                # FIX σ: Use deterministic value (no inference)
                sigma = sigma_expected

            # Likelihood: signal = U @ R + noise
            mu = pm.math.dot(U, R)
            obs = pm.Normal(
                "obs",
                mu=mu,
                sigma=sigma,
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

        # Get sigma statistics (inferred or fixed)
        if infer_sigma:
            sigma_samples = idata.posterior["sigma"].values
            sigma_mean = np.mean(sigma_samples)
            sigma_std = np.std(sigma_samples)
            print(f"\n[NUTS] Inferred noise level:")
            print(f"  σ_data = {sigma_mean:.4f} ± {sigma_std:.4f}")
            print(f"  Expected: {sigma_expected:.4f}")
            if sampler_snr is not None:
                inferred_snr = 1.0 / sigma_mean
                rel_diff = (sigma_mean - sigma_expected) / sigma_expected * 100
                print(
                    f"  Inferred SNR: {inferred_snr:.1f} (expected: {sampler_snr:.1f})"
                )
                print(f"  Relative difference: {rel_diff:+.1f}%")
        else:
            sigma_samples = None
            sigma_mean = sigma_expected
            sigma_std = 0.0
            print(f"\n[NUTS] Fixed noise level:")
            print(f"  σ_data = {sigma_mean:.4f} (not inferred)")

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

        # Also include sigma in the saved data (if inferred)
        if infer_sigma and sigma_samples is not None:
            posterior_data["sigma"] = sigma_samples

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
        # Convert sigma to SNR for storage (sampler_snr field expects SNR, not sigma!)
        inferred_snr = 1.0 / sigma_mean if sigma_mean > 0 else None

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
            prior_type="ridge",  # Simplified - only using ridge now
            prior_strength=prior_strength,
            data_snr=getattr(self.config.dataset, "snr", None),
            sampler_snr=inferred_snr,  # Store inferred SNR (1/sigma)
        )

        return spectrum
