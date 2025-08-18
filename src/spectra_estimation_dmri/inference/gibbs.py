# gibbs_sampler.py
import numpy as np
import arviz as az
from scipy.stats import truncnorm
from tqdm import tqdm
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum
import os
import numpy as np
from cvxopt import matrix, solvers
import numpy.random as npr


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
                u = npr.normal(mu, sigma)
                if u > 0.0:
                    return u
        else:
            sigma_minus = -mu / sigma
            alpha_star = (sigma_minus + np.sqrt(sigma_minus**2 + 4)) / 2.0
            while True:
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
                # R[i] = self._sample_normal_non_neg(mu_i, sigma_i)
                R[i] = self._sample_truncated_normal(mu_i, sigma_i, lower=0.0)
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

    def run_new(
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

        # Get initial R for spectrum init
        initial_R = R
        # Get posterior parameters (for full precision approach)
        try:
            posterior_mean, posterior_precision = self.model.get_posterior_params(
                1 / self.config.inference.sampler_snr, signal
            )
            use_full_precision = True
        except:
            use_full_precision = False

        # Determine SNR for the sampler
        sampler_snr = getattr(self.config.inference, "sampler_snr", None)
        if sampler_snr is None:
            sampler_snr = getattr(self.config.dataset, "snr", None)
        if sampler_snr is not None:
            sampler_sigma = 1.0 / np.sqrt(sampler_snr)
        else:
            sampler_sigma = 1.0
        # Set model sigma for inference
        self.model.snr = sampler_snr
        self.model.sigma = sampler_sigma

        # Sample
        samples = []
        iterator = range(n_iter)
        if show_progress:
            iterator = tqdm(iterator, desc="Gibbs Sampling", unit="iter")

        # to check what sampler version to use
        sandy_sampler = self.config.inference.use_sandy_sampler
        if sandy_sampler:
            print("sandy_sampler1")
            Sigma_inverse = posterior_precision
            M = posterior_mean
            N = Sigma_inverse.shape[0]
            sigma_i = np.empty(N, dtype=object)
            Sigma_inverse_quotient = np.empty(N, dtype=object)
            M_slash_i = np.empty(N, dtype=object)
            for i in range(N):
                Sigma_inverse_ii = Sigma_inverse[i][i]
                sigma_i[i] = np.sqrt(1.0 / Sigma_inverse_ii)
                Sigma_inverse_i_slash_i = np.delete(Sigma_inverse[i], i, axis=0)
                Sigma_inverse_quotient[i] = Sigma_inverse_i_slash_i / Sigma_inverse_ii
                M_slash_i[i] = np.delete(M, i, 0)

        for it in iterator:
            # Choose sampling method
            if (
                use_full_precision and n_dim < 100 and not sandy_sampler
            ):  # Use full precision for small problems
                R = self._gibbs_step(
                    R, signal, posterior_mean, posterior_precision, prior_type
                )
            elif sandy_sampler:  # Use original sampling technique
                R_new = R.copy()
                for i in range(N):
                    R_slash_i = np.delete(R, i)
                    dot_prod = np.dot(
                        Sigma_inverse_quotient[i], M_slash_i[i] - R_slash_i
                    )
                    mu_i = M[i] + dot_prod
                    # Use the original _sample_normal_non_neg method as in the original code
                    R_new[i] = self._sample_normal_non_neg(mu_i, sigma_i[i])
                R = R_new
            else:  # Use element-wise updates for efficiency
                R = self._gibbs_step_elementwise(
                    R, signal, U, self.model.sigma, prior_type, prior_strength
                )

            # Store sample after burn-in
            if it >= burn_in:
                samples.append(R.copy())

        samples = np.array(samples)

        # Normalize samples if using sandy sampler (as in original implementation)
        if sandy_sampler:
            print("Normalizing samples...")
            for i in range(len(samples)):
                sum_val = np.sum(samples[i])
                if sum_val > 0:
                    samples[i] = samples[i] / sum_val

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
            spectrum_init=initial_R.tolist(),
            spectrum_vector=spectrum_vector.tolist(),
            spectrum_samples=samples.tolist(),
            spectrum_std=spectrum_std.tolist(),
            true_spectrum=true_spectrum,
            inference_data=inference_data_path,
            spectra_id=unique_hash,
            init_method=init_method,
            prior_type=prior_type,
            prior_strength=self.model.prior_config.strength,
            data_snr=getattr(self.config.dataset, "snr", None),
            sampler_snr=sampler_snr,
        )

        return spectrum

    ###################
    ### OLD ###########
    ###################

    def compute_trunc_MVN_mode(
        self,
        signal_data,
        diffusivities,
        sigma,
        inverse_prior_covariance=None,
        L1_lambda=0.0,
        L2_lambda=0.0,
    ):
        from cvxopt import matrix, solvers

        signal_values = np.array(signal_data.signal_values)
        b_values = np.array(signal_data.b_values)
        diffusivities = np.array(diffusivities)
        M_count = signal_values.shape[0]
        N = diffusivities.shape[0]
        u_vec_tuple = ()
        for i in range(M_count):
            u_vec_tuple += (np.exp((-b_values[i] * diffusivities)),)
        U_vecs = np.vstack(u_vec_tuple)
        U_outer_prods = np.zeros((N, N))
        for i in range(M_count):
            U_outer_prods += np.outer(U_vecs[i], U_vecs[i])
        Sigma_inverse = (1.0 / (sigma * sigma)) * U_outer_prods
        if L2_lambda > 0.0:
            inverse_prior_covariance = L2_lambda * np.eye(N)
        if inverse_prior_covariance is not None:
            Sigma_inverse += inverse_prior_covariance
        weighted_U_vecs = np.zeros(N)
        for i in range(M_count):
            weighted_U_vecs += signal_data.signal_values[i] * U_vecs[i]
        One_vec = np.ones(N)
        Sigma_inverse_M = (
            1.0 / (sigma * sigma) * weighted_U_vecs
        ) - L1_lambda * One_vec
        M = np.linalg.solve(Sigma_inverse, Sigma_inverse_M)
        P = matrix(Sigma_inverse)
        Q = matrix(-Sigma_inverse_M)
        G = matrix(-np.identity(N), tc="d")
        H = matrix(np.zeros(N))
        sol = solvers.qp(P, Q, G, H)
        mode = np.array(sol["x"]).T[0]
        return mode

    def _calculate_MVN_posterior_params(
        self,
        signal_data,
        recon_diffusivities,
        sigma,
        inverse_prior_covariance=None,
        L1_lambda=0.0,
        L2_lambda=0.0,
    ):
        signal_values = np.array(signal_data.signal_values)
        b_values = np.array(signal_data.b_values)
        recon_diffusivities = np.array(recon_diffusivities)
        M_count = signal_values.shape[0]
        N = recon_diffusivities.shape[0]
        u_vec_tuple = ()
        for i in range(M_count):
            u_vec_tuple += (np.exp((-b_values[i] * recon_diffusivities)),)
        U_vecs = np.vstack(u_vec_tuple)
        U_outer_prods = np.zeros((N, N))
        for i in range(M_count):
            U_outer_prods += np.outer(U_vecs[i], U_vecs[i])
        Sigma_inverse = (1.0 / (sigma * sigma)) * U_outer_prods
        if L2_lambda > 0.0:
            inverse_prior_covariance = L2_lambda * np.eye(N)
        if inverse_prior_covariance is not None:
            Sigma_inverse += inverse_prior_covariance
        Sigma = np.linalg.inv(Sigma_inverse)
        weighted_U_vecs = np.zeros(N)
        for i in range(M_count):
            weighted_U_vecs += signal_data.signal_values[i] * U_vecs[i]
        One_vec = np.ones(N)
        Sigma_inverse_M = (
            1.0 / (sigma * sigma) * weighted_U_vecs
        ) - L1_lambda * One_vec
        M = np.linalg.solve(Sigma_inverse, Sigma_inverse_M)
        return (M, Sigma_inverse, Sigma_inverse_M)

    def _calculate_trunc_MVN_mode(self, M, Sigma_inverse, Sigma_inverse_M):
        N = Sigma_inverse.shape[0]
        P = matrix(Sigma_inverse)
        Q = matrix(-Sigma_inverse_M)
        G = matrix(-np.identity(N), tc="d")
        H = matrix(np.zeros(N))
        sol = solvers.qp(P, Q, G, H)
        mode = sol["x"]
        return mode

    def _sample_normal_non_neg(self, mu, sigma):
        if mu >= 0:
            while True:
                u = npr.normal(mu, sigma)
                if u > 0.0:
                    return u
        else:
            sigma_minus = -mu / sigma
            alpha_star = (sigma_minus + np.sqrt(sigma_minus**2 + 4)) / 2.0
            while True:
                x = npr.exponential(1.0 / alpha_star)
                z = sigma_minus + x
                eta = np.exp(-((z - alpha_star) ** 2) / 2.0)
                u = npr.uniform()
                if u <= eta:
                    return mu + sigma * z

    def sample(self, iterations: int, diffusivities, sigma, initial_R=None):
        signal_data = self.signal_decay
        inverse_prior_covariance = (
            None  # TODO: check older version if at some point this was not None!
        )
        L1_lambda = 0.0  # TODO: check in older version if at some point this was not 0!
        L2_lambda = (
            1e-5  # TODO: check in older version if at some point this was not 0!
        )
        M, Sigma_inverse, weighted_U_vecs = self._calculate_MVN_posterior_params(
            signal_data,
            diffusivities,
            sigma,
            inverse_prior_covariance,
            L1_lambda=L1_lambda,
            L2_lambda=L2_lambda,
        )
        if initial_R is not None:
            R = np.copy(initial_R)
        else:
            R = np.array(
                self._calculate_trunc_MVN_mode(M, Sigma_inverse, weighted_U_vecs)
            ).T[0]
        N = Sigma_inverse.shape[0]
        sigma_i = np.empty(N, dtype=object)
        Sigma_inverse_quotient = np.empty(N, dtype=object)
        M_slash_i = np.empty(N, dtype=object)
        for i in range(N):
            Sigma_inverse_ii = Sigma_inverse[i][i]
            sigma_i[i] = np.sqrt(1.0 / Sigma_inverse_ii)
            Sigma_inverse_i_slash_i = np.delete(Sigma_inverse[i], i, axis=0)
            Sigma_inverse_quotient[i] = Sigma_inverse_i_slash_i / Sigma_inverse_ii
            M_slash_i[i] = np.delete(M, i, 0)
        the_sample = d_spectra_sample(diffusivities)
        the_sample.initial_R = np.copy(R)
        count = 0
        for j in range(iterations):
            if (j % 100) == 0:
                print(f"GibbsSampler iteration {j}/{iterations}")
            count += 1
            for i in range(N):
                R_slash_i = np.delete(R, i, 0)
                dot_prod = np.dot(Sigma_inverse_quotient[i], (M_slash_i[i] - R_slash_i))
                mu_i = M[i] + dot_prod
                value = self._sample_normal_non_neg(mu_i, sigma_i[i])
                R[i] = value
            the_sample.sample.append(np.copy(R))
        the_sample.normalize()
        return the_sample

    def run(
        self,
        return_idata=True,
        show_progress=False,
        save_dir=None,
        true_spectrum=None,
        unique_hash=None,
    ):
        U = self.model.U_matrix()
        prior_type = self.model.prior_config.type
        n_iter = self.config.inference.n_iter
        burn_in = self.config.inference.burn_in
        sampler_snr = self.config.inference.sampler_snr
        init_method = self.config.inference.init
        pair = self.config.dataset.spectrum_pair
        diffusivities = self.config.dataset.spectrum_pairs[
            pair
        ].diff_values  # TODO: this only works for sim case, make sure to make it work with bwh

        # noisy_signal_normalized = generate_signals(
        #     1.0, config.b_values, true_spectrum, 1 / config.data_snr
        # )
        noisy_signal_normalized = self.signal_decay
        L2_lambda = (
            1e-5  # TODO: check in older version if at some point this was not 0!
        )
        # Compute shared initial R (mode)
        initial_R = self.compute_trunc_MVN_mode(
            noisy_signal_normalized,
            diffusivities,
            1 / self.config.inference.sampler_snr,
            L2_lambda=L2_lambda,
        )

        sample = self.sample(
            n_iter,
            diffusivities,
            1 / self.config.inference.sampler_snr,
            initial_R=initial_R,
        )
        sample.sample = sample.sample[burn_in:]  # Discard burn-in
        sample = np.array(sample.sample)

        # Create inference data
        idata = None
        inference_data_path = None
        if return_idata:
            idata = az.from_dict(
                posterior={"R": sample[None, :, :]}  # Add chain dimension
            )
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{unique_hash}.nc"
                inference_data_path = os.path.join(save_dir, fname)
                idata.to_netcdf(inference_data_path)

        # Summary statistics
        spectrum_vector = np.mean(sample, axis=0)
        spectrum_std = np.std(sample, axis=0)

        # Create result object
        pair = self.config.dataset.spectrum_pair
        diffusvities = self.config.dataset.spectrum_pairs[pair].diff_values
        true_spectrum = self.config.dataset.spectrum_pairs[pair].true_spectrum
        spectrum = DiffusivitySpectrum(
            inference_method="gibbs",
            signal_decay=self.signal_decay,
            diffusivities=diffusvities,
            design_matrix_U=U,
            spectrum_init=initial_R.tolist(),
            spectrum_vector=spectrum_vector.tolist(),
            spectrum_samples=sample.tolist(),
            spectrum_std=spectrum_std.tolist(),
            true_spectrum=true_spectrum,
            inference_data=inference_data_path,
            spectra_id=unique_hash,
            init_method=init_method,
            prior_type=prior_type,
            prior_strength=self.model.prior_config.strength,
            data_snr=getattr(self.config.dataset, "snr", None),
            sampler_snr=sampler_snr,
        )

        return spectrum


class d_spectra_sample:
    def __init__(self, diffusivities):
        self.diffusivities = diffusivities
        self.sample = []  # a list of samples
        self.initial_R = None  # store the initial R vector

    def normalize(self):
        for i in range(0, len(self.sample)):
            sum_val = np.sum(self.sample[i])
            self.sample[i] = self.sample[i] / sum_val


class d_spectrum:
    def __init__(self, fractions, diffusivities):
        self.fractions = fractions
        self.diffusivities = diffusivities
