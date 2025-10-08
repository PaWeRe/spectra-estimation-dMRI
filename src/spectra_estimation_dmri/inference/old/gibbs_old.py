import numpy as np
import arviz as az
from scipy.stats import truncnorm
from tqdm import tqdm
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum
import os


class GibbsSampler:
    def __init__(self, model, signal_decay, config):
        self.model = model
        self.signal_decay = signal_decay
        self.config = config

    def _calculate_MVN_posterior_params(
        self, signal, b_values, diffusivities, sigma, L2_lambda=0.0
    ):
        M_count = signal.shape[0]
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
            Sigma_inverse += L2_lambda * np.eye(N)
        weighted_U_vecs = np.zeros(N)
        for i in range(M_count):
            weighted_U_vecs += signal[i] * U_vecs[i]
        Sigma_inverse_M = (1.0 / (sigma * sigma)) * weighted_U_vecs
        M = np.linalg.solve(Sigma_inverse, Sigma_inverse_M)
        return (M, Sigma_inverse, Sigma_inverse_M)

    def _calculate_trunc_MVN_mode(self, M, Sigma_inverse, Sigma_inverse_M):
        from cvxopt import matrix, solvers

        N = Sigma_inverse.shape[0]
        P = matrix(Sigma_inverse)
        Q = matrix(-Sigma_inverse_M)
        G = matrix(-np.identity(N), tc="d")
        H = matrix(np.zeros(N))
        sol = solvers.qp(P, Q, G, H)
        mode = np.array(sol["x"]).T[0]
        return mode

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

    def run(
        self,
        return_idata=True,
        show_progress=False,
        save_dir=None,
        true_spectrum=None,
        unique_hash=None,
    ):
        """
        Run Gibbs sampling for the spectrum fractions.
        Returns:
            DiffusivitySpectrum object
        """
        signal = np.array(self.signal_decay.signal_values)
        U = self.model.U_matrix()
        sigma = self.model.sigma
        prior_type = self.model.prior_config.type
        prior_strength = getattr(
            self.model.prior_config, "strength", 0.0
        )  # previously L2_lambda
        n_iter = self.config.inference.n_iter
        burn_in = self.config.inference.burn_in
        N = U.shape[1]
        # Initialization
        if self.config.inference.init == "map":
            R = self.model.map_estimate(signal)
        elif self.config.inference.init == "random":
            R = np.abs(np.random.rand(N))
            R /= np.sum(R)
        elif self.config.inference.init == "zeros":
            R = np.zeros(N)
        else:
            raise ValueError(f"Unknown init method: {self.config.inference.init}")
        samples = []
        iterator = range(n_iter)
        if show_progress:
            iterator = tqdm(iterator, desc="Gibbs Sampling", unit="iter")
        if prior_type == "uniform":
            # Uniform prior (nonnegativity)
            prec = 1.0 / sigma**2
            UtU = U.T @ U
            Uty = U.T @ signal
            for it in iterator:
                for i in range(N):
                    var_i = 1.0 / (prec * UtU[i, i])
                    std_i = np.sqrt(var_i)
                    r_i = signal - U @ R + U[:, i] * R[i]
                    mu_i = prec * U[:, i].dot(r_i) * var_i
                    a, b = (0 - mu_i) / std_i, np.inf
                    R[i] = truncnorm.rvs(a, b, loc=mu_i, scale=std_i)
                R = R / np.sum(R) if np.sum(R) > 0 else np.ones(N) / N
                if it >= burn_in:
                    samples.append(R.copy())
        elif prior_type == "ridge":
            # Ridge (L2) prior
            l2_lambda = prior_strength
            prec = 1.0 / sigma**2
            UtU = U.T @ U
            for it in iterator:
                for i in range(N):
                    var_i = 1.0 / (l2_lambda + prec * UtU[i, i])
                    std_i = np.sqrt(var_i)
                    r_i = signal - U @ R + U[:, i] * R[i]
                    mu_i = prec * U[:, i].dot(r_i) * var_i
                    R[i] = np.random.normal(mu_i, std_i)
                R = R / np.sum(R) if np.sum(R) > 0 else np.ones(N) / N
                if it >= burn_in:
                    samples.append(R.copy())
        elif prior_type == "full":
            # Full covariance (current implementation, possibly simplified)
            # Use the old _calculate_MVN_posterior_params logic
            M_count = signal.shape[0]
            diffusivities = np.array(self.config.dataset.diff_values)
            N = diffusivities.shape[0]
            u_vec_tuple = ()
            b_values = np.array(self.signal_decay.b_values)
            for i in range(M_count):
                u_vec_tuple += (np.exp((-b_values[i] * diffusivities)),)
            U_vecs = np.vstack(u_vec_tuple)
            U_outer_prods = np.zeros((N, N))
            for i in range(M_count):
                U_outer_prods += np.outer(U_vecs[i], U_vecs[i])
            Sigma_inverse = (1.0 / (sigma * sigma)) * U_outer_prods
            if prior_type == "ridge" and prior_strength > 0.0:
                Sigma_inverse += prior_strength * np.eye(N)
            weighted_U_vecs = np.zeros(N)
            for i in range(M_count):
                weighted_U_vecs += signal[i] * U_vecs[i]
            Sigma_inverse_M = (1.0 / (sigma * sigma)) * weighted_U_vecs
            M = np.linalg.solve(Sigma_inverse, Sigma_inverse_M)
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
                for i in range(N):
                    R_slash_i = np.delete(R, i, 0)
                    dot_prod = np.dot(
                        Sigma_inverse_quotient[i], (M_slash_i[i] - R_slash_i)
                    )
                    mu_i = M[i] + dot_prod
                    # For uniform, sample truncated; for ridge, sample normal
                    if prior_type == "uniform":
                        a, b = (0 - mu_i) / sigma_i[i], np.inf
                        R[i] = truncnorm.rvs(a, b, loc=mu_i, scale=sigma_i[i])
                    else:
                        R[i] = np.random.normal(mu_i, sigma_i[i])
                R = R / np.sum(R) if np.sum(R) > 0 else np.ones(N) / N
                if it >= burn_in:
                    samples.append(R.copy())
        else:
            raise NotImplementedError(
                f"GibbsSampler does not support prior type: {prior_type}"
            )
        samples = np.array(samples)
        idata = None
        inference_data_path = None
        if return_idata:
            idata = az.from_dict(
                posterior={"R": samples.T}
            )  # transpose to match arviz.InferenceData (n_chains, n_draws) format
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{unique_hash}.nc"
                inference_data_path = os.path.join(save_dir, fname)
                idata.to_netcdf(inference_data_path)
        spectrum_vector = np.mean(samples, axis=0)
        spectrum_std = np.std(samples, axis=0)
        spectrum = DiffusivitySpectrum(
            inference_method="gibbs",
            signal_decay=self.signal_decay,
            diffusivities=self.config.dataset.diff_values,
            design_matrix_U=self.model.U_matrix(),
            spectrum_vector=spectrum_vector.tolist(),
            spectrum_samples=samples.tolist(),
            spectrum_std=spectrum_std.tolist(),
            true_spectrum=true_spectrum,
            inference_data=inference_data_path,
            spectra_id=unique_hash,
            init_method=self.config.inference.init,
            prior_type=prior_type,
        )
        return spectrum
