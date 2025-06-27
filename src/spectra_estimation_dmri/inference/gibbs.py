import numpy as np
import arviz as az
from cvxopt import matrix, solvers
import numpy.random as npr
from tqdm import tqdm
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum
import hashlib
import os


class GibbsSampler:
    def __init__(self, model, signal, config=None):
        # TODO: doesn't it make more sense to pull out the config and assign everything directly in the main.py without the complex if else structure here?
        self.model = model
        self.signal = signal
        self.config = config
        self.n_iter = (
            getattr(config, "iterations", 10000) if config is not None else 10000
        )
        self.burn_in = getattr(config, "burn_in", 1000) if config is not None else 1000
        self.L2_lambda = (
            getattr(config, "l2_lambda", 1e-5) if config is not None else 1e-5
        )
        self.init_method = (
            getattr(config, "init", "map") if config is not None else "map"
        )

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
        Sigma = np.linalg.inv(Sigma_inverse)
        weighted_U_vecs = np.zeros(N)
        for i in range(M_count):
            weighted_U_vecs += signal[i] * U_vecs[i]
        Sigma_inverse_M = (1.0 / (sigma * sigma)) * weighted_U_vecs
        M = np.linalg.solve(Sigma_inverse, Sigma_inverse_M)
        return (M, Sigma_inverse, Sigma_inverse_M)

    def _calculate_trunc_MVN_mode(self, M, Sigma_inverse, Sigma_inverse_M):
        N = Sigma_inverse.shape[0]
        P = matrix(Sigma_inverse)
        Q = matrix(-Sigma_inverse_M)
        G = matrix(-np.identity(N), tc="d")
        H = matrix(np.zeros(N))
        sol = solvers.qp(P, Q, G, H)
        mode = np.array(sol["x"]).T[0]
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

    def run(
        self,
        return_idata=True,
        show_progress=False,
        save_dir=None,
        ground_truth_spectrum=None,
        config_hash=None,
        config_tag=None,
    ):
        """
        Run Gibbs sampling for the spectrum fractions.
        Returns:
            DiffusivitySpectrum object
        """
        signal = np.array(self.signal)
        b_values = np.array(self.model.b_values)
        diffusivities = np.array(self.model.diffusivities)
        sigma = self.model.sigma
        L2_lambda = self.L2_lambda
        # Initialization
        n_diff = len(diffusivities)
        if self.init_method == "map":
            R = self.model.map_estimate(signal, regularization=L2_lambda)
        elif self.init_method == "random":
            R = np.abs(np.random.rand(n_diff))
            R /= np.sum(R)
        elif self.init_method == "zeros":
            R = np.zeros(n_diff)
        else:
            raise ValueError(f"Unknown init method: {self.init_method}")
        # Precompute posterior params
        M, Sigma_inverse, Sigma_inverse_M = self._calculate_MVN_posterior_params(
            signal, b_values, diffusivities, sigma, L2_lambda=L2_lambda
        )
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
        samples = []
        iterator = range(self.n_iter)
        if show_progress:
            iterator = tqdm(iterator, desc="Gibbs Sampling", unit="iter")
        for j in iterator:
            for i in range(N):
                R_slash_i = np.delete(R, i, 0)
                dot_prod = np.dot(Sigma_inverse_quotient[i], (M_slash_i[i] - R_slash_i))
                mu_i = M[i] + dot_prod
                value = self._sample_normal_non_neg(mu_i, sigma_i[i])
                R[i] = value
            # Normalize after each full sweep
            R = R / np.sum(R) if np.sum(R) > 0 else np.ones(N) / N
            if j >= self.burn_in:
                samples.append(R.copy())
        samples = np.array(samples)
        idata = None
        inference_data_path = None
        if return_idata:
            idata = az.from_dict(posterior={"R": samples})
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"gibbs_inference"
                if config_hash:
                    fname += f"_{config_hash}"
                if config_tag:
                    fname += f"_{config_tag}"
                fname += ".nc"
                inference_data_path = os.path.join(save_dir, fname)
                idata.to_netcdf(inference_data_path)
        spectrum_vector = np.mean(samples, axis=0)
        spectrum_std = np.std(samples, axis=0)
        spectrum = DiffusivitySpectrum(
            inference_method="gibbs",
            signal_decay=(
                self.model.signal_decay if hasattr(self.model, "signal_decay") else None
            ),
            diffusivities=list(self.model.diffusivities),
            design_matrix_U=self.model.U_matrix().tolist(),
            spectrum_vector=list(spectrum_vector),
            spectrum_samples=samples.tolist(),
            spectrum_std=list(spectrum_std),
            ground_truth_spectrum=(
                list(ground_truth_spectrum)
                if ground_truth_spectrum is not None
                else None
            ),
            inference_data=(
                inference_data_path if inference_data_path is not None else ""
            ),
            config_hash=config_hash,
            config_tag=config_tag,
        )
        return spectrum
