from .base import BaseSampler, d_spectra_sample
import numpy as np
from cvxopt import matrix, solvers
import numpy.random as npr


class GibbsSampler(BaseSampler):
    def _calculate_MVN_posterior_params(
        self,
        signal_data,
        recon_diffusivities,
        sigma,
        inverse_prior_covariance=None,
        L1_lambda=0.0,
        L2_lambda=0.0,
    ):
        M_count = signal_data.signal_values.shape[0]
        N = recon_diffusivities.shape[0]
        u_vec_tuple = ()
        for i in range(M_count):
            u_vec_tuple += (np.exp((-signal_data.b_values[i] * recon_diffusivities)),)
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

    def sample(self, iterations: int):
        signal_data = self.signal_data
        diffusivities = self.diffusivities
        sigma = self.sigma
        inverse_prior_covariance = self.kwargs.get("inverse_prior_covariance", None)
        L1_lambda = self.kwargs.get("l1_lambda", 0.0)
        L2_lambda = self.kwargs.get("l2_lambda", 0.0)
        M, Sigma_inverse, weighted_U_vecs = self._calculate_MVN_posterior_params(
            signal_data,
            diffusivities,
            sigma,
            inverse_prior_covariance,
            L1_lambda=L1_lambda,
            L2_lambda=L2_lambda,
        )
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
            if (count % 100) == 0:
                print(".", end="")
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
