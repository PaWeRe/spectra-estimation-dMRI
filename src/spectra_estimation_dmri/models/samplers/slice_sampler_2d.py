import numpy as np
from .base import BaseSampler, d_spectra_sample
from cvxopt import matrix, solvers


class SliceSampler2D(BaseSampler):
    """
    2D Slice Sampler for truncated multivariate normal distributions in the non-negative orthant.
    Implements the algorithm described in:
    Prange, M. and Song, Y.-Q. (2009). Quantifying uncertainty in NMR T2 spectra using Monte Carlo inversion.
    Journal of Magnetic Resonance, 196(1), 54-60. doi:10.1016/j.jmr.2008.10.008

    The sampler updates pairs of coordinates (R_i, R_j) at each step, using a 2D slice sampling move
    that respects the non-negativity constraint. This improves mixing compared to 1D moves.
    """

    def __init__(self, signal_data, diffusivities, sigma, config=None, **kwargs):
        super().__init__(signal_data, diffusivities, sigma, config=config, **kwargs)
        self.L1_lambda = kwargs.get("l1_lambda", 0.0)
        self.L2_lambda = kwargs.get("l2_lambda", 0.0)
        self.inverse_prior_covariance = kwargs.get("inverse_prior_covariance", None)

    def _log_posterior(self, R):
        # Log-posterior as in Prange et al. (2009), up to a constant
        signal_data = self.signal_data
        diffusivities = self.diffusivities
        sigma = self.sigma
        M_count = signal_data.signal_values.shape[0]
        N = diffusivities.shape[0]
        u_vec_tuple = ()
        for i in range(M_count):
            u_vec_tuple += (np.exp((-signal_data.b_values[i] * diffusivities)),)
        U_vecs = np.vstack(u_vec_tuple)
        recon = U_vecs @ R
        log_likelihood = (
            -0.5 * np.sum((signal_data.signal_values - recon) ** 2) / (sigma**2)
        )
        log_prior = 0.0
        if self.L2_lambda > 0.0:
            log_prior -= 0.5 * self.L2_lambda * np.sum(R**2)
        if self.L1_lambda > 0.0:
            log_prior -= self.L1_lambda * np.sum(R)
        return log_likelihood + log_prior

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

    def _slice_sample_2d(self, R, i, j, max_trials=100):
        """
        Perform a 2D slice sampling update for coordinates (i, j) of R, as in Prange et al. (2009).
        The move is constrained to the non-negative orthant.
        """
        # 1. Draw a vertical slice level
        logp0 = self._log_posterior(R)
        logy = logp0 + np.log(np.random.uniform())
        # 2. Define a 2D window around (R[i], R[j])
        w = 1.0  # step size, can be tuned
        # The window must not go below zero
        L_i, R_i = (
            max(0, R[i] - w * np.random.uniform()),
            R[i] + w * np.random.uniform(),
        )
        L_j, R_j = (
            max(0, R[j] - w * np.random.uniform()),
            R[j] + w * np.random.uniform(),
        )
        # 3. Repeatedly sample from the 2D box, shrink if outside the slice or negative
        for _ in range(max_trials):
            new_i = np.random.uniform(L_i, R_i)
            new_j = np.random.uniform(L_j, R_j)
            R_prop = R.copy()
            R_prop[i] = new_i
            R_prop[j] = new_j
            if np.all(R_prop >= 0) and self._log_posterior(R_prop) >= logy:
                return R_prop
            # Shrink the box if rejected
            if new_i < R[i]:
                L_i = new_i
            else:
                R_i = new_i
            if new_j < R[j]:
                L_j = new_j
            else:
                R_j = new_j
        # If no valid sample found, return the original
        return R

    def sample(self, iterations: int, initial_R=None):
        N = self.diffusivities.shape[0]
        M, Sigma_inverse, weighted_U_vecs = self._calculate_MVN_posterior_params(
            self.signal_data,
            self.diffusivities,
            self.sigma,
            self.inverse_prior_covariance,
            L1_lambda=self.L1_lambda,
            L2_lambda=self.L2_lambda,
        )
        if initial_R is not None:
            R = np.copy(initial_R)
        else:
            R = np.array(
                self._calculate_trunc_MVN_mode(M, Sigma_inverse, weighted_U_vecs)
            ).T[0]
        the_sample = d_spectra_sample(self.diffusivities)
        the_sample.initial_R = np.copy(R)
        for it in range(iterations):
            if (it % 100) == 0:
                print(f"SliceSampler2D iteration {it}/{iterations}")
            idxs = np.arange(N)
            np.random.shuffle(idxs)
            for k in range(N - 1):
                i, j = idxs[k], idxs[k + 1]
                R = self._slice_sample_2d(R, i, j)
            the_sample.sample.append(np.copy(R))
        the_sample.normalize()
        return the_sample
