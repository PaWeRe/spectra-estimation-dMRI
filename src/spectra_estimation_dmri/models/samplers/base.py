import numpy as np


class signal_data:
    def __init__(self, signal_values, b_values):
        self.signal_values = signal_values
        self.b_values = b_values


class d_spectrum:
    def __init__(self, fractions, diffusivities):
        self.fractions = fractions
        self.diffusivities = diffusivities


class d_spectra_sample:
    def __init__(self, diffusivities):
        self.diffusivities = diffusivities
        self.sample = []  # a list of samples
        self.initial_R = None  # store the initial R vector

    def normalize(self):
        for i in range(0, len(self.sample)):
            sum_val = np.sum(self.sample[i])
            self.sample[i] = self.sample[i] / sum_val


from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(self, signal_data, diffusivities, sigma, config=None, **kwargs):
        self.signal_data = signal_data
        self.diffusivities = diffusivities
        self.sigma = sigma
        self.config = config
        self.kwargs = kwargs

    @staticmethod
    def compute_trunc_MVN_mode(
        signal_data,
        diffusivities,
        sigma,
        inverse_prior_covariance=None,
        L1_lambda=0.0,
        L2_lambda=0.0,
    ):
        from cvxopt import matrix, solvers

        M_count = signal_data.signal_values.shape[0]
        N = diffusivities.shape[0]
        u_vec_tuple = ()
        for i in range(M_count):
            u_vec_tuple += (np.exp((-signal_data.b_values[i] * diffusivities)),)
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

    @abstractmethod
    def sample(self, iterations: int):
        """Run the sampler for a given number of iterations and return a d_spectra_sample object."""
        pass
