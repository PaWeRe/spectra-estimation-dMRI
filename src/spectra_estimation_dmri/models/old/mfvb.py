import numpy as np
from scipy.special import gamma, digamma
from scipy.stats import lognorm, gamma as gamma_dist
import h5py
import json
from tqdm import tqdm
from scipy.optimize import minimize
from cvxopt import matrix, solvers

import datetime
import json
import os
import pickle
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cvxopt import matrix, solvers
import pandas as pd
from tqdm import tqdm
import yaml
import h5py

import numpy as np
from scipy.special import gamma, digamma
from scipy.stats import lognorm, gamma as gamma_dist
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture

import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import minimize
from scipy.special import erf, erfc, log_ndtr
from scipy.stats import norm

sys.path.append(os.path.join(os.getcwd() + "/src/utils/"))
import generate_analysis_datasets as gad


class signal_data:
    def __init__(self, signal_values, b_values):
        self.signal_values = signal_values
        self.b_values = b_values
        # self.v_count = v_count
        # self.patient_id = patient_id

    def plot(self, save_filename=None, title=None):
        plt.plot(self.b_values, self.signal_values, linestyle="None", marker=".")
        plt.xlabel("B Value")
        plt.ylabel("Signal")
        if title is not None:
            plt.title(title)
        # plt.show()
        if save_filename is not None:
            plt.savefig(save_filename)
        else:
            plt.show()

    # Plot 3rd dimension (snr/p_count as well)
    def plot3d(self, title=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.b_values, self.signal_values, self.v_count, marker=".")
        ax.set_xlabel("B Value")
        ax.set_ylabel("Signal")
        ax.set_zlabel("V Count")
        if title is not None:
            ax.set_title(title)
        plt.show()


# package up fractions and diffusivities
class d_spectrum:
    def __init__(self, fractions, diffusivities):
        self.fractions = fractions
        self.diffusivities = diffusivities

    def plot(self, title=None):
        # plt.plot(diffusivities, fractions, 'o')
        plt.vlines(self.diffusivities, np.zeros(len(self.fractions)), self.fractions)
        plt.xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)")
        plt.ylabel("Relative Fraction")
        plt.xlim(left=0)
        if title is not None:
            plt.title(title)
        plt.show()


# a collection of diffusivity spectra, probably
# result of Gibbs' sampling
class d_spectra_sample:
    def __init__(self, diffusivities):
        self.diffusivities = diffusivities
        self.sample = []  # a list of samples

    def plot(
        self, save_filename=None, title=None, start=0, end=-1, skip=False, ax=None
    ):
        if not ax:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)")
            ax.set_ylabel("Relative Fraction")
            tick_labels = []
            for number in self.diffusivities:
                tick_labels.append(str(number))
            if skip:
                for i in range(0, len(self.diffusivities)):
                    if i % 2 == 1:
                        tick_labels[i] = ""
            print(tick_labels)
            ax.set_xticklabels(tick_labels, rotation=45)
        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 12})
        sample_arr = np.asarray(self.sample)[start:end]
        ax.boxplot(
            sample_arr,
            showfliers=False,
            manage_ticks=False,
            showmeans=True,
            meanline=True,
        )
        # store = ax.boxplot(sample_arr, showfliers=True, manage_ticks=False, showmeans=True, meanline=True, whis=10)
        # sum_means = np.sum([store['means'][i]._y[0] for i in range(10)])
        # print(sum_means)
        if save_filename is not None:
            plt.savefig(save_filename)

    def diagnostic_plot(self):
        plt.plot(np.asarray(self.sample))
        plt.show()

    def normalize(self):
        for i in range(0, len(self.sample)):
            sum = np.sum(self.sample[i])
            self.sample[i] = self.sample[i] / sum


class MFVB_TruncatedMVN:
    def __init__(
        self, signal_data, recon_diffusivities, sigma, L1_lambda=0.0, L2_lambda=0.0
    ):
        self.signal_data = signal_data
        self.recon_diffusivities = recon_diffusivities
        self.sigma = sigma
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.m = len(recon_diffusivities)
        self.n = len(signal_data.b_values)

        # Precompute U matrix
        self.U = np.exp(-np.outer(signal_data.b_values, recon_diffusivities))

        # Calculate MVN posterior parameters
        self.M, self.Sigma_inv, self.Sigma_inv_M = self.calculate_MVN_posterior_params()

        # Initialize variational parameters using MVN posterior
        self.initialize_from_mvn()

    def initialize_from_mvn(self):
        mode = self.calculate_trunc_MVN_mode(self.M, self.Sigma_inv, self.Sigma_inv_M)

        # Use the mode to initialize alpha and beta
        self.alpha = mode + 1  # Adding 1 to ensure alpha > 0
        self.beta = self.alpha / mode  # This sets E[R] = mode initially

    def calculate_MVN_posterior_params(self):
        U_outer_prods = np.dot(self.U.T, self.U)
        Sigma_inv = (
            1.0 / (self.sigma * self.sigma)
        ) * U_outer_prods + self.L2_lambda * np.eye(self.m)
        weighted_U_vecs = np.dot(self.signal_data.signal_values, self.U)
        Sigma_inv_M = (
            1.0 / (self.sigma * self.sigma) * weighted_U_vecs
        ) - self.L1_lambda * np.ones(self.m)
        M = np.linalg.solve(Sigma_inv, Sigma_inv_M)
        return M, Sigma_inv, Sigma_inv_M

    def calculate_trunc_MVN_mode(self, M, Sigma_inv, Sigma_inv_M):
        P = matrix(Sigma_inv)
        Q = matrix(-Sigma_inv_M)
        G = matrix(-np.identity(self.m), tc="d")
        H = matrix(np.zeros(self.m))
        sol = solvers.qp(P, Q, G, H)
        return np.array(sol["x"]).flatten()

    def objective(self, params):
        alpha, beta = params[: self.m], params[self.m :]

        # Expectation of log-likelihood
        E_R = alpha / beta
        E_R2 = alpha * (alpha + 1) / (beta**2)

        log_likelihood = -0.5 * np.sum(np.diag(self.Sigma_inv) * E_R2) + np.sum(
            self.Sigma_inv_M * E_R
        )

        # KL divergence
        kl_divergence = np.sum(
            alpha * np.log(beta)
            - np.log(gamma(alpha))
            + (alpha - 1) * (digamma(alpha) - np.log(beta))
            - beta * E_R
        )

        return -(log_likelihood - kl_divergence)

    def update_parameters(self):
        x0 = np.concatenate([self.alpha, self.beta])
        res = minimize(
            self.objective, x0, method="L-BFGS-B", bounds=[(1e-10, None)] * (2 * self.m)
        )
        self.alpha, self.beta = res.x[: self.m], res.x[self.m :]

    def sample(self, n_samples):
        samples = np.zeros((n_samples, self.m))
        for i in range(self.m):
            samples[:, i] = gamma_dist.rvs(
                a=self.alpha[i], scale=1 / self.beta[i], size=n_samples
            )
        return samples


class MFVB_TruncatedMVN_LogNormal:
    def __init__(
        self, signal_data, recon_diffusivities, sigma, L1_lambda=0.0, L2_lambda=0.0
    ):
        self.signal_data = signal_data
        self.recon_diffusivities = recon_diffusivities
        self.sigma = sigma
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.m = len(recon_diffusivities)
        self.n = len(signal_data.b_values)

        # Precompute U matrix
        self.U = np.exp(-np.outer(signal_data.b_values, recon_diffusivities))

        # Calculate MVN posterior parameters
        self.M, self.Sigma_inv, self.Sigma_inv_M = self.calculate_MVN_posterior_params()

        # Initialize variational parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        mode = self.calculate_trunc_MVN_mode(self.M, self.Sigma_inv, self.Sigma_inv_M)

        # Initialize mu and sigma for log-normal distributions
        self.mu = np.log(np.maximum(mode, 1e-10))  # Ensure positive values
        self.sigma = np.ones_like(mode)

    def calculate_MVN_posterior_params(self):
        U_outer_prods = np.dot(self.U.T, self.U)
        Sigma_inv = (
            1.0 / (self.sigma * self.sigma)
        ) * U_outer_prods + self.L2_lambda * np.eye(self.m)
        weighted_U_vecs = np.dot(self.signal_data.signal_values, self.U)
        Sigma_inv_M = (
            1.0 / (self.sigma * self.sigma) * weighted_U_vecs
        ) - self.L1_lambda * np.ones(self.m)
        M = np.linalg.solve(Sigma_inv, Sigma_inv_M)
        return M, Sigma_inv, Sigma_inv_M

    def calculate_trunc_MVN_mode(self, M, Sigma_inv, Sigma_inv_M):
        P = matrix(Sigma_inv)
        Q = matrix(-Sigma_inv_M)
        G = matrix(-np.identity(self.m), tc="d")
        H = matrix(np.zeros(self.m))
        sol = solvers.qp(P, Q, G, H)
        return np.array(sol["x"]).flatten()

    def objective(self, params):
        mu, sigma = params[: self.m], params[self.m :]

        # Ensure sigma is positive
        sigma = np.maximum(sigma, 1e-10)

        # Expectation of R and R^2 for log-normal distribution
        E_R = np.exp(mu + 0.5 * sigma**2)
        E_R2 = np.exp(2 * mu + 2 * sigma**2)

        # Log-likelihood
        log_likelihood = -0.5 * np.sum(np.diag(self.Sigma_inv) * E_R2) + np.sum(
            self.Sigma_inv_M * E_R
        )

        # KL divergence for log-normal distribution
        kl_divergence = np.sum(np.log(sigma) + 0.5 * (1 + mu**2 / sigma**2))

        return -(log_likelihood - kl_divergence)

    def update_parameters(self):
        x0 = np.concatenate([self.mu, self.sigma])

        # Bounds for parameters (mu can be any real number, sigma must be positive)
        bounds = [(None, None)] * self.m + [(1e-10, None)] * self.m

        res = minimize(self.objective, x0, method="L-BFGS-B", bounds=bounds)

        self.mu, self.sigma = res.x[: self.m], np.maximum(res.x[self.m :], 1e-10)

        print("After optimization:")
        print("mu:", self.mu)
        print("sigma:", self.sigma)

    def sample(self, n_samples):
        samples = np.zeros((n_samples, self.m))
        for i in range(self.m):
            print(f"Sampling for dimension {i}:")
            print(f"mu: {self.mu[i]}, sigma: {self.sigma[i]}")
            samples[:, i] = lognorm.rvs(
                s=self.sigma[i], scale=np.exp(self.mu[i]), size=n_samples
            )
        return samples


class ImprovedMFVB_TruncatedMVN:
    def __init__(
        self, signal_data, recon_diffusivities, sigma, L1_lambda=0.0, L2_lambda=0.00001
    ):
        self.signal_data = signal_data
        self.recon_diffusivities = recon_diffusivities
        self.sigma = sigma
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.m = len(recon_diffusivities)
        self.n = len(signal_data.b_values)

        # Precompute U matrix
        self.U = np.exp(-np.outer(signal_data.b_values, recon_diffusivities))

        # Calculate MVN posterior parameters
        self.M, self.Sigma_inv, self.Sigma_inv_M = self.calculate_MVN_posterior_params()

        # Initialize variational parameters
        self.initialize_parameters()

    def calculate_MVN_posterior_params(self):
        U_outer_prods = np.dot(self.U.T, self.U)
        Sigma_inv = (
            1.0 / (self.sigma * self.sigma)
        ) * U_outer_prods + self.L2_lambda * np.eye(self.m)
        weighted_U_vecs = np.dot(self.signal_data.signal_values, self.U)
        Sigma_inv_M = (
            1.0 / (self.sigma * self.sigma) * weighted_U_vecs
        ) - self.L1_lambda * np.ones(self.m)
        M = np.linalg.solve(Sigma_inv, Sigma_inv_M)
        return M, Sigma_inv, Sigma_inv_M

    def calculate_trunc_MVN_mode(self, M, Sigma_inv, Sigma_inv_M):
        P = matrix(Sigma_inv)
        Q = matrix(-Sigma_inv_M)
        G = matrix(-np.identity(self.m), tc="d")
        H = matrix(np.zeros(self.m))
        sol = solvers.qp(P, Q, G, H)
        return np.array(sol["x"]).flatten()

    def initialize_parameters(self):
        mode = self.calculate_trunc_MVN_mode(self.M, self.Sigma_inv, self.Sigma_inv_M)

        # Initialize parameters for both Gamma and LogNormal distributions
        self.alpha = mode + 1
        self.beta = self.alpha / mode
        self.mu = np.log(np.maximum(mode, 1e-10))
        self.sigma = np.ones_like(mode)

        # Initialize mixture weights
        self.gamma_weight = 0.5
        self.lognormal_weight = 0.5

    def objective(self, params):
        alpha, beta = params[: self.m], params[self.m : 2 * self.m]
        mu, sigma = params[2 * self.m : 3 * self.m], params[3 * self.m : 4 * self.m]
        gamma_weight = params[-2]
        lognormal_weight = params[-1]

        # Ensure parameters are positive
        alpha = np.maximum(alpha, 1e-10)
        beta = np.maximum(beta, 1e-10)
        sigma = np.maximum(sigma, 1e-10)

        # Expectation of R and R^2 for Gamma distribution
        E_R_gamma = alpha / beta
        E_R2_gamma = alpha * (alpha + 1) / (beta**2)

        # Expectation of R and R^2 for LogNormal distribution
        E_R_lognormal = np.exp(mu + 0.5 * sigma**2)
        E_R2_lognormal = np.exp(2 * mu + 2 * sigma**2)

        # Mixture expectations
        E_R = gamma_weight * E_R_gamma + lognormal_weight * E_R_lognormal
        E_R2 = gamma_weight * E_R2_gamma + lognormal_weight * E_R2_lognormal

        # Log-likelihood
        log_likelihood = -0.5 * np.sum(np.diag(self.Sigma_inv) * E_R2) + np.sum(
            self.Sigma_inv_M * E_R
        )

        # KL divergence for Gamma distribution
        kl_gamma = np.sum(
            alpha * np.log(beta)
            - np.log(gamma(alpha))
            + (alpha - 1) * (digamma(alpha) - np.log(beta))
            - beta * E_R_gamma
        )

        # KL divergence for LogNormal distribution
        kl_lognormal = np.sum(np.log(sigma) + 0.5 * (1 + mu**2 / sigma**2))

        # Total KL divergence
        kl_divergence = gamma_weight * kl_gamma + lognormal_weight * kl_lognormal

        # Entropy of the mixture weights
        entropy = -gamma_weight * np.log(gamma_weight) - lognormal_weight * np.log(
            lognormal_weight
        )

        return -(log_likelihood - kl_divergence + entropy)

    def update_parameters(self):
        x0 = np.concatenate(
            [
                self.alpha,
                self.beta,
                self.mu,
                self.sigma,
                [self.gamma_weight, self.lognormal_weight],
            ]
        )

        # Bounds for parameters
        bounds = [(1e-10, None)] * (4 * self.m) + [(0, 1), (0, 1)]

        # Constraint to ensure mixture weights sum to 1
        def constraint(x):
            return x[-1] + x[-2] - 1

        cons = {"type": "eq", "fun": constraint}

        res = minimize(
            self.objective, x0, method="SLSQP", bounds=bounds, constraints=cons
        )

        self.alpha, self.beta = res.x[: self.m], res.x[self.m : 2 * self.m]
        self.mu, self.sigma = (
            res.x[2 * self.m : 3 * self.m],
            res.x[3 * self.m : 4 * self.m],
        )
        self.gamma_weight, self.lognormal_weight = res.x[-2], res.x[-1]

    def sample(self, n_samples):
        samples = np.zeros((n_samples, self.m))
        for i in range(self.m):
            gamma_samples = gamma_dist.rvs(
                a=self.alpha[i], scale=1 / self.beta[i], size=n_samples
            )
            lognormal_samples = lognorm.rvs(
                s=self.sigma[i], scale=np.exp(self.mu[i]), size=n_samples
            )

            mixture = np.random.choice(
                [0, 1], size=n_samples, p=[self.gamma_weight, self.lognormal_weight]
            )
            samples[:, i] = mixture * lognormal_samples + (1 - mixture) * gamma_samples

        return samples


# does not work correctly!
class MFVB_TruncatedNormal:
    def __init__(
        self, signal_data, recon_diffusivities, sigma, L1_lambda=0.0, L2_lambda=0.00001
    ):
        self.signal_data = signal_data
        self.recon_diffusivities = recon_diffusivities
        self.sigma = sigma
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.m = len(recon_diffusivities)
        self.n = len(signal_data.b_values)

        # Precompute U matrix
        self.U = np.exp(-np.outer(signal_data.b_values, recon_diffusivities))

        # Calculate MVN posterior parameters
        self.M, self.Sigma_inv, self.Sigma_inv_M = self.calculate_MVN_posterior_params()

        # Initialize variational parameters
        self.initialize_parameters()

    def calculate_MVN_posterior_params(self):
        U_outer_prods = np.dot(self.U.T, self.U)
        Sigma_inv = (
            1.0 / (self.sigma * self.sigma)
        ) * U_outer_prods + self.L2_lambda * np.eye(self.m)
        weighted_U_vecs = np.dot(self.signal_data.signal_values, self.U)
        Sigma_inv_M = (
            1.0 / (self.sigma * self.sigma) * weighted_U_vecs
        ) - self.L1_lambda * np.ones(self.m)
        M = np.linalg.solve(Sigma_inv, Sigma_inv_M)
        return M, Sigma_inv, Sigma_inv_M

    def initialize_parameters(self):
        self.mu = np.maximum(self.M, 0)
        self.sigma = np.sqrt(np.diag(np.linalg.inv(self.Sigma_inv)))

    def truncated_normal_moments(self, mu, sigma):
        alpha = -mu / sigma
        Z = log_ndtr(-alpha)  # log of the CDF of standard normal
        log_pdf = norm.logpdf(alpha)

        # Compute E[X] and E[X^2] in a numerically stable way
        E_X = mu + sigma * np.exp(log_pdf - Z)
        E_X2 = mu**2 + sigma**2 + mu * sigma * np.exp(log_pdf - Z)

        return E_X, E_X2

    def objective(self, params):
        mu, log_sigma = params[: self.m], params[self.m :]
        sigma = np.exp(log_sigma)

        # Compute expectations
        E_R, E_R2 = self.truncated_normal_moments(mu, sigma)

        # Log-likelihood (computed in log space for stability)
        log_likelihood = -0.5 * np.sum(np.diag(self.Sigma_inv) * E_R2) + np.sum(
            self.Sigma_inv_M * E_R
        )

        # KL divergence (computed in log space for stability)
        alpha = -mu / sigma
        log_Z = log_ndtr(-alpha)
        kl_divergence = np.sum(log_Z + 0.5 * np.log(2 * np.pi) + log_sigma + 0.5)

        return -(log_likelihood - kl_divergence)

    def update_parameters(self):
        x0 = np.concatenate([self.mu, np.log(self.sigma)])

        # Add bounds to prevent extremely large or small values
        bounds = [(None, None)] * self.m + [(-10, 10)] * self.m

        res = minimize(self.objective, x0, method="L-BFGS-B", bounds=bounds)
        self.mu, log_sigma = res.x[: self.m], res.x[self.m :]
        self.sigma = np.exp(log_sigma)

    def sample(self, n_samples):
        return truncnorm.rvs(
            a=0, b=np.inf, loc=self.mu, scale=self.sigma, size=(n_samples, self.m)
        )


# does not work correctly!
class GaussianMixtureVI:
    def __init__(
        self,
        signal_data,
        recon_diffusivities,
        sigma,
        n_components=3,
        L1_lambda=0.0,
        L2_lambda=0.00001,
    ):
        self.signal_data = signal_data
        self.recon_diffusivities = recon_diffusivities
        self.sigma = sigma
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.m = len(recon_diffusivities)
        self.n = len(signal_data.b_values)
        self.n_components = n_components

        # Precompute U matrix
        self.U = np.exp(-np.outer(signal_data.b_values, recon_diffusivities))

        # Calculate MVN posterior parameters
        self.M, self.Sigma_inv, self.Sigma_inv_M = self.calculate_MVN_posterior_params()

        # Initialize GMM
        self.gmm = GaussianMixture(n_components=n_components, covariance_type="diag")

    def calculate_MVN_posterior_params(self):
        U_outer_prods = np.dot(self.U.T, self.U)
        Sigma_inv = (
            1.0 / (self.sigma * self.sigma)
        ) * U_outer_prods + self.L2_lambda * np.eye(self.m)
        weighted_U_vecs = np.dot(self.signal_data.signal_values, self.U)
        Sigma_inv_M = (
            1.0 / (self.sigma * self.sigma) * weighted_U_vecs
        ) - self.L1_lambda * np.ones(self.m)
        M = np.linalg.solve(Sigma_inv, Sigma_inv_M)
        return M, Sigma_inv, Sigma_inv_M

    def fit(self, n_samples=10000):
        # Generate initial samples from truncated normal
        initial_samples = np.maximum(
            np.random.multivariate_normal(
                self.M, np.linalg.inv(self.Sigma_inv), size=n_samples
            ),
            0,
        )

        # Fit GMM to these samples
        self.gmm.fit(initial_samples)

    def sample(self, n_samples):
        samples, _ = self.gmm.sample(n_samples)
        return np.maximum(samples, 0)  # Ensure non-negativity


def compute_mfvb_object(
    input_dict_path,
    output_hdf5_path,
    output_json_path,
    metadata_path,
    output_print_path,
    recon_diffusivities,
    iters=100000,
    c=150,
    l1_lambda=0.0,
    l2_lambda=0.00001,
):
    # Load input data
    with open(input_dict_path, "r") as f:
        input_dict = json.load(f)

    # Create output dict
    output_dict = {
        "normal_pz_s2": [],
        "normal_tz_s3": [],
        "tumor_pz_s1": [],
        "tumor_tz_s1": [],
        "Neglected!": [],
    }

    # Create HDF5 file
    with h5py.File(output_hdf5_path, "w") as hf:
        for patient_key in tqdm(input_dict, desc="Patients"):
            for roi_key in tqdm(input_dict[patient_key], desc="ROIs"):
                if roi_key.startswith("roi"):
                    b_values_roi = (
                        np.array(input_dict[patient_key][roi_key]["b_values"]) / 1000
                    )
                    max_sig = np.array(
                        input_dict[patient_key][roi_key]["signal_values"][0]
                    )
                    signal_values_roi = (
                        np.array(input_dict[patient_key][roi_key]["signal_values"])
                        / max_sig
                    )
                    sig_obj = signal_data(signal_values_roi, b_values_roi)
                    v_count = input_dict[patient_key][roi_key]["v_count"]
                    snr = np.sqrt(v_count / 16) * c
                    sigma = 1.0 / snr
                    a_region = input_dict[patient_key][roi_key]["anatomical_region"]

                    # # Run MFVB for the current ROI
                    # mfvb = MFVB_TruncatedNormal(
                    #     sig_obj,
                    #     recon_diffusivities,
                    #     sigma,
                    #     L1_lambda=l1_lambda,
                    #     L2_lambda=l2_lambda,
                    # )
                    # mfvb.update_parameters()
                    # samples = mfvb.sample(iters)

                    gmm_vi = GaussianMixtureVI(
                        sig_obj,
                        recon_diffusivities,
                        sigma,
                        n_components=3,
                        L1_lambda=l1_lambda,
                        L2_lambda=l2_lambda,
                    )
                    gmm_vi.fit()
                    samples = gmm_vi.sample(iters)

                    # Normalize samples
                    samples /= samples.sum(axis=1, keepdims=True)

                    # Discard the first 10000 (burn-in)
                    samples = samples[10000:]

                    # Create metadata dict
                    mfvb_sample_dict = {
                        "a_region": a_region,
                        "snr": snr,
                        "patient_key": patient_key,
                        "gs": "NaN",
                        "target": "NaN",
                        "patient_age": "NaN",
                        "diffusivities": recon_diffusivities.tolist(),
                    }

                    # Store large array in HDF5
                    dataset_name = f"{patient_key}_{roi_key}"
                    hf.create_dataset(dataset_name, data=samples, compression="gzip")

                    # Add reference to HDF5 dataset in metadata
                    mfvb_sample_dict["hdf5_dataset"] = dataset_name

                    # Aggregate samples per roi for later plotting
                    output_dict[a_region].append(mfvb_sample_dict)

    # Save metadata as JSON
    with open(output_json_path, "w") as f:
        json.dump(output_dict, f)

    print_all(output_print_path, output_dict, output_hdf5_path, 3, 2)
    print_avg(
        output_print_path, output_dict, output_hdf5_path, recon_diffusivities, 2, 2
    )


def init_plot_matrix(m, n, diffusivities):
    """
    Create graph layout on PDF with adjustments for better fitting
    """
    # Increase figure size
    fig, axarr = plt.subplots(m, n, sharex="col", sharey="row", figsize=(10, 10))

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    arr_ij = list(np.ndindex(axarr.shape))
    subplots = [axarr[index] for index in arr_ij]

    for s, splot in enumerate(subplots):
        last_row = m * n - s < n + 1
        first_in_row = s % n == 0
        splot.grid(color="0.75")
        if last_row:
            splot.set_xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)", fontsize=10)
            str_diffs = [""] + list(map(str, diffusivities)) + [""]
            splot.set_xticks(
                np.arange(len(str_diffs)), labels=str_diffs, rotation=90, fontsize=6
            )
        if first_in_row:
            splot.set_ylabel("Relative Fraction", fontsize=10)

        # Set tick parameters for both axes
        splot.tick_params(axis="both", which="major", labelsize=10)

    # plt.tight_layout()
    return fig, axarr, subplots


def print_all(output_path: str, rois: dict, hdf5_path: str, m: int, n: int) -> None:
    KEY_TO_PDF_NAME = {
        "normal_pz_s2": "/npz.pdf",
        "normal_tz_s3": "/ntz.pdf",
        "tumor_pz_s1": "/tpz.pdf",
        "tumor_tz_s1": "/ttz.pdf",
        "Neglected!": "/neglected.pdf",
    }
    KEY_TO_CSV_NAME = {
        "normal_pz_s2": "/npz.csv",
        "normal_tz_s3": "/ntz.csv",
        "tumor_pz_s1": "/tpz.csv",
        "tumor_tz_s1": "/ttz.csv",
        "Neglected!": "/neglected.csv",
    }

    plt.rcParams.update({"font.size": 6})
    with h5py.File(hdf5_path, "r") as hf:
        for zone_key, zone_list in tqdm(rois.items(), desc="ROIs Zones", position=0):
            if len(zone_list) == 0:
                continue
            with PdfPages(os.path.join(output_path + KEY_TO_PDF_NAME[zone_key])) as pdf:
                f, axarr, subplots = init_plot_matrix(
                    m, n, zone_list[0]["diffusivities"]
                )
                n_pages = 1
                for i, sample_dict in tqdm(
                    enumerate(zone_list), desc="Samples in Zone", position=1
                ):
                    dataset = hf[sample_dict["hdf5_dataset"]]
                    sample = d_spectra_sample(sample_dict["diffusivities"])
                    sample.sample = dataset[()]

                    title = f'{sample_dict["patient_key"]}|{sample_dict["gs"]}|{sample_dict["target"]}|{int(sample_dict["snr"])}|{sample_dict["patient_age"]}'
                    sample.plot(ax=subplots[i - (n_pages - 1) * m * n], title=title)
                    if i == n_pages * m * n - 1:
                        pdf.savefig()
                        plt.close(f)
                        n_pages += 1
                        f, axarr, subplots = init_plot_matrix(
                            m, n, zone_list[0]["diffusivities"]
                        )
                pdf.savefig()
                plt.close(f)

            df = pd.DataFrame()
            for sample_dict in zone_list:
                dataset = hf[sample_dict["hdf5_dataset"]]
                for diff, sample in zip(
                    sample_dict["diffusivities"], np.transpose(dataset[()])
                ):
                    boxplot_stats = {
                        "Patient": sample_dict["patient_key"],
                        "ROI": sample_dict["a_region"],
                        "SNR": sample_dict["snr"],
                        "Gleason Score": sample_dict["gs"],
                        "Target": sample_dict["target"],
                        "Patient Age": sample_dict["patient_age"],
                        "Diffusivity": diff,
                        "Min": np.min(sample),
                        "Q1": np.percentile(sample, 25),
                        "Median": np.median(sample),
                        "Mean": np.mean(sample),
                        "Q3": np.percentile(sample, 75),
                        "Max": np.max(sample),
                    }
                    df = pd.concat(
                        [df, pd.DataFrame([boxplot_stats])], ignore_index=True
                    )
            df.to_csv(
                os.path.join(output_path + KEY_TO_CSV_NAME[zone_key]), index=False
            )


def print_avg(
    output_path: str, rois: dict, hdf5_path: str, diffusivities: list, m: int, n: int
) -> None:
    avg_dict = {}
    with h5py.File(hdf5_path, "r") as hf:
        for zone_key, zone_list in rois.items():
            if len(zone_list) == 0:
                continue
            avg_sample_obj = d_spectra_sample(diffusivities)
            avg_sample_obj.sample = np.mean(
                [hf[d["hdf5_dataset"]][()] for d in zone_list], axis=0
            )
            avg_dict[zone_key] = avg_sample_obj

    with PdfPages(os.path.join(output_path + "roi_avgs.pdf")) as pdf:
        f, axarr, subplots = init_plot_matrix(m, n, diffusivities)
        for i, (key, avg_sample) in enumerate(avg_dict.items()):
            if key != "Neglected!":
                avg_sample.plot(ax=subplots[i - m * n], title=key)
        pdf.savefig()
        plt.close(f)

    df = pd.DataFrame()
    for title, avg_sample in avg_dict.items():
        for diff, sample in zip(diffusivities, np.transpose(avg_sample.sample)):
            boxplot_stats = {
                "ROI": title,
                "Diffusivity": diff,
                "Min": np.min(sample),
                "Q1": np.percentile(sample, 25),
                "Median": np.median(sample),
                "Mean": np.mean(sample),
                "Q3": np.percentile(sample, 75),
                "Max": np.max(sample),
            }
            df = pd.concat([df, pd.DataFrame([boxplot_stats])], ignore_index=True)
    df.to_csv(os.path.join(output_path + "roi_avgs.csv"), index=False)


if __name__ == "__main__":
    # Load configuration and run the MFVB implementation
    import yaml

    configs = {}
    base_config_path = os.path.join(os.getcwd() + "/configs.yaml")
    with open(base_config_path, "r") as file:
        configs.update(yaml.safe_load(file))

    recon_diffusivities = np.array(
        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
    )
    compute_mfvb_object(
        input_dict_path=configs["INPUT_DICT_PATH"],
        output_hdf5_path=configs["OUTPUT_HDF5_PATH"],
        output_json_path=configs["OUTPUT_JSON_PATH"],
        metadata_path=configs["METADATA_PATH"],
        output_print_path=configs["OUTPUT_PRINT_ROI_PATH"],
        recon_diffusivities=recon_diffusivities,
    )
