import numpy as np
import torch
import torch.optim as optim
from .base import BaseSampler, d_spectra_sample


class VariationalSampler(BaseSampler):
    """
    Variational Inference Sampler for non-negative truncated multivariate normal.
    Uses a diagonal Gaussian with softplus transform to enforce non-negativity.
    Optimizes the ELBO using the reparameterization trick.
    """

    def __init__(self, signal_data, diffusivities, sigma, config=None, **kwargs):
        super().__init__(signal_data, diffusivities, sigma, config=config, **kwargs)
        self.N = len(diffusivities)
        # Variational parameters: mean and log-variance
        self.mu = torch.nn.Parameter(torch.zeros(self.N, dtype=torch.float32))
        self.log_var = torch.nn.Parameter(torch.zeros(self.N, dtype=torch.float32))
        self.device = torch.device("cpu")
        self.optimizer = optim.Adam([self.mu, self.log_var], lr=1e-2)
        self.n_elbo_samples = kwargs.get("n_elbo_samples", 10)
        self.n_final_samples = kwargs.get("n_final_samples", 1000)
        self.L1_lambda = kwargs.get("l1_lambda", 0.0)
        self.L2_lambda = kwargs.get("l2_lambda", 0.0)

        # Convert data to torch tensors
        self.signal_values = torch.tensor(
            signal_data.signal_values, dtype=torch.float32, device=self.device
        )
        self.b_values = torch.tensor(
            signal_data.b_values, dtype=torch.float32, device=self.device
        )
        self.diffusivities = torch.tensor(
            diffusivities, dtype=torch.float32, device=self.device
        )
        self.sigma = float(sigma)

    def _log_likelihood(self, R):
        # R: (..., N)
        # Compute predicted signal
        U = torch.exp(-self.b_values.unsqueeze(-1) * self.diffusivities)  # (M, N)
        pred = torch.matmul(U, R.T).T  # (..., M)
        # Gaussian log-likelihood
        ll = (
            -0.5 * torch.sum((self.signal_values - pred) ** 2, dim=-1) / (self.sigma**2)
        )
        return ll

    def _log_prior(self, R):
        # Optional L2 and L1 regularization as Gaussian and Laplace priors
        lp = 0.0
        if self.L2_lambda > 0.0:
            lp = lp - 0.5 * self.L2_lambda * torch.sum(R**2, dim=-1)
        if self.L1_lambda > 0.0:
            lp = lp - self.L1_lambda * torch.sum(R, dim=-1)
        return lp

    def elbo(self, n_samples=10):
        # Sample from q(R) using reparameterization and softplus for non-negativity
        eps = torch.randn(n_samples, self.N, device=self.device)
        std = torch.exp(0.5 * self.log_var)
        R = torch.nn.functional.softplus(self.mu + eps * std)  # (n_samples, N)
        # Log-likelihood and log-prior
        log_lik = self._log_likelihood(R)
        log_prior = self._log_prior(R)
        # Entropy of diagonal Gaussian (before softplus)
        entropy = 0.5 * torch.sum(self.log_var + np.log(2 * np.pi * np.e))
        # ELBO: E_q[log p(data|R) + log p(R)] + entropy
        elbo = torch.mean(log_lik + log_prior) + entropy
        return elbo

    def sample(self, iterations: int, initial_R=None):
        # Optimize ELBO
        for it in range(iterations):
            self.optimizer.zero_grad()
            loss = -self.elbo(self.n_elbo_samples)
            loss.backward()
            self.optimizer.step()
            if (it % 100) == 0:
                print(
                    f"VariationalSampler iter {it}/{iterations}, ELBO: {-loss.item():.2f}"
                )
        # After optimization, sample from q(R) for uncertainty estimates
        with torch.no_grad():
            eps = torch.randn(self.n_final_samples, self.N, device=self.device)
            std = torch.exp(0.5 * self.log_var)
            samples = torch.nn.functional.softplus(self.mu + eps * std).cpu().numpy()
            initial_R = torch.nn.functional.softplus(self.mu).cpu().numpy()
        the_sample = d_spectra_sample(self.diffusivities.cpu().numpy())
        the_sample.initial_R = initial_R
        the_sample.sample = [s for s in samples]
        the_sample.normalize()
        return the_sample
