import numpy as np
import torch
import torch.optim as optim
from .base import BaseSampler, d_spectra_sample


class HierarchicalVariationalSampler(BaseSampler):
    """
    Variational Inference Sampler with Hierarchical Variance Learning.
    Uses softplus-transformed Gaussian to model non-negative truncated posteriors.
    Learns both mean and log-variance for each R_j component.
    """

    def __init__(self, signal_data, diffusivities, sigma, config=None, **kwargs):
        super().__init__(signal_data, diffusivities, sigma, config=config, **kwargs)
        self.N = len(diffusivities)

        # Variational parameters: mean and log-variance (hierarchical)
        self.mu = torch.nn.Parameter(torch.zeros(self.N, dtype=torch.float32))
        self.log_var = torch.nn.Parameter(torch.zeros(self.N, dtype=torch.float32))

        self.device = torch.device("cpu")
        self.optimizer = optim.Adam([self.mu, self.log_var], lr=1e-2)

        self.n_elbo_samples = kwargs.get("n_elbo_samples", 10)
        self.n_final_samples = kwargs.get("n_final_samples", 1000)
        self.L1_lambda = kwargs.get("l1_lambda", 0.0)
        self.L2_lambda = kwargs.get("l2_lambda", 0.0)

        # Use log-spaced diffusivity grid
        self.diffusivities = torch.tensor(
            np.logspace(np.log10(0.05), np.log10(3.0), self.N),
            dtype=torch.float32,
            device=self.device,
        )

        self.signal_values = torch.tensor(
            signal_data.signal_values, dtype=torch.float32, device=self.device
        )
        self.b_values = torch.tensor(
            signal_data.b_values, dtype=torch.float32, device=self.device
        )
        self.sigma = float(sigma)

    def _log_likelihood(self, R):
        U = torch.exp(-self.b_values.unsqueeze(-1) * self.diffusivities)  # (M, N)
        pred = torch.matmul(U, R.T).T  # (..., M)
        ll = (
            -0.5 * torch.sum((self.signal_values - pred) ** 2, dim=-1) / (self.sigma**2)
        )
        return ll

    def _log_prior(self, R):
        lp = 0.0
        if self.L2_lambda > 0.0:
            lp = lp - 0.5 * self.L2_lambda * torch.sum(R**2, dim=-1)
        if self.L1_lambda > 0.0:
            lp = lp - self.L1_lambda * torch.sum(R, dim=-1)
        return lp

    def elbo(self, n_samples=10):
        eps = torch.randn(n_samples, self.N, device=self.device)
        std = torch.exp(0.5 * self.log_var)  # std_j = exp(log_var_j / 2)
        R = torch.nn.functional.softplus(self.mu + eps * std)  # (n_samples, N)

        log_lik = self._log_likelihood(R)
        log_prior = self._log_prior(R)

        # Entropy of diagonal Gaussian before softplus
        entropy = 0.5 * torch.sum(self.log_var + np.log(2 * np.pi * np.e))

        return torch.mean(log_lik + log_prior) + entropy

    def sample(self, iterations: int, initial_R=None):
        for it in range(iterations):
            self.optimizer.zero_grad()
            loss = -self.elbo(self.n_elbo_samples)
            loss.backward()
            self.optimizer.step()
            if (it % 100) == 0:
                print(
                    f"Hierarchical VI iter {it}/{iterations}, ELBO: {-loss.item():.2f}"
                )

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
