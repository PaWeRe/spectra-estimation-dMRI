import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

# Set style
sns.set_theme(style="whitegrid")
np.random.seed(42)


def sample_truncated_mvn_gibbs(M, Sigma_inv, num_samples=1000, burn_in=200):
    n = len(M)
    R = np.maximum(M, 0)
    samples = []

    Lambda = Sigma_inv
    Lambda_ii = np.diag(Lambda)

    for sweep in range(num_samples + burn_in):
        for i in range(n):
            sum_except_i = np.dot(Lambda[i, :], R - M) - Lambda[i, i] * (R[i] - M[i])
            mu_i = M[i] - sum_except_i / Lambda[i, i]
            sigma2_i = 1.0 / Lambda[i, i]
            sigma_i = np.sqrt(sigma2_i)
            a, b = (0 - mu_i) / sigma_i, np.inf
            R[i] = truncnorm.rvs(a, b, loc=mu_i, scale=sigma_i)

        if sweep >= burn_in:
            samples.append(R.copy())

        if sweep < 5 or sweep % 300 == 0:
            print(f"Sweep {sweep}: R = {R}")

    return np.array(samples)


# --- Set 3D parameters
M = np.array([1.0, 2.0, 1.5])
Sigma = np.array([[1.0, 0.6, 0.2], [0.6, 1.2, 0.4], [0.2, 0.4, 1.0]])
Sigma_inv = np.linalg.inv(Sigma)

# Run sampler
samples = sample_truncated_mvn_gibbs(M, Sigma_inv, num_samples=1000)


# Plot diagnostics
def plot_trace(samples):
    fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i in range(3):
        axs[i].plot(samples[:, i])
        axs[i].set_title(f"Trace of R[{i}]")
    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.show()


def plot_acfs(samples, lags=50):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i in range(3):
        plot_acf(samples[:, i], lags=lags, ax=axs[i])
        axs[i].set_title(f"ACF of R[{i}]")
    plt.tight_layout()
    plt.show()


def plot_running_mean(samples):
    running_mean = (
        np.cumsum(samples, axis=0) / np.arange(1, samples.shape[0] + 1)[:, None]
    )
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(running_mean[:, i], label=f"R[{i}]")
    plt.title("Running Mean of Samples")
    plt.xlabel("Iteration")
    plt.ylabel("Mean")
    plt.legend()
    plt.show()


def plot_3d_samples(samples):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.3)
    ax.set_xlabel("R[0]")
    ax.set_ylabel("R[1]")
    ax.set_zlabel("R[2]")
    ax.set_title("3D Scatter Plot of Samples")
    plt.show()


# --- Run diagnostics ---
plot_trace(samples)
plot_running_mean(samples)
plot_acfs(samples)
plot_3d_samples(samples)
