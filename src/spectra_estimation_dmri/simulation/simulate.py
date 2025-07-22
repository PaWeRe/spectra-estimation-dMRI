import numpy as np


def generate_simulated_signal(
    snr, diff_values, b_values, true_spectrum, no_noise=False
):
    """
    Generate simulated diffusion MRI signal with or without Gaussian noise.

    Args:
        snr: Signal-to-noise ratio (float)
        diff_values: Array of diffusivity values (np.ndarray)
        b_values: Array of b-values (np.ndarray)
        true_spectrum: Array of fractions (np.ndarray, shape [n_diff,])
        no_noise: If True, return noiseless signal (bool, default=False)

    Returns:
        signals: Simulated signal (np.ndarray, shape [n_bvals,])
    """
    s0 = 1.0
    # Generate noiseless signal
    signal = np.zeros_like(b_values)
    for i, d in enumerate(diff_values):
        signal += s0 * true_spectrum[i] * np.exp(-b_values * d)

    if no_noise:
        return signal  # Return noiseless signal
    else:
        # Add Gaussian noise
        sigma = 1.0 / snr
        noisy_signal = signal + np.random.normal(0, sigma, size=signal.shape)
        return noisy_signal
