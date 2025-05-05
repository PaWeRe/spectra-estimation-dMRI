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
    def __init__(self, signal_data, diffusivities, sigma, **kwargs):
        self.signal_data = signal_data
        self.diffusivities = diffusivities
        self.sigma = sigma
        self.kwargs = kwargs

    @abstractmethod
    def sample(self, iterations: int):
        """Run the sampler for a given number of iterations and return a d_spectra_sample object."""
        pass
