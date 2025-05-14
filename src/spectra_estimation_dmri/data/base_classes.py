from typing import Optional
from abc import ABC, abstractmethod
from .data_models import DiffusivitySpectrum


class ProbabilisticModel(ABC):
    def __init__(self):
        self.estimated_spectrum: Optional[DiffusivitySpectrum] = None
        self.posterior_type: str = "truncated_mvn"  # TODO: consider Enum
        self.prior_type: str = "dirichlet_prior"  # TODO: consider Enum
        self.sigma: float = 0.0
        self.iterations: int = 0

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass


class VariationalBayes(ProbabilisticModel):
    def fit(self, *args, **kwargs):
        pass  # TODO: implement


class BaseSampler(ProbabilisticModel, ABC):
    def __init__(self):
        super().__init__()
        self.burn_in: int = 0

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class GibbsSampler(BaseSampler):
    def __init__(self):
        super().__init__()
        # TODO: add any Gibbs sampler specific attributes

    def sample(self, *args, **kwargs):
        pass  # TODO: implement


class CancerBiomarker(ABC):
    def __init__(self):
        self.prob_model: Optional[ProbabilisticModel] = None
        self.auc_values: list[float] = []
        self.p_values: list[float] = []

    @abstractmethod
    def plot(self):
        pass


class GleasonDifferentiator(CancerBiomarker):
    def plot(self):
        pass  # TODO: implement


class ROIDifferentiator(CancerBiomarker):
    def plot(self):
        pass  # TODO: implement
