from typing import Optional, Literal
from pydantic import BaseModel, Field
import numpy as np
from abc import ABC, abstractmethod


class SignalDecay(BaseModel):
    patient: str  # TODO: add validator with unique patient ids
    signal_values: list[float]
    b_values: list[float]
    voxel_count: int
    snr: float
    a_region: Literal["pz", "tz"]
    is_tumor: bool = False
    ggg: int  # TODO: add validator for ggg (1-5)
    gs: str  # TODO: add validator for strings e.g. 3+4 or 3+4+5

    # TODO: implement
    def as_numpy(self):
        pass

    # TODO: implement
    def plot(self):
        pass


class DiffusivitySpectrum(BaseModel):
    signal_decay: SignalDecay
    diffusivities: list[
        float
    ]  # TODO: think of usage for finding opt diff discr for design matrix U (List of Lists?)
    # TODO: think of usage during simulation (List of Lists for stability analysis?)
    fractions_mode: list[float]  # posterior map estimate from reg. NNLS
    fractions_mean: list[float]  # bayesian posterior mean from approx. inf.
    fractions_variance: list[float]  # uncertainty / spread of posterior
    estimated_snr: Optional[
        float
    ]  # TODO: check typing to work for simulation (float) and on real data (None)
    noise_realizations: int

    # TODO: implement
    def as_numpy(self):
        pass

    # TODO: implement
    def plot(self):
        pass
