import json
from pathlib import Path
from .data_models import SignalDecay, DiffusivitySpectrum


def load_signal_decay(json_path: str) -> SignalDecay:
    with open(json_path, "r") as f:
        data = json.load(f)
    return SignalDecay(**data)


def load_diffusivity_spectrum(json_path: str) -> DiffusivitySpectrum:
    with open(json_path, "r") as f:
        data = json.load(f)
    return DiffusivitySpectrum(**data)
