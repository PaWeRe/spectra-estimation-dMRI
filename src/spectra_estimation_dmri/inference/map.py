import numpy as np
import arviz as az
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum
import hashlib
import os


class MAPInference:
    def __init__(self, model, signal, config=None):
        self.model = model
        self.signal = signal
        self.config = config
        self.regularization = (
            getattr(config, "l2_lambda", 0.0) if config is not None else 0.0
        )

    # TODO: this should output a DiffusivitySpectraDataset object for easy ref in main.py
    def run(
        self,
        return_idata=True,
        save_dir=None,
        true_spectrum=None,
        unique_hash=None,
    ):
        # Compute MAP/NNLS estimate
        fractions = self.model.map_estimate(
            self.signal, regularization=self.regularization
        )
        idata = None
        inference_data_path = None
        if return_idata:
            idata = az.from_dict(
                posterior={
                    "R": fractions[None, :]  # shape (chain, draw, dim) or (draw, dim)
                }
            )
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                # Use config_hash for filename
                fname = f"{unique_hash}.nc"
                inference_data_path = os.path.join(save_dir, fname)
                idata.to_netcdf(inference_data_path)
        # Build DiffusivitySpectrum object
        spectrum = DiffusivitySpectrum(
            inference_method="map",
            signal_decay=(
                self.model.signal_decay if hasattr(self.model, "signal_decay") else None
            ),
            diffusivities=list(self.model.diffusivities),
            design_matrix_U=self.model.U_matrix().tolist(),
            spectrum_vector=list(fractions),
            spectrum_samples=None,
            spectrum_std=None,
            true_spectrum=(list(true_spectrum) if true_spectrum is not None else None),
            inference_data=(
                inference_data_path if inference_data_path is not None else ""
            ),
            unique_hash=unique_hash,
        )
        return spectrum
