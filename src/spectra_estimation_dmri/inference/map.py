# map.py
import numpy as np
import arviz as az
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum
import os


class MAPInference:
    """MAP inference for spectrum estimation"""

    def __init__(self, model, signal_decay, config):
        self.model = model
        self.signal_decay = signal_decay
        self.config = config

    def run(
        self,
        return_idata=True,
        save_dir=None,
        true_spectrum=None,
        unique_hash=None,
    ):
        """
        Run MAP estimation.

        Returns:
            DiffusivitySpectrum object
        """
        signal = np.array(self.signal_decay.signal_values)

        # Compute MAP estimate
        fractions = self.model.map_estimate(signal)

        # Create inference data (single point estimate)
        idata = None
        inference_data_path = None
        if return_idata:
            idata = az.from_dict(
                posterior={
                    "R": fractions[None, None, :]
                }  # Add chain and draw dimensions
            )
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"{unique_hash}.nc"
                inference_data_path = os.path.join(save_dir, fname)
                idata.to_netcdf(inference_data_path)

        # Create result object
        if (
            hasattr(self.config.dataset, "spectrum_pair")
            and self.config.dataset.spectrum_pair is not None
        ):
            pair = self.config.dataset.spectrum_pair
            diffusivities = self.config.dataset.spectrum_pairs[pair].diff_values
            true_spectrum = self.config.dataset.spectrum_pairs[pair].true_spectrum
        else:
            # For BWH data, get directly from config
            diffusivities = self.config.dataset.diff_values
            true_spectrum = None
        spectrum = DiffusivitySpectrum(
            inference_method="map",
            signal_decay=self.signal_decay,
            diffusivities=diffusivities,
            design_matrix_U=self.model.U_matrix(),
            spectrum_init=fractions.tolist(),  # for map vector is equal to init
            spectrum_vector=fractions.tolist(),
            spectrum_samples=None,
            spectrum_std=None,
            true_spectrum=true_spectrum,
            inference_data=inference_data_path,
            spectra_id=unique_hash,
            prior_type=self.model.prior_config.type,
            prior_strength=self.model.prior_config.strength,
            data_snr=getattr(self.config.dataset, "snr", None),
            sampler_snr=getattr(
                self.config.dataset, "snr", None
            ),  # use same snr in map case
        )

        return spectrum
