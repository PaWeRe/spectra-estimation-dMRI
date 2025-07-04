"""Main script for approximative inference experiments and cancer biomarker development"""

import importlib.resources
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
import arviz as az
import hashlib
from spectra_estimation_dmri.utils.spectra_id import set_spectra_id

from spectra_estimation_dmri.data.loaders import load_bwh_signal_decays
from spectra_estimation_dmri.models.spectra_model import SpectrumModel
from spectra_estimation_dmri.simulation.simulate import generate_simulated_signal
from spectra_estimation_dmri.inference.map import MAPInference
from spectra_estimation_dmri.inference.gibbs import GibbsSampler
from spectra_estimation_dmri.diagnostics.diagnostics import run_all_diagnostics
from spectra_estimation_dmri.data.data_models import (
    SignalDecay,
    DiffusivitySpectrum,
    DiffusivitySpectraDataset,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main experiment pipeline with four main steps:
    - loads user-defined data as SignalDecayDataset objects (in simulated data case only consists of single SignalDecay object)
    - defines spectrum model for s=UR+eps
    - runs user-defined inference methods for spectrum reconstruction (output as DiffusivitySpectraDataset)
    - runs user-defined classification methods for cancer prediction
    - prints user-defined diagnostics
    """
    print(f"Hydra run directory: {os.getcwd()}")

    ### 1) DATA LOADING (REAL OR SIMULATED) ###
    if cfg.dataset.name == "bwh":
        signal_decay_dataset = load_bwh_signal_decays(
            json_path=cfg.dataset.signal_decays_json,
            metadata_path=cfg.dataset.metadata_csv,
        )
        signal_decay_dataset.summary()
        sample_decay = signal_decay_dataset.samples[0]
        spectrum_model = SpectrumModel(
            diffusivities=cfg.dataset.diff_values, b_values=sample_decay.b_values
        )  # SNR has to be set later, as changes for every SignalDecay
    elif cfg.dataset.name == "simulated":
        true_spectrum_name = cfg.dataset.true_spectrum_name
        true_spectrum_dict = cfg.dataset.true_spectra_dict
        true_spectrum = np.array(true_spectrum_dict[true_spectrum_name])
        spectrum_model = SpectrumModel(
            diffusivities=cfg.dataset.diff_values,
            b_values=cfg.dataset.b_values,
            snr=cfg.dataset.snr,
            true_spectrum=true_spectrum,
        )
        signal_decay_dataset = spectrum_model.simulate_signal_decay_dataset()
        signal_decay_dataset.summary()

    ### 2) INFERENCE (MAP OR FULL BAYESIAN) ###
    # Set up results directories
    # TODO: file retrieval - ceate one spectra_id that's unique and consists of all exp related and signal decay related parameters that lead to a different spectra (for inference subfolder)
    # TODO: plotting - create naming convention for spectra plotting and config (types: dist plotting, trace plotting, grouping: same snr, same ground truth spectrum etc. ... should all have dedicated plotting functions in Diffspectradatset)
    # TODO: inference methods - write down list of exp cli commands and implement necessary methods for combo's
    inference_dir = os.path.join(os.getcwd(), "results", "inference")
    plots_dir = os.path.join(os.getcwd(), "results", "plots")
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    spectra = []
    for i, signal_decay in enumerate(signal_decay_dataset.samples):
        # Prepare model for this signal
        model = SpectrumModel(
            diffusivities=cfg.dataset.diff_values,
            b_values=signal_decay.b_values,
            snr=signal_decay.snr,
        )
        model.signal_decay = signal_decay  # Attach for reference
        signal = signal_decay.signal_values
        true_spectrum = getattr(signal_decay, "true_spectrum", None)
        # Compute spectra_id for this spectrum
        spectra_id = set_spectra_id(signal_decay, cfg)
        nc_filename = f"{spectra_id}.nc"
        output_path = os.path.join(inference_dir, nc_filename)
        if os.path.exists(output_path) and not cfg.diagnostics.recompute:
            print(
                f"[INFO] Loaded precomputed result: {output_path} (spectra_id: {spectra_id})"
            )
            # Load spectrum from .nc file
            idata = az.from_netcdf(output_path)
            if cfg.inference.name == "map":
                spectrum_vector = idata.posterior["R"].values[0, :].tolist()
                spectrum_samples = None
                spectrum_std = None
            elif cfg.inference.name == "gibbs":
                samples = idata.posterior["R"].values.reshape(
                    -1, idata.posterior["R"].shape[-1]
                )
                spectrum_vector = samples.mean(axis=0).tolist()
                spectrum_samples = samples.tolist()
                spectrum_std = samples.std(axis=0).tolist()
            else:
                spectrum_vector = []
                spectrum_samples = None
                spectrum_std = None
            spectrum = DiffusivitySpectrum(
                inference_method=cfg.inference.name,
                signal_decay=model.signal_decay,
                diffusivities=list(model.diffusivities),
                design_matrix_U=model.U_matrix().tolist(),
                spectrum_vector=spectrum_vector,
                spectrum_samples=spectrum_samples,
                spectrum_std=spectrum_std,
                true_spectrum=(
                    list(true_spectrum) if true_spectrum is not None else None
                ),
                inference_data=output_path,
                spectra_id=spectra_id,
            )
            spectra.append(spectrum)
            continue
        if cfg.inference.name == "map":
            infer = MAPInference(model, signal, cfg.inference)
            spectrum = infer.run(
                return_idata=True,
                save_dir=inference_dir,
                true_spectrum=true_spectrum,
                unique_hash=spectra_id,
            )
            spectra.append(spectrum)
        elif cfg.inference.name == "gibbs":
            infer = GibbsSampler(model, signal, cfg.inference)
            spectrum = infer.run(
                return_idata=True,
                show_progress=True,
                save_dir=inference_dir,
                true_spectrum=true_spectrum,
                unique_hash=spectra_id,
            )
            spectra.append(spectrum)
        elif cfg.inference == "vb":
            pass  # To be implemented
    spectra_dataset = DiffusivitySpectraDataset(spectra=spectra)

    ### 3) CANCER PREDICTION ###
    # TODO: to be implemented, maybe look at base_classes.py and use custom classes

    ### 4) DIAGNOSTICS AND PLOTTING ###
    spectra_dataset.plot_group_boxplot(
        save_dir=plots_dir,
        config_info={
            "inference": cfg.inference,
        },
        group_by=cfg.diagnostics.group_by,
        show=True,
    )


if __name__ == "__main__":
    main()
