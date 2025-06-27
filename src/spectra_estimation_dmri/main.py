"""Main script for approximative inference experiments and cancer biomarker development"""

import importlib.resources
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
import arviz as az
import hashlib

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
        )
        signal_decay_dataset = spectrum_model.simulate_signal_decay_dataset(
            true_spectrum
        )

    ### 2) INFERENCE (MAP OR FULL BAYESIAN) ###
    spectra = []
    config_str = str(cfg)  # config hash for reproducibility and file organization
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    config_tag = getattr(cfg, "tag", None)
    save_dir = os.path.join(os.getcwd(), f"inference_{config_hash}")
    os.makedirs(save_dir, exist_ok=True)
    for i, signal_decay in enumerate(signal_decay_dataset.samples):
        # Prepare model for this signal
        model = SpectrumModel(
            diffusivities=cfg.dataset.diff_values,
            b_values=signal_decay.b_values,
            snr=getattr(cfg.dataset, "snr", None),
        )
        model.signal_decay = signal_decay  # Attach for reference
        signal = signal_decay.signal_values
        ground_truth = getattr(signal_decay, "ground_truth_spectrum", None)
        if cfg.inference.name == "map":
            infer = MAPInference(model, signal, cfg.inference)
            spectrum = infer.run(
                return_idata=True,
                save_dir=save_dir,
                ground_truth_spectrum=ground_truth,
                config_hash=config_hash,
                config_tag=config_tag,
            )
            spectra.append(spectrum)
        elif cfg.inference.name == "gibbs":
            infer = GibbsSampler(model, signal, cfg.inference)
            spectrum = infer.run(
                return_idata=True,
                show_progress=True,
                save_dir=save_dir,
                ground_truth_spectrum=ground_truth,
                config_hash=config_hash,
                config_tag=config_tag,
            )
            spectra.append(spectrum)
        elif cfg.inference == "vb":
            pass  # To be implemented
    spectra_dataset = DiffusivitySpectraDataset(spectra=spectra)
    index_dict = {f"spec_{i}": s.inference_data for i, s in enumerate(spectra)}
    DiffusivitySpectraDataset.save_index(
        index_dict, os.path.join(save_dir, "spectra_index.json")
    )

    ### 3) CANCER PREDICTION ###
    # TODO: to be implemented, maybe look at base_classes.py and use custom classes

    ### 4) DIAGNOSTICS AND PLOTTING ###
    spectra_dataset.plot_group_boxplot(
        save_dir=save_dir,
        config_info={
            "inference": cfg.inference,
            "hash": config_hash,
            "tag": config_tag,
        },
        show=True,
    )


if __name__ == "__main__":
    main()
