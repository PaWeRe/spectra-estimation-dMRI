"""Main script for approximative inference experiments and cancer biomarker development"""

import os
import hydra
import wandb
import numpy as np
import arviz as az
from omegaconf import DictConfig, OmegaConf
from spectra_estimation_dmri.utils.spectra_id import set_spectra_id

from spectra_estimation_dmri.data.loaders import load_bwh_signal_decays
from spectra_estimation_dmri.models.prob_model import ProbabilisticModel
from spectra_estimation_dmri.simulation.simulate import generate_simulated_signal
from spectra_estimation_dmri.inference.map import MAPInference
from spectra_estimation_dmri.inference.gibbs import GibbsSampler
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
    # Create more descriptive run name and tags for aggregation
    run_name = f"{cfg.inference.name}-{cfg.prior.type}-snr{cfg.dataset.snr if hasattr(cfg.dataset, 'snr') else 'real'}"

    # Add tags for easy filtering and grouping
    tags = [
        f"inference_{cfg.inference.name}",
        f"prior_{cfg.prior.type}",
        f"dataset_{cfg.dataset.name}",
    ]

    if hasattr(cfg.dataset, "snr"):
        tags.append(f"snr_{cfg.dataset.snr}")
    if hasattr(cfg.dataset, "true_spectrum_name"):
        tags.append(f"spectrum_{cfg.dataset.true_spectrum_name}")

    run = wandb.init(
        project="bayesian-dMRI-biomarker",
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        tags=tags,
        group=f"{cfg.dataset.name}_{cfg.inference.name}",  # Group related runs
    )

    ### 1) DATA LOADING (REAL OR SIMULATED) ###
    spectra = []
    if cfg.dataset.name == "bwh":
        signal_decay_dataset = load_bwh_signal_decays(
            json_path=cfg.dataset.signal_decays_json,
            metadata_path=cfg.dataset.metadata_csv,
        )
        signal_decay_dataset.summary()
        spectrum_model = ProbabilisticModel(
            likelihood_config=cfg.likelihood,
            prior_config=cfg.prior,
            b_values=cfg.dataset.b_values,
            diffusivities=cfg.dataset.diff_values,
        )
        signal_decay_datasets = [signal_decay_dataset]
        n_realizations = 1
    elif cfg.dataset.name == "simulated":
        pair = cfg.dataset.spectrum_pair
        diff_values = cfg.dataset.spectrum_pairs[pair].diff_values
        true_spectrum = np.array(cfg.dataset.spectrum_pairs[pair].true_spectrum)
        spectrum_model = ProbabilisticModel(
            snr=cfg.dataset.snr,
            likelihood_config=cfg.likelihood,
            prior_config=cfg.prior,
            true_spectrum=true_spectrum,
            b_values=cfg.dataset.b_values,
            diffusivities=diff_values,
        )
        n_realizations = getattr(cfg.dataset, "noise_realizations", 1)
        signal_decay_datasets = [
            spectrum_model.simulate_signal_decay_dataset()
            for _ in range(n_realizations)
        ]
        for sdd in signal_decay_datasets:
            sdd.summary()

    ### 2) INFERENCE (MAP OR FULL BAYESIAN) ###
    # Set up results directories
    inference_dir = os.path.join(os.getcwd(), "results", "inference")
    plots_dir = os.path.join(os.getcwd(), "results", "plots")
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    for signal_decay_dataset in signal_decay_datasets:
        for i, signal_decay in enumerate(signal_decay_dataset.samples):
            # TODO: reconcile with model redundant model defs in phase 1 (there has to be cleaner way)
            if cfg.dataset.name == "simulated":
                pair = cfg.dataset.spectrum_pair
                diff_values = cfg.dataset.spectrum_pairs[pair].diff_values
                true_spectrum = np.array(cfg.dataset.spectrum_pairs[pair].true_spectrum)
            else:
                diff_values = cfg.dataset.diff_values
                true_spectrum = getattr(signal_decay, "true_spectrum", None)
            model = ProbabilisticModel(
                snr=signal_decay.snr,
                likelihood_config=cfg.likelihood,
                prior_config=cfg.prior,
                true_spectrum=true_spectrum,
                b_values=cfg.dataset.b_values,
                diffusivities=diff_values,
            )
            # Pre-inference diagnostics
            diagnostics_ok = model.pre_inference_diagnostics(signal_decay, cfg, run)
            if getattr(cfg.diagnostics, "only_pre_inference", False):
                continue  # Only run diagnostics, skip inference
            if not diagnostics_ok:
                print("[INFO] Skipping inference due to diagnostics failure.")
                continue
            # Compute spectra_id for this spectrum
            spectra_id = set_spectra_id(signal_decay, cfg)
            nc_filename = f"{spectra_id}.nc"
            output_path = os.path.join(inference_dir, nc_filename)
            if os.path.exists(output_path) and not cfg.recompute:
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
                    signal_decay=signal_decay,
                    diffusivities=diff_values,
                    design_matrix_U=model.U_matrix(),
                    spectrum_vector=spectrum_vector,
                    spectrum_samples=spectrum_samples,
                    spectrum_std=spectrum_std,
                    true_spectrum=model.true_spectrum,
                    inference_data=output_path,
                    spectra_id=spectra_id,
                )
                spectra.append(spectrum)
                continue
            if cfg.inference.name == "map":
                infer = MAPInference(model, signal_decay, cfg)
                spectrum = infer.run(
                    return_idata=True,
                    save_dir=inference_dir,
                    true_spectrum=true_spectrum,
                    unique_hash=spectra_id,
                )
                spectra.append(spectrum)
            elif cfg.inference.name == "gibbs":
                infer = GibbsSampler(model, signal_decay, cfg)
                spectrum = infer.run(
                    return_idata=True,
                    show_progress=True,
                    save_dir=inference_dir,
                    true_spectrum=true_spectrum,
                    unique_hash=spectra_id,
                )
                spectra.append(spectrum)
            elif cfg.inference.name == "vb":
                pass  # To be implemented

    # TODO: fit_adc() needs to be called somewhere and stored for every SignalDecay to compare to spectrum-derived biomarker
    spectra_dataset = DiffusivitySpectraDataset(spectra=spectra)

    ### 3) CANCER PREDICTION ###
    # TODO: to be implemented, maybe look at base_classes.py and use custom classes

    ### 4) DIAGNOSTICS AND PLOTTING ###
    spectra_dataset.run_diagnostics(exp_config=cfg)

    run.finish()


if __name__ == "__main__":
    main()
