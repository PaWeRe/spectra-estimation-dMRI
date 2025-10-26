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
from spectra_estimation_dmri.inference.gibbs import GibbsSamplerClean
from spectra_estimation_dmri.inference.nuts import NUTSSampler
from spectra_estimation_dmri.data.data_models import (
    SignalDecay,
    DiffusivitySpectrum,
    DiffusivitySpectraDataset,
)

# Import biomarker analysis components
from spectra_estimation_dmri.biomarkers import (
    BiomarkerPipeline,
    BiomarkerEvaluator,
    BiomarkerVisualizer,
)

# Import sampler comparison tools
from spectra_estimation_dmri.analysis.sampler_comparison import (
    extract_metrics_from_spectrum,
    log_metrics_to_wandb,
    save_metrics_to_csv,
    print_metrics_summary,
)

# TODO: TRAIN NNET (e.g. DIFFUSION MODEL) AND RELEASE AS PART OF REPO (LIKE EMILY ALSENTZER!! https://arxiv.org/pdf/1904.03323


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Main experiment pipeline with four main steps:
    1. Data loading (real or simulated) as SignalDecayDataset objects
    2. Defines spectrum model for s=UR+eps
    3. Runs user-defined inference methods for spectrum reconstruction (output as DiffusivitySpectraDataset)
    4. Runs cancer biomarker analysis for Gleason score prediction (if enabled)
    5. Prints user-defined diagnostics
    """
    # Create more descriptive run name and tags for aggregation
    run_name = f"{cfg.inference.name}-{cfg.prior.type}-data_snr{cfg.dataset.snr if hasattr(cfg.dataset, 'snr') else 'real'}-{cfg.prior.type}-{cfg.prior.strength}"

    # Add biomarker tag if enabled
    if getattr(cfg, "biomarker_analysis", {}).get("enabled", False):
        run_name += f"-biomarker-{cfg.classifier.name}"

    # Add tags for easy filtering and grouping
    tags = [
        f"inference_{cfg.inference.name}",
        f"prior_{cfg.prior.type}",
        f"dataset_{cfg.dataset.name}",
    ]

    if hasattr(cfg.dataset, "snr"):
        tags.append(f"data_snr_{cfg.dataset.snr}")
    if hasattr(cfg.dataset, "true_spectrum_name"):
        tags.append(f"spectrum_{cfg.dataset.true_spectrum_name}")
    if getattr(cfg, "biomarker_analysis", {}).get("enabled", False):
        tags.append("biomarker_analysis")
        tags.append(f"classifier_{cfg.classifier.name}")

    run = None
    if not cfg.local:
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
        # Apply subset sampling if configured
        max_samples = getattr(cfg.dataset, "max_samples", None)
        if max_samples is not None and max_samples > 0:
            print(f"[INFO] Limiting to first {max_samples} samples for testing")
            signal_decay_dataset.samples = signal_decay_dataset.samples[:max_samples]
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
            data_snr=cfg.dataset.snr,
            likelihood_config=cfg.likelihood,
            prior_config=cfg.prior,
            true_spectrum=true_spectrum,
            b_values=cfg.dataset.b_values,
            diffusivities=diff_values,
            no_noise=cfg.dataset.get("no_noise", False),
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
    biomarker_dir = os.path.join(os.getcwd(), "results", "biomarkers")
    comparison_dir = os.path.join(os.getcwd(), "results", "comparison")
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(biomarker_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Set up sampler comparison metrics CSV
    comparison_csv_path = os.path.join(comparison_dir, "sampler_metrics.csv")

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
            # Set no_noise parameter for simulated data only
            no_noise = (
                cfg.dataset.get("no_noise", False)
                if cfg.dataset.name == "simulated"
                else False
            )
            model = ProbabilisticModel(
                data_snr=signal_decay.snr,
                likelihood_config=cfg.likelihood,
                prior_config=cfg.prior,
                true_spectrum=true_spectrum,
                b_values=cfg.dataset.b_values,
                diffusivities=diff_values,
                no_noise=no_noise,
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
                # TODO: retrieve spectrum init as well and change None in SpectrumDiffusivity object below!
                idata = az.from_netcdf(output_path)
                if cfg.inference.name == "map":
                    spectrum_vector = idata.posterior["R"].values[0, 0, :].tolist()
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
                    # spectrum_init=,
                    spectrum_vector=spectrum_vector,
                    spectrum_samples=spectrum_samples,
                    spectrum_std=spectrum_std,
                    true_spectrum=model.true_spectrum,
                    inference_data=output_path,
                    spectra_id=spectra_id,
                    prior_type=cfg.prior.type,
                    prior_strength=cfg.prior.strength,
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
                infer = GibbsSamplerClean(model, signal_decay, cfg)
                spectrum = infer.run(
                    return_idata=True,
                    show_progress=True,
                    save_dir=inference_dir,
                    true_spectrum=true_spectrum,
                    unique_hash=spectra_id,
                )
                spectra.append(spectrum)

                # Extract and log sampler comparison metrics
                metrics = extract_metrics_from_spectrum(spectrum, cfg)
                print_metrics_summary(metrics)
                save_metrics_to_csv(metrics, comparison_csv_path, append=True)
                if not cfg.local:
                    log_metrics_to_wandb(metrics)

            elif cfg.inference.name == "nuts":
                infer = NUTSSampler(model, signal_decay, cfg)
                spectrum = infer.run(
                    return_idata=True,
                    show_progress=True,
                    save_dir=inference_dir,
                    true_spectrum=true_spectrum,
                    unique_hash=spectra_id,
                )
                spectra.append(spectrum)

                # Extract and log sampler comparison metrics
                metrics = extract_metrics_from_spectrum(spectrum, cfg)
                print_metrics_summary(metrics)
                save_metrics_to_csv(metrics, comparison_csv_path, append=True)
                if not cfg.local:
                    log_metrics_to_wandb(metrics)

            elif cfg.inference.name == "vb":
                pass  # To be implemented

    # Create spectra dataset for biomarker analysis
    spectra_dataset = DiffusivitySpectraDataset(spectra=spectra)

    ### 3) CANCER BIOMARKER ANALYSIS (SIMPLE & LEAN) ###
    biomarker_results = None
    if getattr(cfg, "biomarker_analysis", {}).get("enabled", False):
        print("\n" + "=" * 60)
        print("SIMPLE GLEASON SCORE PREDICTION")
        print("=" * 60)

        try:
            from .biomarkers.simple_gleason_predictor import simple_biomarker_analysis

            # Get metadata path
            metadata_path = cfg.dataset.get(
                "metadata_path", "src/spectra_estimation_dmri/data/bwh/metadata.csv"
            )

            # Run simple biomarker analysis
            predictor, X, y, metrics = simple_biomarker_analysis(
                spectra_dataset=spectra_dataset,
                metadata_path=metadata_path,
                output_dir=biomarker_dir,
                model_type=cfg.classifier.get("name", "logistic"),
                use_uncertainty=cfg.biomarker_analysis.get("use_uncertainty", True),
                use_mc_predictions=cfg.biomarker_analysis.get(
                    "use_mc_predictions", False
                ),
            )

            biomarker_results = {
                "predictor": predictor,
                "features": X,
                "targets": y,
                "metrics": metrics,
            }

            # Log to W&B
            if not cfg.local:
                wandb.log(
                    {
                        "biomarker/accuracy": metrics.get("accuracy", 0),
                        "biomarker/auc": metrics.get("auc", 0),
                        "biomarker/sensitivity": metrics.get("sensitivity", 0),
                        "biomarker/specificity": metrics.get("specificity", 0),
                    }
                )

        except Exception as e:
            print(f"[ERROR] Biomarker analysis failed: {str(e)}")
            print("[INFO] Continuing with spectrum diagnostics...")

    ### 4) SPECTRUM DIAGNOSTICS AND PLOTTING ###
    print("\n" + "=" * 60)
    print("SPECTRUM DIAGNOSTICS")
    print("=" * 60)
    spectra_dataset.run_diagnostics(exp_config=cfg, local=cfg.local)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Inference method: {cfg.inference.name}")
    print(f"Prior: {cfg.prior.type}")
    print(f"Spectra analyzed: {len(spectra_dataset.spectra)}")

    if biomarker_results:
        if isinstance(
            biomarker_results, dict
        ) and "best_model" in biomarker_results.get("comparison_results", {}):
            print(
                f"Best biomarker model: {biomarker_results['comparison_results']['best_model']}"
            )
        else:
            print("Biomarker analysis: Single model analysis completed")
    else:
        print("Biomarker analysis: Not performed")

    if not cfg.local:
        run.finish()


if __name__ == "__main__":
    main()
