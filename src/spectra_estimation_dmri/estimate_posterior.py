"""Main script for approximative inference experiments."""

import importlib.resources
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
import wandb
import arviz as az

from spectra_estimation_dmri.data.loaders import load_bwh_signal_decays
from spectra_estimation_dmri.models.spectra_model import SpectrumModel
from spectra_estimation_dmri.simulation.simulate import generate_simulated_signal
from spectra_estimation_dmri.inference.map import MAPInference
from spectra_estimation_dmri.inference.gibbs import GibbsSampler
from spectra_estimation_dmri.diagnostics.diagnostics import run_all_diagnostics


def maybe_init_wandb(cfg):
    if hasattr(cfg, "wandb") and cfg.wandb.enable:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.experiment_name,
            config=dict(cfg),
            dir=os.getcwd(),
            reinit=True,
        )
        return run
    return None


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Loads the bwh dataset and prints a summary.
    """
    print(f"Hydra run directory: {os.getcwd()}")
    wandb_run = maybe_init_wandb(cfg)
    # 1. Load or simulate data
    if cfg.dataset.name == "bwh":
        data = load_bwh_signal_decays(
            cfg.dataset.signal_decays_json, cfg.dataset.metadata_csv
        )
    elif cfg.dataset.name == "simulated":
        snr = cfg.dataset.snr
        diff_values = np.array(cfg.dataset.diff_values)
        b_values = np.array(cfg.dataset.b_values)
        true_spectrum = np.array(cfg.dataset.true_spectrum[0])
        model = SpectrumModel(diff_values, b_values, snr=snr)
        signal = model.simulate_signal(true_spectrum)
    else:
        model = SpectrumModel(cfg.model)

    # 2. Prepare result file paths
    os.makedirs("results", exist_ok=True)
    map_path = f"results/{cfg.experiment_name}_map.nc"
    gibbs_path = f"results/{cfg.experiment_name}_gibbs.nc"

    # 3. Check for cached results
    if os.path.exists(map_path) and os.path.exists(gibbs_path):
        print(f"Loading cached InferenceData from {map_path} and {gibbs_path}")
        idata_map = az.from_netcdf(map_path)
        idata_gibbs = az.from_netcdf(gibbs_path)
        # For summary
        map_estimate = idata_map.posterior["R"].values.reshape(-1, len(diff_values))[0]
        gibbs_samples = idata_gibbs.posterior["R"].values.reshape(-1, len(diff_values))
    else:
        # 4. Run MAP/NNLS inference
        map_infer = MAPInference(model, signal, cfg.inference)
        map_result = map_infer.run(return_idata=True)
        map_estimate = map_result["map_estimate"]
        idata_map = map_result["idata"]

        # 5. Run Gibbs inference (with progress bar)
        gibbs_sampler = GibbsSampler(model, signal, cfg.inference)
        gibbs_result = gibbs_sampler.run(return_idata=True, show_progress=True)
        gibbs_samples = gibbs_result["samples"]
        idata_gibbs = gibbs_result["idata"]

        # 6. Save InferenceData to disk
        idata_map.to_netcdf(map_path)
        idata_gibbs.to_netcdf(gibbs_path)
        print(f"Saved MAP InferenceData to {map_path}")
        print(f"Saved Gibbs InferenceData to {gibbs_path}")
        if wandb_run is not None:
            wandb_run.save(map_path)
            wandb_run.save(gibbs_path)
            print(f"wandb run URL: {wandb_run.url}")

    # 7. Diagnostics and comparison (save and log plots)
    plot_paths = run_all_diagnostics(
        idata_gibbs,
        idata_map,
        true_spectrum=true_spectrum,
        diffusivities=diff_values,
        title="Simulated Data: True vs MAP vs Gibbs",
        wandb_run=wandb_run,
    )

    # 8. Print summary
    print("MAP/NNLS estimate:", map_estimate)
    print("Gibbs posterior mean:", np.mean(gibbs_samples, axis=0))
    print("True spectrum:", true_spectrum)

    # Save posterior for downstream use
    ...


if __name__ == "__main__":
    main()
