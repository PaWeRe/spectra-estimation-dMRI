"""Main script for approximative inference experiments."""

import importlib.resources
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from spectra_estimation_dmri.data.loaders import load_bwh_signal_decays


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Loads the bwh dataset and prints a summary.
    """
    print("cfg:", cfg)
    print("cfg.dataset:", getattr(cfg, "dataset", None))
    print("Hydra working directory:", os.getcwd())
    print("Signal decays path:", getattr(cfg.dataset, "signal_decays_json", None))
    print("Metadata path:", getattr(cfg.dataset, "metadata_csv", None))
    data = load_bwh_signal_decays(
        cfg.dataset.signal_decays_json, cfg.dataset.metadata_csv
    )
    data.summary()


if __name__ == "__main__":
    main()
