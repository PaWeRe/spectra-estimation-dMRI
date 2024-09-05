import json
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score
import yaml
import sys

sys.path.append(os.path.join(os.getcwd() + "/spectra-estimation-dMRI/src/models/"))
import gibbs


def average_signals():
    """
    descr: Create average tumor and normal signals for gibbs sampler testing.
    """


# TODO: are we only gibbs sampling to see the spread / uncertainty? For the GGG and tumor correlation this is irrelevant.
def gibbs_config_testing():
    """
    descr: Test and visualize sampler performance with different configurations (different SNR, ranges etc.).
    As reference for the "True Spectrum" the mode of the truncated MVN (posterior R) will be used.
    """


def main(configs: dict) -> None:

    # Create config dict for Gibbs Sampler hyperparams search
    gibbs_config_dict = {
        "range" "iters" "c" "lambda_l1" "lamba_l2" "inverse_prior_covariance"
    }


if __name__ == "__main__":
    # load in YAML configuration
    configs = {}
    base_config_path = os.path.join(
        os.getcwd() + "/spectra-estimation-dMRI/configs.yaml"
    )
    with open(base_config_path, "r") as file:
        configs.update(yaml.safe_load(file))
    main(configs)
