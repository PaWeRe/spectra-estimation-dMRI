import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy.random as npr
import sys
import yaml
from spectra_estimation_dmri.models.samplers.base import (
    d_spectrum,
    d_spectra_sample,
    signal_data,
)
from spectra_estimation_dmri.models.samplers.gibbs_sampler import GibbsSampler
from spectra_estimation_dmri.utils import plotting
import importlib.resources

sys.path.append(os.path.join(os.getcwd() + "/src/models/"))


def predict_signals(s_0, b_values, diffusivities, fractions):
    signals = 0
    for i in range(len(diffusivities)):
        signals += s_0 * fractions[i] * np.exp(-b_values * diffusivities[i])
    return signals


def generate_signals(s_0, b_values, d_spectrum, sigma):
    predictions = predict_signals(
        s_0, b_values, d_spectrum.diffusivities, d_spectrum.fractions
    )
    if sigma > 0.0:
        predictions += npr.normal(0, sigma, len(predictions))
    return signal_data(predictions, b_values)


def generate_true_spectrum(diffusivities):
    """Generate a true spectrum for simulation."""
    fractions = np.zeros_like(diffusivities)
    # Make sure fractions sum up to 1
    fractions[0] = 0.35  # Peak at D=0.25
    fractions[1] = 0.25  # Peak at D=0.50
    fractions[-2] = 0.1  # Peak at D=3.00
    fractions[-1] = 0.3  # Peak at D=20.00
    fractions /= np.sum(fractions)  # Normalize
    return d_spectrum(fractions, diffusivities)


def run_simulation(diffusivities, b_values, snr, iterations, l2_lambda):
    """Run a single simulation with given parameters."""
    true_spectrum = generate_true_spectrum(diffusivities)
    noisy_signal_normalized = generate_signals(1.0, b_values, true_spectrum, 1 / snr)

    sampler = GibbsSampler(
        noisy_signal_normalized,
        diffusivities,
        1 / snr,
        L2_lambda=l2_lambda,
    )
    sample = sampler(iterations)
    # Normalize and discard the first 10000 (burn-in)
    sample.normalize()
    sample.sample = sample.sample[10000:]

    return true_spectrum, sample


def plot_results(true_spectrum, all_results, output_pdf_path, output_csv_path):
    """Plot results with true spectrum as vertical lines."""
    n_configs = len(all_results)
    rows = int(np.ceil((n_configs + 1) / 2))
    cols = 2

    with PdfPages(output_pdf_path) as pdf:
        fig = plt.figure(figsize=(15, 5 * rows))
        plt.suptitle("Simulation Results", fontsize=16)

        gs = plt.GridSpec(rows, cols, figure=fig)

        for i, result in enumerate(all_results):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])

            # Plot true spectrum as blue vertical lines
            for j, (diff, frac) in enumerate(
                zip(true_spectrum.diffusivities, true_spectrum.fractions)
            ):
                ax.vlines(
                    j,
                    0,
                    frac,
                    color="blue",
                    alpha=0.5,
                    linewidth=2,
                    label="True Spectrum" if j == 0 else "",
                )

            # Plot Gibbs sampler results
            plotting.plot_d_spectra_sample_boxplot(
                result["sample"],
                ax=ax,
                title=f'SNR: {result["snr"]}\nIterations: {result["iterations"]}',
            )

            # Set x-ticks to match diffusivity values
            ax.set_xticks(range(len(true_spectrum.diffusivities)))
            ax.set_xticklabels(true_spectrum.diffusivities, fontsize=8, rotation=90)
            ax.set_xlabel(r"Diffusivity Value (μm²/ms)")
            ax.set_ylabel("Relative Fraction")

            ax.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    # Save results as CSV
    df = pd.DataFrame()
    for result in all_results:
        for diff, sample in zip(
            true_spectrum.diffusivities,
            np.transpose(result["sample"].sample),
        ):
            stats = {
                "SNR": result["snr"],
                "Iterations": result["iterations"],
                "Diffusivity": diff,
                "Mean": np.mean(sample),
                "Std": np.std(sample),
                "True_Value": true_spectrum.fractions[
                    list(true_spectrum.diffusivities).index(diff)
                ],
            }
            df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)

    df.to_csv(output_csv_path, index=False)


def main(configs: dict) -> None:
    diffusivities = np.array(
        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
    )
    b_values = np.linspace(0, 3500, 15) / 1000

    # Test configurations with high SNR and large iteration counts
    configurations = [
        {
            "snr": 1000,
            "iterations": 20000,
            "l2_lambda": 1e-5,
        },
        # {
        #     "snr": 2000,
        #     "iterations": 300000,
        #     "l2_lambda": 1e-5,
        # },
        # {
        #     "snr": 5000,
        #     "iterations": 400000,
        #     "l2_lambda": 1e-5,
        # },
    ]

    # Run simulations
    all_results = []
    true_spectrum = None
    for config in configurations:
        true_spec, sample = run_simulation(
            diffusivities,
            b_values,
            config["snr"],
            config["iterations"],
            config["l2_lambda"],
        )
        if true_spectrum is None:
            true_spectrum = true_spec

        all_results.append(
            {"sample": sample, "snr": config["snr"], "iterations": config["iterations"]}
        )

    # Plot and save results
    plot_results(true_spectrum, all_results, configs["SIM_PDF"], configs["SIM_CSV"])


if __name__ == "__main__":
    configs = {}
    with importlib.resources.files("spectra_estimation_dmri").joinpath(
        "configs.yaml"
    ).open("r") as file:
        configs.update(yaml.safe_load(file))
    main(configs)
