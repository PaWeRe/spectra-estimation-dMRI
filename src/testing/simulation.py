import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
import os
import yaml
import json
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy.random as npr

sys.path.append(os.path.join(os.getcwd() + "/src/models/"))
import gibbs as gs


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
    return gs.signal_data(predictions, b_values)


def generate_true_spectrum(diffusivities):
    """Generate a true spectrum for simulation."""
    fractions = np.zeros_like(diffusivities)
    # Make sure fractions sum up to 1
    fractions[0] = 0.35
    fractions[1] = 0.25
    fractions[-2] = 0.1
    fractions[-1] = 0.3
    fractions /= np.sum(fractions)  # Normalize
    return gs.d_spectrum(fractions, diffusivities)


def run_simulation(diffusivities, b_values, snr, iterations, l2_lambda):
    """Run a single simulation with given parameters."""
    true_spectrum = generate_true_spectrum(diffusivities)
    noisy_signal_normalized = generate_signals(1.0, b_values, true_spectrum, 1 / snr)

    sampler = gs.make_Gibbs_sampler(
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


def run_simulations(configurations):
    """Run simulations for multiple configurations"""
    results = []
    true_spectrum = None
    for config in configurations:
        true_spec, sample = run_simulation(
            config["diffusivities"],
            config["b_values"],
            config["snr"],
            config["iterations"],
            config["l2_lambda"],
        )
        if true_spectrum is None:
            true_spectrum = true_spec
        results.append(
            {
                "sample": sample,
                "snr": config["snr"],
                "b_max": np.max(config["b_values"]),
                "title": f"Spectrum (SNR={config['snr']}, b_max={np.max(config['b_values'])})",
            }
        )
    return true_spectrum, results


def init_plot_matrix(m, n, diffusivities):
    """Create graph layout on PDF with adjustments for better fitting"""
    fig, axarr = plt.subplots(m, n, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    arr_ij = list(np.ndindex(axarr.shape))
    subplots = [axarr[index] for index in arr_ij]

    for s, splot in enumerate(subplots):
        last_row = m * n - s < n + 1
        first_in_row = s % n == 0
        splot.grid(color="0.75")
        if last_row:
            splot.set_xlabel(r"Diffusivity [μm²/ms]", fontsize=10)
        if first_in_row:
            splot.set_ylabel("Relative Fraction", fontsize=10)
        splot.tick_params(axis="both", which="major", labelsize=8)

    return fig, axarr, subplots


def plot_true_spectrum(ax, true_spectrum, title):
    """Plot the true spectrum on the given axis"""
    ax.vlines(
        range(len(true_spectrum.diffusivities)), 0, true_spectrum.fractions, linewidth=2
    )
    ax.set_ylabel("Relative Fraction")
    ax.set_title(title)

    # Set x-ticks to match diffusivity values
    ax.set_xticks(range(len(true_spectrum.diffusivities)))
    ax.set_xticklabels(true_spectrum.diffusivities, fontsize=10, rotation=90)

    # Adjust y-axis limits
    y_max = max(true_spectrum.fractions)
    ax.set_ylim(0, y_max * 1.1)  # Add 10% padding at the top


def plot_results(true_spectrum, results, output_pdf_path, output_csv_path):
    """Plot the results of multiple simulations and save as PDF and CSV"""
    m = 2  # Rows
    n = 2  # Columns
    total_plots = len(results) + 1  # +1 for true spectrum
    pages = (total_plots - 1) // (
        m * n - 1
    ) + 1  # -1 because true spectrum is always on top-left

    with PdfPages(output_pdf_path) as pdf:
        for page in range(pages):
            fig, axarr, subplots = init_plot_matrix(m, n, true_spectrum.diffusivities)

            # Plot true spectrum on top-left of each page
            plot_true_spectrum(subplots[0], true_spectrum, "True Spectrum")

            start_idx = page * (m * n - 1)
            end_idx = min((page + 1) * (m * n - 1), len(results))

            for i, result in enumerate(results[start_idx:end_idx], start=1):
                ax = subplots[i]
                result["sample"].plot(ax=ax, title=result["title"])

                # be careful to reserve ticks for the first and last (numbering starts at 0!)
                str_diffs = [""] + list(map(str, true_spectrum.diffusivities)) + [""]
                ax.set_xticks(
                    np.arange(len(str_diffs)),
                    labels=str_diffs,
                    fontsize=10,
                    rotation=90,
                )

                # Adjust y-axis limits based on data
                y_max = ax.get_ylim()[1]
                ax.set_ylim(0, y_max * 1.1)  # Add 10% padding at the top

            pdf.savefig(fig)
            plt.close(fig)

    # Save results as CSV
    df = pd.DataFrame()
    for result in results:
        for diff, sample in zip(
            true_spectrum.diffusivities, np.transpose(result["sample"].sample)
        ):
            stats = {
                "Configuration": result["title"],
                "Diffusivity": diff,
                "Min": np.min(sample),
                "Q1": np.percentile(sample, 25),
                "Median": np.median(sample),
                "Mean": np.mean(sample),
                "Q3": np.percentile(sample, 75),
                "Max": np.max(sample),
            }
            df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
    df.to_csv(output_csv_path, index=False)


def main(configs: dict) -> None:
    diffusivities = np.array(
        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
    )
    b_values_3500 = np.linspace(0, 3500, 15) / 1000
    b_values_7000 = np.linspace(0, 7000, 15) / 1000

    configurations = [
        {
            "diffusivities": diffusivities,
            "b_values": b_values_3500,
            "snr": 300,
            "iterations": 100000,
            "l2_lambda": 1e-5,
        },
        {
            "diffusivities": diffusivities,
            "b_values": b_values_3500,
            "snr": 600,
            "iterations": 100000,
            "l2_lambda": 1e-5,
        },
        {
            "diffusivities": diffusivities,
            "b_values": b_values_3500,
            "snr": 10000,
            "iterations": 100000,
            "l2_lambda": 1e-5,
        },
        # {
        #     "diffusivities": diffusivities,
        #     "b_values": b_values_7000,
        #     "snr": 100,
        #     "iterations": 100000,
        #     "l2_lambda": 1e-5,
        # },
        # {
        #     "diffusivities": diffusivities,
        #     "b_values": b_values_7000,
        #     "snr": 600,
        #     "iterations": 100000,
        #     "l2_lambda": 1e-5,
        # },
        # {
        #     "diffusivities": diffusivities,
        #     "b_values": b_values_7000,
        #     "snr": 10000,
        #     "iterations": 100000,
        #     "l2_lambda": 1e-5,
        # },
    ]

    true_spectrum, results = run_simulations(configurations)
    plot_results(true_spectrum, results, configs["SIM_PDF"], configs["SIM_CSV"])


if __name__ == "__main__":
    # load in YAML configuration
    configs = {}
    base_config_path = os.path.join(os.getcwd() + "/configs.yaml")
    with open(base_config_path, "r") as file:
        configs.update(yaml.safe_load(file))
    main(configs)
