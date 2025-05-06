import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Stateless plotting functions for d_spectrum, d_spectra_sample, etc.


def plot_signal_data(signal_data, save_filename=None, title=None):
    plt.plot(
        signal_data.b_values, signal_data.signal_values, linestyle="None", marker="."
    )
    plt.xlabel("B Value")
    plt.ylabel("Signal")
    if title is not None:
        plt.title(title)
    if save_filename is not None:
        plt.savefig(save_filename)
    else:
        plt.show()


def plot_d_spectrum(d_spectrum, title=None):
    plt.vlines(
        d_spectrum.diffusivities,
        np.zeros(len(d_spectrum.fractions)),
        d_spectrum.fractions,
    )
    plt.xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)")
    plt.ylabel("Relative Fraction")
    plt.xlim(left=0)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_d_spectra_sample_boxplot(
    d_spectra_sample,
    save_filename=None,
    title=None,
    start=0,
    end=-1,
    skip=False,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)")
        ax.set_ylabel("Relative Fraction")
        tick_labels = [str(number) for number in d_spectra_sample.diffusivities]
        if skip:
            for i in range(0, len(d_spectra_sample.diffusivities)):
                if i % 2 == 1:
                    tick_labels[i] = ""
        ax.set_xticklabels(tick_labels, rotation=45)
    if title is not None:
        ax.set_title(title, fontdict={"fontsize": 12})
    sample_arr = np.asarray(d_spectra_sample.sample)[start:end]
    ax.boxplot(
        sample_arr,
        showfliers=False,
        manage_ticks=False,
        showmeans=True,
        meanline=True,
    )
    if d_spectra_sample.initial_R is not None:
        x_positions = np.arange(1, len(d_spectra_sample.diffusivities) + 1)
        ax.plot(
            x_positions,
            d_spectra_sample.initial_R,
            "r*",
            label="Initial R (Mode)",
            markersize=8,
        )
        ax.legend()
    if save_filename is not None:
        plt.savefig(save_filename)


def plot_d_spectra_sample_autocorrelation(d_spectra_sample, max_lag=2000, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sample_array = np.array(d_spectra_sample.sample)
    n_dims = len(d_spectra_sample.diffusivities)
    for dim in range(n_dims):
        chain = sample_array[:, dim]
        n = len(chain)
        this_max_lag = min(max_lag, n)
        chain_centered = chain - np.mean(chain)
        acf = np.zeros(this_max_lag)
        variance = np.sum(chain_centered**2) / n
        for k in range(this_max_lag):
            acf[k] = np.sum(chain_centered[k:] * chain_centered[: (n - k)]) / (n - k)
        acf = acf / variance
        ax.plot(
            np.arange(this_max_lag),
            acf,
            label=f"D={d_spectra_sample.diffusivities[dim]}",
        )
    ax.set_xlabel("Sample Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return ax
