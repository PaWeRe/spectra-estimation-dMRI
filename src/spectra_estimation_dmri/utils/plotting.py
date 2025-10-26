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
        tick_labels = [f"{number:.2f}" for number in d_spectra_sample.diffusivities]
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
            label=f"D={d_spectra_sample.diffusivities[dim]:.2f}",
        )
    ax.set_xlabel("Sample Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return ax


def analyze_design_matrix(U, pdf_path):
    """
    Analyze and plot diagnostics for a design matrix U, saving the output as a PDF.
    Includes: heatmap, singular value spectrum, condition number, numerical rank, and column correlation matrix.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    # Compute diagnostics
    U = np.asarray(U)
    cond_number = np.linalg.cond(U)
    singular_values = np.linalg.svd(U, compute_uv=False)
    tol = np.max(U.shape) * np.amax(singular_values) * np.finfo(float).eps
    numerical_rank = np.sum(singular_values > tol)
    col_corr = np.corrcoef(U.T)
    row_norms = np.linalg.norm(U, axis=1)
    col_norms = np.linalg.norm(U, axis=0)

    with PdfPages(pdf_path) as pdf:
        # 1. Heatmap of U
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(U, aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.set_title("Design Matrix U (heatmap)")
        ax.set_xlabel("Columns (diffusivity bins)")
        ax.set_ylabel("Rows (b-values)")
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Singular value spectrum (log scale)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.semilogy(singular_values, "o-")
        ax.set_title("Singular Value Spectrum of U")
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value (log scale)")
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Condition number and numerical rank (text)
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        text = f"Condition number: {cond_number:.2e}\nNumerical rank: {numerical_rank} / {U.shape[1]}"
        ax.text(0.1, 0.5, text, fontsize=14, va="center")
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Correlation matrix of columns
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(col_corr, aspect="auto", cmap="bwr", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title("Column Correlation Matrix of U")
        ax.set_xlabel("Column index")
        ax.set_ylabel("Column index")
        pdf.savefig(fig)
        plt.close(fig)

        # 5. Row and column norms
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(row_norms, label="Row norms")
        ax.plot(col_norms, label="Column norms")
        ax.set_title("Norms of Rows and Columns of U")
        ax.set_xlabel("Index")
        ax.set_ylabel("Norm")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

    return {
        "condition_number": cond_number,
        "singular_values": singular_values,
        "numerical_rank": numerical_rank,
        "column_correlation_matrix": col_corr,
        "row_norms": row_norms,
        "col_norms": col_norms,
    }
