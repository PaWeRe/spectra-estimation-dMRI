"""
BWH-specific plotting and visualization for diffusivity spectra.

Organizes spectra by anatomical region (PZ/TZ, Tumor/Normal) and generates:
- Per-region boxplot PDFs (paginated 3×2 grids)
- Averaged spectra plots
- Statistical summaries as CSV files
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_anatomical_region(signal_decay) -> Tuple[str, str]:
    """
    Parse anatomical region from SignalDecay object.

    Args:
        signal_decay: SignalDecay object with a_region and is_tumor fields

    Returns:
        (tissue_type, zone): e.g., ("tumor", "pz"), ("normal", "tz")
    """
    # Get zone from a_region field
    region_str = getattr(signal_decay, "a_region", "unknown")
    zone = region_str.lower() if region_str in ["pz", "tz"] else "unknown"

    # Get tissue type from is_tumor field
    is_tumor = getattr(signal_decay, "is_tumor", None)
    if is_tumor is True:
        tissue_type = "tumor"
    elif is_tumor is False:
        tissue_type = "normal"
    else:
        tissue_type = "unknown"

    return tissue_type, zone


def group_spectra_by_region(spectra_dataset) -> Dict[str, List]:
    """
    Group spectra by anatomical region.

    Args:
        spectra_dataset: DiffusivitySpectraDataset object

    Returns:
        Dictionary with keys: "normal_pz", "normal_tz", "tumor_pz", "tumor_tz"
        Each value is a list of DiffusivitySpectrum objects
    """
    regions = {
        "normal_pz": [],
        "normal_tz": [],
        "tumor_pz": [],
        "tumor_tz": [],
    }

    for spectrum in spectra_dataset.spectra:
        # Parse anatomical region from signal decay object
        tissue_type, zone = parse_anatomical_region(spectrum.signal_decay)

        # Create region key
        region_key = f"{tissue_type}_{zone}"

        if region_key in regions:
            regions[region_key].append(spectrum)
        else:
            print(
                f"[WARNING] Unknown region combination: tissue={tissue_type}, zone={zone}"
            )

    # Print summary
    print("\n[BWH PLOTTING] Grouped spectra by region:")
    for region, spectra_list in regions.items():
        print(f"  {region}: {len(spectra_list)} spectra")

    return regions


def init_plot_matrix(
    m: int, n: int, diffusivities: List[float], figsize: Tuple[float, float] = (10, 10)
) -> Tuple:
    """
    Create boxplot grid layout for spectra visualization.

    Args:
        m: Number of rows
        n: Number of columns
        diffusivities: List of diffusivity values for x-axis
        figsize: Figure size in inches

    Returns:
        (fig, axarr, subplots): Figure, axes array, flattened subplot list
    """
    fig, axarr = plt.subplots(m, n, sharex="col", sharey="row", figsize=figsize)

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    # Flatten axes for easier indexing
    if m == 1 and n == 1:
        subplots = [axarr]
    elif m == 1 or n == 1:
        subplots = list(axarr)
    else:
        arr_ij = list(np.ndindex(axarr.shape))
        subplots = [axarr[index] for index in arr_ij]

    # Configure subplots
    for s, splot in enumerate(subplots):
        last_row = m * n - s <= n
        first_in_row = s % n == 0

        splot.grid(color="0.75", alpha=0.3)

        if last_row:
            splot.set_xlabel(r"Diffusivity ($\mu$m$^2$/ms)", fontsize=10)
            str_diffs = [""] + [f"{d:.2f}" for d in diffusivities] + [""]
            splot.set_xticks(
                np.arange(len(str_diffs)), labels=str_diffs, rotation=90, fontsize=6
            )

        if first_in_row:
            splot.set_ylabel("Relative Fraction", fontsize=10)

        splot.tick_params(axis="both", which="major", labelsize=10)

    return fig, axarr, subplots


def plot_spectrum_boxplot(
    ax, spectrum_samples: np.ndarray, diffusivities: List[float], title: str = None
):
    """
    Plot boxplot of spectrum posterior samples on given axis.

    Args:
        ax: Matplotlib axis
        spectrum_samples: Array of shape (n_samples, n_diffusivities)
        diffusivities: List of diffusivity values
        title: Optional title for the subplot
    """
    # Normalize each sample
    spectrum_samples_norm = spectrum_samples / (
        np.sum(spectrum_samples, axis=1, keepdims=True) + 1e-10
    )

    # Create boxplot
    ax.boxplot(
        spectrum_samples_norm,
        showfliers=False,
        manage_ticks=False,
        showmeans=True,
        meanline=True,
    )

    # Set fine-grained y-axis
    ax.set_ylim([0, 1.0])
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # Ticks every 0.1
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))  # Minor ticks every 0.05
    ax.grid(which="minor", alpha=0.2, linestyle=":", linewidth=0.5)  # Minor grid

    if title is not None:
        ax.set_title(title, fontdict={"fontsize": 10})


def plot_region_boxplots(
    regions: Dict[str, List],
    output_dir: str,
    m: int = 3,
    n: int = 2,
) -> None:
    """
    Generate paginated boxplot PDFs for each anatomical region.

    Args:
        regions: Dictionary from group_spectra_by_region()
        output_dir: Directory to save PDF files
        m: Number of rows per page
        n: Number of columns per page
    """
    os.makedirs(output_dir, exist_ok=True)

    region_names = {
        "normal_pz": "Normal Peripheral Zone",
        "normal_tz": "Normal Transition Zone",
        "tumor_pz": "Tumor Peripheral Zone",
        "tumor_tz": "Tumor Transition Zone",
    }

    plt.rcParams.update({"font.size": 8})

    for region_key, spectra_list in regions.items():
        if len(spectra_list) == 0:
            print(f"[BWH PLOTTING] Skipping {region_key} (no spectra)")
            continue

        output_path = os.path.join(output_dir, f"{region_key}_spectra.pdf")
        diffusivities = spectra_list[0].diffusivities

        print(f"[BWH PLOTTING] Creating {region_key} boxplots...")

        with PdfPages(output_path) as pdf:
            spectra_per_page = m * n
            n_pages = (len(spectra_list) + spectra_per_page - 1) // spectra_per_page

            for page in range(n_pages):
                fig, axarr, subplots = init_plot_matrix(m, n, diffusivities)

                start_idx = page * spectra_per_page
                end_idx = min((page + 1) * spectra_per_page, len(spectra_list))

                for i, spectrum in enumerate(spectra_list[start_idx:end_idx]):
                    arrays = spectrum.as_numpy()
                    samples = arrays["spectrum_samples"]

                    if samples is None:
                        print(
                            f"[WARNING] No samples for spectrum {i+start_idx}, skipping"
                        )
                        continue

                    # Create title with metadata
                    patient_id = getattr(spectrum.signal_decay, "patient", "unknown")
                    ggg = getattr(spectrum.signal_decay, "ggg", "N/A")
                    gs = getattr(spectrum.signal_decay, "gs", "N/A")
                    snr = getattr(spectrum.signal_decay, "snr", "N/A")
                    snr_str = (
                        f"{snr:.0f}" if isinstance(snr, (int, float)) else str(snr)
                    )

                    title = f"{patient_id} | GS:{gs} | GGG:{ggg} | SNR:{snr_str}"

                    plot_spectrum_boxplot(
                        subplots[i], samples, diffusivities, title=title
                    )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"  ✓ Saved: {output_path}")


def plot_averaged_spectra(
    regions: Dict[str, List],
    output_dir: str,
    m: int = 2,
    n: int = 2,
) -> None:
    """
    Generate averaged spectra boxplots (one per region on single page).

    Args:
        regions: Dictionary from group_spectra_by_region()
        output_dir: Directory to save PDF
        m: Number of rows
        n: Number of columns
    """
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "averaged_spectra.pdf")

    region_names = {
        "normal_pz": "Normal PZ",
        "normal_tz": "Normal TZ",
        "tumor_pz": "Tumor PZ",
        "tumor_tz": "Tumor TZ",
    }

    # Compute averaged samples for each region
    averaged_regions = {}
    diffusivities = None

    for region_key, spectra_list in regions.items():
        if len(spectra_list) == 0:
            continue

        # Get diffusivities from first spectrum
        if diffusivities is None:
            diffusivities = spectra_list[0].diffusivities

        # Collect all samples
        all_samples = []
        for spectrum in spectra_list:
            arrays = spectrum.as_numpy()
            samples = arrays["spectrum_samples"]
            if samples is not None:
                all_samples.append(samples)

        if len(all_samples) > 0:
            # Average across patients (mean of samples)
            averaged_samples = np.mean(np.array(all_samples), axis=0)
            averaged_regions[region_key] = averaged_samples

    if len(averaged_regions) == 0:
        print("[BWH PLOTTING] No regions to plot (averaged spectra)")
        return

    print(f"[BWH PLOTTING] Creating averaged spectra plot...")

    with PdfPages(output_path) as pdf:
        fig, axarr, subplots = init_plot_matrix(m, n, diffusivities, figsize=(12, 10))

        for i, (region_key, samples) in enumerate(averaged_regions.items()):
            if i >= len(subplots):
                break

            title = f"{region_names.get(region_key, region_key)} (n={len(regions[region_key])})"
            plot_spectrum_boxplot(subplots[i], samples, diffusivities, title=title)

        # Hide unused subplots
        for j in range(len(averaged_regions), len(subplots)):
            subplots[j].set_visible(False)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"  ✓ Saved: {output_path}")


def export_region_statistics(
    regions: Dict[str, List],
    output_dir: str,
) -> None:
    """
    Export statistical summaries as CSV files (per region + averaged).

    Args:
        regions: Dictionary from group_spectra_by_region()
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Export per-region statistics
    for region_key, spectra_list in regions.items():
        if len(spectra_list) == 0:
            continue

        output_path = os.path.join(output_dir, f"{region_key}_stats.csv")

        rows = []
        for spectrum in spectra_list:
            arrays = spectrum.as_numpy()
            diffusivities = arrays["diffusivities"]
            samples = arrays["spectrum_samples"]

            if samples is None:
                continue

            # Normalize samples
            samples_norm = samples / (np.sum(samples, axis=1, keepdims=True) + 1e-10)

            # Get metadata
            patient_id = getattr(spectrum.signal_decay, "patient", "unknown")
            ggg = getattr(spectrum.signal_decay, "ggg", None)
            gs = getattr(spectrum.signal_decay, "gs", None)
            snr = getattr(spectrum.signal_decay, "snr", None)

            # Compute statistics for each diffusivity
            for j, diff in enumerate(diffusivities):
                diff_samples = samples_norm[:, j]
                rows.append(
                    {
                        "Patient": patient_id,
                        "Region": region_key,
                        "GS": gs,
                        "GGG": ggg,
                        "SNR": snr,
                        "Diffusivity": diff,
                        "Min": np.min(diff_samples),
                        "Q1": np.percentile(diff_samples, 25),
                        "Median": np.median(diff_samples),
                        "Mean": np.mean(diff_samples),
                        "Q3": np.percentile(diff_samples, 75),
                        "Max": np.max(diff_samples),
                        "Std": np.std(diff_samples),
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"[BWH PLOTTING] Saved statistics: {output_path}")

    # Export averaged statistics
    averaged_path = os.path.join(output_dir, "averaged_stats.csv")
    rows = []

    for region_key, spectra_list in regions.items():
        if len(spectra_list) == 0:
            continue

        # Collect samples
        all_samples = []
        for spectrum in spectra_list:
            arrays = spectrum.as_numpy()
            samples = arrays["spectrum_samples"]
            if samples is not None:
                all_samples.append(samples)

        if len(all_samples) == 0:
            continue

        # Average across patients
        averaged_samples = np.mean(np.array(all_samples), axis=0)
        diffusivities = spectra_list[0].diffusivities

        # Normalize
        averaged_samples_norm = averaged_samples / (
            np.sum(averaged_samples, axis=1, keepdims=True) + 1e-10
        )

        # Compute statistics
        for j, diff in enumerate(diffusivities):
            diff_samples = averaged_samples_norm[:, j]
            rows.append(
                {
                    "Region": region_key,
                    "N_Patients": len(spectra_list),
                    "Diffusivity": diff,
                    "Min": np.min(diff_samples),
                    "Q1": np.percentile(diff_samples, 25),
                    "Median": np.median(diff_samples),
                    "Mean": np.mean(diff_samples),
                    "Q3": np.percentile(diff_samples, 75),
                    "Max": np.max(diff_samples),
                    "Std": np.std(diff_samples),
                }
            )

    df_avg = pd.DataFrame(rows)
    df_avg.to_csv(averaged_path, index=False)
    print(f"[BWH PLOTTING] Saved averaged statistics: {averaged_path}")


def run_bwh_diagnostics(spectra_dataset, output_dir: str = "results/plots/bwh"):
    """
    Run complete BWH visualization pipeline.

    Args:
        spectra_dataset: DiffusivitySpectraDataset object
        output_dir: Base directory for outputs
    """
    print("\n" + "=" * 60)
    print("BWH DATASET VISUALIZATION")
    print("=" * 60)

    # Group spectra by anatomical region
    regions = group_spectra_by_region(spectra_dataset)

    # Generate per-region boxplots
    plot_region_boxplots(regions, output_dir)

    # Generate averaged spectra plot
    plot_averaged_spectra(regions, output_dir)

    # Export statistics
    export_region_statistics(regions, output_dir)

    print("\n✓ BWH visualization complete!")
    print(f"  Results saved to: {output_dir}/")
