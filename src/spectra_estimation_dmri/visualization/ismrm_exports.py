"""
ISMRM Abstract Figure Exports

Creates publication-ready figures for ISMRM abstract submissions:
- 3:2 aspect ratio (e.g., 900×600 pixels)
- PNG format
- <1MB file size
- High-quality but optimized
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import roc_curve


def export_ismrm_plot(
    fig_generator,
    output_path: str,
    width_px: int = 900,
    dpi: int = 150,
) -> None:
    """
    Export a plot in ISMRM-compatible format.

    Args:
        fig_generator: Function that returns a matplotlib figure
        output_path: Path for PNG output
        width_px: Width in pixels (default 900 for 900×600)
        dpi: DPI for rendering (default 150 for good quality)
    """
    # 3:2 aspect ratio
    height_px = int(width_px * 2 / 3)

    # Convert to inches for matplotlib
    width_in = width_px / dpi
    height_in = height_px / dpi

    # Generate figure
    fig = fig_generator(figsize=(width_in, height_in))

    # Save as PNG with optimization
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        format="png",
        pil_kwargs={"optimize": True, "quality": 85},
    )
    plt.close(fig)

    # Check file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(
        f"[ISMRM] Saved {output_path} ({width_px}×{height_px}px, {file_size_mb:.2f} MB)"
    )

    if file_size_mb > 1.0:
        print(f"[WARNING] File size exceeds 1MB limit")


def create_ismrm_averaged_spectra(
    regions: Dict[str, List],
    output_dir: str,
) -> None:
    """
    Create ISMRM-compatible averaged spectra figure.

    Args:
        regions: Dictionary from group_spectra_by_region()
        output_dir: Directory to save PNG
    """
    from .bwh_plotting import plot_spectrum_boxplot

    def fig_generator(figsize):
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten()

        region_names = {
            "normal_pz": "Normal PZ",
            "normal_tz": "Normal TZ",
            "tumor_pz": "Tumor PZ",
            "tumor_tz": "Tumor TZ",
        }

        diffusivities = None

        for idx, (region_key, spectra_list) in enumerate(regions.items()):
            if idx >= 4 or len(spectra_list) == 0:
                continue

            ax = axes[idx]

            # Get diffusivities
            if diffusivities is None:
                diffusivities = spectra_list[0].diffusivities

            # Collect samples
            all_samples = []
            for spectrum in spectra_list:
                arrays = spectrum.as_numpy()
                samples = arrays["spectrum_samples"]
                if samples is not None:
                    all_samples.append(samples)

            if len(all_samples) > 0:
                # Average samples across spectra
                avg_samples = np.mean(all_samples, axis=0)

                # Normalize
                avg_samples_norm = avg_samples / (
                    np.sum(avg_samples, axis=1, keepdims=True) + 1e-10
                )

                # Plot
                ax.boxplot(
                    avg_samples_norm,
                    showfliers=False,
                    manage_ticks=False,
                    showmeans=True,
                    meanline=True,
                )

                # Styling
                ax.set_ylim([0, 1.0])
                ax.set_yticks(np.arange(0, 1.1, 0.2))
                ax.grid(True, alpha=0.3)
                ax.set_title(
                    f"{region_names[region_key]} (n={len(spectra_list)})",
                    fontsize=11,
                    fontweight="bold",
                )

                # X-axis labels (bottom row only)
                if idx >= 2:
                    ax.set_xlabel(r"Diffusivity ($\mu$m$^2$/ms)", fontsize=10)
                    str_diffs = [f"{d:.2f}" for d in diffusivities]
                    ax.set_xticks(
                        np.arange(1, len(str_diffs) + 1),
                        labels=str_diffs,
                        rotation=45,
                        fontsize=8,
                    )

                # Y-axis labels (left column only)
                if idx % 2 == 0:
                    ax.set_ylabel("Relative Fraction", fontsize=10)

        plt.tight_layout()
        return fig

    output_path = os.path.join(output_dir, "averaged_spectra_ismrm.png")
    export_ismrm_plot(fig_generator, output_path)


def create_ismrm_roc_curve(
    results_list: List[Dict],
    output_path: str,
    title: str = "ROC Curve",
    highlight_features: Optional[List[str]] = None,
) -> None:
    """
    Create ISMRM-compatible ROC curve figure.

    Args:
        results_list: List of classification results
        output_path: Path for PNG output
        title: Plot title
        highlight_features: List of feature names to emphasize (thicker lines)
    """

    def fig_generator(figsize):
        fig, ax = plt.subplots(figsize=figsize)

        # Filter valid results
        valid_results = [
            r
            for r in results_list
            if r is not None and not np.isnan(r.get("metrics", {}).get("auc", np.nan))
        ]

        if not valid_results:
            ax.text(0.5, 0.5, "No valid results", ha="center", va="center")
            return fig

        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))

        # Plot curves
        for i, result in enumerate(valid_results):
            y_true = result["y_true"]
            y_pred_proba = result["y_pred_proba"]
            feature_name = result["feature_name"]
            auc = result["metrics"]["auc"]

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

            # Highlight certain features
            linewidth = (
                3 if (highlight_features and feature_name in highlight_features) else 2
            )
            alpha = (
                1.0
                if (highlight_features and feature_name in highlight_features)
                else 0.7
            )

            label = f"{feature_name} (AUC={auc:.3f})"
            ax.plot(fpr, tpr, color=colors[i], lw=linewidth, label=label, alpha=alpha)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")

        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        return fig

    export_ismrm_plot(fig_generator, output_path)


def create_all_ismrm_exports(
    spectra_dataset,
    results_dict: Dict[str, List[Dict]],
    regions: Dict[str, List],
    output_dir: str,
) -> None:
    """
    Create all ISMRM-compatible exports for abstract submission.

    Args:
        spectra_dataset: DiffusivitySpectraDataset
        results_dict: Biomarker classification results
        regions: Regional spectra groupings
        output_dir: Directory to save exports
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("CREATING ISMRM ABSTRACT FIGURES")
    print("=" * 60)

    # 1. Averaged spectra
    print("\n[1/3] Averaged spectra...")
    create_ismrm_averaged_spectra(regions, output_dir)

    # 2. ROC curves (tumor vs normal, both zones)
    print("\n[2/3] ROC curves...")
    for task_name, results_list in results_dict.items():
        if task_name.startswith("tumor_vs_normal"):
            zone = "PZ" if "pz" in task_name else "TZ"
            title = f"Tumor vs Normal Classification ({zone})"
            output_path = os.path.join(output_dir, f"roc_{task_name}_ismrm.png")

            # Highlight Full LR and ADC
            create_ismrm_roc_curve(
                results_list,
                output_path,
                title=title,
                highlight_features=[
                    "Full LR",
                    "ADC (baseline)",
                    "D[0.25]+1/D[2.0]+1/D[3.0]",
                ],
            )

    # 3. Note: SNR posterior and uncertainty calibration plots are in analysis module
    print("\n[3/3] SNR posterior and uncertainty calibration")
    print("  Note: These plots are generated by the analysis module")
    print("  Looking for existing plots to convert...")

    # Try to find and convert existing diagnostic plots
    plot_dir = os.path.join("results", "plots", "plot")
    if os.path.exists(plot_dir):
        # Find SNR posterior plot
        import glob

        snr_plots = glob.glob(os.path.join(plot_dir, "snr_posterior_*.pdf"))
        unc_plots = glob.glob(os.path.join(plot_dir, "uncertainty_calibration_*.pdf"))

        if snr_plots:
            print(f"  Found SNR posterior plots: {len(snr_plots)}")
            print("  Manual conversion needed (PDF -> PNG with 3:2 aspect)")
        if unc_plots:
            print(f"  Found uncertainty calibration plots: {len(unc_plots)}")
            print("  Manual conversion needed (PDF -> PNG with 3:2 aspect)")

    print(f"\n✓ ISMRM figures saved to: {output_dir}/")
    print("  Format: 900×600px PNG, optimized for <1MB")
