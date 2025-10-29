#!/usr/bin/env python
"""
Generate SNR Posterior Plot in ISMRM Format

This script creates a publication-ready SNR posterior plot showing
joint inference of SNR and spectrum using NUTS samples from simulated data.

Uses the first 10 inference results from the results/inference folder.
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path


def find_inference_files(inference_dir: str, max_files: int = 10) -> list:
    """Find NetCDF inference files."""
    nc_files = glob.glob(os.path.join(inference_dir, "*.nc"))
    # Sort by modification time (most recent first)
    nc_files = sorted(nc_files, key=os.path.getmtime, reverse=True)
    return nc_files[:max_files]


def load_snr_samples(nc_file: str):
    """Load SNR samples from inference data."""
    try:
        idata = az.from_netcdf(nc_file)
        if "sigma" in idata.posterior:
            sigma_samples = idata.posterior["sigma"].values.flatten()
            snr_samples = 1.0 / sigma_samples
            return snr_samples
        else:
            return None
    except Exception as e:
        print(f"[WARNING] Could not load {nc_file}: {e}")
        return None


def get_true_snr_from_metadata(nc_file: str):
    """
    Try to extract true SNR from inference data attributes.
    Returns None if not available.
    """
    try:
        idata = az.from_netcdf(nc_file)
        # Check if SNR is stored in attributes
        if hasattr(idata, "attrs") and "true_snr" in idata.attrs:
            return float(idata.attrs["true_snr"])
        # Alternative: check if stored in posterior attrs
        if hasattr(idata.posterior, "attrs") and "snr" in idata.posterior.attrs:
            return float(idata.posterior.attrs["snr"])
    except:
        pass
    return None


def create_ismrm_snr_posterior(
    all_snr_samples: list,
    true_snr: float = None,
    output_path: str = "snr_posterior_ismrm.png",
    width_px: int = 900,
    dpi: int = 150,
):
    """
    Create ISMRM-compatible SNR posterior plot.

    Args:
        all_snr_samples: List of SNR sample arrays (one per realization)
        true_snr: Ground truth SNR (if available)
        output_path: Path for PNG output
        width_px: Width in pixels (default 900 for 900×600)
        dpi: DPI for rendering (default 150)
    """
    # 3:2 aspect ratio
    height_px = int(width_px * 2 / 3)
    width_in = width_px / dpi
    height_in = height_px / dpi

    # Create figure
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    n_realizations = len(all_snr_samples)

    # Create boxplot (one box per realization)
    bp = ax.boxplot(
        all_snr_samples,
        showfliers=False,
        showmeans=True,
        meanline=False,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
        medianprops=dict(color="blue", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Add ground truth line if available
    if true_snr is not None:
        ax.axhline(
            true_snr,
            color="darkgreen",
            linestyle="--",
            linewidth=2.5,
            label="Ground Truth",
            zorder=10,
        )

        # Calculate bias
        snr_means = [np.mean(samples) for samples in all_snr_samples]
        overall_mean = np.mean(snr_means)
        bias = overall_mean - true_snr
        rel_error = (bias / true_snr) * 100

        # Add statistics box (top left corner)
        stats_text = (
            f"Estimated SNR: {overall_mean:.1f}\nBias: {bias:+.1f} ({rel_error:+.1f}%)"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="black",
                linewidth=1.5,
            ),
            zorder=15,
        )

    # Styling
    ax.set_xlabel("Realization", fontsize=11, fontweight="bold")
    ax.set_ylabel("SNR", fontsize=11, fontweight="bold")
    ax.set_xticks(range(1, n_realizations + 1))
    ax.set_xticklabels([f"{i}" for i in range(1, n_realizations + 1)])
    ax.grid(True, alpha=0.3, axis="y")
    if true_snr is not None:
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "SNR Posterior Distributions Across Realizations",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

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
    print(f"\n✓ Saved {output_path}")
    print(f"  Size: {width_px}×{height_px}px, {file_size_mb:.2f} MB")

    if file_size_mb > 1.0:
        print(f"  [WARNING] File size exceeds 1MB ISMRM limit")


def create_ismrm_snr_and_spectrum_combined(
    nc_files: list,
    true_snr: float = None,
    output_path: str = "snr_and_spectrum_ismrm.png",
    width_px: int = 1200,
    dpi: int = 150,
    realization_idx: int = 0,
):
    """
    Create ISMRM-compatible combined SNR + Spectrum plot.

    Two subplots side-by-side:
    - Left: SNR posterior across realizations
    - Right: Spectrum posterior for one realization

    Args:
        nc_files: List of inference NetCDF files
        true_snr: Ground truth SNR (if available)
        output_path: Path for PNG output
        width_px: Width in pixels (default 1200 for 1200×600, 2:1 aspect for two subplots)
        dpi: DPI for rendering (default 150)
        realization_idx: Which realization to show in spectrum subplot
    """
    # 2:1 aspect ratio for two subplots side-by-side
    height_px = int(width_px / 2)
    width_in = width_px / dpi
    height_in = height_px / dpi

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_in, height_in))

    # Load SNR samples and spectrum data
    all_snr_samples = []
    all_spectrum_samples = []
    all_diffusivities = None
    all_true_spectra = []
    all_spectrum_inits = []

    for nc_file in nc_files:
        try:
            idata = az.from_netcdf(nc_file)

            # Extract SNR samples
            if "sigma" in idata.posterior:
                sigma_samples = idata.posterior["sigma"].values.flatten()
                snr_samples = 1.0 / sigma_samples
                all_snr_samples.append(snr_samples)

                # Extract spectrum samples
                # Look for diffusivity variables (e.g., diff_0.25, diff_0.50, etc.)
                diff_vars = [
                    v for v in idata.posterior.data_vars if v.startswith("diff_")
                ]
                if diff_vars:
                    # Sort by diffusivity value
                    diff_vars_sorted = sorted(
                        diff_vars, key=lambda x: float(x.split("_")[1])
                    )

                    # Extract diffusivities
                    if all_diffusivities is None:
                        all_diffusivities = [
                            float(v.split("_")[1]) for v in diff_vars_sorted
                        ]

                    # Extract samples for each diffusivity
                    spectrum_samples = []
                    for var in diff_vars_sorted:
                        samples = idata.posterior[var].values.flatten()
                        spectrum_samples.append(samples)

                    # Transpose to get (n_iterations, n_diffusivities)
                    spectrum_samples = np.array(spectrum_samples).T
                    all_spectrum_samples.append(spectrum_samples)

                    # Try to get true spectrum from multiple possible locations
                    true_spectrum = None
                    spectrum_init = None

                    # Check constant_data group (where observed/constant data is stored)
                    if hasattr(idata, "constant_data"):
                        if "true_spectrum" in idata.constant_data:
                            true_spectrum = np.array(
                                idata.constant_data["true_spectrum"].values
                            )
                        if "spectrum_init" in idata.constant_data:
                            spectrum_init = np.array(
                                idata.constant_data["spectrum_init"].values
                            )

                    # Check attributes
                    if true_spectrum is None and hasattr(idata, "attrs"):
                        if "true_spectrum" in idata.attrs:
                            true_spectrum = np.array(idata.attrs["true_spectrum"])

                    if spectrum_init is None and hasattr(idata, "attrs"):
                        if "spectrum_init" in idata.attrs:
                            spectrum_init = np.array(idata.attrs["spectrum_init"])

                    # Check posterior attrs
                    if true_spectrum is None and hasattr(idata.posterior, "attrs"):
                        if "true_spectrum" in idata.posterior.attrs:
                            true_spectrum = np.array(
                                idata.posterior.attrs["true_spectrum"]
                            )

                    if spectrum_init is None and hasattr(idata.posterior, "attrs"):
                        if "spectrum_init" in idata.posterior.attrs:
                            spectrum_init = np.array(
                                idata.posterior.attrs["spectrum_init"]
                            )

                    # Store if found
                    if true_spectrum is not None:
                        all_true_spectra.append(true_spectrum)
                    else:
                        # Add None placeholder to maintain alignment
                        all_true_spectra.append(None)

                    if spectrum_init is not None:
                        all_spectrum_inits.append(spectrum_init)
                    else:
                        # Add None placeholder to maintain alignment
                        all_spectrum_inits.append(None)

        except Exception as e:
            print(f"[WARNING] Could not load {nc_file}: {e}")

    if not all_snr_samples:
        ax1.text(0.5, 0.5, "No SNR data", ha="center", va="center")
        ax2.text(0.5, 0.5, "No spectrum data", ha="center", va="center")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format="png")
        plt.close(fig)
        return

    n_realizations = len(all_snr_samples)

    # Debug output: show what was loaded
    n_with_true = sum(1 for t in all_true_spectra if t is not None)
    n_with_init = sum(1 for i in all_spectrum_inits if i is not None)
    print(f"  Loaded {n_realizations} realizations")
    print(f"  Found true spectrum in {n_with_true}/{n_realizations} files")
    print(f"  Found spectrum init in {n_with_init}/{n_realizations} files")

    # === LEFT SUBPLOT: SNR Posterior ===
    bp1 = ax1.boxplot(
        all_snr_samples,
        showfliers=False,
        showmeans=True,
        meanline=False,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
        medianprops=dict(color="blue", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Add ground truth SNR line
    if true_snr is not None:
        ax1.axhline(
            true_snr,
            color="darkgreen",
            linestyle="--",
            linewidth=2.5,
            label="Ground Truth",
            zorder=10,
        )

        # Calculate bias
        snr_means = [np.mean(samples) for samples in all_snr_samples]
        overall_mean = np.mean(snr_means)
        bias = overall_mean - true_snr
        rel_error = (bias / true_snr) * 100

        # Add statistics box (TOP LEFT)
        stats_text = (
            f"Estimated SNR: {overall_mean:.1f}\nBias: {bias:+.1f} ({rel_error:+.1f}%)"
        )
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="black",
                linewidth=1.0,
            ),
            zorder=15,
        )

    # SNR subplot styling
    ax1.set_xlabel("Realization", fontsize=10, fontweight="bold")
    ax1.set_ylabel("SNR", fontsize=10, fontweight="bold")
    ax1.set_xticks(range(1, n_realizations + 1))
    ax1.set_xticklabels([f"{i}" for i in range(1, n_realizations + 1)], fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    if true_snr is not None:
        ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(
        "Joint Inference: SNR Across Multiple Realizations",
        fontsize=10,
        fontweight="bold",
    )

    # === RIGHT SUBPLOT: Spectrum Posterior ===
    if all_spectrum_samples and realization_idx < len(all_spectrum_samples):
        samples = all_spectrum_samples[realization_idx]
        diffusivities = all_diffusivities
        n_diff = len(diffusivities)

        # Create boxplot for spectrum
        bp2 = ax2.boxplot(
            samples,
            showfliers=False,
            showmeans=True,
            meanline=True,
            labels=[f"{d:.2f}" for d in diffusivities],
            boxprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        # Add MAP initialization if available
        if (
            realization_idx < len(all_spectrum_inits)
            and all_spectrum_inits[realization_idx] is not None
        ):
            x_positions = np.arange(1, n_diff + 1)
            spectrum_init = all_spectrum_inits[realization_idx]
            ax2.plot(
                x_positions,
                spectrum_init,
                "r*",
                label="Initial R (MAP)",
                markersize=8,
                alpha=0.8,
            )

        # Add true spectrum if available
        if (
            realization_idx < len(all_true_spectra)
            and all_true_spectra[realization_idx] is not None
        ):
            true_spectrum = all_true_spectra[realization_idx]
            x_positions = np.arange(1, n_diff + 1)
            ax2.vlines(
                x_positions,
                0,
                true_spectrum,
                colors="blue",
                linewidth=2.5,
                label="True Spectrum",
                alpha=0.9,
            )

        # Spectrum subplot styling
        ax2.set_xlabel(r"Diffusivity ($\mu$m$^2$/ms)", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Relative Fraction", fontsize=10, fontweight="bold")
        ax2.set_xticklabels(
            [f"{d:.2f}" for d in diffusivities], rotation=45, fontsize=8
        )
        ax2.grid(True, alpha=0.3)
        ax2.set_title(
            f"Joint Inference: Spectrum (Realization {realization_idx+1})",
            fontsize=10,
            fontweight="bold",
        )

        # Show legend if we have any overlays (true spectrum or init)
        has_overlays = (
            realization_idx < len(all_spectrum_inits)
            and all_spectrum_inits[realization_idx] is not None
        ) or (
            realization_idx < len(all_true_spectra)
            and all_true_spectra[realization_idx] is not None
        )
        if has_overlays:
            ax2.legend(fontsize=8, loc="best")
    else:
        ax2.text(0.5, 0.5, "No spectrum samples", ha="center", va="center")

    plt.tight_layout()

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
    print(f"\n✓ Saved {output_path}")
    print(f"  Size: {width_px}×{height_px}px, {file_size_mb:.2f} MB")

    if file_size_mb > 1.0:
        print(f"  [WARNING] File size exceeds 1MB ISMRM limit")


def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate SNR posterior plot in ISMRM format"
    )
    parser.add_argument(
        "--true-snr",
        type=float,
        default=500.0,
        help="Ground truth SNR value (default: 500)",
    )
    parser.add_argument(
        "--max-realizations",
        type=int,
        default=10,
        help="Maximum number of realizations to include (default: 10)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create combined SNR + Spectrum plot (two subplots)",
    )
    parser.add_argument(
        "--realization-idx",
        type=int,
        default=0,
        help="Which realization to show in spectrum subplot for combined plot (default: 0)",
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    inference_dir = project_root / "results" / "inference"
    output_dir = project_root / "results" / "biomarkers" / "ismrm"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    if args.combined:
        print("GENERATING COMBINED SNR + SPECTRUM PLOT (ISMRM FORMAT)")
    else:
        print("GENERATING SNR POSTERIOR PLOT (ISMRM FORMAT)")
    print("=" * 60)

    # Find inference files
    print(f"\nSearching for inference files in: {inference_dir}")
    nc_files = find_inference_files(str(inference_dir), max_files=args.max_realizations)

    if not nc_files:
        print(f"[ERROR] No inference files found in {inference_dir}")
        sys.exit(1)

    print(
        f"Found {len(nc_files)} inference files (using first {args.max_realizations})"
    )

    # Determine ground truth SNR
    print("\nLoading SNR samples from NUTS inference...")
    all_snr_samples = []
    true_snrs = []

    for i, nc_file in enumerate(nc_files):
        print(f"  [{i+1}/{len(nc_files)}] {Path(nc_file).name}...", end=" ")
        snr_samples = load_snr_samples(nc_file)

        if snr_samples is not None:
            all_snr_samples.append(snr_samples)
            true_snr_meta = get_true_snr_from_metadata(nc_file)
            if true_snr_meta is not None:
                true_snrs.append(true_snr_meta)
            print(f"✓ ({len(snr_samples)} samples)")
        else:
            print("✗ (no sigma inference)")

    if not all_snr_samples:
        print("\n[ERROR] No valid SNR samples found in inference files")
        print("Note: SNR inference requires NUTS method with sigma parameter")
        sys.exit(1)

    print(f"\n✓ Loaded SNR samples from {len(all_snr_samples)} realizations")

    # Determine ground truth SNR
    true_snr = None
    if true_snrs:
        # Check if all true SNRs are consistent
        unique_snrs = np.unique(np.round(true_snrs, 1))
        if len(unique_snrs) == 1:
            true_snr = unique_snrs[0]
            print(f"  Ground truth SNR from metadata: {true_snr:.1f}")
        else:
            print(f"  [WARNING] Multiple true SNR values found: {unique_snrs}")
            print(f"  Using mean: {np.mean(true_snrs):.1f}")
            true_snr = np.mean(true_snrs)
    else:
        # Use command-line argument
        true_snr = args.true_snr
        print(f"  Ground truth SNR (from --true-snr): {true_snr:.1f}")

    # Create plot(s)
    print(f"\nCreating ISMRM plot...")

    if args.combined:
        # Create combined SNR + Spectrum plot
        output_path = output_dir / "snr_and_spectrum_combined_ismrm.png"
        create_ismrm_snr_and_spectrum_combined(
            nc_files,
            true_snr=true_snr,
            output_path=str(output_path),
            realization_idx=args.realization_idx,
        )
    else:
        # Create SNR-only plot
        output_path = output_dir / "snr_posterior_ismrm.png"
        create_ismrm_snr_posterior(
            all_snr_samples,
            true_snr=true_snr,
            output_path=str(output_path),
        )

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    snr_means = [np.mean(samples) for samples in all_snr_samples]
    print(f"Number of realizations: {len(all_snr_samples)}")
    print(
        f"Mean SNR (across realizations): {np.mean(snr_means):.1f} ± {np.std(snr_means):.1f}"
    )
    print(
        f"SNR range: [{np.min([np.min(s) for s in all_snr_samples]):.1f}, {np.max([np.max(s) for s in all_snr_samples]):.1f}]"
    )

    if true_snr is not None:
        overall_mean = np.mean(snr_means)
        bias = overall_mean - true_snr
        rel_error = (bias / true_snr) * 100
        print(f"\nGround truth SNR: {true_snr:.1f}")
        print(f"Bias: {bias:+.1f} ({rel_error:+.1f}%)")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
