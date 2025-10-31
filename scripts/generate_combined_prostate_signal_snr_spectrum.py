#!/usr/bin/env python
"""
Generate ISMRM Combined Figure: Prostate + Signal Decay + SNR + Spectrum

This script creates a 2x2 combined figure with:
- Top-left: Prostate MRI annotation image
- Top-right: Signal decay curves (scatter only, no lines)
- Bottom-left: SNR posterior across realizations
- Bottom-right: Spectrum posterior for one realization

The figure follows ISMRM submission guidelines (1800×1800px, 1:1 aspect ratio).
"""

import os
import sys
import glob

# Add project root to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from spectra_estimation_dmri.data.loaders import load_bwh_signal_decays
from spectra_estimation_dmri.data.data_models import (
    DiffusivitySpectraDataset,
    DiffusivitySpectrum,
    SignalDecay,
)
from spectra_estimation_dmri.visualization.ismrm_exports import (
    create_ismrm_combined_prostate_signal_snr_spectrum,
)


def load_inference_spectra_from_directory(inference_dir: str, max_files: int = 10):
    """
    Load inference data from .nc files in the specified directory.

    Args:
        inference_dir: Directory containing .nc inference files
        max_files: Maximum number of files to load

    Returns:
        DiffusivitySpectraDataset with loaded spectra
    """
    import arviz as az
    import numpy as np

    # Find all .nc files
    nc_files = glob.glob(os.path.join(inference_dir, "*.nc"))
    nc_files = sorted(nc_files, key=os.path.getmtime, reverse=True)[:max_files]

    spectra = []
    for nc_file in nc_files:
        try:
            idata = az.from_netcdf(nc_file)

            # Check if this has sigma (SNR inference)
            if "sigma" not in idata.posterior:
                continue

            # Extract diffusivity variables
            diff_vars = sorted(
                [v for v in idata.posterior.data_vars if v.startswith("diff_")],
                key=lambda x: float(x.split("_")[1]),
            )

            if not diff_vars:
                continue

            # Extract diffusivities
            diffusivities = [float(v.split("_")[1]) for v in diff_vars]

            # Extract spectrum samples (transpose to get n_samples × n_diffusivities)
            spectrum_samples_list = []
            for var in diff_vars:
                samples = idata.posterior[var].values.flatten()
                spectrum_samples_list.append(samples)
            spectrum_samples = np.array(spectrum_samples_list).T  # Transpose

            # Get true SNR if available
            true_snr = None
            if hasattr(idata, "attrs") and "true_snr" in idata.attrs:
                true_snr = float(idata.attrs["true_snr"])
            elif hasattr(idata, "attrs") and "snr" in idata.attrs:
                true_snr = float(idata.attrs["snr"])
            elif hasattr(idata, "constant_data") and "snr" in idata.constant_data:
                true_snr = float(idata.constant_data["snr"].values)
            else:
                # Default to 500 for simulation data (standard value)
                true_snr = 500.0

            # Get true spectrum if available
            true_spectrum = None
            if (
                hasattr(idata, "constant_data")
                and "true_spectrum" in idata.constant_data
            ):
                true_spectrum = list(idata.constant_data["true_spectrum"].values)
            elif hasattr(idata, "attrs") and "true_spectrum" in idata.attrs:
                true_spectrum = list(idata.attrs["true_spectrum"])

            # Get spectrum init (MAP) if available
            spectrum_init = None
            if (
                hasattr(idata, "constant_data")
                and "spectrum_init" in idata.constant_data
            ):
                spectrum_init = list(idata.constant_data["spectrum_init"].values)
            elif hasattr(idata, "attrs") and "spectrum_init" in idata.attrs:
                spectrum_init = list(idata.attrs["spectrum_init"])

            # Create placeholder signal decay (required field)
            dummy_signal_decay = SignalDecay(
                patient="simulation",
                signal_values=[1.0] * 10,  # Dummy values
                b_values=[0.0] * 10,  # Dummy values
                snr=true_snr if true_snr else 500.0,
                voxel_count=100,
                a_region="pz",
                is_tumor=False,
            )

            # Create dummy design matrix (required field)
            n_bvals = 10
            n_diff = len(diffusivities)
            dummy_design_matrix = [[0.0] * n_diff for _ in range(n_bvals)]

            # Get spectrum vector (required field) - use posterior mean
            spectrum_vector = list(np.mean(spectrum_samples, axis=0))

            # Create DiffusivitySpectrum object
            spectrum = DiffusivitySpectrum(
                signal_decay=dummy_signal_decay,
                diffusivities=diffusivities,
                design_matrix_U=dummy_design_matrix,
                spectrum_init=spectrum_init,
                spectrum_vector=spectrum_vector,
                spectrum_samples=spectrum_samples.tolist(),
                inference_method="nuts",
                inference_data=nc_file,
                data_snr=true_snr,
                true_spectrum=true_spectrum,
            )
            spectra.append(spectrum)

        except Exception as e:
            print(f"[WARNING] Could not load {nc_file}: {e}")
            continue

    return DiffusivitySpectraDataset(spectra=spectra)


def main():
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # BWH signal decay data
    signal_decays_json = os.path.join(
        project_root, "src/spectra_estimation_dmri/data/bwh/signal_decays.json"
    )
    metadata_csv = os.path.join(
        project_root, "src/spectra_estimation_dmri/data/bwh/metadata.csv"
    )

    # Prostate image
    prostate_image_path = os.path.join(
        project_root, "assets/prostate_mri_annotation_right.jpeg"
    )

    # Simulation inference data (for SNR and spectrum)
    inference_dir = os.path.join(project_root, "results/inference")

    # Output
    output_dir = os.path.join(project_root, "results/biomarkers/ismrm")
    output_path = os.path.join(
        output_dir, "combined_prostate_signal_snr_spectrum_ismrm.png"
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GENERATING ISMRM COMBINED FIGURE")
    print("Prostate + Signal Decay + SNR + Spectrum (2x2 layout)")
    print("=" * 80)

    # Load BWH signal decay data
    print("\n[1/3] Loading BWH patient data...")
    signal_decay_dataset = load_bwh_signal_decays(signal_decays_json, metadata_csv)
    print(f"  ✓ Loaded {len(signal_decay_dataset)} signal decay samples")

    # Load simulation spectra data (for SNR and spectrum inference)
    print("\n[2/3] Loading simulation inference data...")
    try:
        spectra_dataset = load_inference_spectra_from_directory(
            inference_dir, max_files=10
        )
        if len(spectra_dataset.spectra) > 0:
            print(f"  ✓ Loaded {len(spectra_dataset.spectra)} inference results")
        else:
            print(f"  [WARNING] No valid inference files found in {inference_dir}")
            print(
                f"  [INFO] Bottom subplots (SNR and spectrum) will show placeholder text"
            )
    except Exception as e:
        print(f"  [WARNING] Could not load inference data: {e}")
        import traceback

        traceback.print_exc()
        print(f"  [INFO] Bottom subplots (SNR and spectrum) will show placeholder text")
        # Create empty dataset
        spectra_dataset = DiffusivitySpectraDataset(spectra=[])

    # Generate combined figure
    print("\n[3/3] Generating 2x2 combined figure...")
    print(f"  Prostate image: {prostate_image_path}")
    print(f"  Output: {output_path}")

    create_ismrm_combined_prostate_signal_snr_spectrum(
        spectra_dataset=spectra_dataset,
        signal_decay_dataset=signal_decay_dataset,
        prostate_image_path=prostate_image_path,
        output_path=output_path,
        patient_id=None,  # Random patient with both tumor and normal
        max_realizations=10,
        realization_idx=0,  # Show first realization in spectrum subplot
        width_px=1800,
        height_px=1800,
        dpi=150,
    )

    print("\n" + "=" * 80)
    print("✓ COMBINED FIGURE GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutput saved to: {output_path}")
    print("\nFigure details:")
    print("  - Size: 1800×1800 pixels (1:1 aspect ratio, 2x2 layout)")
    print("  - Format: PNG")
    print("  - Top-left: Prostate MRI with ROI annotations")
    print("  - Top-right: Signal decay curves (scatter only, no lines)")
    print("  - Bottom-left: SNR posterior across multiple realizations")
    print("  - Bottom-right: Spectrum posterior for one realization")
    print("\nThis figure is ready for ISMRM abstract submission.")
    print("\nNOTE: If you see placeholder text in bottom subplots, you need to:")
    print(
        "  1. Run simulation inference first (e.g., using configs/dataset/simulated.yaml)"
    )
    print("  2. Ensure inference results are saved in results/inference/")


if __name__ == "__main__":
    main()
