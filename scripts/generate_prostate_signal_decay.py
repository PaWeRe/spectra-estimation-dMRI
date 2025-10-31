#!/usr/bin/env python
"""
Generate ISMRM figure: Prostate MRI + Signal Decay Comparison

This script creates a figure with:
- Left: Prostate MRI annotation image
- Right: Signal decay curves from a random patient (tumor vs normal tissue)

The figure follows ISMRM submission guidelines (1800×900px, 2:1 aspect ratio).
"""

import os
import sys

# Add project root to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from spectra_estimation_dmri.data.loaders import load_bwh_signal_decays
from spectra_estimation_dmri.visualization.ismrm_exports import (
    create_ismrm_prostate_and_signal_decay,
)


def main():
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    signal_decays_json = os.path.join(
        project_root, "src/spectra_estimation_dmri/data/bwh/signal_decays.json"
    )
    metadata_csv = os.path.join(
        project_root, "src/spectra_estimation_dmri/data/bwh/metadata.csv"
    )
    prostate_image_path = os.path.join(
        project_root, "assets/prostate_mri_annotation_right.jpeg"
    )
    output_dir = os.path.join(project_root, "results/biomarkers/ismrm")
    output_path = os.path.join(output_dir, "prostate_signal_decay_ismrm.png")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GENERATING ISMRM PROSTATE + SIGNAL DECAY FIGURE")
    print("=" * 70)

    # Load BWH signal decay data
    print("\n[1/2] Loading BWH patient data...")
    signal_decay_dataset = load_bwh_signal_decays(signal_decays_json, metadata_csv)
    print(f"  ✓ Loaded {len(signal_decay_dataset)} signal decay samples")

    # Generate figure
    print("\n[2/2] Generating figure...")
    print(f"  Prostate image: {prostate_image_path}")
    print(f"  Output: {output_path}")

    create_ismrm_prostate_and_signal_decay(
        signal_decay_dataset=signal_decay_dataset,
        prostate_image_path=prostate_image_path,
        output_path=output_path,
        patient_id=None,  # Random patient with both tumor and normal
        width_px=1800,
        height_px=900,
        dpi=150,
    )

    print("\n" + "=" * 70)
    print("✓ FIGURE GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput saved to: {output_path}")
    print("\nFigure details:")
    print("  - Size: 1800×900 pixels (2:1 aspect ratio)")
    print("  - Format: PNG")
    print("  - Left subplot: Prostate MRI with ROI annotations")
    print("  - Right subplot: Signal decay curves (tumor vs normal)")
    print("\nThis figure is ready for ISMRM abstract submission.")


if __name__ == "__main__":
    main()
