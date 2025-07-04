import json
from pathlib import Path
from .data_models import SignalDecayDataset, DiffusivitySpectraDataset
import csv
import numpy as np
from .data_models import SignalDecay


def load_signal_decays(json_path: str) -> SignalDecayDataset:
    with open(json_path, "r") as f:
        data = json.load(f)
    return SignalDecayDataset(**data)


def load_diffusivity_spectra(json_path: str) -> DiffusivitySpectraDataset:
    with open(json_path, "r") as f:
        data = json.load(f)
    return DiffusivitySpectraDataset(**data)


def load_bwh_signal_decays(json_path: str, metadata_path: str) -> SignalDecayDataset:
    # Load JSON
    with open(json_path, "r") as f:
        signal_data = json.load(f)
    # Load metadata
    metadata = {}
    with open(metadata_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata[row["patient_id"]] = row
    samples = []
    for patient_id, rois in signal_data.items():
        meta = metadata.get(patient_id, {})
        gs = meta.get("gs") if meta else None
        ggg = None
        try:
            ggg = (
                int(meta["targets"])
                if meta and meta.get("targets", "").isdigit()
                else None
            )
        except Exception:
            ggg = None
        for roi_name, roi in rois.items():
            # Parse anatomical_region
            anatomical_region = roi["anatomical_region"]
            # Determine region and tumor status from anatomical_region
            if "tumor" in anatomical_region:
                is_tumor = True
            else:
                is_tumor = False
            if "tz" in anatomical_region:
                region = "tz"
            elif "pz" in anatomical_region:
                region = "pz"
            else:
                print(
                    f"Omitting entry for patient {patient_id}, ROI {roi_name}: unknown region in anatomical_region: {anatomical_region}"
                )
                continue
            # Cross-check with metadata: ensure the region/tumor flag is set for this patient
            meta_key = anatomical_region
            meta_flag = meta.get(meta_key, None)
            if meta_flag not in ("1", ""):  # Only allow 1 or empty
                raise ValueError(
                    f"Mismatch between JSON anatomical_region '{anatomical_region}' and metadata for patient {patient_id}"
                )
            # Map v_count to voxel_count
            voxel_count = roi["v_count"]
            snr = float(np.sqrt(voxel_count / 16) * 150)
            sample = SignalDecay(
                patient=patient_id,
                signal_values=roi["signal_values"],
                b_values=roi["b_values"],
                snr=snr,
                voxel_count=voxel_count,
                a_region=region,
                is_tumor=is_tumor,
                ggg=ggg,
                gs=gs,
            )
            samples.append(sample)
    return SignalDecayDataset(samples=samples)


def create_simulated_signal_decays(
    true_spectrum: list, b_values: list, snr: list
) -> SignalDecayDataset:
    # TODO: create noise signal with torch / pyro (normal)
    # TODO: adapt the SignalDecay class and SignalDecay class to also include snr, true spectrum optionally
    # TODO: construct form 1 to N if I want to vary in hydra configs
    pass
