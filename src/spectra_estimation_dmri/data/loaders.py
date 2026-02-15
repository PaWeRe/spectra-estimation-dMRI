import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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


def load_binary_images(
    folder_path: str,
    shape: Tuple[int, int] = (256, 256),
    dtype: np.dtype = np.int16,
) -> Dict[int, np.ndarray]:
    """
    Load all .bin files from a folder as 2D images.

    Each binary file is expected to contain raw pixel data stored as
    contiguous 2-byte short integers (int16) in row-major order.

    Args:
        folder_path: Path to folder containing .bin files
        shape: Image dimensions (rows, cols). Default (256, 256)
        dtype: NumPy dtype for the binary data. Default np.int16

    Returns:
        Dictionary mapping file number (int) to 2D numpy array
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(folder.glob("*.bin"), key=lambda f: int(f.stem))
    if len(files) == 0:
        raise ValueError(f"No .bin files found in {folder_path}")

    expected_bytes = shape[0] * shape[1] * np.dtype(dtype).itemsize
    images = {}

    for f in files:
        file_size = f.stat().st_size
        if file_size != expected_bytes:
            raise ValueError(
                f"File {f.name}: expected {expected_bytes} bytes, got {file_size}. "
                f"Check shape={shape} and dtype={dtype}."
            )
        data = np.fromfile(f, dtype=dtype).reshape(shape)
        images[int(f.stem)] = data

    print(f"[INFO] Loaded {len(images)} binary images from {folder_path}")
    print(f"  Shape: {shape}, dtype: {dtype}, files: {list(images.keys())[:5]}...")
    return images


def subsample_to_native(
    images: Dict[int, np.ndarray],
    factor: int = 4,
) -> Dict[int, np.ndarray]:
    """
    Subsample images from interpolated resolution to native resolution.

    Args:
        images: Dictionary of file_number -> 2D array (e.g., 256x256)
        factor: Subsampling factor (e.g., 4 means 256->64)

    Returns:
        Dictionary of file_number -> subsampled 2D array (e.g., 64x64)
    """
    return {k: img[::factor, ::factor] for k, img in images.items()}


def compute_mean_intensities(
    images: Dict[int, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """
    Compute mean signal intensity for each image (optionally within a mask).

    Useful for sorting images by b-value (higher b -> lower mean signal).

    Args:
        images: Dictionary of file_number -> 2D array
        mask: Optional boolean mask (True = include pixel)

    Returns:
        Dictionary of file_number -> mean intensity
    """
    means = {}
    for k, img in images.items():
        if mask is not None:
            means[k] = float(np.mean(img[mask]))
        else:
            means[k] = float(np.mean(img[img > 0]))  # Exclude background zeros
    return means


def group_images_by_bvalue(
    images: Dict[int, np.ndarray],
    n_bvalues: int = 16,
    mask: Optional[np.ndarray] = None,
) -> Tuple[List[List[int]], Dict[int, np.ndarray]]:
    """
    Group images by b-value using mean intensity clustering.

    Images at the same b-value (different gradient directions) should have
    similar mean intensities. This function sorts by mean intensity and
    groups into n_bvalues clusters.

    Args:
        images: Dictionary of file_number -> 2D array
        n_bvalues: Expected number of distinct b-values
        mask: Optional boolean mask for intensity calculation

    Returns:
        (groups, averaged_images):
            - groups: List of lists, each containing file numbers in same b-value group
            - averaged_images: Dictionary of group_index -> direction-averaged 2D array
    """
    means = compute_mean_intensities(images, mask)

    # Sort file numbers by descending mean intensity (b=0 is brightest)
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)

    n_files = len(sorted_keys)
    n_per_group = n_files // n_bvalues
    remainder = n_files % n_bvalues

    # Split into groups (approximately equal size)
    groups = []
    idx = 0
    for i in range(n_bvalues):
        # Distribute remainder across first groups
        size = n_per_group + (1 if i < remainder else 0)
        group = sorted_keys[idx : idx + size]
        groups.append(group)
        idx += size

    # Average images within each group
    averaged_images = {}
    for i, group in enumerate(groups):
        avg = np.mean([images[k].astype(np.float64) for k in group], axis=0)
        averaged_images[i] = avg

    return groups, averaged_images


def group_images_b0_plus_directions(
    images: Dict[int, np.ndarray],
    n_directions: int = 3,
    mask: Optional[np.ndarray] = None,
) -> Tuple[List[List[int]], Dict[int, np.ndarray], int]:
    """
    Group images assuming 1 b=0 image + N directions per non-zero b-value.

    For diffusion MRI, b=0 has no gradient direction (1 image), while
    each non-zero b-value has n_directions gradient directions that should
    be averaged to produce isotropic "trace" images.

    For 46 files with 3 directions: 1 b=0 + 15 non-zero b-values x 3 dirs = 46.

    Args:
        images: Dictionary of file_number -> 2D array
        n_directions: Number of gradient directions per non-zero b-value (default: 3)
        mask: Optional boolean mask for intensity calculation

    Returns:
        (groups, trace_images, n_bvalues):
            - groups: List of lists, each containing file numbers in same b-value group
            - trace_images: Dictionary of group_index -> direction-averaged 2D array
            - n_bvalues: Number of unique b-values (including b=0)
    """
    means = compute_mean_intensities(images, mask)
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)

    n_files = len(sorted_keys)
    n_nonzero = (n_files - 1) // n_directions

    if 1 + n_nonzero * n_directions != n_files:
        print(
            f"[WARNING] {n_files} files does not split cleanly into "
            f"1 b=0 + N x {n_directions} directions. "
            f"Remainder: {n_files - 1 - n_nonzero * n_directions}"
        )

    # Group 0: b=0 (brightest single image)
    groups = [[sorted_keys[0]]]

    # Remaining images grouped into sets of n_directions
    remaining = sorted_keys[1:]
    for i in range(0, len(remaining), n_directions):
        group = remaining[i : i + n_directions]
        groups.append(group)

    # Average images within each group (trace computation)
    trace_images = {}
    for i, group in enumerate(groups):
        avg = np.mean([images[k].astype(np.float64) for k in group], axis=0)
        trace_images[i] = avg

    n_bvalues = len(groups)
    print(f"[INFO] Grouped {n_files} files into {n_bvalues} b-value levels:")
    print(f"  b=0: 1 image (file {groups[0][0]:06d})")
    print(f"  Non-zero: {n_bvalues - 1} levels x {n_directions} directions")

    return groups, trace_images, n_bvalues


def build_pixel_signal_array(
    trace_images: Dict[int, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a (n_pixels, n_bvalues) signal array from trace images.

    Args:
        trace_images: Dictionary of bvalue_index -> 2D array (ordered by b-value)
        mask: Optional boolean mask. If None, uses all pixels

    Returns:
        (signal_array, pixel_coords):
            - signal_array: Shape (n_pixels, n_bvalues) signal decay per pixel
            - pixel_coords: Shape (n_pixels, 2) row, col coordinates
    """
    n_bvalues = len(trace_images)
    shape = trace_images[0].shape

    if mask is None:
        mask = np.ones(shape, dtype=bool)

    pixel_coords = np.argwhere(mask)  # (n_pixels, 2)
    n_pixels = len(pixel_coords)

    signal_array = np.zeros((n_pixels, n_bvalues), dtype=np.float64)
    for b_idx in range(n_bvalues):
        img = trace_images[b_idx]
        signal_array[:, b_idx] = img[pixel_coords[:, 0], pixel_coords[:, 1]]

    return signal_array, pixel_coords


def create_simulated_signal_decays(
    true_spectrum: list, b_values: list, snr: list
) -> SignalDecayDataset:
    # TODO: create noise signal with torch / pyro (normal)
    # TODO: adapt the SignalDecay class and SignalDecay class to also include snr, true spectrum optionally
    # TODO: construct form 1 to N if I want to vary in hydra configs
    pass
