import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from .data_models import SignalDecayDataset, DiffusivitySpectraDataset
import csv
import numpy as np
from .data_models import SignalDecay

# Default data path relative to this module
_DATA_DIR = Path(__file__).parent / "8640-sl6-bin"

# Langkilde et al. 2018: 15 b-values, 250 s/mm² spacing, 3 orthogonal directions
B_VALUES_S_MM2 = np.arange(0, 3501, 250, dtype=float)  # [0, 250, ..., 3500]
B_VALUES_MS_UM2 = B_VALUES_S_MM2 / 1000.0  # [0, 0.25, ..., 3.5]
N_BVALUES = 15
N_DIRECTIONS = 3
INTERPOLATED_SHAPE = (256, 256)
NATIVE_SHAPE = (64, 64)
SUBSAMPLE_FACTOR = 4


@dataclass
class ProstateDWI:
    """Complete processed prostate DWI dataset for a single slice.

    Attributes:
        trace_images: Array of shape (n_bvalues, rows, cols) — direction-averaged images
        b_values: Array of b-values in ms/um² (length n_bvalues)
        b_values_s_mm2: Array of b-values in s/mm² (length n_bvalues)
        groups: List of lists — file numbers grouped by b-value
        dropped_reference: File number of the excluded scanner reference image
        raw_images: All loaded images (including reference) keyed by file number
        mean_intensities: Per-file mean intensities used for grouping
    """

    trace_images: np.ndarray
    b_values: np.ndarray
    b_values_s_mm2: np.ndarray
    groups: List[List[int]]
    dropped_reference: int
    raw_images: Dict[int, np.ndarray] = field(repr=False)
    mean_intensities: Dict[int, float] = field(repr=False)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.trace_images.shape[1:]

    @property
    def n_bvalues(self) -> int:
        return self.trace_images.shape[0]

    def prostate_mask(self, threshold_fraction: float = 0.05) -> np.ndarray:
        """Create a simple prostate mask from the b=0 trace image.

        Pixels above threshold_fraction * max(b=0 image) are included.
        """
        b0 = self.trace_images[0]
        return b0 > (threshold_fraction * b0.max())

    def pixel_signal_array(
        self, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract per-pixel signal decays as a 2D array.

        Args:
            mask: Boolean mask (rows, cols). If None, uses prostate_mask().

        Returns:
            (signals, coords): signals is (n_pixels, n_bvalues), coords is (n_pixels, 2)
        """
        if mask is None:
            mask = self.prostate_mask()
        coords = np.argwhere(mask)
        signals = np.zeros((len(coords), self.n_bvalues), dtype=np.float64)
        for b_idx in range(self.n_bvalues):
            signals[:, b_idx] = self.trace_images[b_idx][coords[:, 0], coords[:, 1]]
        return signals, coords


def load_prostate_dwi(
    folder_path: Optional[str] = None,
    subsample: bool = True,
) -> ProstateDWI:
    """Load and process the BWH prostate DWI dataset from binary files.

    This implements the correct data handling for patient 8640, slice 6:
      - 46 binary files: 1 scanner reference + 15 b-values x 3 directions
      - The scanner reference (brightest image, first in DICOM sequence) is
        dropped per Wells et al. ISMRM 2022: "the first signal value was
        not employed" (IVIM / different TE)
      - Remaining 45 images are sorted by mean intensity, grouped into 15
        triplets, and averaged to produce isotropic trace images
      - b-values assigned: [0, 250, 500, ..., 3500] s/mm² (Langkilde 2018)

    Args:
        folder_path: Path to folder with .bin files. Defaults to package data dir.
        subsample: If True, subsample from 256x256 to native 64x64 resolution.

    Returns:
        ProstateDWI with trace images, b-values, grouping info, and raw data.
    """
    folder = Path(folder_path) if folder_path else _DATA_DIR
    images = _load_binary_images(folder)

    if subsample:
        images = {k: img[::SUBSAMPLE_FACTOR, ::SUBSAMPLE_FACTOR] for k, img in images.items()}

    means = _compute_mean_intensities(images)
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)

    # The brightest image is the scanner reference — verify it's an outlier
    ref_key = sorted_keys[0]
    protocol_keys = sorted_keys[1:]
    ref_intensity = means[ref_key]
    next_intensity = means[protocol_keys[0]]
    gap_pct = (ref_intensity - next_intensity) / next_intensity * 100

    if gap_pct < 10:
        raise ValueError(
            f"Expected scanner reference to be significantly brighter than "
            f"protocol b=0. Got {ref_intensity:.1f} vs {next_intensity:.1f} "
            f"({gap_pct:.1f}% gap). Check data integrity."
        )

    n_protocol = len(protocol_keys)
    expected = N_BVALUES * N_DIRECTIONS
    if n_protocol != expected:
        raise ValueError(
            f"After dropping reference, expected {expected} protocol images "
            f"(15 b-values x 3 directions), got {n_protocol}."
        )

    # Group the 45 protocol images into 15 triplets by intensity ranking
    groups = []
    for i in range(0, n_protocol, N_DIRECTIONS):
        groups.append(protocol_keys[i : i + N_DIRECTIONS])

    # Average each triplet to produce trace images
    shape = images[ref_key].shape
    trace_images = np.zeros((N_BVALUES, *shape), dtype=np.float64)
    for b_idx, group in enumerate(groups):
        trace_images[b_idx] = np.mean(
            [images[k].astype(np.float64) for k in group], axis=0
        )

    print(f"Loaded prostate DWI: {n_protocol + 1} files -> {N_BVALUES} trace images")
    print(f"  Dropped scanner reference: file {ref_key:06d} (intensity {ref_intensity:.0f}, {gap_pct:.0f}% above protocol b=0)")
    print(f"  Image shape: {shape}, b-values: {B_VALUES_S_MM2[0]:.0f}-{B_VALUES_S_MM2[-1]:.0f} s/mm²")

    return ProstateDWI(
        trace_images=trace_images,
        b_values=B_VALUES_MS_UM2.copy(),
        b_values_s_mm2=B_VALUES_S_MM2.copy(),
        groups=groups,
        dropped_reference=ref_key,
        raw_images=images,
        mean_intensities=means,
    )


def _load_binary_images(
    folder: Path,
    shape: Tuple[int, int] = INTERPOLATED_SHAPE,
    dtype: np.dtype = np.int16,
) -> Dict[int, np.ndarray]:
    """Load all .bin files from a folder as 2D images."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(folder.glob("*.bin"), key=lambda f: int(f.stem))
    if not files:
        raise ValueError(f"No .bin files found in {folder}")

    expected_bytes = shape[0] * shape[1] * np.dtype(dtype).itemsize
    images = {}
    for f in files:
        if f.stat().st_size != expected_bytes:
            raise ValueError(
                f"File {f.name}: expected {expected_bytes} bytes, got {f.stat().st_size}. "
                f"Check shape={shape} and dtype={dtype}."
            )
        images[int(f.stem)] = np.fromfile(f, dtype=dtype).reshape(shape)

    return images


def _compute_mean_intensities(
    images: Dict[int, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """Compute mean signal intensity for each image (excluding background zeros)."""
    means = {}
    for k, img in images.items():
        if mask is not None:
            means[k] = float(np.mean(img[mask]))
        else:
            means[k] = float(np.mean(img[img > 0]))
    return means


# --- Existing loaders for ROI-level data (BWH JSON + metadata CSV) ---


def load_signal_decays(json_path: str) -> SignalDecayDataset:
    with open(json_path, "r") as f:
        data = json.load(f)
    return SignalDecayDataset(**data)


def load_diffusivity_spectra(json_path: str) -> DiffusivitySpectraDataset:
    with open(json_path, "r") as f:
        data = json.load(f)
    return DiffusivitySpectraDataset(**data)


def load_bwh_signal_decays(json_path: str, metadata_path: str) -> SignalDecayDataset:
    with open(json_path, "r") as f:
        signal_data = json.load(f)
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
            anatomical_region = roi["anatomical_region"]
            is_tumor = "tumor" in anatomical_region
            if "tz" in anatomical_region:
                region = "tz"
            elif "pz" in anatomical_region:
                region = "pz"
            else:
                print(
                    f"Omitting entry for patient {patient_id}, ROI {roi_name}: "
                    f"unknown region in anatomical_region: {anatomical_region}"
                )
                continue
            meta_key = anatomical_region
            meta_flag = meta.get(meta_key, None)
            if meta_flag not in ("1", ""):
                raise ValueError(
                    f"Mismatch between JSON anatomical_region '{anatomical_region}' "
                    f"and metadata for patient {patient_id}"
                )
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


# --- Legacy helpers (kept for backward compatibility) ---


def load_binary_images(
    folder_path: str,
    shape: Tuple[int, int] = (256, 256),
    dtype: np.dtype = np.int16,
) -> Dict[int, np.ndarray]:
    """Load all .bin files from a folder as 2D images."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    files = sorted(folder.glob("*.bin"), key=lambda f: int(f.stem))
    if len(files) == 0:
        raise ValueError(f"No .bin files found in {folder_path}")
    expected_bytes = shape[0] * shape[1] * np.dtype(dtype).itemsize
    images = {}
    for f in files:
        if f.stat().st_size != expected_bytes:
            raise ValueError(f"File {f.name}: expected {expected_bytes} bytes, got {f.stat().st_size}")
        images[int(f.stem)] = np.fromfile(f, dtype=dtype).reshape(shape)
    return images


def subsample_to_native(
    images: Dict[int, np.ndarray],
    factor: int = 4,
) -> Dict[int, np.ndarray]:
    """Subsample images from interpolated to native resolution."""
    return {k: img[::factor, ::factor] for k, img in images.items()}


def group_images_b0_plus_directions(
    images: Dict[int, np.ndarray],
    n_directions: int = 3,
    mask: Optional[np.ndarray] = None,
) -> Tuple[List[List[int]], Dict[int, np.ndarray], int]:
    """Group images: 1 b=0 + N directions per non-zero b-value, return trace images."""
    # Sort by descending mean intensity (b=0 is brightest)
    means = {}
    for k, img in images.items():
        means[k] = float(np.mean(img[img > 0])) if mask is None else float(np.mean(img[mask]))
    sorted_keys = sorted(means.keys(), key=lambda k: means[k], reverse=True)

    groups = [[sorted_keys[0]]]  # b=0
    remaining = sorted_keys[1:]
    for i in range(0, len(remaining), n_directions):
        groups.append(remaining[i : i + n_directions])

    trace_images = {}
    for i, group in enumerate(groups):
        trace_images[i] = np.mean([images[k].astype(np.float64) for k in group], axis=0)

    return groups, trace_images, len(groups)


def build_pixel_signal_array(
    trace_images: Dict[int, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a (n_pixels, n_bvalues) signal array from trace images.

    Prefer ProstateDWI.pixel_signal_array() for new code.
    """
    n_bvalues = len(trace_images)
    shape = trace_images[0].shape

    if mask is None:
        mask = np.ones(shape, dtype=bool)

    pixel_coords = np.argwhere(mask)
    n_pixels = len(pixel_coords)

    signal_array = np.zeros((n_pixels, n_bvalues), dtype=np.float64)
    for b_idx in range(n_bvalues):
        img = trace_images[b_idx]
        signal_array[:, b_idx] = img[pixel_coords[:, 0], pixel_coords[:, 1]]

    return signal_array, pixel_coords
