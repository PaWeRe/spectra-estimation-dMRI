---
name: image-processing
description: Load, preprocess, and manage medical imaging data for the prostate dMRI pipeline. Use when working with binary image files (.bin), signal decay extraction, ROI data, multi-direction gradient data, pixel subsampling, prostate masking, or any data format conversion between imaging and spectral analysis.
compatibility: Requires numpy, scipy, h5py, matplotlib. Uses uv for execution.
---

# Image Processing

## When to use this skill
Use when the task involves:
- Loading binary image data from `.bin` files
- Subsampling interpolated images to native resolution
- Creating prostate masks from b=0 images
- Extracting per-pixel signal decays for spectrum estimation
- Handling multi-direction gradient data
- Converting between data formats (images ↔ signal decays ↔ spectra)

## Data sources

### ROI-level data (current, 149 ROIs)
- Signal decays: `src/spectra_estimation_dmri/data/bwh/signal_decays.json`
- Metadata: `src/spectra_estimation_dmri/data/bwh/metadata.csv`
- Loader: `src/spectra_estimation_dmri/data/loaders.py` → `load_bwh_signal_decays()`

### Pixel-level imaging data (for pixel-wise maps)
- Location: `8640-sl6-bin/` (46 binary images)
- Format: 256×256, int16, raw binary (no header)
- Native resolution: 64×64 (interpolated 4× to 256×256)
- Loader: `src/spectra_estimation_dmri/data/loaders.py` → `load_binary_images()`

## Critical: Native resolution subsampling
The imaging data is stored at 256×256 but was acquired at 64×64.
**Only every 4th pixel in each dimension contains real measured data.**
The rest is interpolation and correlation artifacts.

```python
from spectra_estimation_dmri.data.loaders import load_binary_images, subsample_to_native
images_256 = load_binary_images("8640-sl6-bin/", shape=(256, 256), dtype=np.int16)
images_64 = subsample_to_native(images_256, factor=4)  # 16× fewer pixels!
```

## B-value to image mapping
The 46 binary images need to be mapped to b-values:
- BWH protocol: 15 b-values × 3 gradient directions = 45 images + 1 (b=0 average?)
- **Or**: different organization — check by sorting by mean intensity
- Images at same b-value (different directions) should have similar mean intensity
- Use `compute_mean_intensities()` and `group_images_by_bvalue()` from loaders

**[BLOCKED: Confirm exact mapping with Stephan]**

## Existing exploration code
- `scripts/explore_pixel_data.py` — loads, visualizes, groups images
- `scripts/pixel_wise_heatmap.py` — pixel-wise analysis pipeline (WIP)

## Pixel-wise processing pipeline
1. Load 46 binary images at native 64×64
2. Create prostate mask from b=0 (brightest) image
3. Map images to b-values (group by direction, average)
4. For each pixel in mask: extract signal decay → run spectrum estimation
5. Assemble spectral component maps (one image per D bin)
6. Apply trained classifier → probability heatmap

## Key functions in loaders.py
- `load_binary_images(folder, shape, dtype)` → dict[int, ndarray]
- `subsample_to_native(images, factor)` → dict[int, ndarray]
- `compute_mean_intensities(images)` → dict[int, float]
- `group_images_by_bvalue(images, n_bvalues)` → groups, averaged

## Data format for spectrum estimation
To feed a pixel into the NUTS sampler, create a `SignalDecay` object:
```python
from spectra_estimation_dmri.data.data_models import SignalDecay
decay = SignalDecay(
    signal_values=pixel_signal.tolist(),  # len=15 (one per b-value)
    roi_id="pixel_r32_c45",
    tissue_type="unknown",
    zone="unknown",
)
```

## Multi-direction data
When all-directions data becomes available:
- 3 gradient directions per b-value
- Compare spectra across directions to validate isotropy
- PZ should show consistent spectra (isotropic tissue)
- TZ may show some anisotropy
- Analysis: run NUTS per direction, compare posteriors
