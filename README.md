# Bayesian estimation of diffusivity spectra in diffusion MRI

This repo contains work in progress for estimating diffusivity spectra of normal and tumor tissue compartments of prostate gland diffusion MRI for 63 patients (relating to [Langkilde et al](https://pubmed.ncbi.nlm.nih.gov/28718517/) study), using bayesian methods for quantifying the diffusivity spectrum, defined by the inverse Laplace Transform.

All derived data is free from HIPAA PHI and repo should not contain any sensitive patient information.

## Motivation

There is great interest to quantify the spectrum of diffusivities that underlie the observed diffusion signal decay; separate tissue compartments can be identified by their spectral peaks. This spectrum is defined by the inverse Laplace transform, but unfortunately this transform is very sensitive to noise omnipresent in diffusion MRI. We present a Bayesian method of inverse Laplace transform that uses Gibbs sampling to provide spectra along with an estimate of the noise related to uncertainty in the spectra; this uncertainty information is valuable in interpreting the results. We show preliminary results estimating distributions on diffusivity spectra for normal and tumor tissues.

## Methods

Gibbs Sampler + Mean-Field Variational Bayes (TBD).

## Quick Install & Setup

1. Clone the repo and `cd` into it.
2. Install in editable mode (requires pip ≥ 21.3):
   ```sh
   pip install -e .
   ```
   (or use `uv pip install -e .` if you use uv)

## Structure & Data

- **Source code:** `src/spectra_estimation_dmri/`
- **Curated data (JSON, etc.):** in `src/spectra_estimation_dmri/data/` (distributed with the package, accessed via `importlib.resources`)
- **Generated output (plots, CSVs, etc.):** in top-level `output/` (auto-created, gitignored)

## Usage

Run simulations or tests with:
```sh
python -m spectra_estimation_dmri.testing.simulation
```
## Project Structure

- All source code is under `src/`
- Utilities are in `src/utils/`
- Models and samplers are in `src/models/`
- Tests and simulations are in `src/testing/`

## Environment Setup

### Prerequisites

- Python 3.13 or later
- [uv](https://github.com/astral-sh/uv) package manager
- Homebrew (for macOS users)
- SuiteSparse (required for cvxopt)

For macOS users, install SuiteSparse using Homebrew:
```bash
brew install suite-sparse
```

### Setting up the Python environment

1. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

2. Install all dependencies (from `pyproject.toml`):
```bash
uv pip install -e .
```

3. (macOS only, for cvxopt):
If you have issues with `cvxopt`, you may need to specify SuiteSparse paths:
```bash
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" uv pip install cvxopt
```

### Data

The `processed_patient_dict.json` (output of `process_data.py`) contains the following derived information for all 62 patients of the [Langkilde et al](https://pubmed.ncbi.nlm.nih.gov/28718517/) study:
- `new01`: patient identifier
  - `roi1`: region of interest on diffusion MRI image segmented by radiologists
    - `signal_values`: array of 15 signal values (average over ROI)
    - `b_values`: corresponding 15 b-values
    - `v_count`: number of voxels in ROI (used for average signal values)
    - `anatomical_region`: 4 anatomical regions in prostate gland were segmented - `tumor_tz_s1`, `tumor_pz_s1`, `normal_pz_s2`, `normal_tz_s3`. In cases where no matching anatomical region was found, regions are marked `Neglected!`.
  - `roi2`: ... (max 3 ROIs per patient)
 
`patient_gleason_grading.csv` and `prostate_rois.csv` contain gleason grades that were attributed by pathologists to certain segmented lesions and an overview of which region and tissue type (tumor vs normal) was segmented, respectively.

## Troubleshooting Import Errors

- If you see errors like `ModuleNotFoundError: No module named 'src'`, make sure you installed the package with `pip install -e .` from the repo root.
- Do **not** add `src` to your `PYTHONPATH` manually; the editable install handles this.
- Make sure every directory in `src/` (and `src/` itself) contains an `__init__.py` file (even if empty).
- If you use an IDE, make sure it recognizes the root of the repo as the project root.
