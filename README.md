# Bayesian estimation of diffusivity spectra in diffusion MRI

This repo contains work in progress for estimating diffusivity spectra of normal and tumor tissue compartments of prostate gland diffusion MRI for 63 patients (relating to [Langkilde et al](https://pubmed.ncbi.nlm.nih.gov/28718517/) study), using bayesian methods for quantifying the diffusivity spectrum, defined by the inverse Laplace Transform.

All derived data is free from HIPAA PHI and repo should not contain any sensitive patient information.

## Motivation

There is great interest to quantify the spectrum of diffusivities that underlie the observed diffusion signal decay; separate tissue compartments can be identified by their spectral peaks. This spectrum is defined by the inverse Laplace transform, but unfortunately this transform is very sensitive to noise omnipresent in diffusion MRI. We present a Bayesian method of inverse Laplace transform that uses Gibbs sampling to provide spectra along with an estimate of the noise related to uncertainty in the spectra; this uncertainty information is valuable in interpreting the results. We show preliminary results estimating distributions on diffusivity spectra for normal and tumor tissues.

## How to run sampler

Git clone repository to local environment and simply run `main.py` file. First time execution will take about 25min (standard macbook pro M1). After first execution `.pkl` file will be created caching results to avoid running sampler for every new experiment (if sampler settings are changed, delete `.pkl` file before running for fresh results).

## File structure
````
.
├── README.md
├── data
│   ├── patient_gleason_grading.csv
│   ├── processed_patient_dict.json
│   └── prostate_rois.csv
├── output
│   ├── correlation_matrix.pdf
│   ├── log_classifier_all.pdf
│   ├── log_classifier_db.pdf
│   ├── neglected.pdf
│   ├── npz.pdf
│   ├── ntz.pdf
│   ├── pr_visualization_025.pdf
│   ├── roi_avg.pdf
│   ├── tpz.pdf
│   └── ttz.pdf
└── src
    ├── diffusivity_spectra.py
    ├── main.py
    └── process_data.py
````

### Data

The `processed_patient_dict.json` (output of `process_data.py`) contains the following derived information for all 63 patients of the [Langkilde et al](https://pubmed.ncbi.nlm.nih.gov/28718517/) study:
- `new01`: patient identifier
  - `roi1`: region of interest on diffusion MRI image segmented by radiologists
    - `signal_values`: array of 15 signal values (average over ROI)
    - `b_values`: corresponding 15 b-values
    - `v_count`: number of voxels in ROI (used for average signal values)
    - `anatomical_region`: 4 anatomical regions in prostate gland were segmented - `tumor_tz_s1`, `tumor_pz_s1`, `normal_pz_s2`, `normal_tz_s3`. In cases where no matching anatomical region was found, regions are marked `Neglected!`.
  - `roi2`: ... (max 3 ROIs per patient)
 
`patient_gleason_grading.csv` and `prostate_rois.csv` contain gleason grades that were attributed by pathologists to certain segmented lesions and an overview of which region and tissue type (tumor vs normal) was segmented, respectively.

### Output

`main.py` outputs multiple visualizations of the output of the gibbs sampler. In the main script a `.pkl` file is created for caching sampler results. The `.pkl`file is not part of this repo due to it's significant size (>1GB). The following files are present in this repo:
- Diffusvity spectra boxplots of sampler output per patient categorized by anatomical region, as `npz.pdf`, `ntz.pdf`, `tpz.pdf`, `ttz.pdf` referring to normal peripheral zone, normal transition zone, tumor peripheral zone, tumor transition zone respectively.
- `correlation_matrix.pdf`
- `roi_avg.pdf`: 4 boxplots describing averaged diffusivity spectra per anatomical region for all patients and regions of interest
- Downstream analysis (preliminary!): `correlation_matrix.pdf`, `log_classifier_all.pdf`, `log_classifier_db.pdf`, `pr_visualization_025.pdf`

### Src

Three python files are used:
- `diffusivity_spectra.py`: core sampler logic
- `main.py`: main script responsible for plotting gibbs sampler results (and downstream analyses) on PDFs
- `process_data.py`: getting from image data to `process_data.py` dict
