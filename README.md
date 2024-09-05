# Bayesian estimation of diffusivity spectra in diffusion MRI

This repo contains work in progress for estimating diffusivity spectra of normal and tumor tissue compartments of prostate gland diffusion MRI for 63 patients (relating to [Langkilde et al](https://pubmed.ncbi.nlm.nih.gov/28718517/) study), using bayesian methods for quantifying the diffusivity spectrum, defined by the inverse Laplace Transform.

All derived data is free from HIPAA PHI and repo should not contain any sensitive patient information.

## Motivation

There is great interest to quantify the spectrum of diffusivities that underlie the observed diffusion signal decay; separate tissue compartments can be identified by their spectral peaks. This spectrum is defined by the inverse Laplace transform, but unfortunately this transform is very sensitive to noise omnipresent in diffusion MRI. We present a Bayesian method of inverse Laplace transform that uses Gibbs sampling to provide spectra along with an estimate of the noise related to uncertainty in the spectra; this uncertainty information is valuable in interpreting the results. We show preliminary results estimating distributions on diffusivity spectra for normal and tumor tissues.

## Methods

Gibbs Sampler + Mean-Field Variational Bayes (TBD).


## File structure
````
.
├── README.md
├── data
│   ├── jsons
│   │   ├── ggg_aggressiveness_d2.json
│   │   ├── ggg_aggressiveness_d2_discarded.json
|   |   ├── processed_patient_dict.json
│   │   ├── tumor_normal_d1.json
|   |   └── tumor_normal_d1_discarded.json
│   └── metadata
│   |   ├── patient_gleason_grading.csv
│   |   └── prostate_rois.csv
|   └── processed
|       └── processed_data.pkl
├── output
|   ├── gibbs
|   |   ├── roi_avgs.csv
|   |   ├── roi_avgs.pdf
|   |   ├── npz.csv
|   |   ├── npz.pdf
|   |   ├── ntz.csv
|   |   ├── ntz.pdf
|   |   ├── tpz.csv
|   |   ├── tpz.pdf
|   |   ├── ttz.csv
|   |   ├── ttz.pdf
|   |   ├── neglected.csv
|   |   └── neglected.pdf 
│   └── eval
│   |   ├── ggg_stat_analysis_combined.csv
|   |   ├── gs_stratified_boxplot_multi.pdf
|   |   ├── gs_stratified_boxplot.pdf
|   |   └── normal_v_tumor_stat_analysis.csv
|   └── mfvb
└── src
    ├── models
    │   ├── gibbs.py
    |   └── mfvb.py
    ├── utils
    │   ├── generate_analysis_datasets.py
    │   └── process_data.py
    ├── testing
    │   └── simulation.py
    └── eval
         └── eval.py

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

TBD.

### Src

TBD.

### Instructions on how to run code

TBD.