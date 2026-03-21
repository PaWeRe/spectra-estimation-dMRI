# CLAUDE.md — spectra-estimation-dMRI

## Project Overview

MRM journal paper: **Bayesian spectral decomposition of multi-b diffusion MRI for prostate cancer characterization**. Codebase for inference, classification, and paper figure generation.

**Authors**: Patrick Remerscheid (lead), Sandy (co-author), Stephan (co-author)

## Key Commands

```bash
# Always use uv, never pip
uv run python -m spectra_estimation_dmri.biomarkers.recompute  # Recompute ALL paper numbers
uv run python -m spectra_estimation_dmri.main                  # Full Hydra pipeline
uv run python scripts/verify_findings.py                       # Legacy verification
uv sync                                                         # Install dependencies
```

## Architecture

```
src/spectra_estimation_dmri/
  main.py                    # Hydra-based pipeline entry point
  data/
    loaders.py               # BWH JSON + metadata CSV + pixel binary loaders
    data_models.py           # SignalDecay, DiffusivitySpectrum, etc.
    bwh/
      signal_decays.json     # Raw ROI signal data (56 patients, 149 ROIs)
      metadata.csv           # Patient metadata with GGG labels
  models/prob_model.py       # ProbabilisticModel (design matrix, MAP, priors)
  inference/
    map.py                   # Ridge NNLS point estimation
    nuts.py                  # PyMC NUTS sampler
    gibbs.py                 # Gibbs sampler (legacy)
  biomarkers/
    recompute.py             # SINGLE SOURCE OF TRUTH for all paper numbers
    pipeline.py              # Full biomarker analysis orchestrator (Hydra)
    features.py              # Feature extraction from spectra (MC uncertainty)
    adc_baseline.py          # ADC computation (monoexponential fit)
    mc_classification.py     # LOOCV classification + DeLong test
    biomarker_viz.py         # Visualization
  pixelwise.py               # Pixel-wise estimation (MAP + NUTS, standalone)
  visualization/             # ISMRM + BWH plotting

configs/                     # Hydra configs (dataset, inference, prior, etc.)
results/
  inference_bwh_backup/      # 149 .nc NUTS posterior files (GOLD STANDARD)
  biomarkers/                # Definitive features + AUC tables (Session 8)
paper/                       # LaTeX sections (shell structure exists)
scripts/                     # Paper figure generators
```

## Critical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| b-values | [0, 250, ..., 3500] s/mm² (15 values) | Langkilde 2018 |
| Diffusivity bins | [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] um²/ms | 8 components |
| Ridge lambda (MAP) | 0.1 | configs/prior/ridge.yaml |
| NUTS | 2000 draws, 200 tune, 4 chains, target_accept=0.95 | |
| ADC b_max | 1.0 ms/um² (= 1000 s/mm²) | Best from Session 6 |
| Dataset | 56 patients, 149 ROIs (40 tumor, 109 normal) | |

## Conventions

- **uv only** — never use pip
- **Edit existing files** — don't create standalone scripts unless necessary
- **Overleaf** for LaTeX — no local compiler
- **Max 10 figures** in the paper
- **Go step by step** — don't rush ahead
- All questions to co-authors go through Patrick

## Current State (Session 8, 2026-03-21)

**All blockers resolved.** Clean recomputation done. Paper drafting can begin.

**Definitive results** in `results/biomarkers/` (regenerate with `recompute.py`):
- `features.csv` — MAP + NUTS features + ADC + metadata for all 149 ROIs
- `auc_table.csv` — LOOCV AUCs at C=0.1, 1.0, 10.0
- `adc_discriminant.csv` — ADC vs spectral discriminant correlations
- `identifiability.csv` — per-component posterior CV
- `map_nuts_comparison.csv` — MAP vs NUTS correlation per component

**Gold standard data (do not modify):**
- `results/inference_bwh_backup/` — 149 .nc NUTS posterior files
- `src/spectra_estimation_dmri/data/bwh/signal_decays.json`
- `src/spectra_estimation_dmri/data/bwh/metadata.csv`
