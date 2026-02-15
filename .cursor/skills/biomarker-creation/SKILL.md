---
name: biomarker-creation
description: Run cancer classification pipeline that transforms diffusivity spectra into tumor biomarkers using logistic regression. Use when the task involves feature extraction from spectra, L2-regularized logistic regression with LOOCV, AUC computation, bootstrap uncertainty propagation, or comparing spectral features against ADC baselines.
compatibility: Requires scikit-learn, pandas, numpy. Uses uv for execution.
---

# Biomarker Creation

## When to use this skill
Use when the task involves:
- Extracting classification features from diffusivity spectra
- Running logistic regression with LOOCV
- Computing AUC with bootstrap confidence intervals
- Propagating posterior uncertainty through the classifier
- Comparing spectra-based vs ADC-based classification
- Feature selection and importance analysis

## How to run the full pipeline
```bash
uv run python -m spectra_estimation_dmri.main dataset=bwh inference=nuts prior=ridge local=true recompute=false
```
The biomarker pipeline runs automatically after inference for BWH data.

## Pipeline steps
1. **Feature extraction**: Posterior mean of each D bin → feature vector per ROI
2. **ADC baseline**: Monoexponential fit (b=0-1250 s/mm²) for comparison
3. **Classification tasks**:
   - PZ tumor vs normal (N=81: 54 normal, 27 tumor)
   - TZ tumor vs normal (N=68: 55 normal, 13 tumor)
   - GGG 1-2 vs 3-5 (N=28 PZ tumors with Gleason data)
4. **Evaluation**: LOOCV with AUC, bootstrap 95% CI (N=200)
5. **Uncertainty propagation**: MC samples through classifier (optional)

## Code locations
- Pipeline: `src/spectra_estimation_dmri/biomarkers/pipeline.py`
- Feature extraction: `src/spectra_estimation_dmri/biomarkers/features.py`
- Classification: `src/spectra_estimation_dmri/biomarkers/mc_classification.py`
- ADC baseline: `src/spectra_estimation_dmri/biomarkers/adc_baseline.py`
- Visualization: `src/spectra_estimation_dmri/biomarkers/biomarker_viz.py`

## Key configuration (in dataset/bwh.yaml)
- `biomarker_regularization`: L2 strength C (default: 1.0)
- `biomarker_n_mc_samples`: MC samples for uncertainty (default: 200)
- `biomarker_propagate_uncertainty`: Enable MC through classifier (default: false)
- `biomarker_use_feature_selection`: Top-k feature selection (default: false)
- `biomarker_random_seed`: Reproducibility seed (default: 42)

## Current results (from ISMRM abstract)
| Task | AUC | 95% CI |
|------|-----|--------|
| PZ tumor detection | 0.94 | [0.87, 0.99] |
| TZ tumor detection | 0.93 | [0.81, 1.00] |
| GGG stratification | 0.82 | [0.63, 0.96] |
| ADC PZ baseline | 0.94 | — |
| ADC TZ baseline | 0.96 | — |
| ADC GGG baseline | 0.79 | — |

## Output files
Results saved to `results/biomarkers/`:
- `features.csv`, `feature_uncertainty.csv`
- `predictions_*.csv` (per task, with uncertainty)
- `auc_table_*.csv`, `feature_importance_*.csv`
- ISMRM-style figures in `results/biomarkers/ismrm/`

## Uncertainty propagation
When enabled, for each LOOCV fold:
1. Sample N=200 spectra from posterior
2. Compute feature vectors for each sample
3. Train classifier on mean features, predict on each sample
4. Report mean prediction ± std across samples
- Samples crossing 0.5 threshold → 1.98× higher misclassification rate
