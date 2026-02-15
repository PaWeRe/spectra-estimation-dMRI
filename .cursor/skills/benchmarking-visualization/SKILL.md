---
name: benchmarking-visualization
description: Create publication-quality figures, tables, and diagnostic analyses for the MRM paper. Use when generating ROC curves, spectra plots, trace plots, convergence diagnostics, heatmaps, feature importance plots, patient tables, or any other visualization. Also use for benchmarking sampler performance and creating analysis scripts.
compatibility: Requires matplotlib, seaborn, pandas, numpy, arviz. Uses uv for execution.
---

# Benchmarking & Visualization

## When to use this skill
Use when the task involves:
- Generating publication-quality figures (300 DPI, PDF preferred)
- Creating diagnostic plots (trace, R-hat, ESS)
- Building analysis tables (AUC comparison, patient demographics)
- Benchmarking sampler convergence across conditions
- Designing new analysis experiments (SNR robustness, direction independence)

## Figure budget (10 max for MRM)

| # | Content | Status |
|---|---------|--------|
| 1 | Signal decay + synthetic validation | Regenerate from ISMRM |
| 2 | Sampling diagnostics (trace, convergence vs SNR) | NEW — build |
| 3 | Robustness: inverse spectra at various SNR | NEW — build |
| 4 | Averaged spectra by tissue type | Regenerate from ISMRM |
| 5 | Direction independence (spectra across directions) | BLOCKED on data |
| 6 | ROC curves (PZ, TZ, GGG) | Regenerate from ISMRM |
| 7 | Uncertainty propagation | Regenerate from ISMRM |
| 8 | Feature importance / LR coefficients | Regenerate from ISMRM |
| 9 | Pixel-wise spectral maps + heatmap | NEW — build |
| 10 | Compartment heatmaps on anatomy | NEW — build |

## Publication style guidelines
- Font: 10pt minimum in figures
- Colors: Use colorblind-safe palettes (viridis, tab10)
- Consistent axis labels with SI units
- Error bars or shaded regions for uncertainty
- Save as PDF (vector) for line plots, PNG (300 DPI) for heatmaps
- Figure width: single column (3.5") or double column (7.0")

## Code locations
- ISMRM exports: `src/spectra_estimation_dmri/visualization/ismrm_exports.py`
- BWH plotting: `src/spectra_estimation_dmri/visualization/bwh_plotting.py`
- Biomarker viz: `src/spectra_estimation_dmri/biomarkers/biomarker_viz.py`
- General plots: `src/spectra_estimation_dmri/utils/plotting.py`
- Sampler comparison: `src/spectra_estimation_dmri/analysis/sampler_comparison.py`

## New analyses needed

### Convergence diagnostics figure (Fig 2)
- Run NUTS on synthetic spectra at SNR = {50, 100, 200, 500, 1000}
- Plot: trace plots, R-hat vs SNR, ESS vs SNR
- Show convergence boundaries clearly

### Robustness test (Fig 3)
- Construct "inverse" spectra (opposite pattern to typical tumor/normal)
- Run at multiple SNRs
- Report RMSE, bias, coverage of credible intervals

### Pixel-wise maps (Fig 9-10)
- Apply NUTS to 64x64 native grid
- Each D component → separate image
- Apply logistic regression → heatmap
- Overlay on anatomical b=0 image

## How to generate existing figures
```bash
# Full pipeline with figure generation
uv run python -m spectra_estimation_dmri.main dataset=bwh inference=nuts local=true recompute=false
```
Figures are saved to `results/plots/` and `results/biomarkers/ismrm/`.

## Output directory
All figures go to `paper/figures/` for the manuscript.
Intermediate analysis outputs go to `results/`.
