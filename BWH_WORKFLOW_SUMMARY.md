# BWH Dataset Workflow - Implementation Summary

## Overview

Complete implementation of BWH-specific visualization and biomarker analysis pipeline with Monte Carlo uncertainty propagation.

## Features Implemented

### 1. **Anatomical Region Visualization** (`visualization/bwh_plotting.py`)

**Functionality:**
- Automatic grouping by anatomical region (Normal PZ, Normal TZ, Tumor PZ, Tumor TZ)
- Per-region boxplot PDFs (3×2 grid, paginated)
- Averaged spectra plots
- Statistical summaries (CSV exports)

**Outputs:**
```
results/plots/bwh/
├── normal_pz_spectra.pdf
├── normal_tz_spectra.pdf
├── tumor_pz_spectra.pdf
├── tumor_tz_spectra.pdf
├── averaged_spectra.pdf
├── normal_pz_stats.csv
├── normal_tz_stats.csv
├── tumor_pz_stats.csv
├── tumor_tz_stats.csv
└── averaged_stats.csv
```

### 2. **Feature Extraction** (`biomarkers/features.py`)

**Features:**
- **Individual**: Fraction at each diffusivity bin (8 features: D_0.25, D_0.50, ..., D_20.00)
- **Combo**: `D[0.25] + 1/(D[1.5] + D[2.0])` (empirically validated)
- **MC Propagation**: N=200 posterior samples per patient

**Uncertainty Quantification:**
- Mean features across MC samples
- Std of features (uncertainty)
- Feature samples for downstream prediction uncertainty

### 3. **ADC Baseline** (`biomarkers/adc_baseline.py`)

**Computation:**
- Zone-specific ADC (PZ and TZ separately)
- Configurable b-value range (default: 0-1000 s/mm²)
- Monoexponential fit: `S(b) = S_0 * exp(-b * ADC)`

### 4. **Classification** (`biomarkers/mc_classification.py`)

**Tasks:**
1. **Tumor vs Normal (PZ)**: Zone-specific classification
2. **Tumor vs Normal (TZ)**: Zone-specific classification  
3. **Gleason Grade (<7 vs ≥7)**: Tumor samples only, combined zones

**Methodology:**
- **LOOCV**: Leave-One-Out Cross-Validation (optimal for small N)
- **L2 Regularization**: Ridge penalty to prevent overfitting
- **MC Predictions**: Propagate posterior uncertainty to predictions
  - Train on mean features
  - Predict with N=200 samples per patient
  - Output: mean prediction ± uncertainty

**Statistical Tests:**
- **DeLong test**: Compare AUC values
- **Bootstrap CI**: 95% confidence intervals on AUC (1000 iterations)
- **P-values**: vs ADC baseline

### 5. **Visualization** (`biomarkers/biomarker_viz.py`)

**Outputs:**
```
results/biomarkers/
├── roc_tumor_vs_normal_pz.pdf
├── roc_tumor_vs_normal_tz.pdf
├── roc_ggg.pdf
├── auc_table_tumor_vs_normal_pz.csv
├── auc_table_tumor_vs_normal_tz.csv
├── auc_table_ggg.csv
├── features.csv
└── feature_uncertainty.csv
```

**Plots:**
- ROC curves (all features + ADC baseline)
- AUC tables with confidence intervals
- Prediction uncertainty plots (error bars)

## Pipeline Architecture

```
main.py
  ↓
  [1] Inference (NUTS sampler)
  ↓
  [2] BWH Visualization (run_diagnostics)
      → Group by region
      → Boxplot PDFs
      → Averaged plots
      → CSV statistics
  ↓
  [3] Biomarker Analysis (run_biomarker_analysis)
      → Extract features (MC)
      → Compute ADC baseline
      → LOOCV classification
      → Statistical comparison
      → Generate ROC curves
      → Export tables
```

## Configuration

**BWH Dataset Config** (`configs/dataset/bwh.yaml`):
```yaml
# Diffusivity grid (8 bins)
diff_values: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]

# Biomarker analysis
biomarker_n_mc_samples: 200
biomarker_regularization: 1.0
biomarker_adc_b_range: [0.0, 1.0]  # 0-1000 s/mm²
```

## Usage

### Run Full BWH Analysis:
```bash
uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  inference=nuts \
  inference.n_iter=5000 \
  inference.tune=500 \
  prior=ridge \
  prior.strength=0.1 \
  local=true
```

### Test with Subset:
```bash
uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  dataset.max_samples=10 \
  inference=nuts \
  inference.n_iter=2000 \
  inference.tune=200 \
  prior=ridge \
  prior.strength=0.1 \
  local=true
```

## Key Design Choices

### 1. **LOOCV vs K-Fold CV**
- **Choice**: LOOCV
- **Rationale**: Sample size N~20-40 per task → LOOCV maximizes training data
- **Trade-off**: Higher variance than K-fold, but unbiased estimator

### 2. **MC Sample Size (N=200)**
- **Choice**: 200 samples per prediction
- **Rationale**: 
  - Sufficient to capture uncertainty distribution
  - Fast computation (200 × 8 features × LR)
  - Standard in Bayesian ML literature
- **Alternative**: 100 samples (faster), 500 samples (more accurate)

### 3. **Regularization**
- **Choice**: L2 (Ridge) with C=1.0
- **Rationale**: 
  - 8 features with N~20-40 samples → need regularization
  - L2 preferred over L1 for correlated features (diffusivity bins are correlated)
  - C=1.0 is moderate regularization (can tune if needed)

### 4. **Feature Set**
- **Individual features**: Clinical interpretability (specific diffusivity bins)
- **Combo feature**: Empirically validated from previous experiments
- **Full LR model**: Expected best performance (uses all information)
- **ADC baseline**: Clinical standard for comparison

### 5. **Zone Stratification**
- **Tumor vs Normal**: Separate PZ and TZ
  - **Rationale**: Different tissue characteristics, ADC values vary by zone
- **GGG**: Combined zones
  - **Rationale**: Limited tumor samples, need sufficient N

## Sample Size Considerations

### Tumor vs Normal
- **PZ**: N ≈ 30-40 → Adequate for LOOCV
- **TZ**: N ≈ 30-40 → Adequate for LOOCV

### Gleason Grade Group
- **Total**: N ≈ 15-25 tumor samples → **Small but acceptable**
- **Mitigation**:
  - LOOCV (maximizes training data)
  - L2 regularization (prevents overfitting)
  - Bootstrap CI (quantifies uncertainty)
  - Report as "pilot study" / "feasibility analysis"

### ISMRM Publication Guidance
✅ **Acceptable for ISMRM** if you:
- Use proper cross-validation (LOOCV ✓)
- Report confidence intervals on metrics (Bootstrap CI ✓)
- Compare to established baseline (ADC ✓)
- Frame as "pilot investigation" or "feasibility study"
- Discuss limited sample size as limitation
- Emphasize novel Bayesian uncertainty quantification

⚠️ **Similar studies at ISMRM**: N=15-40 for prostate cancer classification

## Expected Results

### High-Performing Features
1. **Full LR model**: Likely best AUC (uses all 8 bins)
2. **Combo feature**: Strong single-feature performance
3. **D_0.25**: Low diffusivity → tumor signature
4. **ADC**: Clinical baseline (should be competitive)

### Uncertainty Benefits
- Model confidence ↓ when prediction is uncertain
- Calibration: Check if uncertainty correlates with errors
- Clinical value: "High confidence" vs "Low confidence" predictions

## File Structure

```
src/spectra_estimation_dmri/
├── visualization/
│   ├── __init__.py
│   └── bwh_plotting.py          # Regional visualization
├── biomarkers/
│   ├── __init__.py
│   ├── features.py              # MC feature extraction
│   ├── adc_baseline.py          # ADC computation
│   ├── mc_classification.py     # LOOCV + MC predictions
│   ├── biomarker_viz.py         # ROC curves & tables
│   └── pipeline.py              # Main orchestration
├── data/
│   └── data_models.py           # Updated run_diagnostics()
└── main.py                      # BWH workflow integration
```

## Next Steps

1. **Test on full BWH dataset** (remove `max_samples` limit)
2. **Tune hyperparameters** (regularization strength, MC samples)
3. **Validate results** (check AUC values, p-values, calibration)
4. **Publication prep**:
   - Generate figures for paper
   - Write methods section
   - Discuss uncertainty quantification novelty
   - Address limited sample size in discussion

## Notes for Supervisor

### Key Innovations
1. **Bayesian Uncertainty → Prediction Uncertainty**: Direct MC propagation from posterior to predictions
2. **Zone-Specific Analysis**: Accounts for PZ/TZ differences
3. **Comprehensive Comparison**: Individual features + combinations + full model vs ADC

### Expected Paper Contributions
- **Methodological**: Novel Bayesian approach to dMRI biomarkers
- **Clinical**: Zone-specific performance, prediction confidence intervals
- **Validation**: Rigorous LOOCV + statistical testing vs ADC standard

### Potential Issues & Mitigations
- **Small N for GGG**: Frame as pilot, report CIs, emphasize Bayesian rigor
- **Overfitting Risk**: L2 regularization, LOOCV, multiple feature comparisons
- **Computational Cost**: N=200 MC samples → ~10-30 seconds per patient (acceptable)

## References
- DeLong test: DeLong et al., 1988, Biometrics
- Bootstrap CI: Efron & Tibshirani, 1993
- LOOCV: Hastie et al., Elements of Statistical Learning
- Bayesian uncertainty: Gelman et al., Bayesian Data Analysis

