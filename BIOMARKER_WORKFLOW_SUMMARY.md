# Biomarker Workflow Summary

## ✓ Status: Validated and Working

The logistic regression biomarker classification pipeline has been validated with synthetic data and is ready for use with real BWH prostate MRI data.

---

## Pipeline Overview

```
Raw dMRI Signal → NUTS Inference → Posterior Samples → Feature Extraction → Classification
      (s)            p(R,σ|s)        {R^(i), σ^(i)}      MC Propagation      LOOCV LR
```

---

## 1. Bayesian Inference (NUTS)

**Input**: Signal decay \( s \in \mathbb{R}^m \) at multiple b-values  
**Output**: Posterior samples \( \{R^{(i)}, \sigma^{(i)}\}_{i=1}^{N_{\text{samples}}} \)

The NUTS sampler (implemented in `src/spectra_estimation_dmri/inference/nuts.py`) provides:
- ~2000-4000 posterior samples per spectrum (4 chains × 500-1000 draws)
- Full uncertainty quantification over diffusivity spectrum \( R \)
- Joint inference of noise parameter \( \sigma \)

---

## 2. Feature Extraction with Uncertainty Propagation

**File**: `src/spectra_estimation_dmri/biomarkers/features.py`

### Features Extracted

From each spectrum \( R \in \mathbb{R}^n \), we extract:

1. **Individual diffusivity fractions**: 
   - \( D_{0.25}, D_{0.50}, \ldots, D_{3.00} \)
   - These are the normalized weights in each diffusivity bin

2. **Engineered combo feature**: 
   - \( f_{\text{combo}}(R) = R_{D=0.25} + \frac{1}{R_{D=2.0}} + \frac{1}{R_{D=3.0}} \)
   - Captures restricted diffusion (high \( D_{0.25} \)) and penalizes high free diffusion

3. **Baseline comparison**: 
   - ADC (Apparent Diffusion Coefficient) from signal decay
   - Implemented in `src/spectra_estimation_dmri/biomarkers/adc_baseline.py`

### Monte Carlo Uncertainty Propagation

For each patient, we:

1. **Sample** \( N = 200 \) draws from the posterior (randomly subsampled from NUTS samples)
2. **Transform** each sample through feature functions:
   ```
   R^(1) → f(R^(1))
   R^(2) → f(R^(2))
      ⋮
   R^(N) → f(R^(N))
   ```
3. **Characterize** the feature distribution:
   - Mean: \( \bar{f} = \frac{1}{N} \sum_{i=1}^N f(R^{(i)}) \)
   - Std: \( \sigma_f = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (f(R^{(i)}) - \bar{f})^2} \)

This propagates **epistemic uncertainty** (parameter uncertainty) into derived biomarkers.

**Key insight**: The standard deviation \( \sigma_f \) quantifies how uncertain we are about the feature value given the data. High \( \sigma_f \) means the posterior is diffuse over that feature.

---

## 3. Classification

**File**: `src/spectra_estimation_dmri/biomarkers/mc_classification.py`

### Leave-One-Out Cross-Validation (LOOCV)

**Why LOOCV?**
- Small clinical datasets (\( n \approx 20-50 \)) make train/test splits unreliable
- LOOCV uses \( n-1 \) samples for training, 1 for testing, repeated \( n \) times
- Provides **unbiased** performance estimates and per-sample predictions

**Algorithm**:
```
For i = 1 to n:
    1. Hold out sample i
    2. Train on samples {1, ..., i-1, i+1, ..., n}:
        a. Standardize features (zero mean, unit variance)
        b. Fit logistic regression with L2 regularization
    3. Predict probability for sample i
    4. Store prediction
Return: [p_1, p_2, ..., p_n]
```

### Model: L2-Regularized Logistic Regression

**Specification**:
- **Type**: Binary logistic regression
- **Regularization**: L2 penalty with \( C = 1.0 \) (inverse regularization strength)
- **Solver**: L-BFGS (efficient for small datasets, ~1000 iterations)
- **Feature scaling**: StandardScaler (crucial for gradient-based optimization)

**Model equation**:
\[
P(y = 1 | x) = \sigma(w^T x + b)
\]
where \( \sigma(z) = 1/(1 + e^{-z}) \) is the logistic function.

**Regularization penalty**:
\[
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right] + \frac{1}{2C} \|w\|_2^2
\]

### Classification Tasks

We evaluate **three clinical tasks**:

1. **Tumor vs Normal (Peripheral Zone)**
   - **Binary**: 0 = normal, 1 = tumor
   - **Purpose**: Detect cancer in PZ tissue
   - **Clinical significance**: Most aggressive prostate cancers arise in PZ

2. **Tumor vs Normal (Transition Zone)**
   - **Binary**: 0 = normal, 1 = tumor
   - **Purpose**: Detect cancer in TZ tissue
   - **Note**: TZ has different baseline diffusion than PZ

3. **Gleason Grade Stratification** (among tumors)
   - **Binary**: 0 = GGG 1-2 (GS ≤ 7), 1 = GGG 3-5 (GS > 7)
   - **Purpose**: Distinguish low-grade from high-grade cancer
   - **Clinical significance**: High-grade cancers require aggressive treatment

---

## 4. Evaluation Metrics

**File**: `src/spectra_estimation_dmri/biomarkers/mc_classification.py` (lines 195-233)

### Primary Metric: AUC

**Area Under ROC Curve (AUC)**: 
- Measures discrimination ability (can the classifier separate classes?)
- Range: [0, 1], where 0.5 = random, 1.0 = perfect
- Interpretation:
  - AUC > 0.9: Excellent
  - AUC > 0.8: Good
  - AUC > 0.7: Acceptable
  - AUC > 0.6: Poor
  - AUC ≤ 0.5: No discrimination

### Secondary Metrics

Computed from confusion matrix at optimal threshold:

- **Accuracy**: \( \frac{TP + TN}{TP + TN + FP + FN} \)
- **Sensitivity** (recall): \( \frac{TP}{TP + FN} \) — fraction of positives correctly identified
- **Specificity**: \( \frac{TN}{TN + FP} \) — fraction of negatives correctly identified  
- **Precision** (PPV): \( \frac{TP}{TP + FP} \) — fraction of positive predictions that are correct

### Statistical Rigor

1. **Bootstrap Confidence Intervals**:
   - 1000 bootstrap resamples
   - 95% CI for AUC
   - Accounts for small sample uncertainty

2. **DeLong Test** (optional):
   - Statistically compare AUC values between models
   - Tests if AUC differences are significant

---

## 5. Pipeline Orchestration

**File**: `src/spectra_estimation_dmri/biomarkers/pipeline.py`

The `run_biomarker_analysis()` function orchestrates the complete workflow:

```python
def run_biomarker_analysis(
    spectra_dataset,           # DiffusivitySpectraDataset with NUTS samples
    output_dir,                # Where to save results
    n_mc_samples=200,          # MC samples for uncertainty propagation
    regularization=1.0,        # L2 regularization strength
    adc_b_range=(0.0, 1.0),    # b-value range for ADC (ms units)
):
    # Step 1: Extract features from all spectra
    features_df, uncertainty_df = extract_features_from_dataset(
        spectra_dataset, n_mc_samples
    )
    
    # Step 2: Extract ADC baseline
    adc_df = extract_adc_features(spectra_dataset, b_range=adc_b_range)
    features_df = features_df.merge(adc_df, on="patient_id")
    
    # Step 3: Run classification for each task
    for task in ["tumor_vs_normal_pz", "tumor_vs_normal_tz", "ggg"]:
        X_meta, X_features, y = prepare_classification_data(features_df, task)
        
        # Evaluate multiple feature sets
        for feature_set in [individual_features, combo, full_model, adc]:
            result = evaluate_feature_set(X_meta, X_features, y, feature_set)
            # Compute AUC, CI, etc.
    
    # Step 4: Create visualizations and summary report
    create_summary_report(all_results, output_dir)
    
    return results
```

---

## 6. Usage

### Automatic Execution

The biomarker analysis **runs automatically** when you process BWH data:

```bash
cd /Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI
uv run python -m spectra_estimation_dmri.main dataset=bwh inference=nuts
```

See `main.py` lines 288-327:
```python
if cfg.dataset.name == "bwh":
    biomarker_results = run_biomarker_analysis(
        spectra_dataset=spectra_dataset,
        output_dir=biomarker_dir,
        n_mc_samples=200,
        regularization=1.0,
    )
```

### Validation

To test the implementation with synthetic data:

```bash
uv run python scripts/validate_logistic_regression.py
```

This validates:
- ✓ Feature extraction with MC propagation
- ✓ LOOCV training procedure
- ✓ Metric computation
- ✓ Bootstrap CIs

---

## 7. Output Files

After running, check `results/biomarkers/`:

### CSV Files

- **`features.csv`**: Mean feature values per patient with metadata
- **`feature_uncertainty.csv`**: Standard deviations (uncertainties) per feature

### Visualizations

- **`roc_tumor_vs_normal_pz.pdf`**: ROC curves for PZ tumor detection
- **`roc_tumor_vs_normal_tz.pdf`**: ROC curves for TZ tumor detection
- **`roc_ggg.pdf`**: ROC curves for Gleason grade stratification
- **`summary_report.pdf`**: Comprehensive summary with all metrics

### Console Output

The pipeline prints:
```
BIOMARKER ANALYSIS PIPELINE
========================================
[STEP 1/4] Extracting features from spectra...
  Extracted 45 samples
  Features: 8

[STEP 2/4] Running classification tasks...
  Task 1: Tumor vs Normal (PZ)
    Samples: 30 (Normal: 15, Tumor: 15)
    Individual features...
    Full LR: AUC = 0.85 [0.72, 0.95]
    ADC (baseline): AUC = 0.72 [0.57, 0.85]

  Task 2: Tumor vs Normal (TZ)
    ...

  Task 3: Gleason Grade Group
    ...

[STEP 3/4] Saving feature tables...
[STEP 4/4] Creating visualizations...

✓ BIOMARKER ANALYSIS COMPLETE
```

---

## 8. Interpretation Guide

### What to Look For

1. **ROC Curves**:
   - Curves closer to top-left corner = better discrimination
   - Compare spectrum-based models to ADC baseline
   - Check if Full LR outperforms individual features

2. **Feature Importance**:
   - Which diffusivity bins have highest AUC?
   - Does combo feature outperform individual bins?
   - Does Full LR model improve over best individual feature?

3. **Uncertainty Tables**:
   - High uncertainty (large std) → posterior is diffuse
   - Low uncertainty → tight posterior distribution
   - Can we use uncertainty as a biomarker itself?

### Expected Results (Hypotheses)

- **PZ tumor vs normal**: Expect AUC > 0.75 (tumors have restricted diffusion)
- **TZ tumor vs normal**: Potentially lower AUC (TZ is more heterogeneous)
- **Gleason grade**: Expect AUC > 0.65 (high-grade has more restriction)
- **Spectrum vs ADC**: Expect spectrum-based models to outperform ADC (richer information)

---

## 9. Key Advantages

✅ **Uncertainty-aware**: Propagates Bayesian posterior uncertainty into predictions  
✅ **Robust validation**: LOOCV maximizes data efficiency for small cohorts  
✅ **Interpretable**: Logistic regression coefficients show which features matter  
✅ **Clinical grounding**: Compares against ADC (current standard of care)  
✅ **Comprehensive**: Three clinically relevant classification tasks  
✅ **Statistically rigorous**: Bootstrap CIs and significance tests  

---

## 10. Code Structure

```
src/spectra_estimation_dmri/biomarkers/
├── features.py              # Feature extraction + MC propagation
├── adc_baseline.py          # ADC computation for comparison
├── mc_classification.py     # LOOCV logistic regression
├── pipeline.py              # Orchestrate workflow
└── biomarker_viz.py         # Visualizations

scripts/
└── validate_logistic_regression.py  # Validation with synthetic data
```

---

## 11. Configuration

In `configs/dataset/bwh.yaml`:

```yaml
# Biomarker analysis settings
biomarker_n_mc_samples: 200        # MC samples for feature uncertainty
biomarker_regularization: 1.0       # L2 regularization strength
biomarker_adc_b_range: [0.0, 1.0]  # b-value range for ADC (ms units)
```

---

## 12. Future Extensions

**Potential improvements**:

1. **Use uncertainty in classification**:
   - Train on mean + std features
   - Weight samples by inverse uncertainty
   - Ensemble predictions across MC samples

2. **Feature selection**:
   - Use LASSO (L1) for sparse models
   - Recursive feature elimination (RFE)

3. **Advanced models**:
   - Random forests (non-linear boundaries)
   - Gradient boosting (better for small data)
   - Gaussian processes (uncertainty-aware)

4. **Clinical validation**:
   - External validation cohort
   - Longitudinal analysis (progression prediction)
   - Multi-site validation

---

## Summary for Supervisor

**In brief**: We train L2-regularized logistic regression classifiers on diffusivity spectrum features extracted via Monte Carlo propagation from Bayesian posteriors. Leave-One-Out Cross-Validation provides robust performance estimates for small clinical cohorts. We predict prostate cancer diagnosis and Gleason grade, comparing against ADC baselines. The pipeline is validated, working correctly, and ready for real BWH data analysis.

---

## References

**Implementation files**:
- `LOGISTIC_REGRESSION_EXPLANATION.md`: Detailed methodology explanation
- `ISMRM_ABSTRACT_UNCERTAINTY_PROPAGATION.md`: Abstract text for ISMRM submission
- `scripts/validate_logistic_regression.py`: Validation script

**Key functions**:
- `run_biomarker_analysis()`: Main pipeline orchestrator
- `extract_mc_features()`: MC-based feature extraction
- `loocv_predictions()`: LOOCV training loop
- `evaluate_feature_set()`: Complete evaluation for one feature set

---

**Status**: ✓ Validated and ready for BWH data analysis

