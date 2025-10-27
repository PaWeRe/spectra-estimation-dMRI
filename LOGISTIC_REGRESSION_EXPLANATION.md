# Logistic Regression for Prostate Cancer Biomarker Analysis

## Overview

We use **logistic regression with Leave-One-Out Cross-Validation (LOOCV)** to predict prostate cancer diagnosis and grade from diffusivity spectrum features extracted from dMRI data.

## The Approach

### 1. Feature Extraction
From each patient's reconstructed diffusivity spectrum \( R \in \mathbb{R}^n \), we extract:
- **Individual diffusivity fractions**: \( D_{0.25}, D_{0.50}, \ldots, D_{3.00} \) (the relative contribution of each diffusivity bin)
- **Engineered features**: e.g., \( D_{0.25} + \frac{1}{D_{2.0}} + \frac{1}{D_{3.0}} \) to capture restricted diffusion patterns
- **Baseline**: Apparent Diffusion Coefficient (ADC) for comparison

### 2. Uncertainty Propagation (Monte Carlo)
Since we use Bayesian inference (NUTS), each spectrum has **posterior samples** \( \{R^{(1)}, R^{(2)}, \ldots, R^{(M)}\} \) that quantify uncertainty.

For each patient:
1. Sample \( N \) posterior draws (default: 200)
2. Extract features from each draw → \( N \) feature vectors
3. Compute **mean** and **standard deviation** of each feature
4. The std captures our uncertainty about that feature's value

This allows us to propagate posterior uncertainty through the feature extraction pipeline.

### 3. Classification with LOOCV

**Why LOOCV?**  
With small clinical datasets (\( n \approx 20-50 \)), traditional train/test splits are unreliable. LOOCV provides:
- Maximum use of limited data (trains on \( n-1 \) samples)
- Unbiased performance estimates
- Per-sample predictions for ROC analysis

**Training procedure:**
```
For each patient i = 1, ..., n:
    1. Hold out patient i
    2. Train logistic regression on remaining n-1 patients
       - Standardize features (zero mean, unit variance)
       - Fit: P(y = tumor | features) via L2-regularized LR
    3. Predict on held-out patient i
    4. Store predicted probability
```

**Model specification:**
- **Regularization**: L2 penalty with strength \( C = 1.0 \) (inverse regularization)
- **Solver**: L-BFGS (efficient for small datasets)
- **Feature scaling**: StandardScaler (essential for gradient-based optimization)

### 4. Classification Tasks

We evaluate three clinically relevant tasks:

1. **Tumor vs Normal (Peripheral Zone)**: Detect cancer in PZ tissue
2. **Tumor vs Normal (Transition Zone)**: Detect cancer in TZ tissue  
   *(Note: TZ has different diffusion properties than PZ)*
3. **Gleason Grade Stratification**: Among tumors, predict GGG < 3 vs ≥ 3  
   *(Corresponds to Gleason Score < 7 vs ≥ 7, i.e., low vs high grade)*

### 5. Performance Evaluation

For each feature set (individual features, combinations, full model, ADC):
- **Primary metric**: Area Under ROC Curve (AUC)
- **Secondary metrics**: Accuracy, sensitivity, specificity, precision
- **Confidence intervals**: Bootstrap resampling (1000 iterations) for AUC CIs
- **Statistical comparison**: DeLong test to compare AUC values between models

### 6. Key Advantages

✅ **Uncertainty-aware**: Propagates Bayesian posterior uncertainty into predictions  
✅ **Robust validation**: LOOCV maximizes data efficiency for small cohorts  
✅ **Interpretable**: Logistic regression coefficients show which diffusivity bins matter  
✅ **Clinical grounding**: Compares against ADC baseline (current clinical standard)

## Implementation Summary

The pipeline is implemented in modular fashion:

- **`features.py`**: MC-based feature extraction from posterior samples
- **`mc_classification.py`**: LOOCV logistic regression training and evaluation
- **`pipeline.py`**: Orchestrates workflow (extraction → classification → visualization)
- **`adc_baseline.py`**: Computes ADC from signal decay for comparison

The workflow runs automatically on BWH prostate data when `cfg.dataset.name == "bwh"` in `main.py` (lines 288-327).

## Results Interpretation

After running, check:
- **ROC curves** (`results/biomarkers/roc_*.pdf`): Visual comparison of classifiers
- **Feature table** (`results/biomarkers/features.csv`): Extracted features per patient
- **Metrics summary**: Console output shows AUC ± CI for each task

**What to look for:**
- AUC > 0.7: Good discrimination
- Full LR outperforming ADC: Spectrum features add value beyond standard ADC
- Individual feature AUCs: Identifies most informative diffusivity bins

