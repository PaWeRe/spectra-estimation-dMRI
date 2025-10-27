# Simple Biomarker Analysis Guide

## Overview

This module provides a **lean, practical approach** to Gleason score prediction from diffusivity spectra with **uncertainty quantification**.

### Key Features

1. **Uncertainty as Features** - Add std and CI width alongside point estimates
2. **Monte Carlo Predictions** - Use posterior samples directly for prediction uncertainty
3. **Simple API** - No complex classes, just functions
4. **Clinical Biomarkers** - Low/mid/high diffusivity fractions and ratios

---

## Quick Start

### Option 1: Standalone Script (Recommended for Testing)

```bash
# Run complete pipeline on BWH data with Gibbs sampler
uv run python scripts/run_bwh_biomarker_analysis.py \
  --sampler gibbs \
  --n_iter 5000 \
  --classifier logistic

# With Monte Carlo predictions for uncertainty
uv run python scripts/run_bwh_biomarker_analysis.py \
  --sampler nuts \
  --n_iter 2000 \
  --use_mc \
  --classifier random_forest
```

**Arguments:**
- `--sampler`: `gibbs` or `nuts` (default: gibbs)
- `--n_iter`: MCMC iterations (default: 5000)
- `--n_chains`: Number of chains (default: 4)
- `--prior`: Prior type (default: ridge)
- `--prior_strength`: Prior strength (default: 0.01)
- `--use_mc`: Enable Monte Carlo predictions with uncertainty
- `--classifier`: `logistic` or `random_forest` (default: logistic)
- `--output_dir`: Output directory (default: results/bwh_biomarker_analysis)

### Option 2: Using main.py with Hydra

```bash
# Run with biomarker analysis enabled
uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  inference=gibbs \
  inference.n_iter=5000 \
  prior=ridge \
  prior.strength=0.01 \
  biomarker_analysis.enabled=true \
  biomarker_analysis.use_uncertainty=true \
  biomarker_analysis.use_mc_predictions=false \
  classifier=logistic \
  local=true
```

---

## How It Works

### 1. Biomarker Extraction

From each reconstructed diffusivity spectrum, we extract:

**Point Estimates:**
- `low_frac`: Sum of spectrum for D < 1.0 μm²/ms (restricted diffusion, tumor marker)
- `mid_frac`: Sum for 1.0 ≤ D < 2.5 μm²/ms
- `high_frac`: Sum for D ≥ 2.5 μm²/ms (free water)
- `low_high_ratio`: **Key biomarker** - ratio of restricted to free diffusion
- `low_mid_ratio`: Secondary ratio feature

**Uncertainty Features** (if `use_uncertainty=True`):
- `low_frac_std`: Standard deviation of low fraction across MCMC samples
- `low_high_ratio_std`: Uncertainty in the ratio
- `low_frac_ci_width`: Width of 95% credible interval
- `low_high_ratio_ci_width`: Ratio CI width
- `spectrum_uncertainty`: Global uncertainty metric

### 2. Two Prediction Modes

#### Mode A: Point Estimates + Uncertainty Features (Fast)

```python
from spectra_estimation_dmri.biomarkers.simple_gleason_predictor import SimpleGleasonPredictor

# Extract features (includes uncertainty)
X = extract_features_from_dataset(spectra_dataset, include_uncertainty=True)

# Train classifier
predictor = SimpleGleasonPredictor(model_type='logistic', use_uncertainty_features=True)
predictor.fit(X, y)

# Predict
y_pred = predictor.predict(X)
y_prob = predictor.predict_proba(X)  # P(high-grade cancer)
```

**Pros:**
- Fast (single forward pass)
- Classifier learns to use uncertainty (e.g., "be cautious if uncertain")
- Easy to interpret feature importance

**Cons:**
- No prediction uncertainty
- Assumes features are independent

#### Mode B: Monte Carlo Predictions (Principled)

```python
# Use posterior samples directly
pred_probs, pred_std = predictor.predict_with_uncertainty(
    spectra_dataset, 
    n_mc_samples=100
)

# pred_probs: Mean prediction across 100 posterior samples
# pred_std: Standard deviation → prediction uncertainty!
```

**Pros:**
- Full uncertainty propagation through pipeline
- Gives prediction confidence intervals: `P(high-grade) = 0.75 ± 0.12`
- More principled Bayesian workflow

**Cons:**
- Slower (100× more forward passes)
- Higher variance in predictions

### 3. Why This Helps

**Clinical Insight:**
- Tumor tissue has **restricted diffusion** (low D) → high `low_frac`
- Normal tissue has **free diffusion** (high D) → high `high_frac`
- `low_high_ratio` captures this directly

**Uncertainty Helps:**
- **Scenario 1**: `low_high_ratio = 2.5 ± 0.1` → High confidence, likely tumor
- **Scenario 2**: `low_high_ratio = 2.5 ± 1.2` → High uncertainty, need more data
- Classifier learns: "If `low_high_ratio_std` is high, be more conservative"

---

## Output Files

All results saved to `{output_dir}/`:

1. **`biomarker_results.csv`** - Features, predictions, true labels for each ROI
2. **`feature_importance.pdf`** - Bar plot of most important features
3. **`mc_predictions.pdf`** (if MC mode) - Predictions with uncertainty bars

Console output includes:
- Cross-validation AUC
- Confusion matrix metrics (sensitivity, specificity)
- Top 5 most important features

---

## Tips for Your Use Case

### Quick Exploration
```bash
# Start with simple point estimates + uncertainty features
uv run python scripts/run_bwh_biomarker_analysis.py \
  --sampler gibbs \
  --n_iter 5000 \
  --classifier logistic
```

**Expected runtime:** ~10-15 minutes for full BWH dataset

### Principled Analysis
```bash
# Use Monte Carlo predictions for full uncertainty quantification
uv run python scripts/run_bwh_biomarker_analysis.py \
  --sampler nuts \
  --n_iter 10000 \
  --use_mc \
  --classifier random_forest
```

**Expected runtime:** ~1-2 hours (MC predictions are slower)

### Biomarker Discovery

After running, check `feature_importance.pdf`:
- If `low_high_ratio` is top feature → confirms clinical hypothesis
- If `low_high_ratio_std` is important → uncertainty matters!
- If `spectrum_uncertainty` is important → noisy data, need more samples

### Quick Test (5 ROIs only)

The standalone script limits to 5 ROIs for testing. Remove this limit in `scripts/run_bwh_biomarker_analysis.py`:

```python
# Line 88: Change from
for i, signal_decay in enumerate(samples_with_labels[:5]):

# To
for i, signal_decay in enumerate(samples_with_labels):
```

---

## Comparison to MAP/Point Estimates

| Approach | Reconstruction | Prediction | Uncertainty | Speed |
|----------|---------------|------------|-------------|-------|
| **MAP only** | Point estimate | Point estimate | ❌ None | ⚡⚡⚡ Fastest |
| **MCMC + Point** | Posterior mean | Point estimate | ❌ Lost | ⚡⚡ Fast |
| **MCMC + Uncertainty Features** | Posterior mean | Point estimate | ⚠️ Partial | ⚡⚡ Fast |
| **MCMC + MC Predictions** | Posterior samples | Distribution | ✅ Full | ⚡ Slower |

**Recommendation:** Start with **MCMC + Uncertainty Features** (Mode A). If uncertainty features are important (check `feature_importance.pdf`), consider Mode B for final analysis.

---

## Troubleshooting

### "No samples with Gleason scores"
- Check that `metadata.csv` has valid `targets` column
- Some patients have missing Gleason scores (normal tissue)

### "AUC = 0.5" (random guessing)
- Not enough samples (need ~20+ per class)
- Features not informative (check if spectra look reasonable)
- Try different prior strength or more MCMC iterations

### Monte Carlo predictions crash
- Some spectra might not have `spectrum_samples` (e.g., MAP estimates)
- Code falls back to 0.5 probability (uncertain)

---

## Next Steps

1. **Run quick test** (5 ROIs) to verify pipeline works
2. **Run full BWH dataset** (~60 ROIs with labels)
3. **Check feature importance** - does `low_high_ratio` dominate?
4. **Try MC predictions** if uncertainty features are important
5. **Compare Gibbs vs NUTS** - which gives better uncertainty estimates?
6. **Optimize classifier** - try different models, feature combinations

---

## Questions to Answer

- [ ] Which sampler (Gibbs/NUTS) gives better biomarker discrimination?
- [ ] Are uncertainty features important for classification?
- [ ] Does MC prediction improve over point estimates + uncertainty features?
- [ ] What's the minimum MCMC iterations needed for stable biomarkers?
- [ ] Which specific diffusivity buckets matter most for Gleason prediction?

**Simple, fast, and principled. Let's predict some cancer!** 🎯

