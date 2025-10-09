# Understanding MCMC Uncertainty Calibration for Clinical Use

## Your Goal (Clinical Perspective)

When a clinician sees a reconstructed diffusivity spectrum, they need to know:
1. **Point estimate**: What's our best guess for this patient?
2. **Uncertainty**: How confident should we be in this estimate?
3. **Trustworthiness**: Are these uncertainty intervals reliable?

For example:
```
Patient X: 
  - d=0.5 fraction: 0.30 ± 0.05 (95% CI: [0.20, 0.40])
  - Clinician question: "Can I trust this ±0.05 uncertainty?"
```

## What You're Currently Measuring

### Current Approach (Pooled Realizations)
```python
# With 5 noise realizations:
1. Pool all MCMC samples from all realizations
2. Compute ONE interval from pooled samples
3. Check if true value is in this interval
4. Result: 100% coverage (too wide!)
```

**What this measures**: 
- "How much does the estimate vary across DIFFERENT noise instances?"
- This is NOT what clinicians need!

### Why This Doesn't Work

Clinicians see ONE patient's data, not an average across multiple noisy versions.
They need: "Given THIS patient's data, how uncertain are we?"

## What You SHOULD Measure: Calibration

### Definition of Calibration

**Well-calibrated credible intervals:**
> If you claim a "95% credible interval" contains the true value,
> it should actually contain the true value in 95% of cases.

### How to Test This

#### Method 1: Simulation-Based Calibration (SBC) ✅ GOLD STANDARD

**Protocol:**
1. For i = 1 to 100 (or more):
   a. Simulate true spectrum from prior
   b. Generate data from this true spectrum + noise
   c. Run MCMC on this data
   d. Compute 95% credible interval
   e. Check if true spectrum is in interval
2. Count how many times true value was captured
3. Should be ~95 out of 100

**This tests:** "Across many patients, are my intervals trustworthy?"

#### Method 2: Coverage vs Width Plot (What You Want)

For EACH diffusivity bucket:
1. Run MCMC on N independent datasets
2. For each dataset i:
   - Compute 95% CI from MCMC samples
   - width_i = CI_upper - CI_lower
   - coverage_i = 1 if true value in CI, else 0
3. Plot: coverage vs width across datasets

**Ideal result:**
- Coverage ≈ 95% (across datasets)
- Narrower intervals = more informative
- Consistency across diffusivities

## Current vs Correct Implementation

### What Your Code Does (INCORRECT for calibration)

```python
# Pools all realizations
combined_samples = all_realizations.flatten()
ci_lower = percentile(combined_samples, 2.5)
ci_upper = percentile(combined_samples, 97.5)
coverage = 1 if true_value in [ci_lower, ci_upper] else 0
```

**Problem:** 
- Only ONE test (pooled across realizations)
- Interval is too wide (includes between-realization variance)
- Result: Always 100% coverage

### What You SHOULD Do (CORRECT)

```python
# Test each realization separately
coverages = []
widths = []
for realization in range(N_realizations):
    # Get samples for THIS realization only
    samples = mcmc_samples[realization]
    
    # Compute interval for THIS dataset
    ci_lower = percentile(samples, 2.5)
    ci_upper = percentile(samples, 97.5)
    
    # Test coverage for THIS dataset
    coverage = 1 if true_value in [ci_lower, ci_upper] else 0
    width = ci_upper - ci_lower
    
    coverages.append(coverage)
    widths.append(width)

# Average across independent experiments
empirical_coverage = mean(coverages)  # Should be ~0.95
mean_width = mean(widths)
```

**This tests:** "For a single patient's data, are my intervals reliable?"

## What Makes MCMC Valuable for Clinicians?

### NOT Valuable:
❌ "We ran 100 simulations and pooled them"
❌ "This interval covers all possible noise"

### VALUABLE:
✅ "Given THIS patient's data, we're 95% confident the value is in [a, b]"
✅ "The narrower the interval, the more certain we are"
✅ "These intervals have been validated to be trustworthy 95% of the time"

## Practical Assessment Strategy

### Step 1: Convergence Diagnostics (You Have This!)
- R-hat < 1.05 ✅
- ESS > 100 ✅
- Stable across initializations ✅

**Status:** EXCELLENT (R-hat ≈ 1.0, ESS > 9,000)

### Step 2: Posterior Predictive Checks

Check if MCMC posterior can generate data similar to observed:
```python
# For each MCMC sample:
y_pred = U @ R_sample + noise
# Compare distribution of y_pred vs observed y
```

If posterior can reproduce the data → model is reasonable.

### Step 3: Simulation-Based Calibration (SBC)

Run 50-100 independent experiments:
- Each with different true spectrum
- Each with one noise realization
- Count coverage of 95% credible intervals
- Should be ~95%

### Step 4: Sensitivity Analysis

Test how intervals change with:
- Different SNR (lower SNR → wider intervals?) ✓
- Different prior strength (stronger prior → narrower intervals?) ✓
- Different initialization (same intervals?) ✓

## Recommended Fix for Your Code

### Modify `_calculate_multi_level_calibration_metrics`

Change from:
```python
# Pool all realizations (WRONG)
combined_posterior = diff_samples.flatten()
```

To:
```python
# Test each realization separately (RIGHT)
coverages = []
widths = []
for realization_idx in range(n_realizations):
    realization_samples = diff_samples[realization_idx, :]
    ci_lower = np.percentile(realization_samples, lower_pct)
    ci_upper = np.percentile(realization_samples, upper_pct)
    coverage = 1.0 if (true_val >= ci_lower and true_val <= ci_upper) else 0.0
    width = ci_upper - ci_lower
    coverages.append(coverage)
    widths.append(width)

# Report averages across realizations
empirical_coverage = np.mean(coverages) * 100
mean_width = np.mean(widths)
std_width = np.std(widths)
```

## Bottom Line

### For Clinical Use, You Need:

1. ✅ **Convergence** (have it: R-hat ≈ 1.0)
2. ✅ **Mixing** (have it: ESS > 9,000)
3. ⚠️  **Calibration** (need to fix: test per-realization)
4. 🔄 **Validation** (need to add: SBC with many experiments)

### The Key Insight

MCMC gives you uncertainty for **a specific dataset**.
To validate this uncertainty is trustworthy, test on **many datasets**.

Your current pooling approach conflates:
- Uncertainty within one dataset (what you want)
- Variability across datasets (different question)

Fix: Test each dataset separately, then average the coverage!

