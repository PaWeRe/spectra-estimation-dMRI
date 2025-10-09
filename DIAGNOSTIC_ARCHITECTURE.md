# Diagnostic Architecture: Chains vs Realizations

## The Correct Structure

### Single Noise Realization
```
Data: y = signal + noise₁  (one instance of noise)

Multiple chains sampling the SAME posterior:
├── Chain 1: [45,000 samples from p(R|y)]
├── Chain 2: [45,000 samples from p(R|y)]  
├── Chain 3: [45,000 samples from p(R|y)]
└── Chain 4: [45,000 samples from p(R|y)]

Purpose: Assess convergence via R-hat, ESS
```

**Key:** All 4 chains should converge to the SAME distribution (the posterior given that specific noisy data).

**Metrics:**
- R-hat: Should be < 1.01 (chains have converged)
- ESS: Should be > 400 per chain (good mixing)
- Trace plots: Should look stationary and well-mixed

### Multiple Noise Realizations
```
Realization 1: y₁ = signal + noise₁
├── Chain 1-1: [45,000 samples from p(R|y₁)]
├── Chain 1-2: [45,000 samples from p(R|y₁)]
├── Chain 1-3: [45,000 samples from p(R|y₁)]
└── Chain 1-4: [45,000 samples from p(R|y₁)]
→ R-hat₁, ESS₁ (convergence for realization 1)
→ CI₁ = [2.5%, 97.5%] percentiles of combined samples
→ Coverage₁ = 1 if true_value ∈ CI₁, else 0

Realization 2: y₂ = signal + noise₂
├── Chain 2-1: [45,000 samples from p(R|y₂)]
├── Chain 2-2: [45,000 samples from p(R|y₂)]
├── Chain 2-3: [45,000 samples from p(R|y₂)]
└── Chain 2-4: [45,000 samples from p(R|y₂)]
→ R-hat₂, ESS₂ (convergence for realization 2)
→ CI₂ = [2.5%, 97.5%] percentiles of combined samples
→ Coverage₂ = 1 if true_value ∈ CI₂, else 0

...

Realization 20: y₂₀ = signal + noise₂₀
└── [4 chains, R-hat₂₀, ESS₂₀, CI₂₀, Coverage₂₀]

Purpose: Assess calibration across different noise instances
```

**Metrics:**
- **Per-realization convergence**: R-hat₁, R-hat₂, ..., R-hat₂₀ (should ALL be < 1.01)
- **Overall calibration**: Coverage = mean(Coverage₁, ..., Coverage₂₀) ≈ 95% for 95% CI

## Current Data Structure (WRONG)

```python
# In main.py, for noise_realizations=20:
for realization_idx in range(20):
    # Generate different noisy data
    noisy_signal = signal + noise[realization_idx]
    
    # Run inference (currently only 1 chain in old Gibbs)
    spectrum = sampler.run(noisy_signal)  # Returns 1 chain worth of samples
    
    spectra_list.append(spectrum)

# spectra_list has 20 elements
# Each element has samples from a DIFFERENT posterior (different data!)

# WRONG DIAGNOSTIC (what current code does):
idata = create_inference_data(spectra_list)  
# Treats 20 realizations as "20 chains" of same posterior
# R-hat = 1.47 because they're different posteriors!
```

## New Data Structure (CORRECT)

```python
# In main.py, for noise_realizations=20:
all_realizations = []

for realization_idx in range(20):
    # Generate different noisy data
    noisy_signal = signal + noise[realization_idx]
    
    # Run inference with 4 chains (NEW!)
    spectrum = sampler.run(noisy_signal, n_chains=4)
    # spectrum.inference_data is ArviZ InferenceData with 4 chains
    # spectrum.spectrum_samples is (4*45000, n_dim) flattened for plotting
    
    all_realizations.append(spectrum)

# all_realizations has 20 elements
# Each element represents one noise realization with its own 4-chain posterior

# CORRECT DIAGNOSTICS:

# 1. Convergence (per-realization):
for realization in all_realizations:
    idata = load_idata(realization.inference_data)  # Has 4 chains for this realization
    rhat = az.rhat(idata)  # Should be < 1.01
    ess = az.ess(idata)    # Should be > 400
    print(f"Realization {i}: R-hat={rhat.max():.3f}, ESS={ess.min():.0f}")

# 2. Calibration (across realizations):
coverages = []
for realization in all_realizations:
    # Use combined samples from all 4 chains
    samples = realization.spectrum_samples  # Already flattened (4*45000, n_dim)
    ci_lower = np.percentile(samples[:, d_idx], 2.5)
    ci_upper = np.percentile(samples[:, d_idx], 97.5)
    coverage = 1 if (true_val >= ci_lower and true_val <= ci_upper) else 0
    coverages.append(coverage)

empirical_coverage = np.mean(coverages)  # Should be ~95% for 95% CI
```

## What Needs to Change

### 1. ✅ Gibbs Sampler (DONE!)
- Now runs `n_chains=4` by default
- Creates proper ArviZ InferenceData with (n_chains, n_iterations, n_dim)
- Prints R-hat and ESS per realization
- Stores flattened samples for compatibility

### 2. ❌ Diagnostic Functions (TODO!)

**`_save_arviz_summary`** (line 2292):
- Currently: Pools all realizations as "chains"
- Should: Load idata PER realization, compute diagnostics, aggregate

**`_create_inference_data`** (line 2363):
- Currently: Stacks realizations as chains
- Should: Not needed! Each realization already has its own idata

**`_plot_multichain_trace`** (line 2156):
- Currently: Treats realizations as chains
- Should: Plot chains WITHIN a single realization

**`_plot_rank`** (line 2237):
- Currently: Treats realizations as chains
- Should: Rank plot for chains WITHIN a single realization

**`_calculate_multi_level_calibration_metrics`** (line 1602):
- Currently: ✓ CORRECT! Already loops over realizations separately
- Keep as-is

### 3. ❌ Main Loop (TODO!)
- Need to ensure noise_realizations creates different data
- Each realization runs with n_chains=4

## Number of Chains: Recommendation

**Standard: 4 chains** (used by PyMC, Stan, etc.)

Why 4?
- Enough to detect convergence issues (R-hat needs ≥2)
- Not too expensive computationally
- Industry standard

**For well-conditioned problems (κ < 10):**
- 4 chains × 50,000 iterations = plenty
- Can even reduce to 4 × 10,000 if fast convergence

**For ill-conditioned problems (κ > 100,000):**
- Still use 4 chains
- May need more iterations (100,000+)
- Or stronger prior to help convergence

## Summary

### For Convergence Diagnostics:
- Use: 1 realization, 4 chains
- Metric: R-hat < 1.01, ESS > 400
- Purpose: "Did my sampler converge for THIS dataset?"

### For Calibration:
- Use: 20+ realizations, each with 4 chains (for robustness)
- Metric: 95% CI → ~95% coverage across realizations
- Purpose: "Are my credible intervals correctly calibrated?"

### Complete Test Run:
```bash
# Convergence test (1 realization, 4 chains, many iterations)
dataset.noise_realizations=1 \
inference.n_chains=4 \
inference.n_iter=100000

# Calibration test (20 realizations, 4 chains each)
dataset.noise_realizations=20 \
inference.n_chains=4 \
inference.n_iter=50000
```

Total samples: 20 realizations × 4 chains × 45,000 samples = 3.6M samples
- For convergence: Check each realization's R-hat
- For calibration: Check coverage across realizations

