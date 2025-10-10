# MCMC Sampler Comparison Experiments

## Overview

This document outlines the systematic comparison between Gibbs and NUTS samplers for diffusivity spectrum estimation. The goal is to determine the optimal sampler configuration before applying to real BWH prostate MRI data for Gleason score classification.

## Experimental Design

### Parameters to Vary

1. **SNR values**: [100, 300, 500, 1000]
2. **Spectrum configurations**: [debug_3buckets, ttz_newdiff1, tpz_default]
3. **Iteration counts**: [5000, 10000, 50000]
4. **Noise realizations**: 15 (for uncertainty calibration)
5. **Samplers**: [gibbs, nuts]

### Total Experiments

- **Convergence Study**: 2 samplers × 4 SNRs × 3 iteration counts × 5 realizations = 120 runs
- **Full Comparison**: 2 samplers × 4 SNRs × 3 spectra × 15 realizations = 360 runs (using optimal iterations from convergence study)
- **Grand Total**: ~480 runs

## Phase 1: Convergence Study

**Goal**: Determine minimum iterations needed for convergence at each SNR level.

**Fixed Parameters**:
- Spectrum: `debug_3buckets` (simplest case - 3 well-separated peaks)
- Noise realizations: 5
- Prior: Ridge with strength 0.01
- Chains: 4 (for convergence diagnostics)

**Variable Parameters**:
- SNR: [100, 300, 500, 1000]
- Iterations: [5000, 10000, 50000]
- Sampler: [gibbs, nuts]

### Command

```bash
# Note: Both gibbs.yaml and nuts.yaml already default to n_chains=4
# No need to override it in the sweep command
uv run python src/spectra_estimation_dmri/main.py -m \
  dataset=simulated \
  dataset.spectrum_pair=debug_3buckets \
  dataset.snr=100,300,500,1000 \
  dataset.noise_realizations=5 \
  inference=gibbs,nuts \
  inference.n_iter=5000,10000,50000 \
  prior=ridge \
  prior.strength=0.01 \
  local=false
```

**Total runs**: 2 samplers × 4 SNRs × 3 iterations × 5 realizations = **120 runs**

### Expected Outputs

- W&B dashboard with R-hat and ESS metrics
- `results/comparison/sampler_metrics.csv` with all metrics
- Convergence diagnostic plots in `results/plots/plot/`

### Analysis Steps

After completion, analyze results to determine:
1. Minimum iterations for R-hat < 1.01 at each SNR
2. Minimum iterations for ESS > 400 at each SNR
3. Whether Gibbs or NUTS converges faster
4. Recommended iteration counts for Phase 2

**Example Analysis Command** (to be created):
```bash
uv run python scripts/analyze_convergence_study.py \
  --input results/comparison/sampler_metrics.csv \
  --output results/comparison/convergence/
```

---

## Phase 2: Spectrum Complexity Study

**Goal**: Compare sampler performance across different spectrum configurations.

**Fixed Parameters**:
- Iterations: <optimal_from_phase1> (e.g., 10000 if that's sufficient)
- Noise realizations: 15
- Prior: Ridge with strength 0.01
- Chains: 4

**Variable Parameters**:
- SNR: [100, 300, 500, 1000]
- Spectrum: [debug_3buckets, ttz_newdiff1, tpz_default]
- Sampler: [gibbs, nuts]

### Command Template

**Replace `<OPTIMAL_ITER>` with value from Phase 1** (e.g., 10000):

```bash
# Note: Both configs default to n_chains=4, no need to override
uv run python src/spectra_estimation_dmri/main.py -m \
  dataset=simulated \
  dataset.spectrum_pair=debug_3buckets,ttz_newdiff1,tpz_default \
  dataset.snr=100,300,500,1000 \
  dataset.noise_realizations=15 \
  inference=gibbs,nuts \
  inference.n_iter=<OPTIMAL_ITER> \
  prior=ridge \
  prior.strength=0.01 \
  local=false
```

**Total runs**: 2 samplers × 4 SNRs × 3 spectra × 15 realizations = **360 runs**

### Expected Outputs

- Comprehensive metrics for all configurations
- Uncertainty calibration plots showing coverage vs. width
- ArviZ diagnostic summaries

---

## Phase 3: Analysis and Visualization

After all experiments complete, generate comparison plots and summaries.

### 3.1 Generate Comparison Plots

```bash
# To be implemented
uv run python scripts/analyze_sampler_comparison.py \
  --csv results/comparison/sampler_metrics.csv \
  --output results/comparison/
```

**Generated outputs**:
- `results/comparison/convergence/rhat_vs_iterations.pdf`
- `results/comparison/convergence/ess_vs_iterations.pdf`
- `results/comparison/accuracy/reconstruction_error_by_snr.pdf`
- `results/comparison/uncertainty/coverage_by_config.pdf`
- `results/comparison/efficiency/time_vs_accuracy.pdf`
- `results/comparison/summary/comparison_metrics.csv`

### 3.2 Create Interactive Dashboard

```bash
# To be implemented  
uv run python scripts/create_comparison_dashboard.py \
  --csv results/comparison/sampler_metrics.csv \
  --output results/comparison/summary/dashboard.html
```

---

## Metrics Tracked

All metrics are automatically extracted and logged for each run:

### Convergence
- Max R-hat across all parameters
- Min ESS (bulk and tail) across all parameters
- Convergence status (converged/marginal/not_converged)

### Accuracy
- L2 error (RMSE) vs. true spectrum
- L1 error (MAE) vs. true spectrum
- Max absolute error

### Uncertainty Calibration
- Mean credible interval width
- Interval sharpness (width / reconstruction error)
- **Coverage** (requires multiple noise realizations):
  - Does the 95% CI contain the true value 95% of the time?

### Efficiency
- Sampling time (seconds)
- Samples per second
- ESS per second (effective samples per unit time)

---

## Quick Test Run

Before running full experiments, test the pipeline with a single configuration:

```bash
uv run python src/spectra_estimation_dmri/main.py \
  dataset=simulated \
  dataset.spectrum_pair=debug_3buckets \
  dataset.snr=1000 \
  dataset.noise_realizations=1 \
  inference=gibbs \
  inference.n_iter=5000 \
  prior=ridge \
  prior.strength=0.01 \
  local=true
```

Check that:
1. Metrics are printed to console
2. CSV file is created: `results/comparison/sampler_metrics.csv`
3. Plots have config-specific names (not overwriting)

---

## Expected Timeline

- **Phase 1 (Convergence Study)**: ~2-4 hours (depending on hardware)
  - 120 runs × 1-2 min/run average
- **Phase 2 (Full Comparison)**: ~6-12 hours
  - 360 runs × 1-2 min/run average
- **Phase 3 (Analysis)**: ~1-2 hours
  - Script development and plot generation

**Total**: ~1-2 days of compute time

---

## Notes

1. **W&B Tags**: All runs are automatically tagged with:
   - `inference_{gibbs|nuts}`
   - `prior_ridge`
   - `dataset_simulated`
   - `data_snr_{100|300|500|1000}`
   - `spectrum_{debug_3buckets|ttz_newdiff1|tpz_default}`

2. **Local vs W&B**: 
   - Use `local=false` for W&B logging (recommended for sweeps)
   - Use `local=true` for local-only testing
   - CSV metrics are always saved regardless of `local` setting

3. **Resuming Interrupted Sweeps**:
   - Hydra creates unique run directories
   - Metrics CSV appends (doesn't overwrite)
   - To resume, just re-run the command - Hydra will skip completed runs

4. **Monitoring Progress**:
   - Check W&B dashboard for real-time metrics
   - Monitor `results/comparison/sampler_metrics.csv` for completed runs
   - Check Hydra multirun logs in `multirun/YYYY-MM-DD/HH-MM-SS/`

---

## Decision Criteria

After analysis, select the best sampler based on:

1. **Primary**: Convergence reliability (R-hat < 1.01, ESS > 400)
2. **Secondary**: Reconstruction accuracy (low RMSE)
3. **Tertiary**: Efficiency (ESS per second)
4. **Quaternary**: Uncertainty calibration (95% coverage)

**Recommendation**: Choose the sampler that consistently converges across all SNR levels and spectra with the fewest iterations.

---

## Next Steps After Comparison

Once the optimal sampler is determined:

1. Run on real BWH data:
   ```bash
   uv run python src/spectra_estimation_dmri/main.py \
     dataset=bwh \
     inference=<best_sampler> \
     inference.n_iter=<optimal_iterations> \
     prior=ridge \
     prior.strength=<optimal_strength> \
     local=false
   ```

2. Proceed to biomarker analysis for Gleason score classification

3. Share results folder with supervisors:
   - `results/comparison/` - All comparison results
   - `SAMPLER_COMPARISON_RESULTS.md` - Summary report
   - W&B project link - Interactive dashboard

