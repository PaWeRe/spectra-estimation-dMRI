---
name: spectrum-estimation
description: Run Bayesian inference to estimate diffusivity spectra from multi-b-value dMRI signal decays. Use when the task involves running NUTS sampling, Gibbs sampling, or MAP estimation to solve the inverse Laplace transform problem s=UR+noise. Also use for convergence diagnostics (R-hat, ESS, trace plots) and synthetic validation experiments.
compatibility: Requires pymc>=5.6.1, arviz, numpy, scipy. Uses uv for execution.
---

# Spectrum Estimation

## When to use this skill
Use when the task involves:
- Running NUTS/Gibbs/MAP inference on signal decay data
- Convergence diagnostics (R-hat, ESS, trace plots)
- Synthetic data experiments (known ground truth)
- Benchmarking sampler performance across SNR scenarios
- Single-pixel or batch spectrum estimation

## The inverse problem
Given multi-b-value signal decay s, estimate diffusivity spectrum R:
- Forward model: s = U @ R + noise, where U[i,j] = exp(-b[i] * D[j])
- R >= 0 (non-negative spectrum fractions)
- σ is jointly inferred (fully Bayesian)
- Priors: R ~ HalfNormal(0, 1/√λ), σ ~ HalfCauchy(β)

## How to run inference

### Full pipeline (all 149 ROIs)
```bash
uv run python -m spectra_estimation_dmri.main dataset=bwh inference=nuts prior=ridge local=true recompute=false
```

### Key parameters
- `inference.n_iter`: MCMC draws (default: 2000)
- `inference.n_chains`: Number of chains (default: 4)
- `inference.tune`: Warmup steps (default: 1000)
- `inference.target_accept`: NUTS target acceptance (default: 0.95)
- `prior.strength`: Ridge λ (default: 0.1, σ_R = 1/√λ ≈ 3.16)
- `dataset.max_samples`: Limit ROIs for testing (null = all)

### Simulated data (for validation)
```bash
uv run python -m spectra_estimation_dmri.main dataset=simulated inference=nuts prior=ridge dataset.snr=500
```

## Code locations
- NUTS sampler: `src/spectra_estimation_dmri/inference/nuts.py` (NUTSSampler class)
- Probabilistic model: `src/spectra_estimation_dmri/models/prob_model.py`
- Data models: `src/spectra_estimation_dmri/data/data_models.py`
- Simulation: `src/spectra_estimation_dmri/simulation/simulate.py`

## Diffusivity grid (BWH)
D = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] μm²/ms
- D ≤ 1.0: restricted diffusion (tumor tissue, dense cellularity)
- 1.0 < D ≤ 2.0: normal tissue, benign cells
- D = 3.0: free water (lacunar spaces)
- D = 20.0: intravascular (perfusion/IVIM)

## B-value protocol
15 b-values: [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500] s/mm²
(stored in ms units in config: divide by 1000)

## Convergence criteria
- R-hat < 1.05: converged
- R-hat < 1.10: nearly converged
- ESS_bulk > 400: adequate mixing
- ESS_tail > 400: reliable tail estimates

## Output format
Each ROI produces a `DiffusivitySpectrum` object saved as NetCDF (.nc):
- Posterior samples: shape (n_chains × n_draws, n_diffusivities)
- Inferred SNR: 1/σ_posterior
- Stored in `results/inference/` with spectra_id filename

## Speed considerations for pixel-wise
- Single ROI (15 b-values, 8 diffusivities): ~2-5 sec with 4 chains × 2000 draws
- For pixel-wise mapping: reduce to 1 chain × 500 draws, or use MAP init + short NUTS
- Native 64x64 grid → ~4096 pixels × ~1 sec = ~1 hour feasible
