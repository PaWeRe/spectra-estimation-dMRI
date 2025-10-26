# BWH Dataset Workflow

## Quick Test on Subset (5 samples)

```bash
uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  dataset.max_samples=5 \
  inference=nuts \
  inference.n_iter=5000 \
  inference.tune=500 \
  prior=ridge \
  prior.strength=0.5 \
  local=true
```

**Expected:** ~25 minutes, 5 .nc files in `results/inference/`

---

## Full BWH Analysis (149 ROIs)

```bash
uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  inference=nuts \
  inference.n_iter=10000 \
  inference.tune=1000 \
  prior=ridge \
  prior.strength=0.5 \
  local=false
```

**Expected:** ~12-15 hours

---

## Configuration

**Discretization:** `geometric_8bins`
- D = [0.25, 0.75, 1.0, 1.5, 2.25, 2.5, 3.0, 20.0] μm²/ms
- Formula: S(D) = exp(-b_max*D), geometric spacing in log(S)
- Condition number: κ = 3.46e5 (optimized for NUTS)

**Subset Control:**
- Edit `configs/dataset/bwh.yaml`: set `max_samples: 5` for testing
- Set `max_samples: null` for full dataset

---

## Outputs

**Posterior Samples:** `results/inference/*.nc`
- Each file contains full posterior: R ~ p(R|data)
- Load with: `arviz.from_netcdf(path)`

**Metadata:** Already loaded in each SignalDecay
- `is_tumor`: bool (parsed from anatomical_region)
- `ggg`: Gleason Grade Group (0-5)
- `gs`: Gleason Score string
- `a_region`: "pz" or "tz"

---

## Next: Uncertainty Propagation for Classification

Your existing biomarker pipeline already handles posterior samples!

**Enable biomarker analysis:**
```bash
uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  dataset.max_samples=10 \
  inference=nuts \
  biomarker_analysis.enabled=true \
  classifier.name=logistic \
  local=true
```

The pipeline will:
1. Extract spectral features from posterior samples
2. Train classifier (with uncertainty)
3. Generate ROC curves
4. Compare to ADC baseline

**See:** `src/spectra_estimation_dmri/biomarkers/` for implementation details

---

## Monitor Progress

```bash
# Count completed ROIs
ls -1 results/inference/*.nc | wc -l

# View latest log
tail -f outputs/$(ls -t outputs/ | head -1)/main.log

# Check convergence
grep "R-hat" outputs/$(ls -t outputs/ | head -1)/main.log
```

