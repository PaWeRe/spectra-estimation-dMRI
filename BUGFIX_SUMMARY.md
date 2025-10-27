# Bug Fixes - BWH Workflow

## Date: 2025-10-27

### Issues Fixed

#### 1. **BWH Visualization - Region Parsing Error**

**Problem:**
- `SignalDecay.a_region` is typed as `Literal["pz", "tz", "sim"]`
- BWH loader stores tumor/normal info in separate `is_tumor` field
- Visualization code tried to parse tumor/normal from `a_region` string
- Result: All regions parsed as "unknown_pz" or "unknown_tz"

**Solution:**
- Modified `parse_anatomical_region()` to accept full SignalDecay object
- Extract zone from `a_region` field
- Extract tissue type from `is_tumor` boolean field
- Now correctly groups spectra into 4 categories: normal_pz, normal_tz, tumor_pz, tumor_tz

**Files Modified:**
- `src/spectra_estimation_dmri/visualization/bwh_plotting.py`

---

#### 2. **Biomarker Classification - Single Class Error**

**Problem:**
- GGG classification failed when all samples belong to same class
- With test data (5 samples, all GGG=3), only one class present: GGG >= 7
- Logistic regression requires at least 2 classes
- Error: `ValueError: This solver needs samples of at least 2 classes`

**Solution:**
- Added check in `evaluate_feature_set()` before LOOCV
- If only one class present, skip classification and return None
- Prevents crash, allows other classification tasks to proceed
- Warning message printed for transparency

**Files Modified:**
- `src/spectra_estimation_dmri/biomarkers/mc_classification.py`

---

#### 3. **NUTS/Gibbs - Signal Normalization Issue**

**Problem:**
- Signal values in raw MRI units (magnitude 1000-10000)
- NUTS/Gibbs samplers expected normalized signal (magnitude ~1)
- Prior σ_expected = 0.01 assumes normalized signal
- Result:
  - Inferred σ_posterior = 660-4000 (huge!)
  - SNR_posterior printed as 0.0 (should be ~100-500)
  - Numerical instability, poor prior matching

**Root Cause:**
- Old Gibbs code normalized by S_0: `signal_values_roi = signal / signal[0]`
- New NUTS/Gibbs implementations did not include normalization
- Signal loaded directly from JSON without preprocessing

**Solution:**
- Added signal normalization in both NUTS and Gibbs samplers:
  ```python
  S_0 = signal[0] if signal[0] > 0 else 1.0
  signal_normalized = signal / S_0
  ```
- Use `signal_normalized` for:
  - MAP initialization
  - Likelihood computation
  - Gibbs sampling residual calculation
- Print S_0 value for transparency
- Updated log messages to clarify "normalized signal"

**Expected Result:**
- σ_posterior should now be ~0.002-0.01 (similar to σ_expected)
- SNR_posterior should match SNR from data (~100-500)
- Better prior-data agreement
- Improved numerical stability

**Files Modified:**
- `src/spectra_estimation_dmri/inference/nuts.py`
- `src/spectra_estimation_dmri/inference/gibbs.py`

---

## Testing Recommendations

### 1. Run Small Test (5 samples)
```bash
uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  dataset.max_samples=5 \
  inference=nuts \
  inference.n_iter=2000 \
  inference.tune=200 \
  prior=ridge \
  prior.strength=0.1 \
  local=true
```

**Expected Output:**
- ✅ BWH visualization: Correctly grouped regions (not "unknown")
- ✅ NUTS diagnostics: 
  - σ_posterior ≈ 0.002-0.01
  - SNR_posterior ≈ 100-500 (not 0.0)
  - S_0 values printed (~1000-10000)
- ✅ Biomarker analysis: Gracefully skips GGG task (single class warning)
- ✅ Tumor vs Normal tasks: Should work if both classes present

### 2. Run Full Dataset
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

**Expected Output:**
- All regions should have samples
- GGG classification should work (sufficient class diversity)
- ROC curves and AUC tables generated
- Regional boxplots and statistics exported

---

## Notes

### Why S_0 Normalization?

1. **Numerical Stability**: 
   - Signal magnitude ~10000 → after normalization ~1
   - Avoids numerical precision issues
   - Matches typical ML preprocessing

2. **Prior Specification**:
   - σ ~ HalfCauchy(0.01) assumes normalized signal
   - Ridge penalty λ assumes unit-scale data
   - Without normalization, priors are mismatched

3. **Historical Consistency**:
   - Old Gibbs code normalized by S_0
   - Standard practice in dMRI analysis
   - Maintains compatibility with previous results

### Impact on Results

- **Spectra estimates**: Unchanged (algorithm-wise equivalent)
- **Convergence**: Improved (better numerical conditioning)
- **Diagnostics**: More interpretable (σ, SNR values make sense)
- **Biomarkers**: Unchanged (uses posterior samples directly)

---

#### 4. **Precomputed Results Loading - Missing spectrum_init**

**Problem:**
- Loading precomputed `.nc` files failed with Pydantic validation error
- `spectrum_init` field was commented out: `# spectrum_init=,`
- `DiffusivitySpectrum` model requires `spectrum_init` (not Optional)
- NUTS results were not recognized (only "map" and "gibbs" handled)

**Error:**
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for DiffusivitySpectrum
spectrum_init
  Field required [type=missing, ...]
```

**Additional Issue - Variable Naming:**
- NUTS saves variables as `diff_0.25`, `diff_0.50`, etc. (for diagnostics)
- Loading code expected single `R` variable (old format)
- KeyError when trying to load NUTS results

**Solution:**
- Added "nuts" alongside "gibbs" in loading logic
- Compute `spectrum_init` on-the-fly using MAP estimate
- Normalize signal before MAP (consistent with inference code)
- **Handle both variable formats:**
  - Old format: `R` variable (single matrix)
  - New format: `diff_X.XX` variables (NUTS output)
  - Automatically detect and reconstruct appropriately

**Files Modified:**
- `src/spectra_estimation_dmri/main.py`

---

## Bug #6: Empty Results Handling in Visualization

**Error:**
```
KeyError: 'AUC'
```

**Location:**
- `src/spectra_estimation_dmri/biomarkers/biomarker_viz.py` (line 141)

**Problem:**
- When all classifiers return `None` (e.g., GGG task with only one class present)
- `create_auc_table` creates empty DataFrame with no columns
- Attempting to sort by non-existent "AUC" column causes KeyError
- `plot_roc_curves` would create useless plot with only diagonal reference line

**Solution:**
- **`create_auc_table`**: Check if DataFrame is empty before sorting
  ```python
  if not df.empty and "AUC" in df.columns:
      df = df.sort_values("AUC", ascending=False)
  ```
- **`plot_roc_curves`**: Filter valid results upfront, skip plotting if none
- **`create_summary_report`**: Print informative message for empty results
- Now gracefully handles edge cases with insufficient data

**Files Modified:**
- `src/spectra_estimation_dmri/biomarkers/biomarker_viz.py`

---

## Related Files

### Core Changes
- `src/spectra_estimation_dmri/visualization/bwh_plotting.py` (Bug #1)
- `src/spectra_estimation_dmri/biomarkers/mc_classification.py` (Bug #2)
- `src/spectra_estimation_dmri/biomarkers/biomarker_viz.py` (Bug #6)
- `src/spectra_estimation_dmri/inference/nuts.py` (Bug #3)
- `src/spectra_estimation_dmri/inference/gibbs.py` (Bug #3)
- `src/spectra_estimation_dmri/main.py` (Bugs #4, #5)

### Unchanged (but relevant)
- `src/spectra_estimation_dmri/data/loaders.py` - Already correctly parses `is_tumor`
- `src/spectra_estimation_dmri/data/data_models.py` - SignalDecay structure correct
- `src/spectra_estimation_dmri/biomarkers/pipeline.py` - No changes needed

