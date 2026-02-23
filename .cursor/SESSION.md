# Session State — MRM Paper Collaboration

> **READ THIS FIRST** when starting a new session.
> Updated: 2026-02-22 (Session 4)

---

## Quick Context

We are writing an MRM journal paper on **Bayesian spectral decomposition of multi-b diffusion MRI for prostate cancer characterization**. The human (Patrick) provides high-level direction; the AI handles implementation. See `paper/PAPER_PLAN.md` for full plan.

**Branch:** `paper/mrm-manuscript`
**Key directories:** `paper/` (LaTeX), `src/spectra_estimation_dmri/` (Python package), `results/` (outputs)
**Patient pixel data:** `src/spectra_estimation_dmri/data/8640-sl6-bin/` (46 .bin files, gitignored)

---

## Session 4 Summary (2026-02-22)

### What we accomplished:

1. **Cleaned up agent/skills architecture**: Deleted `.cursor/skills/` and `.cursor/agents/` directories — they were unused and overcomplicated. Removed references from `.cursorrules`. SESSION.md is the only shared memory needed.

2. **Walked through existing pipeline end-to-end**: Documented the complete data flow:
   - Step 1: `load_bwh_signal_decays()` → 149 ROI-averaged `SignalDecay` objects
   - Step 2: `ProbabilisticModel` builds design matrix U (15×8)
   - Step 3: Per-sample inference loop → `MAPInference` or `NUTSSampler` → `DiffusivitySpectrum`
   - Step 4: Biomarker pipeline (LR, LOOCV, AUC)
   
   **Identified normalization bug**: NUTS normalizes signal by S0 before fitting (spectra in [0,1]). MAP does NOT (spectra in raw signal units). Inconsistent — fixed in pixelwise.py.

3. **Researched prostate segmentation**: Neural nets (TotalSegmentator, Prostate158, nnU-Net) won't work for our data — they need T2w and/or 3D volumes. We have single 2D DWI slice at 64×64. Decision: **manual segmentation in 3D Slicer**.

4. **Created manual prostate mask**: Patrick drew it in 3D Slicer → `results/prostate_mask.nii` (146 pixels, binary, 64×64×1). Exported b=0 as NIfTI for Slicer: `results/b0_trace_64x64.nii.gz`.

5. **Built `pixelwise.py`** — new module at `src/spectra_estimation_dmri/pixelwise.py`:
   - Pure functions, no config objects, no Pydantic wrappers
   - `build_design_matrix()` — U matrix
   - `compute_adc()` — vectorized ADC for all pixels (monoexponential fit)
   - `compute_map_spectra()` — closed-form Ridge NNLS: `R = (U'U + λI)^{-1} U' S_norm`
   - `run_nuts_pixel()` / `run_nuts_all()` — NUTS with checkpointing
   - `train_tumor_lr()` — train LR on ROI data, apply to pixels
   - `compute_tumor_probability()` / `compute_tumor_probability_with_uncertainty()`
   - `assemble_map()` — scatter pixel values back into 2D image
   - All functions normalize by S0 consistently

6. **Computed all pixel-wise maps**:
   - **ADC**: 0.47–1.49 ×10⁻³ mm²/s (correct physiological range)
   - **MAP spectra**: all 8 diffusivity fractions, spectra sum ≈ 1.0 per pixel
   - **NUTS**: completed on all 146 pixels, all converged (R-hat = 1.000)
     - SNR range: 12–172 across pixels
     - Checkpoint-based, resumable
   - **Tumor probability**: trained LR on ROI data (PZ AUC=0.919, TZ AUC=0.945)

7. **Results files created**:
   - `results/prostate_mask.nii` — manual mask from 3D Slicer
   - `results/b0_trace_64x64.nii.gz` — b=0 NIfTI for Slicer
   - `results/pixelwise_all_fast.npz` — ADC + MAP spectra + tumor prob (fast methods)
   - `results/pixelwise/nuts_results.npz` — full NUTS results (spectrum mean/std, sigma, SNR)
   - `results/pixelwise/nuts_checkpoint.npz` — checkpoint (146/146 done)
   - `results/pixelwise_adc_map_preview.png` — ADC + MAP preview
   - `results/pixelwise_all_fast_overview.png` — all fast maps overview
   - `results/mask_comparison.png` — threshold mask comparison

### Key findings and decisions:

**LR coefficients tell the spectral story**:
- Positive (tumor): D=0.25 (+0.71), D=0.5 (+0.55), D=0.75 (+0.40)
- Near zero: D=1.0 (+0.07) — transition
- Negative (normal): D=1.5 (−0.36), D=2.0 (−0.44), D=3.0 (−0.55), D=20 (−0.76)
- This decomposition is what ADC cannot provide

**Tumor probability maps are saturated** — most pixels → P≈1.0. This is NOT a bug:
- Domain shift: LR trained on ROI averages from 40 patients, applied to pixels from 1 patient
- StandardScaler fitted on ROI feature distribution; pixel features land in a different range
- The logit scores all push to extremes
- **Fix options for next session**: (a) skip StandardScaler for pixel application, (b) recalibrate, (c) just show raw spectral fractions instead of P(tumor)
- Patrick asked: "why PZ and TZ models?" — because normal tissue looks different in the two zones, so separate classifiers are standard. For pixel-wise, we'd need zone segmentation to apply the right model.

**For the journal figure**: Show raw spectral fractions (not saturated P(tumor)) + NUTS uncertainty + NUTS SNR. The spectral decomposition itself is the contribution, not a pixel-level classifier.

### Correct parameters (confirmed, unchanged from Session 3):

| Parameter | Value | Source |
|-----------|-------|--------|
| b-values | [0, 250, ..., 3500] s/mm² (15 values) | Langkilde 2018 |
| Diffusivity bins | [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] μm²/ms | ISMRM 2024 |
| Ridge λ | 0.1 | ISMRM 2024 / configs/prior/ridge.yaml |
| HalfCauchy β | 0.01 | pixelwise.py (generic, no SNR estimate for pixels) |
| NUTS | 2000 draws, 200 tune, 4 chains, target_accept=0.95 | configs/inference/nuts.yaml |
| ADC | monoexponential, b ≤ 1250 s/mm² | ISMRM 2024 |
| Prostate mask | Manual, 146 pixels | 3D Slicer, Session 4 |

### Key file map (updated):

| File | Purpose |
|------|---------|
| `src/spectra_estimation_dmri/pixelwise.py` | **NEW**: Pixel-wise ADC, MAP, NUTS, tumor prob |
| `src/spectra_estimation_dmri/main.py` | Main Hydra pipeline (ROI-level) |
| `src/spectra_estimation_dmri/data/loaders.py` | Data loading (`load_prostate_dwi()` + `ProstateDWI`) |
| `src/spectra_estimation_dmri/models/prob_model.py` | Probabilistic model (U matrix, MAP, posterior) |
| `src/spectra_estimation_dmri/inference/nuts.py` | NUTS sampler (ROI-level, config-coupled) |
| `src/spectra_estimation_dmri/inference/map.py` | MAP inference (ROI-level) |
| `src/spectra_estimation_dmri/biomarkers/pipeline.py` | Full biomarker analysis |
| `src/spectra_estimation_dmri/biomarkers/features.py` | Feature extraction |
| `src/spectra_estimation_dmri/biomarkers/mc_classification.py` | LOOCV classification |
| `src/spectra_estimation_dmri/biomarkers/adc_baseline.py` | ADC baseline |
| `configs/dataset/bwh.yaml` | BWH dataset config |
| `configs/prior/ridge.yaml` | Ridge prior (λ=0.1) |
| `configs/inference/nuts.yaml` | NUTS config (2000 draws, 200 tune, 4 chains) |
| `results/prostate_mask.nii` | Manual prostate mask (146 pixels) |
| `results/pixelwise/nuts_results.npz` | NUTS pixel results (all 146 done) |
| `results/pixelwise_all_fast.npz` | ADC + MAP + tumor prob results |
| `results/inference_bwh_backup/` | 149 NUTS .nc files for ROI-level (keep!) |

### Architecture note (from Session 4 cleanup):

- Deleted `.cursor/skills/` (6 skill folders) and `.cursor/agents/` (4 agent configs)
- SESSION.md is the only cross-session memory
- Patrick expressed interest in specialized Cursor agents for plotting and pipeline running — tabled for now, wants to understand Cursor's actual capabilities first (docs needed)

### Patrick's preferences (carry across all sessions):

- **Methodology first**: Think about what to show before coding
- **Unified pipeline**: Pixel-level = generalization of ROI-level, same code path
- **Clinical utility focus**: Why is Bayesian better than ADC?
- **Uncertainty is key**: This is our main differentiator
- **uv only**: Never use pip, always `uv run python`, `uv add`, `uv sync`
- **Edit existing files**: Don't create standalone scripts, integrate into `src/`
- **Max 10 figures**: Every MRM figure must earn its place
- **No direct co-author contact**: All questions through Patrick
- **Go step by step**: Don't rush ahead — walk through each decision
- **Simplify**: If current code is overcomplicated, say so and propose simpler alternatives
- **nibabel** is now installed (added in Session 4)

---

## Session 5 TODO (Priority Order)

### Phase 1: Build the journal-ready comparison figure
1. **Fix tumor probability saturation**: Either (a) drop StandardScaler for pixel application, (b) use Platt recalibration, or (c) just show raw spectral fractions. Discuss with Patrick.
2. **Design figure layout**: Patrick approved working on layout. Proposed panels:
   - Row 1: b=0 | ADC | NUTS D=0.25 mean | NUTS D=0.25 uncertainty
   - Row 2: Selected spectral fractions (D=0.5, D=1.0, D=3.0, D=20) or all 8
   - Row 3: NUTS SNR map | maybe tumor probability (if fixed) | LR coefficients bar chart
3. **Generate publication-quality figure**: 300 DPI, PDF, consistent colormaps, proper labels
4. **All data is ready**: ADC, MAP spectra, NUTS spectra+uncertainty+SNR all computed

### Phase 2: Investigate Full LR < ADC anomaly
5. **Debug why full LR (8 features) underperforms ADC**: Likely overfitting with 8 correlated features on small N. Check regularization, feature selection, try fewer features.
6. **Consider**: D=0.25 alone (AUC=0.88 PZ) vs ADC (AUC=0.95 PZ) — why does ADC still win for tumor detection even though D=0.25 carries more specific information?

### Phase 3: Refactor toward unified pipeline
7. **Long-term**: Refactor `main.py` and inference classes to use `pixelwise.py` patterns (pure functions, consistent normalization, no config coupling). Patrick expressed interest in this.

### Phase 4: Paper writing
8. **Draft methods section** describing pixel-wise extension
9. **Draft results section** with figure

---

## How to Start Session 5

1. Read this file
2. Run `git log --oneline -5` for latest commits
3. All pixel-wise results are already computed — load from:
   - `results/pixelwise_all_fast.npz` (ADC, MAP, tumor prob)
   - `results/pixelwise/nuts_results.npz` (NUTS spectra, uncertainty, SNR)
4. Start with figure layout design
5. Ask Patrick: "Ready for Session 5? Starting with the comparison figure. Sound right?"
