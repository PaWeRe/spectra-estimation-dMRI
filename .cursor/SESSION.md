# Session State — MRM Paper Collaboration

> **READ THIS FIRST** when starting a new session.
> This file is the handoff document between collaborative sessions.
> Updated: 2026-02-21 (Session 2)

---

## Quick Context

We are writing an MRM journal paper on **Bayesian spectral decomposition of multi-b diffusion MRI for prostate cancer characterization**. The human (Patrick) provides high-level direction; the AI handles all implementation. See `paper/PAPER_PLAN.md` for full plan.

**Branch:** `paper/mrm-manuscript`
**Key directories:** `paper/` (LaTeX), `scripts/` (analysis), `results/` (outputs), `8640-sl6-bin/` (pixel data, gitignored)

---

## Session 2 Summary (2026-02-21)

### What we accomplished:
1. **Deep paper analysis**: Read all three reference papers in `assets/`:
   - `Abstract #02290.pdf` — Our ISMRM 2024 abstract (5 figures, the building block for MRM)
   - `ISMRM-2022-abstract.pdf` — Wells/Maier/Westin 2022 (original Gibbs sampling approach)
   - `Evaluation of Fitting Models...pdf` — Langkilde et al. 2018 (definitive dataset description)

2. **Resolved b-value question**: Langkilde et al. is the definitive source:
   - **15 b-values**: 0, 250, 500, 750, ..., 3500 s/mm² (uniform 250 s/mm² steps)
   - In ms/μm²: [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]
   - **3 orthogonal diffusion encoding directions** per b-value
   - 15 × 3 = 45 DW images + 1 extra b=0 reference = **46 files** in `8640-sl6-bin/`

3. **Identified data grouping bug**: Current `group_images_b0_plus_directions` creates 16 groups
   (1 singleton + 15 triplets), incorrectly treating 3 DW-protocol b=0 images as a separate b-value.
   Then `np.linspace(0, 3.5, 16)` assigns wrong b-values. **Fix needed.**

4. **Identified parameter mismatches** between ISMRM paper and pixel scripts:
   - Ridge λ: paper uses **0.1**, pixel scripts use **0.5** (5× too strong)
   - Inference: paper uses NUTS (joint R, σ), pixel scripts use closed-form ridge
   - Full LR underperforms ADC in current ISMRM ROC plots — needs investigation

5. **Reviewed all ISMRM figures** (in `results/biomarkers/ismrm/`):
   - Fig 1: `combined_prostate_signal_snr_spectrum_ismrm.png` — 4-panel overview
   - Fig 2: `averaged_spectra_ismrm.png` — Boxplots per tissue type (good)
   - Fig 3: `roc_combined_ismrm.png` — ROC curves (Full LR < ADC for PZ/TZ, needs investigation)
   - Fig 4: `combined_uncertainty_ismrm.png` — Uncertainty propagation (strong: 2.34× ratio PZ)
   - Fig 5: `loocv_feature_importance_ismrm.png` — LR coefficients (good)

6. **Prioritization decisions**:
   - De-prioritized: Gmail/email integration, co-researcher agent (scope creep for now)
   - Focus: Get core pipeline correct → heatmap figure → compare ADC vs spectral biomarkers
   - All questions go through Patrick, not Stephan directly

### Correct data handling for 8640-sl6-bin (46 files):
The pixel data analysis should use the **45 DW-protocol images** (skip file 6, the extra b=0):
- Sort remaining 45 files by mean intensity (descending)
- Group into 15 b-value levels × 3 directions
- Average 3 directions per b-value → 15 trace images
- Assign known b-values: [0, 0.25, 0.5, ..., 3.5] ms/μm²

OR: keep all 46, detect the 4 b=0 images by intensity clustering, average all 4 as b=0,
then 14 non-zero × 3 dirs → 14 traces. Total: 15 trace images.
**Either way: 15 trace images at 15 known b-values.**

### Correct model parameters (from ISMRM abstract):
- **Diffusivity bins**: D ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0} μm²/ms (8 bins)
- **Ridge prior**: λ = 0.1 (NOT 0.5)
- **HalfCauchy noise prior**: β = 1.0 (for joint σ estimation)
- **NUTS**: 2000 iterations, 200 warmup
- **ADC baseline**: monoexponential, b = 0–1250 s/mm²
- **LR classifier**: L2-regularized (C=1.0), LOOCV, top-5 feature selection

### Correct data parameters (from Langkilde 2018):
- **Matrix**: 64 × 64 native (not 256×256)
- **FOV**: 280 × 280 mm → 4.375 mm pixel size
- **Slice thickness**: 5 mm, no gap
- **TE**: ~100 ms (δ=37ms, Δ=47ms, diffusion time=35ms)
- **Endorectal coil** (high but nonuniform SNR)
- **SNR at b=0**: Normal PZ ~180, Normal TZ ~119, Tumor PZ ~108, Tumor TZ ~74

### Results folder cleanup plan:
**DELETE** (worthless/empty):
- `results/robustness_test/` — plots not useful for MRM
- `results/uncertainty_propagation/` — empty
- `results/comparison/` — not helpful
- `results/direction_comparison/` — topic important but these plots aren't useful
- `results/flagship_figure/` — needs complete rework
- `results/pixel_heatmaps/` — redundant (note: plural)

**KEEP**:
- `results/inference/` — .nc files from NUTS (expensive to recompute)
- `results/inference_bwh_backup/` — 149 .nc files, keep!
- `results/inference_bwh_temp/` — 149 .nc files, keep!
- `results/inference_sim_backup/` — simulation .nc files, keep!
- `results/biomarkers/` — ROI-level analysis + ISMRM figures (our baseline)
- `results/plots/` — diagnostic plots (some useful)
- `results/pixel_exploration/` — data exploration (interesting but review)
- `results/pixel_heatmap/` — has some good plots but needs curation

### Architecture decision:
- Do NOT create new standalone scripts for pixel analysis
- Integrate pixel-wise pipeline into existing `main.py` and src modules
- Move `8640-sl6-bin/` data to `src/spectra_estimation_dmri/data/` folder
- All figure generation should flow through existing pipeline

### What's next (Session 3 priorities):
1. **Fix data loader**: Correct b-value grouping (15 values, handle extra b=0)
2. **Move data**: `8640-sl6-bin/` → `src/spectra_estimation_dmri/data/`
3. **Clean results**: Delete flagged folders
4. **Fix pipeline parameters**: λ=0.1, correct b-values in design matrix
5. **Investigate Full LR < ADC**: Check feature selection, regularization, overfitting
6. **Design heatmap figure**: Think before coding. What story does it tell?
   - Consider: what derivation of estimated spectra best compares to ADC?
   - Options: D=0.25 fraction, restricted/free ratio, full LR probability, entropy
7. **Integrate into main.py**: Add pixel-wise mode to existing pipeline

### ROC investigation notes:
Current ISMRM ROC results show Full LR underperforming ADC and D 0.25:
- PZ: ADC=0.95 > Full LR=0.93 > D 0.25=0.88
- TZ: ADC=0.98 > Full LR=0.92 > D 0.25=0.91
- GGG: D 0.25=0.81 > ADC=0.80 > Full LR=0.77
Possible causes: overfitting (8 features, small n), regularization too weak/strong,
combo feature unstable. The ISMRM abstract reported higher numbers — check if
the feature selection (top-5 vs all-8) or LOOCV implementation differs.

### Evolution from 2022 to 2024 ISMRM:
- 2022: D = [0.0, 0.25, 0.5, ..., 3.0] (includes D=0.0, no D=20.0), Gibbs sampling
- 2024: D = [0.25, 0.5, ..., 3.0, 20.0] (dropped D=0.0, added D=20.0 for IVIM), NUTS
- This evolution is deliberate: D=0.0 is physically meaningless; D=20.0 captures intravascular

### Key file map:
| File | Purpose |
|------|---------|
| `src/spectra_estimation_dmri/main.py` | Main Hydra pipeline (data → inference → biomarkers) |
| `src/spectra_estimation_dmri/models/prob_model.py` | Probabilistic model (U matrix, MAP, posterior) |
| `src/spectra_estimation_dmri/inference/nuts.py` | NUTS sampler |
| `src/spectra_estimation_dmri/inference/map.py` | MAP inference |
| `src/spectra_estimation_dmri/biomarkers/features.py` | Feature extraction from spectra |
| `src/spectra_estimation_dmri/biomarkers/pipeline.py` | Full biomarker analysis pipeline |
| `src/spectra_estimation_dmri/data/loaders.py` | Data loading (binary images, BWH, simulation) |
| `src/spectra_estimation_dmri/visualization/ismrm_exports.py` | ISMRM figure generation |
| `results/biomarkers/ismrm/` | Current ISMRM figures (5 figures) |
| `results/inference_bwh_backup/` | 149 NUTS .nc files (expensive, keep!) |
| `configs/` | Hydra configs for all parameters |
| `assets/Abstract #02290.pdf` | Our ISMRM 2024 abstract |
| `assets/Evaluation of Fitting Models...pdf` | Langkilde 2018 (dataset description) |
| `assets/ISMRM-2022-abstract.pdf` | Wells/Maier/Westin 2022 (original method) |
| `8640-sl6-bin/` | 46 binary pixel images (to be moved to src/data/) |

### Intensity analysis of 46 files (for data loader fix):
File 6: intensity=972 (the extra b=0, outlier — skip or handle separately)
Files by intensity rank (after file 6):
- Rank 1-3: ~617-632 (b=0 from DW protocol, 3 "directions")
- Rank 4-6: ~447-473 (b=250)
- Rank 7-9: ~349-358 (b=500)
- ... (groups of 3 with decreasing intensity)
- Rank 43-45: ~119-130 (b=3500)

---

### Session 3 TODO list:
1. **Investigate file 6 mystery** (priority — blocks correct pipeline):
   - File 6 is the FIRST file in sequence (files: 6, 15, 24, ..., 411, step=9)
   - 35% brighter than next group (972 vs 617-632)
   - Hypotheses to test:
     a) GE scanner calibration/reference b=0 (common: first volume is a ref scan)
     b) b=0 from a different clinical DWI sequence (but those are 96×96, not 64×64)
     c) Different TE (clinical b=0 at shorter TE leaked into same series)
     d) It IS b=0 from the same protocol but without diffusion preparation pulses
   - Langkilde mentions "16 averages" for clinical DWI — probably not related (different matrix)
   - Exploration ideas:
     - Compare spatial pattern of file 6 vs files 141/276/411 (if same anatomy = same b-value)
     - Check if file 6 has different intensity DISTRIBUTION (histogram) vs the triplet
     - Compute pixel-wise correlation between file 6 and the triplet average
     - If file 6 correlates perfectly with triplet but is just scaled → same b, different gain/TE
     - If file 6 shows different contrast → different b or sequence
   - Patrick may also ask Stephan for clarification
2. **Fix data loader** with correct b-value grouping (15 values)
3. **Move 8640-sl6-bin/** → `src/spectra_estimation_dmri/data/`
4. **Clean results folders** (delete 6 flagged folders)
5. **Fix ridge λ** from 0.5 → 0.1 in pixel pipeline
6. **Investigate Full LR < ADC** in ROC plots
7. **Design heatmap figure** (think first, code second)
8. **Integrate pixel pipeline** into existing main.py / src modules

---

## How to Start Session 3

1. Read this file first
2. Check `git log --oneline -5` for latest commits
3. Start with TODO #1 (file 6 investigation) — quick empirical tests
4. Then fix data loader and clean up
5. Ask Patrick: "Ready for Session 3? Here's where we left off. What's the priority?"
