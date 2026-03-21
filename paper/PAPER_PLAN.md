# MRM Paper Plan: Fully Bayesian Uncertainty-Aware Biomarkers for Image-Based Cancer Diagnosis: Application to
Prostate Diffusion MRI

## Status Legend
- [x] Done (code + results exist)
- [~] Partially done (code exists, needs refinement)
- [ ] Not started
- [BLOCKED] Needs external data/input

---

## Paper Structure & Section Plan

### 1. Abstract (write last)
- Structured: Purpose / Methods / Results / Conclusion
- Status: [ ] Write after all sections finalized

### 2. Introduction
- Status: [ ] Draft ready to write from existing literature
- Key points:
  - Clinical context: prostate cancer, PI-RADS, ADC limitations
  - Multi-b DWI and compartmental modeling (Langkilde, Conlin)
  - Gap: ADC is single-number summary, loses microstructural detail
  - Our contribution: fully Bayesian spectral analysis + uncertainty-aware classification
  - Novel pixel-wise mapping capability

### 3. Theory
- Status: [~] Math is well-defined in code, needs LaTeX formalization
- Sections:
  - 3.1 Signal model (Stejskal-Tanner, Laplace transform)
  - 3.2 Bayesian formulation (priors, posterior)
  - 3.3 NUTS sampler (why HMC, advantages over Gibbs/grid search)
  - 3.4 Uncertainty propagation (MC through logistic regression)

### 4. Methods
- Status: [~] Most exists in code, needs systematic writeup
- Sections:
  - 4.1 Patient data and ROI annotations
    - [BLOCKED: Patient demographics table from Stephan]
    - [BLOCKED: Additional GGG case from Stephan]
  - 4.2 MRI acquisition protocol (from Langkilde 2018 reference)
  - 4.3 Inference configuration (NUTS params, convergence criteria)
  - 4.4 Classification pipeline (LOOCV, feature selection, L2-regularized LR)
  - 4.5 ADC baseline computation

### 5. Results (10 figures max)
- 5.1 Spectral estimation validation
  - [x] Fig 1: Signal decay + NUTS synthetic validation (ISMRM Fig 1)
  - [ ] Fig 2: Sampling diagnostics - trace plots, convergence under various SNRs (NEW)
  - [ ] Fig 3: Robustness test - inverse spectra under various SNR (NEW)
- 5.2 Patient spectra
  - [x] Fig 4: Averaged spectra by tissue type (ISMRM Fig 2)
  - [ ] Fig 5: Direction independence - spectra consistency across gradient directions (NEW)
- 5.3 Classification performance
  - [x] Fig 6: ROC curves for PZ, TZ, GGG (ISMRM Fig 3)
  - [x] Fig 7: Uncertainty propagation (ISMRM Fig 4)
  - [x] Fig 8: Feature importance / LR coefficients (ISMRM Fig 5)
- 5.4 Pixel-wise mapping
  - [ ] Fig 9: Pixel-wise spectral component maps + heatmap (NEW, flagship figure)
  - [ ] Fig 10: Compartment-size heatmaps overlaid on anatomy (NEW)
- Tables:
  - [ ] Table 1: Patient demographics [BLOCKED: from Stephan]
  - [~] Table 2: Classification AUC comparison (exists in CSV)
  - [ ] Table 3: Convergence diagnostics summary

### 6. Discussion
- Status: [ ] Draft ready to write
- Key points:
  - Spectra vs ADC: competitive detection, richer information
  - Direction independence validates measurement reliability
  - Uncertainty calibration: misclassified cases have higher uncertainty
  - Pixel-wise maps: potential for whole-gland characterization
  - Limitations: sample size, endorectal coil, computation time
  - Future: neural network amortized inference, T2 mapping integration

### 7. Conclusion
- Status: [ ] Brief, write after discussion

---

## New Analyses Required (Stephan's Feedback)

### A. Direction Independence Analysis [DONE - Session 1]
- **What**: Compare spectra across 3 gradient directions for same pixels
- **Result**: D=0.25 (tumor marker) has best direction consistency; CV<5% at high SNR
- **Data**: Used 8640-sl6-bin/ data (confirmed 3 directions × 15 b-values + 1 b=0)
- **Code**: `scripts/direction_comparison.py`
- **Output**: `results/direction_comparison/`

### B. Sampling Diagnostics & Robustness [DONE (fast) - Session 1]
- **What**: NUTS recovery of 6 synthetic spectral shapes at multiple SNR
- **Result**: All shapes converge (R̂<1.05); RMSE decreases with SNR; inverse tumor recovered well
- **Fast pass done**: 2 SNR × 2 realizations × 2 chains. Full run pending (~30 min).
- **Code**: `scripts/robustness_test.py` (use `--fast` for quick test, no flag for paper quality)
- **Output**: `results/robustness_test/`

### C. Pixel-wise Prostate Mapping [MAP DONE, NUTS PENDING - Session 1]
- **What**: Per-pixel spectral decomposition + biomarker heatmap
- **Result (MAP)**: 1719 pixels processed in <1 sec, 8 component maps + biomarker heatmap
- **Pending**: 
  - Laplace approximation for per-pixel uncertainty (fast, ~0.01ms/pixel)
  - NUTS for select ROI pixels (gold standard uncertainty, ~8 sec/pixel)
  - LR probability map using trained classifier weights
  - ADC comparison map
  - Prostate segmentation overlay
- **Data**: 8640-sl6-bin/ (46 files = 1 b=0 + 15 b-vals × 3 dirs, native 64×64)
- **Code**: `scripts/pixel_wise_heatmap.py`
- **Output**: `results/pixel_heatmaps/`

### D. Patient Demographics Table [BLOCKED: Stephan]
- Need complete patient demographics
- Additional GGG case

---

## Task Priority (What I Can Do RIGHT NOW)

### Immediate (no blockers):
1. Write Theory section (all math is in the code)
2. Write Methods section (minus patient table details)
3. Write Introduction (from literature + abstract)
4. Build sampling diagnostics analysis (synthetic data)
5. Benchmark NUTS pixel speed
6. Start pixel-wise pipeline (data already in 8640-sl6-bin/)
7. Generate publication-quality versions of ISMRM figures
8. Write Discussion skeleton

### Blocked (need Stephan/external):
1. All-directions data → direction independence analysis
2. Patient demographics table
3. Additional GGG case
4. LaTeX installation on this machine

---

## Figure Budget (10 max)

| # | Content | Source | Status |
|---|---------|--------|--------|
| 1 | Signal decay + synthetic validation | ISMRM Fig 1 | [x] Regenerate |
| 2 | Sampling diagnostics (trace, convergence vs SNR) | NEW | [ ] Build |
| 3 | Robustness: inverse spectra at various SNR | NEW | [~] Fast pass done |
| 4 | Averaged spectra by tissue type | ISMRM Fig 2 | [x] Regenerate |
| 5 | Direction independence (spectra across directions) | NEW | [x] Done |
| 6 | ROC curves (PZ, TZ, GGG) | ISMRM Fig 3 | [x] Regenerate |
| 7 | Uncertainty propagation | ISMRM Fig 4 | [x] Regenerate |
| 8 | Feature importance / LR coefficients | ISMRM Fig 5 | [x] Regenerate |
| 9 | Pixel-wise spectral maps + heatmap | NEW | [x] MAP done |
| 10 | ADC vs LR probability vs uncertainty (flagship) | NEW | [ ] Build next |

---

## Paper Storyline (Agreed Session 1)

**Working title:** "Bayesian Spectral Decomposition of Multi-b Diffusion MRI: Pixel-wise Tissue Characterization with Intrinsic Uncertainty Quantification"

**Narrative arc:**
1. **Problem:** ADC collapses rich multi-b signal into one number, losing microstructural detail
2. **Method:** Bayesian inverse problem (NUTS) → full diffusivity spectrum + calibrated uncertainty per pixel
3. **Validation:** Direction independence, robustness across spectral shapes, convergence diagnostics
4. **Clinical utility:**
   - (a) Per-pixel tumor probability maps that outperform ADC
   - (b) Uncertainty maps as a NOVEL biomarker correlating with Gleason grade
   - (c) Multi-parametric tissue characterization from a single dMRI acquisition
5. **Headline:** "Bayesian posterior uncertainty in restricted diffusion correlates with tumor aggressiveness"

**Key differentiation from bitter lesson concern:**
- NOT competing with neural nets on classification AUC
- Providing interpretable, physics-grounded, uncertainty-aware characterization
- Works with small sample sizes (~25 patients) where neural nets cannot
- Uncertainty itself is a biomarker, not noise

**Flagship figure:** Side-by-side: Anatomy | ADC map | LR probability map | Uncertainty map

---

## Communication Items for Stephan

1. Request all-directions data from Dropbox
2. Request patient demographics table
3. Ask about additional GGG case
4. Send first pixel-wise result for feedback
5. Confirm b-value mapping for 46 binary images (15 b-values × 3 directions + 1?)
