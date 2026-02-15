# MRM Paper Plan: Fully Bayesian Uncertainty-Aware Biomarkers for Prostate dMRI

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

### A. Direction Independence Analysis [BLOCKED: Dropbox data]
- **What**: Compare spectra across 3 gradient directions for same ROIs
- **Goal**: Show spectra are consistent within uncertainty bounds (PZ focus)
- **Hypothesis**: PZ should be isotropic, TZ may show some anisotropy
- **Data needed**: All-directions data from Stephan's Dropbox
- **Code needed**: New analysis script to load multi-direction data, run NUTS per direction, compare

### B. Sampling Diagnostics & Robustness [CAN START NOW]
- **What**: Trace plots, R-hat, ESS across SNR scenarios
- **Goal**: Define clear convergence boundaries
- **Approach**: 
  1. Construct various synthetic spectra (including "inverse" of current)
  2. Run NUTS at SNR = {50, 100, 200, 500, 1000}
  3. Report R-hat, ESS, RMSE, bias
- **Code**: Extend simulation module, new analysis script

### C. Pixel-wise Prostate Mapping [PARTIALLY READY]
- **What**: Apply NUTS to every pixel in prostate region
- **Data**: 8640-sl6-bin/ folder already in repo (46 binary images, 256x256)
- **Key insights from Stephan**:
  - Native resolution is 64x64, read every 4th pixel (16x faster)
  - ~10k pixels for prostate, need <1 sec/pixel
  - Each spectral component → separate image
  - Apply logistic regression → probability heatmap
- **Steps**:
  1. Benchmark NUTS speed on single pixel
  2. Run pixel-wise on native 64x64 grid (prostate mask)
  3. Generate spectral component maps
  4. Apply trained LR discriminator → heatmap
- **Code**: explore_pixel_data.py exists, need pixel_wise_heatmap.py

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
| 3 | Robustness: inverse spectra at various SNR | NEW | [ ] Build |
| 4 | Averaged spectra by tissue type | ISMRM Fig 2 | [x] Regenerate |
| 5 | Direction independence (spectra across directions) | NEW | [BLOCKED] |
| 6 | ROC curves (PZ, TZ, GGG) | ISMRM Fig 3 | [x] Regenerate |
| 7 | Uncertainty propagation | ISMRM Fig 4 | [x] Regenerate |
| 8 | Feature importance / LR coefficients | ISMRM Fig 5 | [x] Regenerate |
| 9 | Pixel-wise spectral maps + heatmap | NEW | [~] Pipeline exists |
| 10 | Compartment heatmaps on anatomy | NEW | [ ] Build |

---

## Communication Items for Stephan

1. Request all-directions data from Dropbox
2. Request patient demographics table
3. Ask about additional GGG case
4. Send first pixel-wise result for feedback
5. Confirm b-value mapping for 46 binary images (15 b-values × 3 directions + 1?)
