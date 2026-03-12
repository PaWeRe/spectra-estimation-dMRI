# Findings & Open Questions

> **Read alongside SESSION.md at the start of every session.**
> Concise bullet-point learnings, quantitative findings, and open questions.
> Updated: 2026-03-11 (Session 6)

---

## Key Quantitative Findings

### ADC as Special Case of Spectral Discriminant
- Spectral discriminant (LR coef · normalized fractions) anti-correlates with ADC at **r = −0.97**
- **NEW (Session 6)**: ADC sensitivity vector dADC/dRⱼ correlates with LR feature vector at **r = −0.97**
  - This is an independent verification: ADC's implicit spectral weighting matches the optimal tumor discriminant
  - ADC sensitivity at tumor operating point: D=0.25 → -1.70, D=0.5 → -1.03, D=3.0 → +0.72
  - ADC sensitivity at normal operating point: D=0.25 → -4.00, D=0.5 → -2.73, D=3.0 → +0.67
  - **Key**: ADC sensitivity is spectrum-dependent (changes with tissue type); LR discriminant is a fixed linear projection
  - Closed-form derivation is pending (numerical result already confirmed)
- D=0.25 has higher **spatial CV = 0.61** vs ADC **spatial CV = 0.26** (measures tissue contrast)
- LR coefficients: D ≤ 1.0 → tumor-associated (+), D ≥ 1.5 → normal-associated (−)

### MAP vs NUTS Comparison

#### Pixel-level (146 voxels, 1 patient)
- **D=0.25**: MAP and NUTS highly correlated (r=0.99), NUTS ~17% higher. Best-constrained (CV=0.25)
- **D=0.5–1.0**: MAP and NUTS disagree (r=0.27–0.65), poorly identifiable (CV=0.75–0.81)
- **Discriminant**: MAP and NUTS agree (r=0.997) — NUTS value is uncertainty, not point estimate
- NUTS in-sample RMSE lower for 91% of pixels but NOT a fair comparison (NUTS has flexible noise model)
- MAP LOO-b CV shows 78% RMSE increase → even MAP overfits slightly

#### ROI-level (149 ROIs, 56 patients) — **NEW in Session 6**
- **D=0.25**: r=0.99, NUTS/MAP ratio=1.59 (larger gap than pixel-level!)
- **D=0.5–1.0**: r=0.48–0.79, large redistributions between methods
- **D=1.5**: r=-0.11 (essentially uncorrelated between MAP and NUTS!)
- **D=3.0**: r=0.82, NUTS/MAP ratio=1.64
- Components redistribute dramatically but discriminant stays the same (r=0.997)

### Classification Performance (ROI-level) — **UPDATED Session 6**

Comprehensive table with optimized regularization (C=10) and ADC b-value sensitivity:

| Method | PZ (n=81) | TZ (n=68) | GGG (n=29) |
|--------|:---------:|:---------:|:----------:|
| ADC b≤1000 | **0.940** | **0.964** | **0.778** |
| ADC b≤1250 | 0.928 | 0.946 | 0.778 |
| MAP Full LR C=1 | 0.911 | 0.853 | 0.372 |
| MAP Full LR C=10 | 0.935 | 0.941 | 0.722 |
| NUTS Full LR C=1 | 0.918 | 0.888 | 0.411 |
| NUTS Full LR C=10 | 0.933 | 0.925 | 0.722 |
| NUTS Top-5 C=1 | 0.921 | 0.878 | 0.411 |

**Key insights:**
- **Full LR debugging SOLVED**: C=1.0 was too strong regularization. With C=10, Full LR nearly matches ADC (PZ: 0.935 vs 0.940)
- ADC b≤1000 is BETTER than b≤1250 (opposite of Stephan's expectation!)
- MAP and NUTS converge at higher C — the C=1 gap was a regularization artifact, not a NUTS benefit
- GGG: all methods struggle (n=29, only 9 high-grade), ADC still best (0.778)
- Adding uncertainty as features: no improvement (0.918 → 0.916)
- **Patient new39** now included: GS 2+3 → GGG 1 (per 2025 reclassification to 3+3)

### NUTS-Specific Value — **Revised in Session 6**
- **Prediction uncertainty identifies unreliable classifications**: misclassified ROIs have 2.34x (PZ), 1.96x (TZ), 1.45x (GGG) higher uncertainty
- Per-component **posterior CV** reveals identifiability: D=0.25 CV=0.20, D=3.0 CV=0.32, D=0.5-1.0 CV>0.80
- **Joint noise estimation**: SNR 12–172 across prostate voxels
- All 146 pixels converged: R-hat = 1.000
- **Does NOT improve classification AUC** beyond MAP

### Feature Collinearity — **UPDATED Session 6**
- 8-feature correlation matrix is numerically singular (3 eigenvalues ≈ 0)
- Worst pairs: D=0.5 vs D=2.0 (r=-0.975), D=0.25 vs D=1.5 (r=-0.959)
- 81 samples / 7 effective features = 11.6 samples/feature (borderline)
- **RESOLVED**: Full LR underperformance was due to C=1.0 being too strong. C=10 allows the model to exploit feature correlations → nearly matches ADC

### Boundary Uncertainty — **NEW in Session 6**
- Discriminant uncertainty does NOT correlate with tissue boundaries (r=-0.18)
- Boundary pixels: mean unc=0.096, Interior: 0.092 (no difference)
- Uncertainty is intrinsic to spectral identifiability, not spatial position

### D=0.25 MAP vs NUTS Bias — **NEW in Session 6**
- NUTS/MAP ratio correlates with SNR (r=0.86) — SNR-dependent
- Difference (0.035) is 0.89 posterior SDs — not per-pixel significant
- Mechanism: MAP Ridge + clip-at-zero shrinkage, corrected by NUTS joint noise model

### LR Classifier Pixel Application
- StandardScaler caused domain shift → saturated P(tumor)
- Fix: use raw discriminant score without StandardScaler
- PZ model: 81 samples (27 tumor, 54 normal), LOOCV AUC = 0.919

---

## Open Questions

### Methodology (updated after Session 6 analysis)
- [x] ~~Is in-sample RMSE a fair comparison for MAP vs NUTS?~~ → **No.** MAP LOO-b shows 78% increase. Don't claim NUTS is "better" based on RMSE.
- [x] ~~Does discriminant uncertainty correlate with tissue boundaries?~~ → **No.** r=-0.18, null result.
- [x] ~~Why does Full LR underperform ADC?~~ → Extreme multicollinearity (cond# ≈ ∞) + small N.
- [x] ~~Is NUTS D=0.25 being 17% higher meaningful?~~ → SNR-dependent shrinkage bias in MAP. Not significant per-pixel.
- [ ] **NEW**: Can dADC/dRⱼ be derived in closed form? (Stephan expects yes)
- [ ] **NEW**: How does ADC sensitivity vector change across the b-value range used for fitting?
- [ ] **NEW**: Encoding directions — geometric vs arithmetic mean in current data processing?

### Paper Narrative (refined after Stephan meeting)
- [ ] Central claim: ADC sensitivity vector ≈ LR feature vector (r=-0.97) — the analytical proof
- [ ] Scope decision: ADC-focused only (Stephan) vs ADC + uncertainty (Patrick)?
- [ ] ISMRM rejection feedback: "figures not well explained" — improve captions and annotations
- [ ] Frame spectral decomposition as adding interpretability ON TOP of ADC, not replacing it

### For Stephan/Sandy
- [ ] Patient demographics table — still needed from Stephan
- [x] ~~Patient new39: GS 2+3 → what GGG?~~ → **GGG 1** (reclassified as 3+3 under current ISUP)
- [ ] Are NUTS trace plots useful for paper? (ask Sandy)
- [ ] Encoding directions in original Langkilde data — geometric mean confirmed?
- [ ] **NEW**: Sandy email drafted (`results/email_draft_sandy.md`) — NUTS justification questions

### Dataset Notes
- 56 patients, 149 ROIs (PZ: 27T/54N, TZ: 13T/55N)
- GGG now available for 29 tumors (including new39): 20 low-grade (GGG 1-2), 9 high-grade (GGG 3-5)
- Pixel heatmap patient (8640) is NOT in the ROI training set — good for generalization argument
- Langkilde et al. 2018: GE 3T Discovery MR750, endorectal coil, 15 b-values (0–3500 s/mm²)
- ADC b≤1000 outperforms b≤1250 for tumor detection (less contamination from restricted diffusion at high b)
