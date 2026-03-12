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

### Classification Performance (ROI-level) — **NEW in Session 6**

| Method | PZ AUC | TZ AUC |
|--------|--------|--------|
| ADC | 0.940 | 0.964 |
| NUTS Full LR (8 feat) | 0.918 | 0.888 |
| MAP Full LR (8 feat) | 0.911 | 0.853 |
| NUTS Top-5 (feat sel) | 0.921 | — |
| NUTS D=0.25 + D=3.0 | 0.920 | — |
| NUTS D=0.25 only | 0.776 | 0.733 |
| MAP D=0.25 only | 0.758 | 0.687 |
| NUTS + uncertainty (16 feat) | 0.916 | — |

- ADC wins on tumor detection (single feature, well-calibrated)
- NUTS marginal over MAP (+0.7% PZ, +3.5% TZ)
- Adding uncertainty as features: no improvement (0.918 → 0.916)
- GGG (Gleason grade): too few high-grade cases (n=4) for reliable analysis

### NUTS-Specific Value — **Revised in Session 6**
- **Prediction uncertainty identifies unreliable classifications**: misclassified ROIs have 2.34x (PZ), 1.96x (TZ), 1.45x (GGG) higher uncertainty
- Per-component **posterior CV** reveals identifiability: D=0.25 CV=0.20, D=3.0 CV=0.32, D=0.5-1.0 CV>0.80
- **Joint noise estimation**: SNR 12–172 across prostate voxels
- All 146 pixels converged: R-hat = 1.000
- **Does NOT improve classification AUC** beyond MAP

### Feature Collinearity — **NEW in Session 6**
- 8-feature correlation matrix is numerically singular (3 eigenvalues ≈ 0)
- Worst pairs: D=0.5 vs D=2.0 (r=-0.975), D=0.25 vs D=1.5 (r=-0.959)
- 81 samples / 7 effective features = 11.6 samples/feature (borderline)
- Explains why Full LR (8 feat) underperforms ADC: overfitting on collinear features

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
- [ ] Patient demographics table — still needed
- [ ] Patient new39: GS 2+3 → what GGG? (Gleason 5 predates GGG system)
- [ ] Are NUTS trace plots useful for paper? (ask Sandy)
- [ ] Encoding directions in original Langkilde data — geometric mean confirmed?

### Dataset Notes
- 56 patients, 149 ROIs (PZ: 27T/54N, TZ: 13T/55N)
- GGG available for ~20 PZ tumors, only 4 high-grade → underpowered for GGG analysis
- Pixel heatmap patient (8640) is NOT in the ROI training set — good for generalization argument
- Langkilde et al. 2018: GE 3T Discovery MR750, endorectal coil, 15 b-values (0–3500 s/mm²)
