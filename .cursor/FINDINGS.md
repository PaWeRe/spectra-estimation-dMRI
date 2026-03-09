# Findings & Open Questions

> **Read alongside SESSION.md at the start of every session.**
> Concise bullet-point learnings, quantitative findings, and open questions.
> Updated: 2026-03-08 (Session 5)

---

## Key Quantitative Findings

### ADC vs Spectral Decomposition
- Spectral discriminant (LR coef · normalized fractions) anti-correlates with ADC at **r = −0.97**
- Interpretation: ADC is a particular weighted sum of spectral fractions; spectral decomposition recovers ADC while revealing which compartments contribute
- D=0.25 has higher **spatial CV = 0.61** vs ADC **spatial CV = 0.26** (spatial CV = std across pixels / mean across pixels — measures how much the map varies spatially, i.e., tissue contrast)
- LR coefficients show clean biological separation: D ≤ 1.0 → tumor-associated (+), D ≥ 1.5 → normal-associated (−)

### MAP vs NUTS Comparison
- **D=0.25** (restricted diffusion): MAP and NUTS highly correlated (r=0.99), NUTS ~17% higher. Best-constrained component (posterior CV=0.17)
- **D=0.5–1.0** (intermediate): MAP and NUTS disagree (r=0.27–0.65), poorly identifiable (posterior CV=0.75–0.81). MAP essentially decomposes ADC (r=−0.97 with ADC), NUTS decorrelates (r=−0.20)
- **Discriminant**: MAP and NUTS agree on the composite score (r=0.997) — NUTS value is in uncertainty, not point estimate
- NUTS in-sample RMSE lower for 91% of pixels (100% high-SNR), but this is expected: MAP's Ridge penalty intentionally trades data fit for regularization. Not evidence of "better" spectra without ground truth

### NUTS-Specific Value
- Per-component **posterior CV** reveals identifiability: what the data can vs cannot resolve
- **Discriminant uncertainty**: std = 0.06–0.14 (mean 0.094) on range [−0.27, +0.48] — intrinsic quality metric
- **Joint noise estimation**: SNR 12–172 across prostate voxels, correlates with signal amplitude (sanity check)
- All 146 pixels converged: R-hat = 1.000

### LR Classifier Pixel Application
- StandardScaler caused domain shift (pixel D=0.25 mean=0.202 vs ROI tumor mean=0.119) → saturated P(tumor)
- Fix: use raw discriminant score (coef · fractions) without StandardScaler
- Coefficients remain valid as importance weights — they encode biological relationships, not pixel-specific calibration
- PZ model: 81 samples (27 tumor, 54 normal), LOOCV AUC = 0.919

---

## Open Questions

### Methodology
- [ ] Is in-sample RMSE a fair comparison for MAP vs NUTS? Should we do leave-one-b-value-out CV instead?
- [ ] Does discriminant uncertainty correlate with tissue boundaries? (would strengthen clinical utility)
- [ ] Why does Full LR (8 features) underperform ADC? Likely overfitting — try feature selection or stronger regularization
- [ ] Is there value in NUTS D=0.25 being 17% higher than MAP? Or just a prior effect?

### Paper Narrative
- [ ] "ADC as special case of spectral discriminant" — strong enough central narrative?
- [ ] Should we lead with interpretability (what ADC hides) or uncertainty (Bayesian value-add)?
- [ ] Reviewer concern: per-pixel NUTS is computationally expensive (~8s/pixel). Justify vs MAP (<1s total)?

### For Stephan
- [ ] Does the coefficient interpretation from 81 PZ ROIs make clinical sense?
- [ ] Need anatomical annotations on pixel-wise figure? PZ/TZ boundary?
- [ ] Patient demographics table — still needed
- [ ] R-hat = 1.000 sufficient, or do reviewers want trace plots?
