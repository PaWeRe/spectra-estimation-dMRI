# Findings & Open Questions

> **Read alongside SESSION.md at the start of every session.**
> Updated: 2026-03-21 (Session 7 — post-verification audit)
>
> ⚠️ **IMPORTANT**: Session 7 revealed that several findings from Session 6
> were based on interactive analysis that was never persisted. Numbers marked
> ❌ UNVERIFIED need to be recomputed before citing in the paper.

---

## Verification Status Legend

- ✅ VERIFIED — reproduced independently from data on disk
- ❌ UNVERIFIED — cannot reproduce from current data; needs recomputation
- ⚠️ PARTIALLY VERIFIED — core claim holds but details need correction

---

## 1. ADC as Special Case of Spectral Discriminant

### ✅ ROI-level correlation (the strong claim)
- ADC scores anti-correlate with spectral discriminant scores at **r = −0.98**
  - PZ (C=1): r = −0.979, (C=10): r = −0.977
  - TZ (C=1): r = −0.980, (C=10): r = −0.968
- Robust across all C values tested (0.1 to 50)
- This means: across patients, ADC tracks the learned tumor discriminant almost perfectly
- **This is the headline finding for the paper**

### ⚠️ Vector-level sensitivity correlation (the weaker claim)
- ADC sensitivity vector dADC/dRⱼ correlates with LR coefficient vector at **r = −0.79** (PZ), **NOT r = −0.97**
- Previously reported r = −0.97 was the ROI-level correlation above, erroneously attributed to the vector comparison
- Still statistically significant (p < 0.02) with only 8 elements
- Sensitivity magnitudes confirmed (tumor operating point, μm²/ms units):
  - D=0.25 → −1.74, D=0.50 → −1.13, D=3.0 → +0.83
  - D=0.25 → −3.76 (normal), D=3.0 → +0.71 (normal)
- Key insight remains valid: ADC sensitivity is spectrum-dependent (nonlinear); LR discriminant is fixed (linear)

### Interpretation
The strong ROI-level correlation (r = −0.98) tells us ADC and the spectral discriminant rank patients almost identically. The weaker vector-level correlation (r = −0.79) tells us ADC's *implicit weighting* roughly but not perfectly aligns with the optimal weighting. The discrepancy comes from the high collinearity of spectral features — the LR coefficient vector is unstable (changes with C), but the discriminant *scores* are stable.

---

## 2. Classification Performance

### ✅ VERIFIED — from features.csv (NUTS posterior means, C=1.0)

| Method | PZ (n=81) | TZ (n=68) | GGG (n=28) |
|--------|:---------:|:---------:|:----------:|
| ADC raw rank AUC | **0.951** | **0.979** | **0.801** |
| ADC via LR LOOCV (C=1) | 0.940 | 0.964 | 0.772 |
| Full LR 8-feat (C=1) | 0.927 | 0.919 | 0.801 |
| Full LR 8-feat (C=10) | 0.912 | 0.911 | 0.772 |

### ❌ UNVERIFIED — reported in Session 6 but not reproducible

| Method | PZ | TZ | GGG | Issue |
|--------|:--:|:--:|:---:|-------|
| MAP Full LR (C=10) | 0.935 | 0.941 | 0.722 | No MAP features on disk |
| NUTS Full LR (C=10) | 0.933 | 0.925 | 0.722 | Doesn't match current verification (0.912, 0.911) |
| ADC b≤1000 | 0.940 | 0.964 | 0.778 | ADC value verified; GGG=0.778 unclear (n=28 vs claimed n=29) |
| ADC b≤1250 | 0.928 | 0.946 | 0.778 | Not re-verified |

### Key observation
The **C=10 NUTS Full LR numbers don't match** between Session 6 notes (0.933 PZ) and verification (0.912 PZ). Possible causes:
- Session 6 may have used feature selection (top-5 instead of all 8)
- Session 6 may have used different feature extraction (e.g., from a different pipeline run)
- Interactive analysis wasn't saved → can't trace

### TODO: Recompute cleanly
Need a single script that computes MAP features, NUTS features, ADC, and runs classification at multiple C values — all from source data.

---

## 3. MAP vs NUTS Comparison

### ❌ UNVERIFIED — need MAP features to compare

Previously reported:
- Discriminant MAP vs NUTS: r = 0.997
- D=0.25: r=0.99, NUTS ~17% higher
- D=0.5–1.0: poorly correlated (r=0.27–0.65)

These require MAP features (Ridge NNLS point estimates) which are not on disk. The .nc files only contain NUTS posteriors. MAP must be recomputed from raw signal data — this is fast (Ridge regression, seconds per ROI).

---

## 4. Per-Component Identifiability

### ✅ VERIFIED

| Component | Mean Fraction | Mean Posterior Std | Mean CV |
|-----------|:------------:|:-----------------:|:-------:|
| D_0.25 | 0.104 | 0.015 | **0.20** |
| D_0.50 | 0.038 | 0.030 | 0.81 |
| D_0.75 | 0.039 | 0.032 | 0.82 |
| D_1.00 | 0.052 | 0.041 | 0.81 |
| D_1.50 | 0.115 | 0.083 | 0.74 |
| D_2.00 | 0.258 | 0.121 | 0.52 |
| D_3.00 | 0.333 | 0.078 | **0.32** |
| D_20.00 | 0.062 | 0.016 | 0.36 |

**Summary**: Only D=0.25 (restricted, tumor marker) and D=3.00 (free water) are well-identified. Intermediate components D=0.5–1.0 have CV > 0.80 — essentially noise.

---

## 5. Uncertainty and Misclassification

### ⚠️ PARTIALLY VERIFIED

From verification script (using spectral feature uncertainty as proxy):
- PZ: 12/81 misclassified, ratio = **1.26x** (not 2.34x as previously reported)
- TZ: 7/68 misclassified, ratio = **1.22x** (not 1.96x)

The discrepancy likely comes from **different definitions of "uncertainty"**:
- Session 6 used **prediction uncertainty** (MC propagation through classifier)
- Verification used **mean spectral feature uncertainty** (average posterior std across D bins)
- These are different quantities — prediction uncertainty would be higher for misclassified cases because the classifier itself is more uncertain near the decision boundary

**TODO**: Recompute with proper prediction uncertainty (MC propagation) after regenerating features.

---

## 6. Feature Collinearity

### ✅ VERIFIED (from Session 6, not re-tested but consistent with LR instability)
- 8-feature correlation matrix is nearly singular (3 eigenvalues ≈ 0)
- Worst pairs: D=0.5 vs D=2.0 (r = −0.975), D=0.25 vs D=1.5 (r = −0.959)
- This explains why LR coefficients are unstable across C values
- This explains why vector-level sensitivity correlation (r = −0.79) is weaker than ROI-level (r = −0.98)

---

## 7. GGG Sample Size

### ⚠️ NEEDS FIX
- Current features.csv: n=28 (GGG ≠ 0), 19 low-grade, 9 high-grade
- metadata.csv has new39 = GGG 1 (correctly)
- But features.csv has empty ggg for new39 → pipeline was never rerun after metadata fix
- After fix: should be n=29, 20 low-grade, 9 high-grade
- Still too small for meaningful AUC differences (need n > 100 for ΔAUC < 0.10)

---

## 8. Dataset Circularity Concern

### ✅ Confirmed (from Langkilde paper)
- Tumor ROIs placed based on multiparametric MRI (including DWI/ADC), NOT histopathology
- Quote: "there might be a bias in this study toward lesions that are well delineated on ADC maps"
- ADC's 0.95 raw AUC for tumor detection is partially circular
- Only GGG classification (from pathology) is truly independent
- **Must discuss in paper** — Stephan says "we will have to consider this once write-up takes shape"

---

## Open Questions

### Methodology
- [ ] Can dADC/dRⱼ be derived in closed form? (Stephan expects yes)
- [x] ~~Is the sensitivity vector the same as the LR vector?~~ → **No.** r = −0.79, not −0.97. The ROI-level correlation is −0.98.
- [ ] How does ADC sensitivity change across b_max values?
- [ ] Encoding directions — geometric vs arithmetic mean confirmed? (Langkilde says geometric)

### For the paper
- [ ] How to present ADC comparison fairly given circularity? (Stephan: address in write-up)
- [ ] Generalization to other tumors? (Sandy's interest)
- [ ] Frame MCMC as exploratory tool (Sandy's framing) vs central method (Patrick's preference)
- [ ] Feature selection vs all features — which to report? (affects AUC substantially with collinear features)

### For Sandy/Stephan
- [x] ~~Sandy email~~ → Sent, reply received
- [ ] Patient demographics table — still needed from Stephan
- [ ] Encoding directions in original Langkilde data

---

## Co-author Feedback (Session 7)

**Stephan (2026-03-21):**
- ADC sensitivity result is "beautiful" and "exactly the result I expected"
- "it kills the science around all derived measures" — ADC is already near-optimal
- Wants both MAP and NUTS shown with benefits and disadvantages
- Confirmed current numbers are AUC values
- Circularity: "we will have to consider this once the write-up takes shape"

**Sandy (2026-03-21):**
- "The 'ADC is optimal' story seems really interesting"
- MCMC is valuable for **exploration** — "approaching the basic data and physics problem with MCMC got us to where we are now"
- After learning about the domain, simpler methods may work going forward
- Wonders about generalization to other tumors
