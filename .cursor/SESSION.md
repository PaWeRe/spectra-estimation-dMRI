# Session State — MRM Paper Collaboration

> **READ THIS FIRST** when starting a new session.
> Updated: 2026-03-21 (Session 7)

---

## Quick Context

We are writing an MRM journal paper on **spectral decomposition of multi-b diffusion MRI for prostate cancer characterization**. The human (Patrick) provides high-level direction; the AI handles implementation.

**Branch:** `main`
**Key directories:** `paper/` (LaTeX), `src/spectra_estimation_dmri/` (Python package), `results/` (outputs)
**Patient pixel data:** `src/spectra_estimation_dmri/data/8640-sl6-bin/` (46 .bin files, gitignored)

---

## Session 7 Summary (2026-03-21)

### What we accomplished:

1. **Sent email to Sandy/Stephan** with ADC sensitivity analysis and current numbers. Both replied positively.
   - **Stephan**: Confirms ADC sensitivity result is "beautiful" and belongs in paper. Wants both MAP and NUTS shown with benefits/disadvantages. Notes it "kills the science around all derived measures."
   - **Sandy**: Frames MCMC as an **exploratory tool** — "approaching the basic data and physics problem with MCMC got us to where we are now." Less general methods may work going forward. Interested in generalization to other tumors.

2. **Thorough verification of all key findings** — revealed critical discrepancies between what was reported in Session 6 notes and what can actually be reproduced from the data on disk.

### ⚠️ VERIFICATION RESULTS — CRITICAL

We ran `scripts/verify_findings.py` on the current `results/biomarkers/features.csv` (generated Oct 30, NUTS features only, C=1.0 defaults). Here is what we found:

#### CONFIRMED ✅

| Finding | Verified Value | Reported Value | Status |
|---------|---------------|----------------|--------|
| ADC vs discriminant correlation (PZ, across 81 ROIs) | r = −0.979 | r = −0.97 | ✅ Confirmed |
| ADC vs discriminant correlation (TZ, across 68 ROIs) | r = −0.980 | r = −0.97 | ✅ Confirmed |
| D_0.25 identifiability (posterior CV) | 0.20 | 0.20 | ✅ Confirmed |
| D_0.50–1.00 poorly identified (CV) | 0.81, 0.82, 0.81 | >0.80 | ✅ Confirmed |
| D_3.00 identifiability (CV) | 0.32 | 0.32 | ✅ Confirmed |
| Dataset: 56 patients, 149 ROIs, 40 tumor, 109 normal | ✅ | ✅ | ✅ Confirmed |
| ADC sensitivity magnitudes (tumor D=0.25) | −1.74 μm²/ms | −1.70 | ✅ ~Confirmed |

#### NOT REPRODUCED / INCORRECT ❌

| Finding | Verified Value | Reported Value | Issue |
|---------|---------------|----------------|-------|
| **Sensitivity vector vs LR coef vector** | **r = −0.79** (PZ, 8 elements) | **r = −0.97** | ❌ The r=−0.97 is actually the ROI-level ADC-vs-discriminant correlation, NOT the vector-level sensitivity-vs-LR-coef correlation. These are different things. |
| **PZ ADC AUC** | 0.940 (LR LOOCV C=1) or 0.951 (raw) | 0.940 | ⚠️ Verified, but method matters: 0.940 uses LR wrapper, 0.951 is raw rank AUC |
| **PZ Full LR C=10 AUC (NUTS)** | **0.912** | **0.935 (MAP), 0.933 (NUTS)** | ❌ Cannot reproduce. Reported numbers from Session 6 interactive analysis, not persisted. |
| **TZ Full LR C=10 AUC (NUTS)** | **0.911** | **0.941 (MAP), 0.925 (NUTS)** | ❌ Same issue. |
| **GGG sample size** | **n=28** | **n=29** | ❌ new39 metadata.csv has GGG=1, but features.csv has empty ggg for new39. Pipeline was never rerun after metadata fix. |
| **MAP vs NUTS AUCs (separate)** | Cannot verify | Various | ❌ features.csv contains only NUTS features. MAP features not on disk — must be recomputed. |
| **MAP vs NUTS discriminant r=0.997** | Cannot verify | r=0.997 | ❌ Same: need MAP features to compare. |

#### ROOT CAUSES

1. **Stale features.csv**: Generated Oct 30 with C=1.0 defaults and old metadata. The "C=10" numbers from Session 6 were computed interactively and never persisted to any file.
2. **No MAP features on disk**: The .nc files only store NUTS posteriors. MAP estimates (`spectrum_init`) are computed at runtime but not saved. There is no MAP features.csv.
3. **Conflated correlations**: Two different r=−0.97 claims (ROI-level and vector-level) were reported as the same number. The ROI-level one (ADC scores vs discriminant scores, N=81+) is real; the vector-level one (8-element sensitivity vs 8-element LR coefs) is r=−0.79.
4. **new39 fix partial**: metadata.csv was updated (GGG=1) but pipeline was never rerun → features.csv still has empty ggg for new39.

### What needs to happen before we can write the paper:

**MUST DO (blocking paper draft):**

1. **Compute MAP features** from raw signal data (Ridge NNLS on all 149 ROIs). Fast — seconds, not hours.
2. **Rerun classification** with a single clean script that:
   - Loads raw signal data + updated metadata
   - Computes MAP features (Ridge NNLS)
   - Loads NUTS posterior means from .nc files
   - Computes ADC from raw signal (both b≤1000 and b≤1250)
   - Runs LOOCV classification for PZ, TZ, GGG at C=1 and C=10
   - Reports all AUCs in one clean table
   - Saves everything to CSV
3. **Fix new39 in features pipeline** — ensure metadata.csv GGG flows through to features.csv
4. **Clarify the two r=−0.97 claims** — be precise in the paper:
   - ROI-level: ADC anti-correlates with discriminant at r ≈ −0.98 ✅
   - Vector-level: sensitivity vector vs LR coefs at r ≈ −0.80 (weaker but still significant, p<0.02)

**NICE TO HAVE (can draft around):**
- Closed-form derivation of dADC/dRⱼ
- Updated figures per Stephan's feedback

---

## Correct parameters:

| Parameter | Value | Source |
|-----------|-------|--------|
| b-values (config units) | [0, 0.25, ..., 3.5] ms/μm² | Langkilde 2018 |
| b-values (s/mm²) | [0, 250, ..., 3500] s/mm² (15 values) | Langkilde 2018 |
| Diffusivity bins | [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] μm²/ms | ISMRM 2024 |
| Ridge λ (MAP) | 0.1 | configs/prior/ridge.yaml |
| HalfCauchy β (NUTS noise) | 0.01 | pixelwise.py |
| NUTS | 2000 draws, 200 tune, 4 chains, target_accept=0.95 | configs/inference/nuts.yaml |
| ADC | monoexponential, b ≤ 1000 s/mm² (= 1.0 ms/μm²) | Best from Session 6 analysis |
| Dataset | 56 patients, 149 ROIs (40 tumor, 109 normal) | Langkilde 2018 |
| PZ | 81 ROIs (27 tumor, 54 normal) | |
| TZ | 68 ROIs (13 tumor, 55 normal) | |
| GGG | 28 ROIs (19 low-grade, 9 high-grade) — should be 29 after new39 fix | |

## Key file map:

| File | Purpose | Status |
|------|---------|--------|
| `src/spectra_estimation_dmri/main.py` | Full pipeline: data → inference → biomarkers | Working but uses Hydra, complex |
| `src/spectra_estimation_dmri/data/loaders.py` | Data loaders (ROI JSON + pixel binary) | Working, reads metadata.csv correctly |
| `src/spectra_estimation_dmri/data/bwh/metadata.csv` | Patient metadata with GGG | Updated (new39 = GGG 1) |
| `src/spectra_estimation_dmri/data/bwh/signal_decays.json` | Raw ROI signal data (56 patients) | Unchanged |
| `src/spectra_estimation_dmri/biomarkers/pipeline.py` | ROI-level biomarker analysis | Working but C=1 default |
| `src/spectra_estimation_dmri/biomarkers/mc_classification.py` | LOOCV classification code | Working |
| `src/spectra_estimation_dmri/biomarkers/features.py` | Feature extraction from spectra | Working |
| `src/spectra_estimation_dmri/biomarkers/adc_baseline.py` | ADC computation | Working |
| `results/biomarkers/features.csv` | ⚠️ STALE — NUTS features, old metadata, C=1 | Needs regeneration |
| `results/biomarkers/feature_uncertainty.csv` | NUTS posterior stds, 149 ROIs | OK for identifiability |
| `results/biomarkers/auc_table_*.csv` | ⚠️ STALE — C=1 only, NUTS only | Needs regeneration |
| `results/inference_bwh_backup/` | 149 .nc files (NUTS posteriors) | OK — gold standard |
| `results/inference/` | 20 .nc files (partial run) | Ignore |
| `scripts/verify_findings.py` | Verification script (Session 7) | Current |
| `configs/dataset/bwh.yaml` | Pipeline config | Default C=1, b_range=[0,1] |

## Paper narrative (post Sandy/Stephan feedback):

1. Spectral decomposition reveals what ADC captures and why it works
2. ADC scores anti-correlate with spectral discriminant at r ≈ −0.98 — ADC is near-optimal
3. MAP and NUTS as two complementary inference approaches, honestly compared
4. NUTS adds uncertainty quantification; was the exploratory tool that led to these insights (Sandy's framing)
5. Both methods shown with pros/cons (Stephan's request)
6. The inverse Laplace is ill-posed — only D=0.25 is well-identified (CV=0.20)

## Patrick's preferences (carry across all sessions):

- **Methodology first**: Think about what to show before coding
- **Clinical utility focus**: Why is Bayesian better than ADC?
- **Uncertainty is key**: This is our main differentiator
- **uv only**: Never use pip, always `uv run python`, `uv add`, `uv sync`
- **Edit existing files**: Don't create standalone scripts, integrate into `src/`
- **Max 10 figures**: Every MRM figure must earn its place
- **No direct co-author contact**: All questions through Patrick
- **Go step by step**: Don't rush ahead — walk through each decision
- **Overleaf** for LaTeX compilation (no local LaTeX compiler)
- **Better figure explanations**: ISMRM feedback was that figures were poorly explained
- **Iterate fast**: Get to a full draft ASAP so Sandy/Stephan can comment on concrete text

---

## How to Start Next Session

1. Read this file + `.cursor/FINDINGS.md`
2. **Priority 1**: Compute MAP features + rerun all classification in one clean script
3. **Priority 2**: Once numbers are trustworthy, start drafting paper sections
4. All NUTS inference is done (149 .nc files). Focus is on feature extraction, analysis, and writing.
