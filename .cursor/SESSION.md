# Session State — MRM Paper Collaboration

> **READ THIS FIRST** when starting a new session.
> Updated: 2026-03-08 (Session 5)

---

## Quick Context

We are writing an MRM journal paper on **Bayesian spectral decomposition of multi-b diffusion MRI for prostate cancer characterization**. The human (Patrick) provides high-level direction; the AI handles implementation. See `paper/PAPER_PLAN.md` for full plan.

**Branch:** `paper/mrm-manuscript`
**Key directories:** `paper/` (LaTeX), `src/spectra_estimation_dmri/` (Python package), `results/` (outputs)
**Patient pixel data:** `src/spectra_estimation_dmri/data/8640-sl6-bin/` (46 .bin files, gitignored)

---

## Session 5 Summary (2026-03-08)

### What we accomplished:

1. **Comprehensive diagnostic analysis of MAP vs NUTS vs ADC**: Generated 7 diagnostic figures comparing all estimation methods across 146 prostate voxels. Key quantitative results:
   - D=0.25 (restricted diffusion): MAP and NUTS highly correlated (r=0.99), best-constrained component (CV=0.17)
   - D=0.5–1.0 (intermediate): MAP and NUTS disagree substantially (r=0.27–0.65), poorly identifiable (CV=0.75–0.81)
   - NUTS gives better signal reconstruction for 91% of pixels (100% of high-SNR)

2. **Resolved LR tumor probability saturation**: Root cause = StandardScaler domain shift (pixel D=0.25 mean=0.202 vs ROI tumor mean=0.119). Fix: removed StandardScaler, use raw discriminant score instead of P(tumor).

3. **Key discovery — ADC as special case of spectral discriminant**:
   - Spectral discriminant (LR coef · normalized fractions) correlates r = −0.971 with ADC
   - This means ADC is a particular weighted sum of spectral fractions
   - The decomposition adds interpretability: which compartments drive ADC at each voxel
   - MAP spectral components are highly correlated with ADC (D=0.5: r=-0.97), but NUTS decorrelates (r=-0.20) — NUTS provides genuinely different information for poorly-identified components

4. **LR coefficient importance analysis**: Coefficients tell a clean biological story:
   - Tumor-associated (+): D=0.25 (+0.71), D=0.5 (+0.55), D=0.75 (+0.40)
   - Normal-associated (−): D=20 (−0.76), D=3.0 (−0.55), D=2.0 (−0.44)
   - Created per-component contribution heatmaps showing where each spectral fraction drives the classification

5. **Discriminant uncertainty via MC propagation**: 
   - Drew 500 MC samples from NUTS posterior per pixel, propagated through discriminant
   - Uncertainty range: std = 0.06–0.14 (mean 0.094) on a discriminant range of [-0.27, +0.48]
   - Provides intrinsic quality metric for the spectral classification

6. **First-draft publication figure** (`paper/figures/fig_pixelwise_v2.pdf`):
   - Clean 2×3 layout: ADC | D=0.25 | D=3.0 | LR coefficients | Discriminant | Uncertainty
   - Zoomed to prostate region, consistent colorbars

7. **Started LaTeX paper sections**:
   - `paper/sections/results.tex`: Pixel-wise results with spectral maps, MAP vs NUTS comparison, discriminant, uncertainty
   - `paper/sections/methods.tex`: Pixel-wise methods (MAP, NUTS, ADC, discriminant score, MC uncertainty)
   - `paper/sections/figures.tex`: Figure environment with detailed caption for pixel-wise figure

8. **Email draft for supervisor** in `results/email_draft_supervisor.md`

### Key findings and decisions:

**Spectral decomposition value proposition (for the paper narrative):**
1. **Interpretability**: Spectral fractions reveal which tissue compartments (restricted, glandular, free water) contribute to the observed diffusion signal — ADC collapses this into one number
2. **Feature importance**: LR coefficients trained on ROIs give biological meaning to each spectral bin
3. **Uncertainty**: NUTS posterior provides (a) per-component identifiability (CV), (b) per-voxel discriminant uncertainty, (c) joint noise estimation
4. **ADC recovery**: The spectral discriminant recovers ADC (r=−0.97) as a special case, but additionally decomposes it

**What NUTS adds beyond MAP:**
- Better signal reconstruction (91% of voxels)
- Honest uncertainty (poorly-identified components get high CV)
- Joint noise estimation per voxel (SNR 12–172)
- For the discriminant itself, MAP and NUTS agree (r=0.997) — the value is in the uncertainty, not the point estimate

### Correct parameters (unchanged from Session 4):

| Parameter | Value | Source |
|-----------|-------|--------|
| b-values | [0, 250, ..., 3500] s/mm² (15 values) | Langkilde 2018 |
| Diffusivity bins | [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] μm²/ms | ISMRM 2024 |
| Ridge λ | 0.1 | ISMRM 2024 / configs/prior/ridge.yaml |
| HalfCauchy β | 0.01 | pixelwise.py |
| NUTS | 2000 draws, 200 tune, 4 chains, target_accept=0.95 | configs/inference/nuts.yaml |
| ADC | monoexponential, b ≤ 1250 s/mm² | ISMRM 2024 |
| Prostate mask | Manual, 146 pixels | 3D Slicer, Session 4 |
| LR (PZ) | C=1.0, 81 samples (27 tumor, 54 normal), LOOCV AUC=0.919 | Session 4 |

### Key file map (updated):

| File | Purpose |
|------|---------|
| `src/spectra_estimation_dmri/pixelwise.py` | Pixel-wise ADC, MAP, NUTS, tumor prob |
| `paper/sections/results.tex` | **UPDATED**: Pixel-wise results draft |
| `paper/sections/methods.tex` | **UPDATED**: Pixel-wise methods draft |
| `paper/sections/figures.tex` | **UPDATED**: Pixel-wise figure + caption |
| `paper/figures/fig_pixelwise_v2.pdf` | **NEW**: Publication figure draft (2×3) |
| `paper/figures/fig_pixelwise_v1.pdf` | **NEW**: Extended figure (10 panels, backup) |
| `results/email_draft_supervisor.md` | **NEW**: Email draft for Stephan |
| `results/diag_comprehensive.png` | **NEW**: All-in-one diagnostic |
| `results/diag_map_vs_nuts.png` | **NEW**: MAP vs NUTS per component |
| `results/diag_nuts_uncertainty.png` | **NEW**: NUTS mean/std/CV |
| `results/diag_lr_importance_heatmap.png` | **NEW**: Per-component LR contribution |
| `results/diag_classifier_story.png` | **NEW**: Fractions to composite biomarker |
| `results/diag_lr_investigation.png` | **NEW**: LR scaler fix |
| `results/diag_snr_sigma.png` | **NEW**: SNR and noise overview |
| `results/pixelwise/nuts_results.npz` | NUTS pixel results (146 done) |
| `results/pixelwise_all_fast.npz` | ADC + MAP + tumor prob results |

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
- **Overleaf** for LaTeX compilation (no local LaTeX compiler)
- **Build narrative alongside figures**: Write findings into LaTeX incrementally

---

## Session 6 TODO (Priority Order)

### Phase 1: Paper narrative and figure iteration
1. **Review fig_pixelwise_v2 with Stephan** — get feedback on panel selection, need for annotations
2. **Iterate on figure** based on feedback (add anatomy? PZ/TZ boundary? tumor annotation?)
3. **Consider local LaTeX preview** — install tectonic or BasicTeX via homebrew for local builds
4. **Continue Results/Methods writing** — add ROI-level results sections, complete Methods

### Phase 2: Strengthen the story
5. **Test whether discriminant uncertainty correlates with tissue boundaries** — would strengthen clinical utility argument
6. **Explore whether NUTS D=0.25 has additional value over MAP D=0.25** — the 17% higher mean could matter
7. **Debug Full LR < ADC anomaly** (carried from Session 4) — overfitting with 8 features on small N

### Phase 3: Additional figures
8. **Generate remaining ISMRM-quality figures**: tissue spectra boxplots, ROC curves, signal decay examples
9. **Consider a second pixel-wise figure**: per-component LR importance heatmap (diag_lr_importance_heatmap.png was promising)

### Phase 4: Logistics
10. **Install tectonic** (`brew install tectonic`) for local LaTeX preview — then `tectonic main.tex` in paper/ to build PDF locally
11. **Send email to Stephan** (draft in results/email_draft_supervisor.md)
12. **Patient demographics table** — still BLOCKED on Stephan

---

## How to Start Session 6

1. Read this file
2. Run `git log --oneline -5` for latest commits
3. **Install tectonic**: `brew install tectonic` then test with `cd paper && tectonic main.tex`
4. Ask Patrick: "Did Stephan respond? Any feedback on the figure or narrative direction?"
5. All data is computed — figure iteration is fast
6. If writing: continue with Introduction or Theory sections
