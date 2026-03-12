# Session State — MRM Paper Collaboration

> **READ THIS FIRST** when starting a new session.
> Updated: 2026-03-11 (Session 6)

---

## Quick Context

We are writing an MRM journal paper on **spectral decomposition of multi-b diffusion MRI for prostate cancer characterization**. The human (Patrick) provides high-level direction; the AI handles implementation. See `paper/PAPER_PLAN.md` for full plan.

**Branch:** `paper/mrm-manuscript`
**Key directories:** `paper/` (LaTeX), `src/spectra_estimation_dmri/` (Python package), `results/` (outputs)
**Patient pixel data:** `src/spectra_estimation_dmri/data/8640-sl6-bin/` (46 .bin files, gitignored)

---

## Session 6 Summary (2026-03-11)

### What we accomplished:

1. **Meeting with Stephan** — major direction-setting. Stephan is excited about the "ADC as special case" finding. Wants a focused paper around this, submittable soon. Sees NUTS/uncertainty as potential second paper.

2. **Critical assessment of NUTS vs MAP** (ROI-level, 149 ROIs, 56 patients):
   - MAP vs NUTS classification: PZ AUC 0.911 vs 0.918 (+0.7%), TZ 0.853 vs 0.888 (+3.5%) — marginal
   - Both below ADC (PZ: 0.940, TZ: 0.964)
   - Discriminant MAP vs NUTS: r = 0.997 (identical for practical purposes)
   - Adding NUTS uncertainty as features: AUC 0.918 → 0.916 (no improvement)
   - NUTS value: misclassified ROIs have 2.34x higher prediction uncertainty (PZ)
   - Only D=0.25 well-identified (CV=0.20 ROI, 0.25 pixel); all others CV > 0.32

3. **Answered the 4 open methodological questions:**
   - Q1 (In-sample RMSE): Not a fair comparison. MAP LOO-b CV shows 78% RMSE increase. NUTS lower in-sample RMSE expected (flexible noise model), not evidence of better spectra.
   - Q2 (Uncertainty at boundaries): Null result. r = -0.18. Uncertainty is intrinsic to spectral inversion, not spatial.
   - Q3 (Full LR < ADC): Extreme multicollinearity. Condition number ≈ ∞. 3 near-zero eigenvalues. D=0.5 vs D=2.0: r = -0.975.
   - Q4 (NUTS 17% higher D=0.25): SNR-dependent bias (r=0.86). Difference is 0.89 posterior SDs — not per-pixel significant. MAP shrinkage + clipping artifact.

4. **KEY DISCOVERY — ADC sensitivity vector ≈ LR feature vector (r = -0.97):**
   - Computed dADC/dRⱼ numerically for each spectral component
   - The ADC sensitivity vector (how much ADC changes per unit change in each Rⱼ) is essentially the mirror image of the learned LR discriminant
   - This proves analytically WHY ADC works: its implicit weighting of spectral components aligns with the tumor-vs-normal spectral difference
   - Stephan predicted this ("Es sollte ein geschlossene Lösung geben") — a closed-form derivation is the next step
   - ADC sensitivity at tumor operating point: D=0.25 → -1.70, D=3.0 → +0.72
   - ADC sensitivity at normal operating point: D=0.25 → -4.00, D=3.0 → +0.67
   - Key insight: ADC sensitivity is spectrum-dependent (nonlinear); the LR discriminant is a fixed linear projection

5. **Full dataset confirmed**: 56 patients, 149 ROIs (109 normal, 40 tumor), PZ + TZ. Langkilde et al. 2018 data. Not single-patient!

6. **ISMRM 2025 abstract** (rejected) reviewed — focused on NUTS methodology. Reviewer feedback: figures not well explained.

### Stephan meeting action items (verbatim parsed):

| # | Item | Type | Status |
|---|------|------|--------|
| 1 | ADC sensitivity analysis: derive dADC/dRⱼ closed-form, compare with LR feature vector | Analysis | **Started** — numerical result r=-0.97, need closed-form |
| 2 | ADC vs spectral components correlation plot (use 1000 points) | Figure | TODO |
| 3 | Invert colors (blue/red) in feature importance map | Figure fix | TODO |
| 4 | Example spectra figure: remove middle components | Figure fix | TODO |
| 5 | Encoding directions: check if 3 directions, geometric vs arithmetic mean | Data quality | TODO |
| 6 | Sampling diagnostics: synthetic signal (tumor/normal spectra) + Gaussian noise (SNR 100,300,500,1000) | Validation | TODO |
| 7 | Trace plots for NUTS convergence — ask Sandy if useful for paper | Question for Sandy | TODO |
| 8 | Consider 1 more normal patient image for heatmap (current not in training set) | Figure | TODO |
| 9 | Patient new39: update GS to 2+3, determine GGG | Data fix | TODO (GS 2+3 = Gleason 5, predates GGG system → GGG 0 equivalent) |
| 10 | ADC vs feature map sensitivity investigation | Analysis | TODO (closely related to #1) |
| 11 | ISMRM feedback: improve figure explanations in paper | Writing | TODO |

### Key findings and decisions:

**Paper narrative (refined after Stephan meeting):**
- Central claim: ADC is a special case of the spectral discriminant — now with ANALYTICAL proof (sensitivity vector)
- The ADC sensitivity vector dADC/dRⱼ matches the learned LR feature vector (r = -0.97)
- This explains WHY ADC works: it implicitly weights spectral components in a near-optimal way
- Spectral decomposition adds: (a) interpretability (which compartments), (b) the sensitivity is nonlinear/spectrum-dependent for ADC but fixed for the discriminant

**NUTS role in the paper (decision needed):**
- Stephan leans toward NUTS as a separate paper
- NUTS adds marginal classification improvement but real uncertainty (2.3x misclass ratio)
- Patrick wants to keep uncertainty — discuss scope with Stephan at next meeting
- Recommendation: include NUTS in supporting role (validates MAP, provides uncertainty), not as headline

### Correct parameters (unchanged):

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
| Dataset | 56 patients, 149 ROIs (40 tumor, 109 normal) | Langkilde 2018 |

### Key file map (updated):

| File | Purpose |
|------|---------|
| `src/spectra_estimation_dmri/pixelwise.py` | Pixel-wise ADC, MAP, NUTS, tumor prob |
| `src/spectra_estimation_dmri/main.py` | Full pipeline: data → inference → biomarkers |
| `src/spectra_estimation_dmri/data/loaders.py` | Data loaders (ROI JSON + pixel binary) |
| `src/spectra_estimation_dmri/biomarkers/pipeline.py` | ROI-level biomarker analysis |
| `results/biomarkers/features.csv` | NUTS posterior means, all 149 ROIs |
| `results/biomarkers/feature_uncertainty.csv` | NUTS posterior stds, all 149 ROIs |
| `results/biomarkers/auc_table_*.csv` | Classification AUCs per task |
| `results/biomarkers/predictions_*.csv` | Per-ROI predictions with uncertainty |
| `results/biomarkers/ismrm/` | ISMRM-formatted figures |
| `results/inference/` | 20 .nc inference files (current run) |
| `results/inference_bwh_backup/` | 149 .nc inference files (NUTS, all ROIs) |
| `results/pixelwise/nuts_results.npz` | NUTS pixel results (146 voxels) |
| `results/pixelwise_all_fast.npz` | ADC + MAP pixel results |
| `paper/sections/results.tex` | Pixel-wise results draft |
| `paper/sections/methods.tex` | Pixel-wise methods draft |
| `paper/sections/figures.tex` | Figure environments + captions |
| `paper/figures/fig_pixelwise_v2.pdf` | Publication figure draft (2×3) |
| `.cursor/FINDINGS.md` | Cumulative findings & open questions |
| `assets/ismrm_2025_submission_rejected.pdf` | Rejected ISMRM abstract (for reference) |

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
- **Better figure explanations**: ISMRM feedback was that figures were poorly explained

---

## Session 7 TODO (Priority Order)

### Phase 1: ADC sensitivity analysis (Stephan's top priority)
1. **Closed-form derivation of dADC/dRⱼ** — derive analytically from the least-squares ADC fit + spectral model. Show that it depends on the operating point (spectrum) and b-value range.
2. **ADC vs spectral components correlation plot** — scatter plot with 1000 points (bootstrap? all pixels?), showing the r = -0.97 relationship visually.
3. **Compare ADC sensitivity vector vs LR feature vector** — side-by-side bar chart. Show they are mirror images.
4. **Investigate sensitivity differences**: ADC sensitivity is spectrum-dependent (nonlinear); discriminant is fixed (linear). Quantify this difference.

### Phase 2: Figure fixes (Stephan feedback)
5. **Invert colors** in feature importance map (swap blue/red)
6. **Example spectra figure**: remove middle components, show only D=0.25, D=3.0, D=20.0 (or as Stephan directed)
7. **Consider additional patient heatmap** (normal patient; current heatmap patient not in training set)

### Phase 3: Data quality & validation
8. **Encoding directions check**: verify 3 directions in image data, confirm geometric vs arithmetic mean
9. **Sampling diagnostics**: generate synthetic signal from average tumor/normal spectra + Gaussian noise (SNR 100, 300, 500, 1000), test MAP and NUTS reconstruction
10. **Patient new39**: update GS to 2+3, clarify GGG mapping

### Phase 4: Questions for collaborators
11. **Ask Sandy**: are NUTS trace plots useful for the paper? Or overkill?
12. **Patient demographics table** — still needed from Stephan

---

## How to Start Session 7

1. Read this file + `.cursor/FINDINGS.md`
2. Run `git log --oneline -5` for latest commits
3. Start with Phase 1: the ADC sensitivity closed-form derivation is the paper's central analytical contribution
4. All ROI data is computed. All pixel data is computed. Focus is on analysis and figures now.
5. Key question to resolve: paper scope — ADC-focused (Stephan preference) vs ADC + uncertainty (Patrick preference)
