# Session State — MRM Paper Collaboration

> **READ THIS FIRST** when starting a new session.
> Updated: 2026-02-21 (Session 3)

---

## Quick Context

We are writing an MRM journal paper on **Bayesian spectral decomposition of multi-b diffusion MRI for prostate cancer characterization**. The human (Patrick) provides high-level direction; the AI handles all implementation. See `paper/PAPER_PLAN.md` for full plan.

**Branch:** `paper/mrm-manuscript`
**Key directories:** `paper/` (LaTeX), `src/spectra_estimation_dmri/` (Python package), `results/` (outputs)
**Patient pixel data:** `src/spectra_estimation_dmri/data/8640-sl6-bin/` (46 .bin files, gitignored)

---

## Session 3 Summary (2026-02-21)

### What we accomplished:

1. **Resolved the file 6 mystery**:
   - Wells et al. 2022 ISMRM abstract explicitly states: "the first signal value was not employed"
   - File 000006.bin is a scanner-generated reference b=0 (54% brighter than protocol b=0)
   - GE scanners produce this reference for inline ADC calculation; it has different TE
   - **Resolution: Drop file 6. Use remaining 45 files = 15 b-values × 3 directions.**

2. **Built clean `ProstateDWI` data loader** in `src/spectra_estimation_dmri/data/loaders.py`:
   - `load_prostate_dwi()` → returns `ProstateDWI` dataclass
   - Automatically loads .bin files, drops scanner reference, groups into 15 triplets
   - Direction-averages to produce trace images
   - Includes `.prostate_mask()` and `.pixel_signal_array()` methods
   - Constants: `B_VALUES_MS_UM2`, `B_VALUES_S_MM2`, `N_BVALUES=15`, `N_DIRECTIONS=3`
   - Verified: clean monotonic decay 622→124 across 15 b-values

3. **Moved data** from `8640-sl6-bin/` → `src/spectra_estimation_dmri/data/8640-sl6-bin/`
   - Updated `.gitignore` to cover new location

4. **Created verification visualization**: `results/dwi_data_verification.png`
   - 15 trace images, scanner ref comparison, decay curves, mean intensity plot

5. **Verified configs are correct**: `configs/prior/ridge.yaml` already has `strength: 0.1`
   (matches ISMRM paper; the "0.5" in bwh.yaml was just a stale comment)

6. **Redesigned agent architecture**:
   - Updated all 4 subagent files with proper frontmatter and SESSION.md reading
   - Updated ARCHITECTURE.md with lean design + Patrick's preferences
   - Subagents: researcher (fast, readonly), computation (inherit, background),
     verifier (fast, readonly), latex-writer (inherit)

### Methodology discussion and decisions:

**The fundamental paper question**: Why use Bayesian spectral decomposition when ADC performs comparably and is much simpler?

**Patrick's position**: He wants to prove clinical utility of the full Bayesian approach, even if slow. Speed can be addressed later (neural nets, amortized inference). First, show the method provides real value.

**Selling points (in order of strength)**:
1. **Uncertainty quantification** — ADC gives no confidence; NUTS gives calibrated uncertainty.
   The ISMRM result: 1.98× higher misclassification for uncertain predictions.
2. **Interpretability** — ADC = one number. Spectra show WHICH components change
   (restricted D=0.25, free water D=3.0, perfusion D=20.0).
3. **Joint noise estimation** — NUTS estimates sigma per ROI/pixel via HalfCauchy,
   so uncertainty adapts to actual local SNR. ADC has no noise awareness.
4. **Gleason grading** — For GGG, D=0.25 already beats ADC (0.81 vs 0.80).
   Intermediate bins (1.5-2.0) carry grade-specific info ADC can't capture.

**Key methodological clarification** (Patrick was confused about this):
- **Simulated data**: known truth → tests whether NUTS recovers spectrum under noise
- **Real data**: single noise realization → NUTS posterior IS the uncertainty
  (you don't need multiple realizations — the Bayesian posterior from ONE measurement
  tells you how uncertain you are about the spectrum)
- The sigma jointly estimated by NUTS tells you local SNR without separate estimation

**Maps to generate for comparison figure**:

| Map | What it shows | Source |
|-----|--------------|--------|
| ADC | Monoexponential fit (b=0-1250) | fast, per-pixel |
| MAP D=0.25 | Restricted diffusion (ridge NNLS) | fast, per-pixel |
| NUTS D=0.25 mean | Bayesian restricted diffusion | slow (~s/pixel) |
| NUTS sigma | Jointly estimated noise level | from NUTS |
| Tumor probability | LR on spectrum → P(tumor) | from spectrum + trained LR |
| Uncertainty | Bootstrap from posterior samples | from NUTS posterior |

**Pipeline integration**: Pixel-level analysis is NOT separate from main.py — it's the
same ProbabilisticModel + inference + biomarkers, just applied to every pixel instead
of ROI averages. The code should share the same path.

### Current ROI-level results (from ISMRM, for reference):

| Task | ADC AUC | Full LR AUC | D=0.25 AUC |
|------|---------|-------------|------------|
| PZ tumor | 0.95 | 0.93 | 0.88 |
| TZ tumor | 0.98 | 0.92 | 0.91 |
| GGG | 0.80 | 0.77 | 0.81 |

Full LR underperforming ADC needs investigation (possible overfitting with 8 features + small n).

### Correct parameters (confirmed):

| Parameter | Value | Source |
|-----------|-------|--------|
| b-values | [0, 250, ..., 3500] s/mm² (15 values) | Langkilde 2018 |
| Diffusivity bins | [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] μm²/ms | ISMRM 2024 |
| Ridge λ | 0.1 | ISMRM 2024 / configs/prior/ridge.yaml |
| HalfCauchy β | 1.0 | ISMRM 2024 |
| NUTS | 2000 iter, 200 warmup | ISMRM 2024 |
| ADC baseline | monoexponential, b=0-1250 s/mm² | ISMRM 2024 |
| LR | L2 (C=1.0), LOOCV, top-5 features | ISMRM 2024 |
| Matrix | 64×64 native (stored as 256×256 interpolated) | Langkilde 2018 |
| Pixel size | 4.375 mm | 280mm FOV / 64 pixels |
| Slice | 5 mm, no gap | Langkilde 2018 |

### Key file map:

| File | Purpose |
|------|---------|
| `src/spectra_estimation_dmri/main.py` | Main Hydra pipeline |
| `src/spectra_estimation_dmri/data/loaders.py` | Data loading (NEW: `load_prostate_dwi()` + `ProstateDWI`) |
| `src/spectra_estimation_dmri/models/prob_model.py` | Probabilistic model (U matrix, MAP, posterior) |
| `src/spectra_estimation_dmri/inference/nuts.py` | NUTS sampler |
| `src/spectra_estimation_dmri/inference/map.py` | MAP inference |
| `src/spectra_estimation_dmri/biomarkers/pipeline.py` | Full biomarker analysis |
| `src/spectra_estimation_dmri/biomarkers/features.py` | Feature extraction |
| `src/spectra_estimation_dmri/visualization/ismrm_exports.py` | ISMRM figures |
| `configs/dataset/bwh.yaml` | BWH dataset config |
| `configs/prior/ridge.yaml` | Ridge prior (λ=0.1) |
| `results/inference_bwh_backup/` | 149 NUTS .nc files (keep!) |
| `results/biomarkers/ismrm/` | ISMRM figures |
| `results/dwi_data_verification.png` | Data verification figure (Session 3) |

### Agent architecture:

**Subagents** (4, all read SESSION.md first):
- `researcher` — fast, readonly. Understand codebase, verify claims.
- `computation` — inherit, background. Run pipelines, generate figures.
- `verifier` — fast, readonly. Validate work products.
- `latex-writer` — inherit. Draft/edit paper sections.

**Skills** (6, domain knowledge):
- mrm-paper-writing, spectrum-estimation, biomarker-creation,
  image-processing, benchmarking-visualization, general-coding

**Shared memory**: This file (SESSION.md), committed to git after each session.

### Patrick's preferences (carry across all sessions):

- **Methodology first**: Think about what to show before coding
- **Unified pipeline**: Pixel-level = generalization of ROI-level, same code path
- **Clinical utility focus**: Why is Bayesian better than ADC?
- **Uncertainty is key**: This is our main differentiator
- **uv only**: Never use pip, always `uv run python`, `uv add`, `uv sync`
- **Edit existing files**: Don't create standalone scripts, integrate into `src/`
- **Max 10 figures**: Every MRM figure must earn its place
- **No direct co-author contact**: All questions through Patrick
- **Context efficiency**: Delegate context-heavy tasks to subagents
- **Learn across sessions**: Update SESSION.md with decisions and preferences

---

## Session 4 TODO (Priority Order)

### Phase 1: Understand the existing pipeline deeply
1. **Trace the full pipeline end-to-end**: Use `/researcher` to map data flow from
   `load_bwh_signal_decays` → `ProbabilisticModel` → NUTS → `DiffusivitySpectrum` →
   biomarker pipeline → figures. Document the exact transformation at each step.
2. **Understand the model math**: Review `prob_model.py` — how U is built, how MAP works,
   how NUTS is configured. Document the prior structure (HalfNormal on R, HalfCauchy on σ).
3. **Verify cached inference**: Check that the 149 .nc files in `results/inference_bwh_backup/`
   are loadable and contain valid posterior samples.

### Phase 2: Build the pixel-wise pipeline
4. **Add pixel-wise mode to main.py**: Use `load_prostate_dwi()` → create `ProbabilisticModel`
   per pixel → run MAP (fast) and/or NUTS (slow) → store spectra.
5. **Compute ADC map**: Monoexponential fit per pixel, b=0-1250. This is the baseline.
6. **Compute MAP spectrum maps**: Ridge NNLS per pixel. Extract D=0.25 fraction map.
7. **Apply trained LR**: Use ROI-level LR coefficients on pixel-level spectra → tumor probability map.

### Phase 3: Run NUTS on pixel subset + generate comparison figure
8. **NUTS on subset**: Pick ~50-100 prostate pixels, run full NUTS. Compare NUTS mean vs MAP.
9. **Create comparison figure**: ADC | D=0.25 | tumor probability | uncertainty | sigma
10. **Evaluate**: Does the comparison tell a compelling story?

### Phase 4: Paper writing
11. **Regenerate ISMRM figures** from cached .nc files (baseline)
12. **Investigate Full LR < ADC** — check feature selection, regularization
13. **Draft methods section** using `/latex-writer`

---

## How to Start Session 4

1. Read this file
2. Check `git log --oneline -5` for latest commits
3. Start with Phase 1 (understand pipeline) using `/researcher` subagent
4. Then move to Phase 2 (pixel-wise maps)
5. Ask Patrick: "Ready for Session 4? Starting with pipeline deep-dive. Sound right?"
