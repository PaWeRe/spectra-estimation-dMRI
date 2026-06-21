# PROJECT_STATE — MRM submission

**Single source of truth for the manuscript state. Read this file first every session. `notes/archive/MEETING_PREP_2026-05-25.md` is the archived Q&A from the 2026-05-25 coauthor meeting.**

- **Last update:** 2026-06-21 — **Reviewer-feedback round on the sensitivity figure (Fig 6) + new §0 manuscript-finalization tracker.** A non-domain ML reviewer read Fig 6 as ADC and the spectral classifier being two *different* estimators (ADC sensitive to the middle bins; classifier ignores them). Resolved as a communication issue, not a results issue: (1) added the **mirror ablation** — inner-6 LR with the two outer bins removed drops to NUTS PZ 0.817 / TZ 0.775, vs outer-2 ≈ full ≈ ADC — to Table 1, the ROC (Fig 4, `fig2_v3`, new orange dotted curve), and `recompute.py`; also a **4-poorly-id (CV>0.7) probe** → AUC 0.82/0.81 cited in Results as the "redundancy not uselessness" punchline. (2) Reframed Results "Why ADC works" around **sensitivity × contrast** (ADC *is* sensitive to the mid bins, but that sensitivity is diagnostically inert there because the mid bins carry little tumor–normal contrast and are poorly identified) — replaces the inaccurate "carry little weight in either method" line; ROC + Fig 6 captions updated. (3) Expanded **Methods → Classification** standardization rationale (z-scoring how + why; Stephan had been confused) — **FINALIZED** this pass. Commits 66ca436 + 002f36b, pushed; Overleaf sync pending (Patrick). **New §0 below** tracks per-subsection finalization going forward.
- **2026-06-07** — **Manuscript pass + figure reorder. Detailed progress log, decision record, and outstanding items now live in `paper/drafting/manuscript_blueprint.md` (committed) — read it alongside this file.** Highlights: (1) **Uncertainty story RESOLVED** — posterior uncertainty adds NO downstream diagnostic value, triangulated 3 ways (logit-space propagation p=0.42 / CI incl. 0; whole-spectrum spread dies under distance-to-boundary control; uncertainty-as-features ΔAUC≈0). The 2.4× "wider for misclassified" is sigmoid geometry, not the spectral posterior. Reframed honestly; Fig (uncertainty) → "prediction confidence"; "uncertainty-aware biomarker" framing dropped; Bayesian value is upstream (identifiability + joint noise). (2) **Figures REORDERED to the narrative arc — this SUPERSEDES §11's renumbering**: now **Fisher=1**, spectra=2, validation=3, ROC=4, ADC-discriminant=5, sensitivity=6, GGG=7, uncertainty=8, pixel-wise=9 (verified mention-order compliant). (3) **Fisher (now Fig 1) reworked** — 2+1 layout, improvement factors on bars, inline SNR labels; **panel (b) van-Trees numbers STILL Sandy-gated**. (4) Abstract rewritten (passive, 247 words, no formulae, two-axis grading claim dropped); Theory cleaned (all 18 inline comments cleared, κ verified); Results/Discussion de-redundified; Methods classification rewritten (**DeLong dropped** → orphaned cite); **λ=0.1 smearing-artifact narrative purged throughout**. (5) Per-bin AUCs added to Table 1 + recompute.py; MRM compliance (IRB statement, Data Availability Statement w/ Zenodo-DOI+SHA-1 placeholders, keywords 7→6, title fixed). Committed **f0476cc**, pushed, Overleaf synced (resolved an Overleaf-branch conflict that was only a stray space). **OUTSTANDING (see blueprint):** Sandy CRLB validation (Fig 1b); **grading PARKED** — abandoned "detection vs grading axis" framing still in Fig 7 caption / Results-GGG / Discussion-histology, reconcile as one unit w/ Stephan + literature; ~24 inline `@patrick` comments left (Methods, Results spectra/pixel-wise, Intro); ~~recompute uncertainty-stat wiring~~ **DONE 2026-06-07** (`uncertainty_propagation()` in recompute.py → `results/biomarkers/uncertainty_propagation.csv`; reproduces 2.41×/1.27×/controlled-p=0.45/ΔAUC≈0 exactly; gold CSVs byte-identical); references pass (orphaned `delong` cite); Zenodo DOI + SHA-1 + ORCID + body word count (texcount on Overleaf); Fig 3/8 caption-sizing.
- **2026-06-04** — **Stefan figure-overhaul meeting (2026-06-03) digested → new §11.** 3-hour review of every figure ahead of the **hard Saturday 2026-06-06 submission deadline**. §11 holds: the per-figure change list, the **main-figure renumbering** (promote Fisher/CRLB → Fig 7, swap the simulation battery → Fig 8, demote the directional figure → SI as a new 2×4), a **shared-style spec** (build `visualization/paper_style.py` — fonts/legends/colours/bins are currently inconsistent across scripts), the **open decisions Patrick must lock before fan-out**, and the **Sandy dependencies** (CRLB van-Trees validity, Gibbs-vs-NUTS justification scope). Plan stays at the 10-item cap (9 figs + Table 1). **Nothing rebuilt yet** — this entry records the plan; figure fan-out begins once Patrick answers the §11 open decisions.
- **Update 2026-06-02:** **Pre-submission convergence session** (Stephan 06-01 email steer: don't over-rework figures, fix fonts). Done: captions drastically shortened across all figures (detail → text); **Fig 8 trimmed → `fig8_v3`** (recovery+noise 2×2, Gibbs boxes + ESS panel dropped, no WIP todo); **Fig 5 → `fig5_v4`** single grade ladder Normal/GGG1/GGG2/GGG≥3 (the two-boundary version was redundant — right panel reused left's ROIs); **Fig 2 → `fig2_v2`** (degenerate D=20 curve dropped → kills the AUC=0.000/0.224 red flags); **Fig 7 → `fig_directions_v2`** whole-ROI patient 9283 PZ via new `scripts/fig7_directions_roi.py` (decoded Stephan's .dat export, validated r=0.977; D=0.25 most direction-stable at 9% CV); Fig 9 → PZ (Stephan's 8640 answer); trace + robustness SI figs dropped; atlas confirmed correct (149/149, 20pp — the "not shown" was a stale Overleaf PDF + 1..17 comment, now 1..20). Theory Gibbs sentence corrected (Gibbs *works* — truncated-normal + conjugate inv-gamma σ²; NUTS chosen for mixing). Results ADC-sensitivity subsection reconciled to honest NUTS r=−0.79/−0.88 (the r≈−0.98 is a λ=0.1 regularizer property); direction aggregate stats added; Methods MAP λ corrected 0.1→1e-3 + zone rationale sharpened (Stephan: zonal contrast is a *normal*-tissue property); pixel/direction provenance split. Abstract reframed to 4-pillar "why ADC works" (dropped 100–8000× CRLB, |r|>0.93 sensitivity-vector, and MAP-acinar-underestimation claims). All inline @Stephan markers cleared. **Agenda → `notes/MEETING_PREP_2026-06-02.md`.** OPEN for meeting: Gibbs cut-vs-SI panel (Sandy), CRLB van Trees reframe (Sandy), Fig 7 zone/NUTS confirm, abstract sign-off. OPEN to write: Methods λ-sweep paragraph, Results 4-pillar restructure, ridge.yaml/CLAUDE.md λ drift (still say 0.1). **Overleaf NOT yet synced (repo only).**
- **Prior update:** 2026-06-01 — Fig 8 (method-validation) session: `fig8_v2` built via `scripts/fig8_validation.py` using the REAL pipeline classes + exact configs (no drift), style Patrick-approved. **BUT the Gibbs-vs-NUTS story is UNRESOLVED**: at 100k iters Gibbs converges and the recovery comparison is confounded (Gibbs handed true σ, NUTS infers it). Theory rewrite + figures.tex wiring are ON HOLD pending two experiments — (a) implement Sandy's inverse-gamma σ² Gibbs for a fair joint comparison, (b) reproduce Gibbs trapping on a bimodal config. See §6 Fig 8 row + §8 item 2. Prior entry: 2026-05-31 figure session — Fig 4 `fig4_v3` (bootstrap CIs; two-bin-detector finding F4c; DONE/Patrick-approved — `*` significance stars kept, caption clarifies star=cohort-coef-stability vs colour=per-ROI-identifiability as distinct axes, no sub-legends), S1 atlas colourblind-safe+hatch.
- **Target submission:** next week (was 2026-05-31). Today's goal: finalized draft for Sandy + Stephan review.
- **Status:** Figure scope locked (§6). Fig 3 + Fig 5 done and wired into the manuscript. Reworking remaining figures sequentially (Fig 1 → Fig 4 → ...), then manuscript text. Two-camps literature review running in background (→ `notes/lit_review_two_camps.md`). Old Fisher/CRLB Fig 8 dissolved into supplementary; new main Fig 8 = method-validation (spectrum recovery + joint noise inference).
- **Locked claims** (no longer under revision):
  - F1 — Tuned MAP (λ=1e-3) recovers log-normal spectra ≥0.98 fraction, matching NUTS. Confirmed with corrected solver.
  - F4b — ADC-sensitivity ≈ inverted-LR-coef r ≈ −0.98 elegance was a regulariser artifact (drops to −0.80 at tuned MAP and NUTS). Comparison kept in revised form (per Stephan/Sandy 2026-05-25 meeting).
  - F8 — Intermediate bins are data-limited, not prior-limited.
  - F9 (NEW) — MAP ridge branch in prob_model.py was projecting unconstrained Gaussian MAP onto non-negative orthant rather than solving constrained QP. Bug active on 58% of ROIs, p99 bin diff 0.37. Fixed 2026-05-26 to NNLS on augmented [U; √λI] system. Cohort re-fitted; ROI-scalar ADC-vs-discriminant r ≈ −0.97 robust.

**The manuscript narrative is reshaped around what NUTS uniquely contributes:** per-bin posterior uncertainty (with F6 coverage caveat). Tuned MAP is the point-estimate workhorse.

---

## 0. Manuscript finalization tracker (added 2026-06-21)

**Process.** Walk every prose unit below in document order; polish one, then mark it. Statuses: ☐ pending · 🔶 revised this session, needs a final pass · ✅ reviewed + finalized (the bar: like Methods→Classification, 2026-06-21). Captions in `figure_legends.tex` / `table_legends.tex` get polished alongside their parent float, not separately. Manuscript figure numbering (PROJECT_STATE reorder): 1 Fisher · 2 spectra · 3 validation · 4 ROC · 5 ADC-discriminant · 6 sensitivity · 7 GGG · 8 uncertainty · 9 pixel-wise.

**Progress: 1 ✅ / 33 body units** (+2 🔶 awaiting final pass).

### Abstract — `abstract.tex` (structured, 4 parts)
| St | Unit |
|----|------|
| ☐ | Purpose |
| ☐ | Methods |
| ☐ | Results |
| ☐ | Conclusion |

### Introduction — `introduction.tex` (4 paragraphs)
| St | Unit |
|----|------|
| ☐ | ¶1 ADC as PI-RADS cornerstone biomarker |
| ☐ | ¶2 ADC discards microstructure → multi-compartment models (IVIM/biexp/VERDICT, Langkilde) |
| ☐ | ¶3 Ill-posed inverse Laplace → Bayesian per-component uncertainty motivation |
| ☐ | ¶4 This work: contributions (1) identifiability (2) spectral-vs-ADC (3) sensitivity-alignment + pixel demo |

### Theory — `theory.tex` (5 subsections)
| St | Unit |
|----|------|
| ☐ | Signal Model |
| ☐ | Identifiability and Discretization |
| ☐ | MAP Estimation |
| ☐ | Bayesian Formulation |
| ☐ | ADC as a Spectral Functional |

### Methods — `methods.tex` (6 subsections)
| St | Unit |
|----|------|
| ☐ | Patient Cohort and MRI Acquisition |
| ☐ | Spectral Estimation |
| ☐ | ADC Computation |
| ✅ | Classification — finalized 2026-06-21 (standardization how+why; inner-6 representation added) |
| ☐ | ADC Sensitivity Analysis |
| ☐ | Pixel-wise and Direction-wise Spectral Estimation |

### Results — `results.tex` (6 subsections)
| St | Unit |
|----|------|
| ☐ | Diffusivity spectra and identifiability |
| 🔶 | Tumor detection — added mirror ablation + redundancy framing 2026-06-21; needs final pass |
| 🔶 | Why ADC works — rewrote the "two estimators" / sensitivity×contrast para 2026-06-21; needs final pass |
| ☐ | Spectral Decomposition by Gleason Grade Group |
| ☐ | Prediction confidence |
| ☐ | Pixel-wise Spectral Decomposition |

### Discussion — `discussion.tex` (7 paragraphs)
| St | Unit |
|----|------|
| ☐ | ¶1 Central finding: ADC ≈ optimal spectral discriminant |
| ☐ | ¶2 ADC explained not obsoleted; 3 added-value areas |
| ☐ | ¶3 Histology interpretation of grade-dependent shifts **[Stephan-gated]** |
| ☐ | ¶4 MAP vs NUTS complementary; posterior value is upstream |
| ☐ | ¶5 Fisher/CRLB identifiability; van-Trees decomposition **[Sandy-gated]** |
| ☐ | ¶6 Limitations (circularity, n=29, fixed bins, fixed TE/TD, single pixel-wise patient) |
| ☐ | ¶7 Future directions (other organs, amortized inference, T2 integration) |

### Conclusion — `conclusion.tex` (1 paragraph, 6 sentences)
| St | Unit |
|----|------|
| ☐ | Whole conclusion |

### Back matter (lower priority; polish with the parent float)
| St | Unit |
|----|------|
| ☐ | Figure legends — `figure_legends.tex` (9 figs) |
| ☐ | Table legend — `table_legends.tex` (Table 1) |
| ☐ | Supporting Information — `supporting.tex` |

---

## 1. Locked one-line thesis (under revision, see §4)

> **Bayesian spectral decomposition with per-bin posterior uncertainty resolves the compartment-volume mechanism underlying ADC's clinical success.** At the ROI level (N=149), the spectrum is diagnostically equivalent to ADC for tumor detection and grading. The spectrum's added value is (a) per-bin posterior uncertainty (well-calibrated on smooth ground truths; over-confident on δ-spectra — see F6), (b) a model-free fixed grid that — distinct from prior free-floating biexponential fits — expresses tissue change as a *fraction shift at canonical D values* rather than as a drift in fitted D, and (c) an explicit decomposition of the detection axis (outer bins) from the grading axis (intermediate + lumen bins) in the univariate per-bin profile.

**Title under consideration (already in repo, may need refresh):** "Why ADC Works: Bayesian Spectral Decomposition of Prostate Multi-b Diffusion MRI"

**Scope honesty.** All quantitative results are at ROI level (149 ROIs). Pixel-wise heatmap (Fig 9) is a single-slice feasibility demo (NUTS posterior-mean + per-voxel uncertainty) on one patient — *not* a delivered result. Per-voxel multi-compartment is already done by HM-MRI/VERDICT/rVERDICT/RSI/LWI with parametric constraints; **our novelty is model-free grid + Bayesian uncertainty + axis separation at the ROI level**, not per-voxel mapping.

**Open: what does NUTS uniquely contribute?** With F1 + F4b, the case for NUTS over tuned MAP is narrower than the manuscript currently claims. Working answer (to confirm with Stephan): NUTS provides per-bin posterior σ_j that MAP at any λ cannot. The "MAP smears 35%" and "ADC sensitivity ≈ inverted LR coefs" narratives are largely λ-dependent and should be soft-pedalled.

---

## 2. Frozen headline numbers (post-MAP-fix, regenerated 2026-05-26)

After F9 (MAP solver fix) the full recompute pipeline was rerun. New values:

| Method | PZ tumor-vs-normal (n=81) | TZ tumor-vs-normal (n=68) | GGG≥3 (n=29) |
|---|:---:|:---:|:---:|
| ADC raw rank | **0.951** | **0.979** | 0.811 |
| ADC LR LOOCV | 0.940 (C=1) | 0.964 (C=1) | 0.778 (C=1) |
| MAP Full LR (tuned λ=1e-3) | 0.917 (C=1) | 0.952 (C=1) | 0.767 (C=1); **0.878 (C=10)** |
| NUTS Full LR | 0.926 (C=1) | 0.923 (C=1) | 0.772 (C=1) |

ROI-scalar ADC-vs-discriminant correlation (Fig 3, post-fix):

| Zone | NUTS | tuned-MAP (λ=1e-3) |
|---|---|---|
| PZ | r = −0.977 | r = −0.972 |
| TZ | r = −0.981 | r = −0.967 |

ADC reference is std-ADC (b ≤ 1000, PI-RADS-compliant). The Fig 3 correlation is robust to method and λ — confirmed across NUTS, tuned MAP, and (previously) MAP@λ=0.1.

**Cohort:** 56 patients, 149 ROIs (40 tumor, 109 normal), 29 with valid GGG (PZ 21, TZ 8, low=20, high=9). SNR median 303 (IQR 176–478). 15 b-values up to 3500 s/mm².

---

## 3. Locked findings — do not re-investigate

**F1 — MAP-vs-NUTS gap is mostly a λ=0.1 artifact.** Experiment A, `scripts/map_lambda_sweep.py` + `scripts/map_lambda_bwh.py`. Original (manuscript) simulation ran MAP and NUTS both at λ=0.1, where MAP loses 0.34–0.76 of δ-mass and dumps it into 2–3 neighbours. Adding a λ ∈ {1e-6, ..., 3} dimension to MAP changes the story:

- **Log-normal GTs** (closest to real prostate): MAP @ λ ≈ 1e-3 recovers 0.98–0.99 of mass; *MSE actually slightly better than NUTS* (0.003 MAP vs 0.014 NUTS, GT-H @ SNR=400).
- **Bimodal GTs**: MAP @ λ ≈ 1e-4 recovers 0.75–0.86; NUTS recovers 0.93–0.97. **NUTS retains 10–15 pp advantage.**
- **δ-GTs** (unrealistic for prostate): MAP @ λ ≈ 1e-4 recovers 0.73–0.84; NUTS 0.96–0.99. **NUTS retains ~15 pp advantage.**
- **On BWH data** (`scripts/map_lambda_bwh.py`): PZ-normal D=3.0 median — MAP@0.1 = 0.239 (51% below NUTS 0.484); MAP@1e-4 = 0.426 (12% below). TZ-normal D=3.0: MAP@0.1 = 0.206 (31% below NUTS 0.299); MAP@1e-4 = 0.296 (1% below). On D=0.25, MAP@1e-4 slightly overshoots NUTS (+9% PZ-tumour, +7% TZ-tumour).

Implication: the "MAP underestimates lumen by ~35%" Discussion paragraph as currently written is **largely an artifact of λ=0.1**. MAP at tuned λ is a viable primary point estimator. The remaining and irreducible NUTS contribution is **per-bin posterior σ_j**, not better point estimates.

Files: `results/simulation/map_lambda_sweep{,_summary}.csv`, `results/simulation/map_lambda_sweep_{fraction,mse}.png`, `results/biomarkers/map_lambda_bwh{,_summary}.csv`.

**F2. Intermediate bins carry no independent tumor-vs-normal signal.** `scripts/bin_information_sweep.py` (1782 rows = 198 feature sets × 3 zones × 3 C values, output `results/biomarkers/bin_information_sweep.csv`): no subset beats reference {μ_D=0.25, μ_D=3.00} with CI separation. Intermediate-only LR collapses by ΔAUC ≈ −0.13 in every zone. σ_D=0.25 in TZ marginally helps (Δ +0.010, CI overlaps).

**F3. Detection and grading live on approximately orthogonal axes — but only via univariate Spearman ρ.** `scripts/ggg_continuous_sweep.py`, `results/biomarkers/ggg_continuous_sweep.csv`. Per-bin ρ vs continuous GGG (pooled N=29, Bonferroni α=0.0031):
- μ_D=0.50: +0.565 (Bonferroni-sig)
- σ_D=0.50: +0.566 (Bonferroni-sig)
- ADC: −0.546 (Bonferroni-sig)
- μ_D=0.25: +0.437 (not Bonferroni-sig)
- μ_D=3.00: −0.16 (n.s.)

Detection lives at outer bins (D=0.25, D=3.00); grading at intermediate (D=0.50) + lumen (D=2.0) bins. The univariate picture is the clean evidence.

**F4. The LR-coefficient version of axis-separation is NOT clean at N=29 (Exec 4P1).** `scripts/lr_coef_decomp.py`, `results/biomarkers/lr_coef_decomp{,_cross}.csv`. cos(w, D_vec) tissue-only is small with CIs straddling 0 for both tasks. cos(w_T, w_G) is +0.34 [−0.17, +0.75] — not orthogonal in LR-vector space. Wide CIs from bin collinearity + N=29. *Implication:* lead F-new-2 with the univariate Spearman ρ profile (F3 above), report cos(w, D_vec) only as a secondary high-variance check. Do **not** headline LR-vector orthogonality.

**F4b — ADC-sensitivity ≈ inverted LR coefficient elegance was a regulariser artifact.** Experiment C, `scripts/adc_sensitivity_at_tuned_lambda.py`, `results/biomarkers/adc_sens_vs_lr_tuned_lambda.csv`. Vector-level test: 8-element ∂ADC/∂R_j (at avg_tumor / avg_normal operating point) vs 8-element LR coef w (LR fit on all ROIs in zone). λ sweep on PZ tumour operating point:

| Estimator | r_pearson |
|---|---|
| MAP λ=1e-4 (best on BWH) | **−0.85** |
| MAP λ=1e-3 | −0.76 |
| MAP λ=1e-2 | −0.97 |
| MAP λ=0.1 (manuscript) | **−0.98** |
| MAP λ=0.5 | −0.95 |
| **NUTS** | **−0.79** |

Sanity check at λ=0.1 reproduces the paper's r=−0.979 exactly. The r ≈ −0.98 elegance holds *only* in a narrow band of moderately-high λ where MAP smears mass into intermediates, which makes the LR coefficient profile across D smoother and more monotonic-like. ADC sensitivity ∂ADC/∂R_j is by construction monotonic in D, so smoother LR profiles correlate more strongly. **At tuned λ and at NUTS, r ≈ −0.80** — still anti-correlated, still meaningful directionally, but not "near-perfect." **Distinct from** the ROI-level scalar ADC-vs-discriminant correlation (n=81/n=68, r ≈ −0.98 with bootstrap CI) that the abstract claims — the manuscript currently muddles the two.

**F4c — Bootstrap on the per-bin LR coefficients confirms the discriminant is a TWO-BIN detector (2026-05-31, Patrick feedback on Fig 4).** Implemented in `scripts/fig4_lr_coefs_and_sensitivity.py` (`fig4_v3`): resample ROIs within zone w/ replacement (2000×), refit the standardised tumor-vs-normal LR (NUTS feats, C=1, balanced — same as Fig 3), per-bin 95% CI. **Only D=0.25 (positive) and D=3.0 (negative) have CIs excluding zero, in BOTH zones, 100% sign-stable.** PZ: every intermediate bin (0.5–2.0) + D=20 straddle zero. TZ: marginal exceptions D=0.75 (CI [0.03,0.81]) and D=1.0 (CI [0.13,1.03]) — small, in poorly-identified bins, 98–100% sign-stable (the "weird TZ uptick" Patrick flagged — marginally real, not pure artifact, but not to be leaned on). r(w,∂ADC/∂R) bootstrap CI: **PZ −0.79 [−0.89,−0.50]** (wide!), **TZ −0.88 [−0.95,−0.70]**. *Interpretation for prose:* the whole-profile r is carried by the two outer bins; intermediate detection coefficients are individually unstable (bin collinearity + poor identifiability) → **do NOT interpret intermediate-bin detection signs**. This sharpens F2 (no independent intermediate detection signal) and is consistent with F4 (don't headline LR-vector structure). The intermediate bins' grading signal is a SEPARATE, univariate finding (F3 / Fig 5), not a detection-coefficient claim — keep the two firmly distinct in the text.

*Non-redundancy (Patrick worry 2026-05-31): Fig 4 ≠ Fig 2.* **Fig 2 = performance ablation** (2-fraction {0.25,3.0} LR ≈ full LR ≈ ADC — the ablation is ALREADY there, don't repeat it in Fig 4). **Fig 4 = per-bin mechanism/anatomy** (WHICH bins carry stable weight + how that maps to ADC sensitivity + identifiability). **Fig 3 = ROI-level score equivalence.** Results prose should lead Fig 4's paragraph with the mechanism and *reference* Fig 2 for performance, not re-prove it. Honesty point Patrick insisted on: do NOT call the middle "worthless" — TZ 0.75/1.0 marginally significant + grading signal = "some juice"; reason we report the full spectrum. v3 styling refined per feedback: short titles + in-panel r box, taller symmetric y (D=0.25 CI not clipped), `*` stars on CI-excludes-0 bins.

**F5. Diagnostic equivalence is triangulated.** Four independent tests all agree: no spectral feature meaningfully lifts AUC over std-ADC at N=29 for grading.
- Partial Spearman ρ(μ_D=0.50, GGG | std-ADC) = +0.42, p=0.026 *uncorrected*, does not survive Bonferroni. (`scripts/partial_corr_ggg.py`)
- 2-feat LR + paired DeLong: all ΔAUC ≤ 0, all DeLong p ≥ 0.17. (`scripts/two_feature_lr_vs_adc.py`)
- ADC variants: std-ADC is fair (DKI_D narrowly best at +0.003 detection, ρ=−0.555 grading). (`scripts/adc_variants_sweep.py`)
- Spectrum first moment (spec_M1): PZ AUC = 0.71 vs std-ADC 0.94 — spectrum-then-collapse is information-lossy as a scalar.

**F6. NUTS posterior coverage caveat.** Ground-truth simulation at SNR=400 (`results/simulation/sim_summary.csv`): NUTS 90% credible-interval coverage ≈ 0 on δ-spectra (GT-A...GT-D), 0.01 on bimodal, 0.07–0.12 on trimodal, 0.77–0.87 on log-normals. Well-calibrated on smooth ground truths; over-confident on concentrated ones. Cause: HalfNormal(σ_R=√10) pulls each bin's posterior toward zero; when truth is concentrated, posterior contracts at a slightly biased location whose 90% interval is narrower than the bias. **Must be flagged in Discussion as a limitation of the "calibrated per-bin uncertainty" claim.**

**F7. σ calibration is not the bottleneck.** Investigation A (2026-05-16, in `notes/archive/notes_session_2026-05-14_classifier_deepdive.md`): NUTS σ is pulled *down* from HalfCauchy prior median by data. Pinning σ at Stephan's legacy formula narrows outer-bin posteriors 20–40% but does not unlock middle bins.

**F8 — Half-normal R-prior is not shrinking intermediates either.** Experiment B, `scripts/wider_prior_check.py`, `results/simulation/wider_prior_check{,_summary}.csv`. Re-ran NUTS on 7 representative ROIs (PZ tumour, PZ normal, TZ tumour, TZ normal) at σ_R ∈ {3.16, 10, 30, 100} — up to 30× wider than manuscript's σ_R = 3.16. Across all 8 bins, posterior R_mean changes by ≤3% across σ_R; intermediate-bin CVs stay at 0.75–0.88; outer-bin CVs stay at 0.05–0.27. **The "intermediate bins not identifiable" Discussion point is robust to prior choice — a genuine data limit at this b-grid and SNR.**

**F9 — MAP solver bug (Sandy 2026-05-25, fixed 2026-05-26).** The ridge MAP branch in `src/.../models/prob_model.py` and the duplicate path in `biomarkers/recompute.py` were computing the unconstrained Gaussian MAP `(UᵀU + λI)⁻¹Uᵀs` then clipping with `np.maximum(·, 0)`. This is NOT the constrained MAP whenever the unconstrained optimum lies outside the non-negative orthant (Sandy's 2-D Gaussian counter-example in meeting). Empirical impact across 151 ROIs at λ=1e-3: clip activates on 58.3%; max negative coefficient before clip is −0.48; median per-ROI max-bin difference 0.004, p90=0.08, **p99=0.37**. Fix: NNLS on augmented `[U; √λI]` system, equivalent to `argmin_{R≥0} ‖UR − s/S0‖² + λ‖R‖²`. Cohort and simulation re-run 2026-05-26 with fix: F1 preserved (log-normal MAP@λ=1e-3 still frac=0.985–0.99, matching NUTS); Fig 3 ROI-scalar correlations preserved (PZ MAP r=−0.972 vs −0.975 pre-fix; TZ MAP r=−0.967 vs −0.982 pre-fix). Bin-level changes for individual tumor ROIs in the tail can be substantial. Eq. 5 rewritten in `theory.tex` lines 65–80 as constrained QP.

**F10 — Bayesian CRLB resolves "factor-2000" gap (2026-05-26 diagnostic).** Stephan and Sandy flagged the magnitude of the unconstrained-CRLB vs NUTS posterior std gap (sometimes 2000×) as suspicious. Cause: comparison is apples-to-oranges. Diagnostic at SNR=303 (cohort median, NOT the hard-coded 150 in current Fig 8 script): unconstrained CRLB ranges 0.08–131; Bayesian CRLB (van Trees with HalfNormal prior λ=0.1) ranges 0.028–2.33 (2 orders of magnitude tighter); empirical NUTS std ranges 0.014–0.121 (another 5–50× tighter due to non-negativity + one-sided HalfNormal). Fix: re-build the figure with 3-bar comparison and reframe caption. See `notes/CRLB_NOTE_FOR_SANDY.md` for the full write-up and diagnostic script. Sandy will validate the van Trees derivation.

**F11 — Meeting outcomes 2026-05-25 (Sandy + Stephan).**
- Sensitivity analysis kept (per Stephan/Sandy): not headline but useful intuition for readers. Replaced old Fig 4 vector-correlation with new LR-coefficient-per-bin + comparison-to-∂ADC/∂R figure (now Fig 4, agent in flight).
- Spectrum-by-GGG figure (created in-meeting) is excitedly received. Promoted to main as Fig 5: shows normal baseline + GGG=1 + GGG≥2 spectrum overlay. Histology interpretation (Stephan): D=3.0 = lacunae (free-water pools 100–300 μm); GGG=1 displacement spares D=2.0 glandular; GGG≥2 collapses both D=3.0 and D=2.0, mass shifts to D=0.25. **This replaces the noisy N=29 ROC subplot in old Fig 2.**
- Fig 7 directional needs full rework: previous version used only 1 patient; Stephan's tarball has 10. New plan: 1 representative patient as figure + aggregate per-direction-variance stats reported textually for all 10.
- Fig 8 (Fisher) updated with Bayesian CRLB framing (see F10) and combined with promoted simulation comparison (was F-new-1).
- Methods/results restructure: drop GGG ROC entirely (N=29 too small for reliable AUC); use Spearman ρ + spectrum-by-GGG instead.
- Stephan asked for the spectrum-by-GGG figure for an unrelated grant submission. Sent 2026-05-26.

---

## 4. Stephan's 2026-05-22 response and Patrick's open questions

### Stephan's three points (verbatim summary)

1. **Biexponential reduction is a contribution, not a retreat.** Free-floating biexp fits in prior prostate work absorbed compartment mixtures into intermediate-D drift; our fixed-grid Bayesian decomposition expresses change as fraction shift at canonical D values. Novel positioning.

2. **"Why ADC works" is a real contribution.** "Nobody has revealed why ADC performs so well." Three-reason thesis (biological collinearity, estimator efficiency floor, label-resolution bottleneck) stays as Discussion centerpiece.

3. **Circularity concern is overstated.** Clinicians draw ROIs from T2W + DWI visual contrast, not from quantitative ADC values. Side note: Stephan's T2W+DWI vs T2W+DWI+ADC AI experiment on ~1000 patients showed dropping ADC *improved* performance.

He also wants: ROI-level directional dependence data (re-asked — answer: already in Fig 7 since May; confirm), a meeting (today), soon submission.

### Patrick's open discussion points for today

**Q1 — Framing collision.** "Why ADC works" (Stephan) and "compartment-volume mechanism" (Patrick's draft) feel like the same thing relabelled. Settle on one before prose.

**Q2 — MAP demotion is now LESS justified.** Per F1, tuned MAP ≈ NUTS on realistic spectra. The "MAP smearing 35%" Discussion paragraph mostly dissolves. Three options:
- (a) Keep MAP as primary point estimator at tuned λ; NUTS for uncertainty story only.
- (b) Keep NUTS primary; explain that "MAP smearing" is conditional on λ.
- (c) Compute MAP-tuned-LR AUCs (~5 min) before deciding.

**Q3 — ADC-sensitivity panel (Fig 4).** Per F4b, the r ≈ −0.98 elegance was an artifact. Three honest framings:
- (a) Conservative: drop Fig 4; note r in [−0.95, −0.76] across regularisations.
- (b) Medium: keep Fig 4 with both MAP-tuned and NUTS panels + caveat.
- (c) Permissive: keep MAP λ=0.1 as the headline, add one disclosure sentence.

**Q4 — 8-bin grid justification.** Stephan's fixed-grid-vs-free-floating-biexp framing: the 8-bin grid is the *test*, the 2-bin collapse is the *result*. Confirm Patrick is articulating it as intended.

**Q5 — Half-normal prior (F8 closed).** Quickly note: re-checked with 30× wider prior; intermediate bins still wide. Data limit confirmed, no manuscript change needed.

**Q6 — Fig 7 directional data.** Stephan re-asked, but it was already done in May (one normal + one tumour ROI, patient 9283, per-direction NUTS + MAP, from his tarball). Confirm match.

**Q7 — Sandy.** No response yet to 2026-05-16 email. Even a one-liner.

**Q8 — Selection bias.** Still goes in Limitations regardless of (3). Stephan's "no circularity" reassurance is reasonable but ROIs were still drawn on mpMRI.

---

## 5. Manuscript TODO list — narrative (to revise after meeting)

Categorised. "P" = Patrick decides alone. "S" = needs Stephan/Sandy input. *Some items below may shift materially after today's meeting.*

### 5a. Abstract — REWRITE (P, after meeting locks framing)

Current `paper/sections/abstract.tex` hinges on:
- "optimal multi-component spectral classifier for tumor detection" → soften
- "ADC sensitivity vector aligned with the learned classification direction at |r| > 0.93" → reconsider given F4b
- MAP-vs-NUTS divergence at D=3.0 (acinar lumen) → soften per F1; "ridge smoothing" was conditional on λ=0.1

**Proposed new abstract spine (Path A', subject to meeting):**
1. **Background.** ADC is the de facto prostate dMRI biomarker. Compartment-volume models explain *why* it works but are not directly measured per voxel.
2. **Methods.** Bayesian spectral decomposition on 8-bin grid with HalfCauchy noise prior, NUTS inference for per-bin posterior uncertainty. Comparison to closed-form ridge MAP at matched and tuned regularisation.
3. **Results.** (a) 2-feature spectral classifier matches ADC for detection (PZ 0.933 vs 0.951, TZ 0.937 vs 0.979). (b) Spectral grid expresses tumour/normal change as fraction shift at canonical D values (D=0.25 epithelium proxy ↑, D=3.0 lumen proxy ↓), distinct from prior free-floating biexp fits where intermediate D-values drift. (c) Per-bin posterior uncertainty: only D=0.25 and D=3.0 well identified (CV < 0.4) regardless of prior width; intermediates CV > 0.8, robust data limit. (d) At N=29 valid GGG, spectrum and ADC are diagnostically equivalent for grading — triangulated across four tests.
4. **Conclusion.** Bayesian spectral decomposition mechanistically explains ADC's clinical success while providing per-bin uncertainty and quantifying which compartments are reliably recoverable. Tuned MAP and NUTS produce equivalent point estimates on realistic spectra; the Bayesian gain is calibrated uncertainty per compartment, not better point accuracy.

### 5b. Introduction — INSERT FRAMEWORK UPFRONT (P)

- Open second paragraph with Chatterjee 2015 compartment-volume mechanism.
- Cite Wang Y 2024 (`assets/s00261-024-04684-z.pdf`) as the closest published version of our mechanistic claim.
- Acknowledge per-voxel multi-compartment methods (HM-MRI, VERDICT, rVERDICT, RSI, LWI). Position our work as model-free + Bayesian uncertainty at ROI level.
- Biopsy-replacement motivation → demote to Future Directions, not Intro lede.
- Mulkern 2006 citation already added (2026-05-23).

### 5c. Theory — DEMOTE FISHER (P)

- Move Fig 8 (Fisher) → supplementary.
- Remove "the grid is informed by ΔD ∝ D^(3/2)" implicit claim.
- 4 @Stephan cosmetic comments already resolved (2026-05-23). 2 figure-ordering markers resolve when Fisher moves to SI.
- Audit MAP Eq. 5 prior consistency (3rd-pass Exec #5/#6) — still pending.

### 5d. Methods — ADD λ-SWEEP PARAGRAPH (P)

- **New paragraph:** simulation methodology + λ sweep on the MAP arm. Point at `results/simulation/bias_heatmap.png` and `results/simulation/map_lambda_sweep_*.png`. ~150 words. **Framing must be honest:** at the original λ=0.1, MAP smears mass; at λ ≈ 1e-3, MAP recovers realistic spectra as well as NUTS but with ~10–15 pp gap on bimodal/concentrated GTs. NUTS's irreducible contribution is per-bin posterior σ.
- **New paragraph:** ADC variants sweep methodology. Briefly justify std-ADC as primary reference.
- 5 @Stephan cosmetic comments resolved 2026-05-23.
- **Pixel/direction provenance — done** (Stephan's tarball, May 2026); confirm Fig 7 at meeting.

### 5e. Results — REWRITE AROUND FOUR PILLARS (P, after meeting locks framing)

- **Pillar 1 — Tumour detection.** 2-feat NUTS-LR {D=0.25, D=3.00} matches ADC. MAP-tuned-LR as comparison (NOT just MAP@λ=0.1).
- **Pillar 2 — Fraction-shift mechanism + axis separation.** Lead with per-bin Spearman ρ profile (F3). Map to Bourne 2018 compartments. Report LR-vector cosines (F4) only as secondary check with wide CIs.
- **Pillar 3 — Diagnostic equivalence as triangulated finding.** Four tests (F5).
- **Pillar 4 — Methodology validation, with the honest λ-sensitivity story.** The original "MAP smearing" narrative needs revision: at tuned λ, MAP and NUTS converge on realistic spectra. The Bayesian gain is per-bin σ_j calibration (with the F6 coverage caveat), not point-estimate accuracy.

Cut from current `paper/sections/results.tex`:
- "ADC and the Spectral Discriminant" subsection — fold into Pillar 3.
- "ADC Sensitivity Analysis" subsection — **demote or drop** per Q3 / F4b.
- "LR coefficients become unstable" wording (resolved).
- "intermediate bins are not identifiable" repetition — say once, link to F8.
- 3 @Stephan "Taken up in Discussion" forward-references — rewrite in Pillar restructure.

### 5f. Discussion — REVISED 4-POINT NOVELTY LIST (P)

1. **Opener — what the spectrum uniquely contributes.** Per-bin posterior uncertainty (with F6 coverage caveat for honesty); model-free 8-bin grid recovering compartment-volume story (cite Bourne 2018); fraction-shift framing distinct from free-floating biexp (cite Mulkern 2006, VERDICT, IVIM). **Drop "MAP smearing caveat" as a centerpiece** — it's now an aside about regulariser tuning, not a core finding.
2. **Why ADC succeeds — three-reason thesis (Stephan-endorsed).** Biological collinearity (Chatterjee 2015 ρ=−0.78); estimator efficiency floor (spec_M1 underperforms direct ADC by ΔAUC ≈ 0.23); label-resolution bottleneck (GGG is coarse, N=29).
3. **What the spectrum adds — at ROI level.** Per-bin σ_j (the irreducible Bayesian gain). Model-free decomposition without fixed-compartment assumptions. Fraction-shift framing. Honest comparison to HM-MRI/VERDICT/rVERDICT/RSI.
4. **DKI kurtosis as parametric analog.** DKI_K ρ=+0.476 vs GGG aligns with intermediate-bin grading signal.
5. **NUTS coverage caveat** (F6).
6. **MAP/NUTS regulariser sensitivity** (F1, F4b). Honest paragraph: tuned MAP and NUTS converge on realistic spectra; the difference at λ=0.1 was a regulariser smoothing artifact, and the elegant ADC-sensitivity ≈ LR-coef correlation depends on the same smoothing. Worth a transparent disclosure.
7. **Limitations.** Selection bias, N=29, no whole-mount histology, NUTS coverage on δ-spectra, fixed TE/TD.
8. **Future directions.** Computationally tractable Bayesian per-voxel inference; finer reference labels; image-guided biopsy correlation.

### 5g. Conclusion + Abstract harmonisation (P)

Match Conclusion language to revised Abstract. Soften "ADC is a near-optimal linear projection" to reflect F4b.

---

## 6. Figure plan — REWORKED 2026-05-31 (figure-scoping session)

**Narrative pillars** (every figure must serve one):
- **P1 — Why ADC works (the collapse).** Spectrum reduces to two well-identified outer compartments (D=0.25, D=3.0) moving together → one scalar (ADC) captures detection. Figs 1, 2, 3, 4. Cite Wang Y 2024 / Wang Q 2018 (MC≈ADC precedent); reconcile RSI/VERDICT via the PI-RADS-ADC-reference disambiguation (lit review running → `notes/lit_review_two_camps.md`).
- **P2 — Beyond the outer bins (grading/biology).** Spectrum shifts with Gleason grade in intermediate+lumen bins (D=0.5, D=2.0) where ADC is least sensitive — real but identifiability-limited. Fig 5.
- **P3 — Uncertainty-aware diagnostics (exploratory).** Propagate NUTS posterior through the classifier → calibrated cancer probability + uncertainty (Sandy's idea). Fig 6.
- **Cross-cutting — identifiability.** NOT its own figure: woven into Fig 4 (bar colouring) + supplementary individual spectra (S1).

**Main figures (9 + 1 table = 10, AT CAP):**

| Fig | Status | What it shows / decision |
|---|---|---|
| 1 | ✅ v4 (iterating) | `fig1_v4`: cohort box plots tumor/normal × PZ/TZ. MAP green-hatched / NUTS orange-solid (style + colour, grayscale-safe). Mean within-ROI NUTS CV as light grey annotation. PZ/TZ titles, legend top, no title, shared y. (Box-plot concept kept — NOT the fig1_v3 8-ROI concept.) |
| 2 | ✅ DONE | `fig2_v1`: PZ + TZ detection ROC (GGG dropped). **All curves via the same LOOCV pipeline** (apples-to-apples fix — kills the in-sample raw-rank vs out-of-sample LR asymmetry that made raw singles look better). Thick: ADC (black), 8-fraction spectral LR (orange solid), 2-fraction {D=0.25,3.0} LR (orange dashed), NUTS features. Outer singles D=0.25 (teal) + D=3.0 (brown) highlighted; other 6 = faint grey bundle (D=20 is a degenerate near-empty-bin artifact, kept unlabelled). Top identity legend (2 rows of 3, not literally 1 row — 6 entries). **Empirics back the collapse:** ADC ≳ 2-feat ≳ 8-feat > outer singles ≫ grey; 2-feat AUCs (PZ 0.933 / TZ 0.937) reproduce abstract. AUC numbers → Table 1. Generator `scripts/fig2_roc_detection.py`. |
| 3 | ✅ DONE | ADC vs discriminant, 2×2 (PZ/TZ × NUTS/MAP), shared axes, legend top, no title. `fig3_v3`. |
| 4 | ✅ DONE (v3) | `fig4_v3`: 2 panels (PZ, TZ). Standardised LR coef bars (same fit as Fig 3, NUTS, C=1, balanced) norm to max\|w\|=1, **with 95% bootstrap CIs** (2000×); coloured+hatched by within-ROI CV (colourblind-safe, shared module `visualization/identifiability.py`). −∂ADC/∂R as charcoal **diamonds (NO line)** with their own bootstrap CIs (tight — sensitivity is precise). CV mean±std strip below. Estimator (NUTS) explicit. Short titles + in-panel r box (w/ bootstrap CI); taller symmetric y; **`*` star = coef CI excludes 0** (significance). **Reframed to "two-bin detector"** (F4c): PZ only D=0.25,3.0 starred; TZ also marginal 0.75/1.0 (kept as honest "some juice", not over-read). Caption clarifies star (cohort coef stability) vs colour (per-ROI identifiability) are **distinct axes**. NO sub-legends in subplots (single top fig.legend). Wired into figures.tex (`\label{fig:sensitivity}`). |
| 5 | ✅ DONE (`fig5_v3`, 2026-06-01) | Spectrum by GGG, **2 panels** (full \textwidth, legends on top, Fig 2/3 styling): (left) emergence GGG=1 vs ≥2; (right) aggressiveness GGG≤2 vs ≥3, normal as shared gray baseline in both. **Tells the detection-vs-grading axis dissociation:** lumen D=3.0 collapses at tumor onset then saturates (detection axis); restricted D=0.25 ↑ monotonically + glandular D=2.0 collapses with grade (grading axis, the intermediate bins). Caption + Results subsection rewritten. Supersedes single-panel `fig5_v2`. |
| 6 | ✅ v1 (first draft, may iterate) | `fig6_v1` (`scripts/fig6_uncertainty_classifier.py`). **Uncertainty-aware classifier** (Sandy's "probability of cancer" idea). **Recompute fixed:** old ISMRM fig drew error bars = mean per-bin posterior std ×2 (feature-space heuristic). Now PROPAGATES all 8000 NUTS draws of each held-out ROI's spectrum through the LOOCV LR (same Fig-2 pipeline, 8 NUTS feats, C=1) → posterior *distribution* over P(tumor) + real 90% CI. **GGG panel DROPPED** (consistency w/ Fig 2, N=29 too small). Layout = **2×2**: PZ + TZ sorted-prediction (top), correct-vs-misclassified CI-width box (bottom-left), **4th quadrant empty + minimal legend on TOP** (Patrick pref, matches Fig 2/3 convention). **Headline = misclassified CI 2.4× wider** (pooled, p=0.003; was 1.2–1.3×). **ρ(dist, CI width)=−0.94 DEMOTED** (largely logistic-link geometry — see decomposition below; caption now only says "intervals widest near boundary, partly link geometry", no ρ number). **Logit-space decomposition (the honest answer to "how much is geometric"):** in logit space (removes sigmoid P(1−P) slope), ρ(dist, spread) drops −0.94→−0.29 and misclassified ratio drops 2.4×→**1.3×** — so boundary-widening is mostly geometric, but misclassified excess is **partly genuine** (1.3× in logit space, amplified to 2.4× in prob space b/c errors cluster near boundary). This is now a "slight discussion point" in discussion.tex. **Confident-misclassified failure mode** = low-grade(GGG=1)/ungraded tumors spectrally ≈ normal (new62,new52 PZ; new14,new46 TZ) — contrast limit, ties to Fig 5, not miscalibration. **Wired in:** figures.tex (caption), discussion.tex (para incl. logit-geometry), **results.tex new `\subsection{Uncertainty-aware Classification}`** (first-draft Pillar-3 para — FOLD into the eventual §5e Pillar-3 restructure). |
| 7 | REWORK (MAIN) | Direction independence (trace-averaging validation). 1 representative patient + aggregate per-direction stats in text. Needs Stephan's 10-patient tarball. |
| 8 | ⚠️ DRAFT BUILT, framing UNRESOLVED (2026-06-01) | `fig8_v2` (`scripts/fig8_validation.py`). 2×3: (a-c) recovery on normal-like / tumour-like / δ@D=0.75 (truth=black marks, NUTS+Gibbs=box plots, tuned-MAP=green ×); (d) joint σ̂ over 25 reps vs true σ; (e) ESS/draw NUTS vs Gibbs on δ. Driven by the REAL pipeline classes (NUTSSampler/GibbsSamplerClean/compute_map_spectrum) w/ exact nuts.yaml+gibbs.yaml+ridge.yaml configs — no drift. **Style locked & Patrick-approved** (slate-blue Gibbs OK). **BUT the Gibbs-vs-NUTS scientific story is NOT settled — do NOT write Theory yet.** See §8 item 2 for the open issues + plan. **WIRED AS DRAFT 2026-06-01** into `figures.tex` as main Fig 8 (`fig:validation`, caption lists all configs + a visible red `\todo{}` WIP note). Old **Fisher figure MOVED to `supporting.tex`** (kept `\label{fig:fisher}` → the 7 in-text refs in theory/results/discussion resolve to a supp "Figure S‹n›", no breakage). Main = 9 figs + Table 1 = 10 (MRM cap). Full Fisher excision needs the Theory Fisher/CRLB-subsection rewrite (deferred). **Overleaf sync: figures.tex + supporting.tex + new asset `paper/figures/fig8_v2.pdf`.** |
| 9 | ✅ DONE (2026-06-01) | `fig9_v1` (`scripts/fig9_pixelwise.py`). 3×3 NUTS heatmap grid: A anatomy / B,C restricted+free-water fractions / D ADC (grayscale, conventional tumor-dark) / E,F 2-bin + 8-bin discriminant scores (absolute, decision-boundary-centred) / G windowed 2-bin score / H,I 2-bin + 8-bin per-voxel score uncertainty. Old subplot D (LR coefs) + signal-decay QC dropped. NUTS throughout (Fig 2 recipe, C=1). **8-bin score uncertainty ≈1.8× the 2-bin** (it weights the unidentifiable intermediate bins); both scores reproduce ADC (r=−0.93/−0.96). **Per-voxel tumor-skew finding (§8).** Wired into figures.tex (A–I caption) + methods.tex (per-zone justification) + discussion.tex (limitation). Anonymised; **PZ classifier PROVISIONAL pending zone label (§8 @Stephan).** |
| Table 1 | ✅ REWORKED | Detection-only AUC table (GGG column DROPPED — n=29 too imprecise; grading → Fig 5 + Spearman). 6 rows: ADC raw-rank, ADC LR, 8-fraction LR (MAP/NUTS), **2-fraction {0.25,3.0} LR (MAP/NUTS)** grouped by midrules. **Bootstrap 95% CIs** replace Hanley–McNeil SE (now the manuscript-wide interval convention; DeLong for paired AUC tests). Fixed 2 pre-existing bugs: stale MAP Full LR (was pre-MAP-fix 0.919/0.945 → 0.917/0.952) + GGG-column SEs understated ~2×. Source: `auc_table.csv` (regenerated; now has auc_lo/auc_hi + 2-feat rows). |

**Supplementary:**
- S1 — ✅ **DONE (2026-05-31).** `scripts/figS1_all_roi_spectra.py` → `figS1_all_roi_spectra.pdf` (20 pages, 2 cols, **4 ROIs/page = taller panels / more y-resolution** per Patrick), embedded via `\includepdf` (needs `\usepackage{pdfpages}`). **Box plots of the NUTS posterior** per bin (median + IQR box + 5–95% whiskers, no fliers), box face coloured **+ hatched** by within-ROI CV (**colourblind-safe purple+hatch, shared module `visualization/identifiability.py` — identical scheme to Fig 4**), tuned-MAP green ×. Layout bug fixed (legend + y-axis label no longer clipped). Titles carry **public reproducibility IDs** (patient_id + zone/tissue + GS/GGG) — map to released signal_decays.json / metadata.csv; companion `figS1_roi_key.csv` lists anatomical_region + SNR. Posterior draws read live from `results/inference_bwh_backup/*.nc`. Caption in supporting.tex rewritten.
- S2 — NUTS posterior diagnostics (trace, R̂). NEW DESIGN (current arviz default rejected).
- S3 — MAP λ-tuning: fraction-of-mass recovered vs λ (DROP the redundant MSE panel).
- S4 — Simulation recovery battery (easy bimodal / hard concentrated / log-normal), NUTS + tuned-MAP vs truth.
- S5 — Fisher information matrix + intermediate-bin collinearity (old Fig 8a).
- S6 — Bayesian CRLB (van Trees) vs unconstrained vs NUTS (old Fig 8b). TENTATIVE: supp figure or fold into text — revisit.

**Cosmetic conventions (locked):**
- Match APPARENT font size, not point size: single-panel figs render larger at \textwidth than 2×2 grids, so size via LaTeX width (e.g. Fig 5 at width=0.85\textwidth ≈ Fig 3 at full width). 2×2 grids use axis labels 20 / ticks 18 / panel titles 17 / legend 17.
- NO in-figure titles — caption's first sentence is the title (MRM convention; matches fig_roc, fig1).
- Colour palette: NUTS = orange, tuned MAP = green, ADC = grey/black, tumor = red, normal = blue.
- **Identifiability / CV colour = PURPLE SEQUENTIAL** (light lavender CV<0.4 → dark purple CV>0.8). NOT green→red (clashes with tumor/normal + MAP/NUTS).
- 300 dpi PNG + PDF; anonymise patient IDs.

**Build order (2026-05-31):** Fig 1 → Fig 4 → Fig 6 → Fig 2 → Fig 7 → Fig 8(new) → Fig 9 → supplementary. Then manuscript text.

---

## 7. @Stephan inline-comment tracker (originally 15 markers)

| File | Comment | Status |
|---|---|---|
| introduction.tex:9 | Insert Mulkern 2006 biexponential citation | **DONE 2026-05-23** |
| theory.tex:37 | Out-of-order figure ref (Fisher) | OPEN — resolves when Fig 8 → SI |
| theory.tex:38 | Same out-of-order issue | OPEN — same |
| theory.tex:67 | `\emph{a posteriori}` font emphasis | **DONE 2026-05-23** |
| theory.tex:84 | Long sentence with ", and" — split | **DONE 2026-05-23** |
| theory.tex:97 | "HalfNormal" → "half-normal" | **DONE 2026-05-23** |
| theory.tex:98 | `\emph{a priori}` font emphasis | **DONE 2026-05-23** |
| results.tex:12 | "Taken up in Discussion" forward-reference | OPEN — Path A' Results rewrite |
| results.tex:20 | Same | OPEN — same |
| results.tex:59 | Same | OPEN — same |
| methods.tex:21 | Too many sub-headers | **DONE 2026-05-23** |
| methods.tex:23 | Unused shorthand explanation | **DONE 2026-05-23** |
| methods.tex:41 | Add Maier 2022 ADC b-range reference | **DONE 2026-05-23** |
| methods.tex:53 | Confused about SE vs CI | **DONE 2026-05-23** |
| supporting.tex:78 | Fig S2 — fonts, color, MAP overlay | OPEN — figure regen |
| figures.tex:176 | Text/figure mismatch on directional | OPEN — blocked on Fig 7 confirmation with Stephan |

**Resolved: 8/15.** Remaining 7 tied to Path A' Results rewrite (3), figure decisions post-meeting (2), figure regen (1), text/figure sync after Fig 7 confirmation (1).

---

## 8. Open actions (2026-05-31 session)

**Goal today:** finalized draft to send Sandy + Stephan for review next week. Submit next week.

**Decided this session:**
- Figure scope + pillars locked (§6). Old Fisher/CRLB Fig 8 → supplementary. New main Fig 8 = method validation (recovery + joint noise).
- Fig 3 ✅ (2×2 shared axes, legend top, no title) and Fig 5 ✅ (GGG=1 vs ≥2, width 0.85) wired into figures.tex; Results/Discussion text synced.
- Identifiability distributed (Fig 4 colour + supp S1), not a standalone figure.
- Identifiability/CV colour → purple sequential (was green→red).

**Done 2026-05-31 (figure session, cont.):**
- Supp S1 reworked to box-plot atlas (`fig4_v2` colour scheme), layout bug fixed, public reproducibility IDs in titles + key CSV. See §6 supplementary.
- Fig 4 reworked → `fig4_v2` (LR bars coloured by CV + −∂ADC/∂R line on one normalised axis + CV mean±std strip). Wired into figures.tex with honest caption. See §6 Fig 4.
- **Fig 2 reworked → `fig2_v1`** (detection ROC, apples-to-apples LOOCV, 2-feat collapse). See §6 Fig 2. Wired into figures.tex (caption rewritten, refs Table 1). **Table 1 reworked** (detection-only, bootstrap CIs, 2-feat MAP+NUTS rows; fixed stale MAP + wrong GGG SEs) — see §6 Table 1. **Manuscript-wide interval convention set: percentile bootstrap CIs + DeLong for paired AUC tests; Hanley–McNeil retired** (updated `methods.tex`, `results.tex` Classification Performance para, `tables.tex`; added `delong1988comparing` to references.bib). `recompute.py` extended (2-feat rows + auc_lo/auc_hi columns); `auc_table.csv` regenerated, all other frozen CSVs byte-identical. Per Patrick (2nd pass): tried a per-panel in-panel AUC box but **Patrick reverted it** (looked worse than clean) — AUC/CI numbers live only in Table 1. Caption sharpened to state the ranking explicitly (ADC ≳ 2-bin ≳ 8-bin, 2-bin ≥ 8-bin). **Abstract partially updated** (Methods → unified LOOCV pipeline; Results → 2-feature collapse + bootstrap detection numbers). **§5a abstract full rewrite STILL PENDING** (kept as action item per Patrick): abstract lines 16/20 still carry the F4b sensitivity ‖r‖>0.93 overclaim + F1 "MAP underestimation" — left for that rework. `figures.tex` Fig 9 subplot-D AUC=0.919 is stale (subplot D slated for deletion anyway).
- **OPEN (Patrick flagged, deferred — possibly minor):** the faint grey "other single bins" curves dip **below the diagonal** (e.g. D=20 LOOCV AUC → 0.0/0.22). Diagnosed as a **single-feature LOOCV-LR instability artifact** on near-empty/weak bins, NOT real inverse signal — raw single-bin AUCs are all near chance (D=20 0.46–0.52; D=2.0 0.21–0.59). Reader may find sub-diagonal curves suspicious. Two options if revisited: (a) leave as-is (faint, unlabeled) + one caption clause; (b) plot single-component curves as standard **raw single-feature ROC in tumor-positive (max) orientation** (what the old `fig_roc` did — stable, no sub-diagonal, but introduces a thin=raw / thick=LOOCV split that the caption must note). Patrick leaning "minor"; not changed this session.
- **Data-quality flag for Patrick (visible in published S1):** released `metadata.csv` Gleason scores include `2+3` (1×, primary-pattern-2 is obsolete clinically), `4+3+5` (2× — tertiary-pattern notation, recorded as GGG 5), and `hgpin` (2× — not cancer). These now print verbatim in S1 panel titles / key CSV. Decide before submission whether to normalise or footnote them.

**Done earlier this session:**
- Two-camps literature review complete → `notes/lit_review_two_camps.md` (web-verified DOIs/PMIDs). **Reconciliation = detection vs grading:** fair PI-RADS ADC is hard to beat for DETECTION (Camp A — Fennessy & Maier 2023 Eur J Radiol is *Stephan's own* paper; He et al. 2025 = `assets/s00261...` is the cleanest fair head-to-head, ADC = MC for detection); cellular-compartment metrics (VERDICT fIC, DKI, MC restricted fractions) beat ADC for GRADING (Camp B), because grading signal lives in intracellular/intermediate diffusivities. **KEY CORRECTION:** the RSIrs detection gap (ADC AUC 0.48–0.54) is from an automated whole-gland *minimum* ADC vs radiologist-localized lesion ADC — NOT high-b vs PI-RADS b-values (all headline Camp B papers used b≤1000). Lead the Discussion reconciliation with Maier 2023.
- Fig 1 rebuilt as `fig1_v4` (cohort box plots, MAP green-hatched / NUTS orange-solid, grey mean-CV annotation, PZ/TZ titles, legend top). Wired into figures.tex (caption rewritten, "ridge smoothing" softened to λ-dependent).

**Done 2026-06-01 (Fig 9 — pixel-wise feasibility, this session):**
- Reworked → `fig9_v1` (`scripts/fig9_pixelwise.py`), 3×3 NUTS heatmap grid (see §6 Fig 9). Built entirely from cache (`results/pixelwise/nuts_results.npz` + `pixelwise_all_fast.npz`) — no re-inference. New figure files + script committed + pushed in `95d669f`; `.tex` wiring + this PROJECT_STATE update in the bulk commit.
- **NEW FINDING (per-voxel tumor-skew) — candidate F12:** applying the ROI-trained detector voxel-wise yields scores that skew systematically tumor-like (median voxel logit +2.5 (2-bin) / +4.4 (8-bin) vs median *tumor*-ROI logit +1.3). Cause: a single voxel's much lower SNR inflates the restricted (D=0.25) bin — the high-b noise floor mimics slow decay. The old Fig E red/blue split was partly an artifact of the non-tuned MAP (λ=0.1) smearing that mass away (Patrick's hypothesis — partly confirmed). Implication: reinforces ROI-level scope; faithful per-voxel classification would need explicit voxel-noise modelling or spatial regularisation. Shown honestly via the "absolute" panels (E,F); panel G windows it to recover ADC-like contrast.
- **▶ OPEN — @Stephan: confirm patient 8640's lesion zone (PZ vs TZ).** Fig 9 applies the **PZ** detector PROVISIONALLY. 8640 is NOT recoverable from the repo (absent from `metadata.csv` anon ids new01–new56, `signal_decays.json`, and the directional tarball, which holds 7 patients — 8804/8805/8864/9283/9322/9675/10203 — not 8640); the raw→anon ID mapping is not in-repo. On receipt of the zone: flip `ZONE` in `scripts/fig9_pixelwise.py`, the "peripheral-zone" wording + `% NOTE(provisional)` in the figures.tex caption. This is the long-standing pixel/direction provenance [ASK].

**TODO — near-final literature deep-dive (one of the LAST steps, before submission):**
- Extend the lit review beyond medical MR: general signal-reconstruction / inverse-problem literature + ML approaches (variational inference, amortized / learned inference for spectral or parameter estimation). Goal: position our fully-Bayesian *joint spectrum + noise* inference against the wider methods landscape. Do once main text + final figures are locked.

**Figure reworks:** ~~Fig 1 (fig1_v4)~~ ✅, ~~supp S1 box-plot atlas~~ ✅ (2026-05-31), ~~Fig 4 (fig4_v2)~~ ✅ (2026-05-31), ~~Fig 2 (fig2_v1)~~ ✅ (2026-05-31), **Fig 8 (fig8_v2)** ⚠️ draft built + style approved 2026-06-01 — Gibbs framing UNRESOLVED, NOT wired, Theory NOT written (see §8 item 2), ~~Fig 6 (fig6_v1)~~ ✅ (2026-06-01, first draft wired into figures.tex + discussion.tex; may iterate), ~~Fig 9 (fig9_v1)~~ ✅ (2026-06-01, wired into figures.tex + methods.tex + discussion.tex; PZ provisional) → **Fig 7** → supplementary S2–S6.

**▶ NEXT SESSION — Fig 4 (centrepiece) spec — LOCKED this session:**
- Rework `scripts/fig4_lr_coefs_and_sensitivity.py` → output `fig4_v2`. 2 panels (PZ, TZ), mirroring Fig 3.
- Per bin: standardised LR coefficient (bars) + $-\partial$ADC$/\partial R_j$ (charcoal line+markers) on a SINGLE unit-normalised axis — NO dual-axis (v1's offset problem). Drop the raw-vs-standardised duality (standardised only).
- Bars COLOURED by identifiability = purple sequential, same bands as supp S1 (light CV<0.4 → dark CV>0.8). Consider annotating mean±std of per-ROI CV per bin (Patrick's idea — shows identifiability spread, not just the mean).
- Colours: charcoal sensitivity line + purple CV bars. No red/blue/orange/green clashes.
- **Story it must tell:** outer bins (D=0.25, 3.0) = high LR weight + ADC-aligned + well-identified = the DETECTION axis; intermediate bins = where LR and ADC diverge AND poorly identified — yet Fig 5 shows they still shift with grade = the GRADING axis. Complement to Fig 3 (Fig3 = "spectrum ≈ ADC"; Fig4 = per-bin anatomy + where/whether it breaks).
- **Reconciliation framing** (`notes/lit_review_two_camps.md`): detection vs grading. Lead with Fennessy & Maier 2023 (Camp A = Stephan's own). RSIrs detection gap = whole-gland-min-ADC artifact, NOT b-values.
- Data: LR fit per zone on the 8 features (features.csv, C=1.0, standardised, class_weight balanced — same as fig3); $\partial$ADC$/\partial R$ from `adc_sensitivity.csv` / `adc_sens_vs_lr_tuned_lambda.csv`; CV from `nuts_std_D_*` / `nuts_D_*`. Existing helpers: `scripts/{plot_lr_weights_per_bin,plot_lr_weights_vs_adc_sensitivity,adc_sensitivity_at_tuned_lambda}.py`.

**▶ NEXT SESSION — also deferred (Patrick flagged 2026-05-31, do AFTER main figures):**

*1. Supplementary all-ROI atlas (S1) — REWORK (Patrick not satisfied):*
- Revert to **box plots of the posterior samples** per bin (as in the old `output/gibbs` Gibbs output), NOT the current bars + CV-colour. Box-plot shows the full within-ROI posterior distribution.
- Fix PDF layout: the **top legend is cut off** and the **y-axis label is missing** in the rendered PDF — the current `\includepdf` / page layout is broken.
- Reconsider showing a **patient number / index** after all — currently there is no way to trace a panel back to a patient. Use a non-identifying sequential id + a private key for the authors (traceability matters; reverses the earlier "no patient id").
- Keep **2 columns** (Patrick reaffirmed 2, not 3).
- Generator: `scripts/figS1_all_roi_spectra.py`.

*2. Simulation/validation figure (main Fig 8) — DRAFT BUILT 2026-06-01, Gibbs framing UNRESOLVED:*
- **Built:** `fig8_v2` via `scripts/fig8_validation.py` (cache `results/simulation/fig8_validation.npz`, .nc in `results/simulation/fig8_nc/`). Uses the REAL pipeline classes + exact configs (nuts.yaml: 2000 draws/200 tune/4 chains/target 0.95, σ~HalfCauchy(β=1/SNR) inferred, init=map; gibbs.yaml: **100k iters/10k burn/4 chains, σ FIXED=1/SNR**, truncated-normal conditionals, init=map; ridge.yaml strength 0.1→σ_R=3.162; tuned-MAP λ=1e-3). SNR=303. δ@D=0.75. Style approved (Fig-1 box style, slate-blue Gibbs, green-× MAP, discrete truth marks, boxed top legend, no in-panel legends, fonts = Fig 3).
- **KEY FINDINGS (change the story):**
  - At 100k iters **Gibbs CONVERGES** (R̂≈1.0–1.02). "Gibbs doesn't converge" is FALSE here. Only clean NUTS edge = per-draw efficiency (~200–260× on correlated bins D=0.5–1.0; Gibbs fine on near-independent D=20).
  - **The comparison is CONFOUNDED:** Gibbs is handed true σ (fixed); NUTS infers σ. On δ NUTS overestimates σ (mean 1.6×, median 1.5× true) → wider/biased R posterior → Gibbs looks *tighter & slightly better* on δ. Not a fair fight.
  - σ̂ is right-skewed (report median, not mean → small improvement) AND genuinely inflated on concentrated truths (misfit residual). On realistic spectra within ~1–11% (tumour ≈ unbiased). Honest F6-consistent limitation.
  - NUTS is already converged at 2000 draws → running longer will NOT improve recovery (Patrick asked). The δ behaviour is the σ-inference confound, not undersampling.
- **▶ OPEN — resolve BEFORE writing Theory (Patrick + me agreed 2026-06-01):**
  - **(a) Fair fight:** implement Sandy's **inverse-gamma σ² conjugate update** so Gibbs ALSO infers σ (full joint Gibbs over R,σ²). Never been tried. Directly supports the §8-item-3 correction (closed-form conditionals for BOTH R and σ²). Only then do both solve the same problem.
  - **(b) Reproduce Gibbs getting STUCK:** Patrick recalls a **bimodal** config trapping even at ~1M iters (genuine non-convergence, not just slow). Prime suspect: bimodal {0.25:_, 3.0:_} (two separated modes + correlated middle). If Gibbs traps where NUTS doesn't → THAT is the only legit "Gibbs fails" claim.
  - **(c) Do NOT over-claim:** even if Gibbs is competitive, MAP is the fast diagnostic estimator; NUTS/Gibbs are exploratory full-posterior tools. Position NUTS as: jointly infers σ w/o specification + uniform mixing efficiency + (if (b)) robust where Gibbs stalls. Efficiency alone is a weak argument (both slow; neither real-time).
- **Caption MUST list all configs** (Patrick: reproducibility). Draft config block ready (in this session's chat) — fold into `figures.tex` caption when wiring.

*3. Manuscript CORRECTION (Gibbs) — current text is WRONG:*
- The claim is at **`theory.tex:101`**: "Standard Gibbs sampling is ineffective for this model because the half-normal priors **are not conjugate to the Gaussian likelihood** and, more fundamentally, the strong correlations … mix extremely slowly." The "not conjugate" half is **WRONG** — half-normal × Gaussian → closed-form **truncated-normal** full conditionals (and σ² is inverse-gamma conjugate — Sandy's derivation); Sandy implemented Gibbs, Patrick ran it for months.
- **Refined by 2026-06-01 data:** the failure is NOT non-convergence — at 100k iters Gibbs converges (R̂≈1.0). The honest correction = "Gibbs has closed-form conditionals but is **computationally crippled**: per-draw efficiency on the correlated bins is ~200× below NUTS, so it needs impractically many iterations." **Hold the rewrite until §8-item-2 (a)+(b) resolve** — if (b) shows Gibbs trapping on bimodal, the wording can be stronger ("stalls / fails to reach the target on some configurations").
- Keep the "mix extremely slowly from adjacent-bin correlation" half (it's correct + Fig 8e shows it). Add a short Gibbs→NUTS-progression paragraph tied to Fig 8.

**Then manuscript:** Results around the three pillars; Discussion two-camps reconciliation (await lit review); Abstract + Conclusion; resolve remaining @Stephan inline comments (§7); Methods MAP-tuning + joint-σ paragraphs; Gibbs correction + paragraph (item 3 above).

**Blocked / dependencies:**
- Fig 7 — needs Stephan's full 10-patient tarball (per-direction data).
- Discussion reconciliation paragraph — awaits background lit review.
- Eq. 5 final wording — Sandy review (draft good enough to submit if silent).

**Parked:** see §9 below; no changes there.

---

## 9. Out of scope / parked

- Continuous-spectrum (GP prior) — parked indefinitely.
- VI / amortised inference comparison — one-paragraph mention in Theory only.
- Identifiability features in classifier (aggressive variant from 3rd-pass §4) — future work.
- Pixel-wise per-voxel classification AUC — no whole-mount histology to validate.
- TCIA cross-dataset validation — no public counterpart for extended b-range.
- ADC-variants sweep as main-text content — supplementary at most.
- LR-coefficient axis-separation as a *headline* claim (F4 wide CIs).
- Biopsy-replacement Intro motivation — Future Directions paragraph only.
- Re-running NUTS on the cohort at the wider σ_R values (F8 closed; no benefit).
- Searching for an even smaller MAP λ than 1e-6 (the λ=1e-4 sweet spot is robust across SNRs in the simulation; no need to push further).

---

## 10. Pointers

- **Memory dir** (`~/.claude/projects/-Users-PWR-Documents-Professional-Papers-Paper3-code-spectra-estimation-dMRI/memory/`): 9 files — user profile, principle/feedback rules, MRM policy, LLM policy, MRM guidelines, lit pointers, MAP/NUTS validity note, state pointer back to this file.
- **Figure handover checklist:** `HANDOVER_FIGURE_SESSION.md` (kept in repo root). Note: figure plan in §6 supersedes the figure list there.
- **Meeting Q&A refresher:** `MEETING_PREP_2026-05-25.md`. Archive after today.
- **Email draft:** `_email_draft_2026-05-23.md` (confirm with Patrick whether sent).
- **Frozen result CSVs:** `results/biomarkers/*.csv`, `results/inference_bwh_backup/*.nc`. Regenerate via `uv run python -m spectra_estimation_dmri.biomarkers.recompute`.
- **New 2026-05-24/25 results:**
  - `results/simulation/map_lambda_sweep{,_summary}.csv` + `map_lambda_sweep_{fraction,mse}.png`
  - `results/simulation/wider_prior_check{,_summary}.csv`
  - `results/biomarkers/map_lambda_bwh{,_summary}.csv`
  - `results/biomarkers/adc_sens_vs_lr_tuned_lambda.csv`
- **New scripts:** `scripts/map_lambda_sweep.py`, `scripts/plot_lambda_sweep.py`, `scripts/map_lambda_bwh.py`, `scripts/wider_prior_check.py`, `scripts/adc_sensitivity_at_tuned_lambda.py`.

---

## 11. Stefan figure-overhaul meeting (2026-06-03) — pre-submission sprint

**Context.** 3-hour meeting with Stefan reviewing all figures. Goal: rebuild/polish every figure, review them together, then finalise the manuscript text + MRM-guideline compliance, producing a **final draft today (2026-06-04)** to send Sandy + Stefan. **Hard submission deadline: Saturday 2026-06-06.** This section supersedes §6 figure statuses where they conflict; §6 remains the prior baseline.

### 11.1 Main-figure renumbering (PROPOSED — confirm with Patrick)

Labels are semantic, so renumbering = reordering figure blocks + moving two blocks between `figures.tex` ↔ `supporting.tex`; all `\ref`s auto-update. **Net count unchanged: 9 figures + Table 1 = 10 (MRM cap).**

| New # | Label | Was | Change |
|---|---|---|---|
| 1 | `fig:spectra` | Fig 1 | edit (drop CV) |
| 2 | `fig:roc` | Fig 2 | edit (D=20, NUTS, fonts) |
| 3 | `fig:adc_discriminant` | Fig 3 | edit (PZ-left/TZ-right) |
| 4 | `fig:sensitivity` | Fig 4 | edit (strip stripes/stars/CV-subplots) |
| 5 | `fig:spectrum_ggg` | Fig 5 | layout decision |
| 6 | `fig:uncertainty` | Fig 6 | rebuild (4-panel, violins, recolour) |
| **7** | `fig:fisher` | **SI** | **PROMOTE → main** (van-Trees CRLB, SNR 303, fonts) |
| **8** | `fig:validation` | Fig 8 | **SWAP content → simulation battery (~12 panels)** |
| 9 | `fig:pixelwise` | Fig 9 | edit (delete G, recolour, colourbars) |
| Table 1 | `tab:auc` | Table 1 | revisit (classification waterproofing) |
| **S2** | `fig:directions` | **Fig 7** | **DEMOTE → SI** as new 2×4 directional-spectra panel |

Promoting Fisher → main also requires updating the **theory.tex CRLB paragraph** (lines 61–62, the "100–8000×" framing → van-Trees, per F10) and the out-of-order figure refs (theory.tex:37,38; §7 tracker).

### 11.2 Shared style spec — build `src/.../visualization/paper_style.py` FIRST

Every figure script must import this so the meeting's recurring asks (consistent fonts, legend = title size, legend on top, no angled text, no in-figure titles, consistent D-bins, consistent colours, **PZ always left / TZ always right**) hold uniformly. Currently each script hardcodes its own (inconsistent) values.

- **Fonts (locked):** legend font size **== panel-title font size** (Stefan, repeated). Standardise 2×N grids to axis-label 20 / ticks 18 / title 17 / **legend 17**; single-panel to label 19 / ticks 16 / title 16 / legend 16.
- **Colours (locked, from `figures.tex` header):** normal = blue `#1f77b4`, tumor = red `#d62728`, NUTS = orange `#ff7f0e`, MAP = green `#2ca02c`, ground-truth/reference/CRLB = black/grey `#1a1a1a`. CV/identifiability = purple sequential (`visualization/identifiability.py`). **New, unused colours** needed for: Fig 7 directional-spectra direction lines + ground truth; Fig 9 "score" maps (NOT red/blue — reserved for restricted/free-water B,C) and uncertainty maps (purple→yellow, NOT violet→white which clashes with anatomy).
- **D-bins (locked, consistent w/ Fig 1):** D = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20] µm²/ms; tick labels `"{:g}"` → "0.25","0.5",…,"3","20". No angled tick labels.
- **Layout:** PZ left / TZ right everywhere (Fig 1 sets the convention); legend `fig.legend(loc="upper center", bbox_to_anchor=(0.5, ~0.99), frameon=True)`; no `suptitle`; 300-dpi PNG + PDF, `bbox_inches="tight"`.

### 11.3 Per-figure change list (digested from Stefan notes)

- **Fig 1 (`fig:spectra`):** remove the grey mean-CV annotation **and** the CV sentence from the caption (CV is covered in Fig 4). Else unchanged.
- **Fig 2 (`fig:roc`):** *re-add* D=20 as a coloured, **legend-labelled** curve to demonstrate it is irrelevant to classification (reverses the 06-02 "omit D=20" decision); check whether any intermediate bins are worth colour-highlighting too (Stefan: show the middle can help a little). Make explicit it is **NUTS** (caption + legend). Legend fonts → title size. *Watch the sub-diagonal LOOCV artifact* (D=20 AUC→0.0/0.22) — plot raw single-feature ROC in tumor-positive orientation if needed so curves don't dip below chance, and say so in caption.
- **Fig 3 (`fig:adc_discriminant`):** enforce **PZ left, TZ right** (currently zone = rows). Transpose to zone = columns (PZ | TZ), estimator = rows (NUTS / MAP). Keep this PZ/TZ convention across ALL figures incl. the new directional one.
- **Fig 4 (`fig:sensitivity`):** remove grey CV mean±std strip; remove significance stars (too crowded); remove extra CV subplots (redundant with the bar colour-coding); legend = title size. **OPEN: standardized vs raw LR coefficients** (Stefan frustrated that dividing by σ_j shrinks middle bins away from the monotonic ADC-sensitivity decay; Patrick leans "it's the truth"). The "train a classifier on the ADC-sensitivity vector and compare coefficients" idea = explicitly **explorative, park it**.
- **Fig 5 (`fig:spectrum_ggg`):** current = single ladder (Normal/GGG1/GGG2/GGG≥3). **OPEN layout** (Stefan: current too crowded — shades of green + bands hard to read): (1) keep single panel, recolour for clarity; (2) two panels — **GGG1 vs ≥2** (definitely keep, left) and **GGG2 vs ≥3** (right), normal as shared baseline (with/without normal TBD); (3) worst case, left panel only (GGG1 vs ≥2). Optimise for visible differences **and** max minimum group n. NB §6 history: a two-boundary v3 was dropped as redundant — the new boundaries differ slightly.
- **Fig 6 (`fig:uncertainty`):** colours instead of black rings — light blue/red for correct, **dark** blue/red for misclassified (rings currently overlap). **Violin plots** (not box), ordered (no horizontal jitter). **Split PZ/TZ → 4 subplots** (fill all quadrants; the bottom CI-width panel splits per zone), PZ left. Caption must correctly distinguish the **several** uncertainties (per-bin posterior σ vs propagated P(tumor) credible interval; the boundary-widening that is partly logit-link geometry — see §6 Fig 6 row).
- **Fig 7 → NEW MAIN: Fisher/CRLB (`fig:fisher`, promoted from SI).** Generator `scripts/generate_fisher_figure.py` (1×3: a=Fisher correlation matrix [Stefan wants the matrix kept], b=CRLB vs NUTS std, c=component decay vs noise floor). Fixes: panel-c x-axis **not angled**; panel-b extend range + move legend into free space (currently covers bars); panel-c **SNR labels bigger**; bigger fonts overall (3-in-a-row is small — Stefan says don't over-worry). **SCIENCE: panel b CRLB values are suspiciously high (100–8000×).** Rebuild per **F10**: 3-bar comparison (unconstrained CRLB / **Bayesian van-Trees CRLB** / NUTS std) at **SNR=303** (not hardcoded 150). → **Sandy must validate the van-Trees derivation** (`notes/CRLB_NOTE_FOR_SANDY.md`). Decide placement in sequence (proposed Fig 7).
- **Fig 8 → NEW MAIN: simulation recovery battery (`fig:validation`, swapped in for the 2×2 validation).** Stefan was least happy with current Fig 8; wants the old "S2" with **lots of ground-truth variety**. **This does not exist as one script — must be built** (~12 panels: δ at several D, bimodal, log-normal, normal-like, tumor-like). Each: truth (black marks) + NUTS recovery (box plots) + **MAP** (green ×), consistent Fig-1 style, no angled, legend on top, no title, **R̂ in each subplot subtitle** (explain R̂ in caption), call the spike "δ". Use **normal-like / tumor-like** (cohort-mean, like current Fig 8) rather than NPZ/TPZ. **Try to fold in noise (σ) recovery** per panel or as a companion violin (Patrick: we infer two RVs — fractions + σ). **OPEN/Sandy: Gibbs-vs-NUTS justification.** Patrick wants NUTS rigorously justified (Sandy's inverse-gamma σ² Gibbs makes joint σ inference possible for Gibbs too). Options: (a) cosmetic battery NUTS+MAP+R̂ as main Fig 8 now, defer Gibbs justification to Theory text / SI panel; (b) attempt the full fair-fight Gibbs experiments now (joint-σ Gibbs + bimodal-trapping repro — risky for the deadline, needs Sandy). See §8 item 2 for the unresolved Gibbs story.
- **Fig 9 (`fig:pixelwise`):** **delete subplot G** (windowed 2-bin — Stefan didn't see the absolute-vs-windowed value; leaves one empty 3×3 slot, OK). New colour scale for the **score** maps E,F (NOT red/blue — reserved for restricted/free-water B,C). **Uncertainty maps H,I → purple→yellow** (not violet→white; white clashes with anatomy; maximise contrast). **Colourbars on every subplot** incl. the middle ones. **OPEN: B,C show raw spectral fraction vs weighted?** + the **classification apples-to-apples** question (raw single-fraction classifier vs logistic; in-sample vs out-of-sample) — ties directly to **Table 1 waterproofing** below.
- **Table 1 (`tab:auc`):** Patrick wants the **whole classification story re-verified "waterproof"** before submission — raw-fraction-as-classifier vs logistic, in-sample vs out-of-sample fairness vs the ADC reference. Do this carefully (not fanned out) during the Results/Table pass; reconcile with Fig 2 + Fig 9.
- **SI S2 → directional spectra 2×4 (`fig:directions`, demoted from main Fig 7).** Generator `scripts/fig7_directions_roi.py` (already decodes the .dat export). Selected ROIs (Stefan + Patrick): **9675 (NPZ, TTZ), 9322 (NPZ, TTZ), 9283 (NTZ, TPZ), 10203 (NPZ, TTZ)** — 4 patients × 2 ROIs = **2×4 grid**. Per-direction spectra, **consistent with Fig 1** (no angled, same legend/fonts), **new unused colours** for directions + black/grey ground truth, number panels 1–4 (pairs), keep zone label, **remove the tiny "DIR-CV 63%" annotation** (or move to caption), no overall title, legend on top. (Less critical — ROI analysis uses 3-direction average anyway.) The 4 patients are all in the directional tarball (8804/8805/8864/9283/9322/9675/10203). Exploration overview already in `fig7_explore_all_rois_{spectra,decays}.{pdf,png}`.

### 11.4 Open decisions for Patrick (gate the fan-out)

1. **Renumbering** (§11.1) — confirm scheme + Fisher at position 7.
2. **Fig 8 scope** — cosmetic battery now (Gibbs justification deferred) vs full Gibbs experiments now.
3. **Fig 5 layout** — single recoloured / two-panel (which boundaries, normal y/n) / left-only.
4. **Fig 4** — standardized vs raw coefficients (or show both for review).
5. (defaulting unless told otherwise) **Fig 9** — delete G ✓; keep B,C as fractions; recolour scores + purple→yellow uncertainty + colourbars everywhere. Classification waterproofing handled in the Results/Table pass.

### 11.5 Co-author dependencies

- **Sandy:** (a) van-Trees Bayesian-CRLB derivation for new Fig 7 (F10); (b) inverse-gamma σ² conjugate Gibbs for a fair Gibbs-vs-NUTS comparison (Fig 8 / Theory). All co-author questions route through Patrick.
- **Stefan:** confirm Fig 9 lesion zone (patient 8640 PZ/TZ — still provisional, §8); asked whether his **Obsidian paper** can be cited — Patrick to check fit (candidate: Discussion / Methods ADC-reference or biology). 

### 11.6 General / MRM / submission checklist

- 10-item main cap respected (9 figs + Table 1). SI unlimited; Stefan: SI figure size doesn't matter as long as it reads on one page — individual-spectra atlas (S1) already there.
- Convergence/trace plots in SI: **likely no** (sampler performance covered by the new Fig 8) — confirm.
- After figures locked: rewrite Results around the 4 pillars (§5e), Discussion (§5f), Abstract + Conclusion harmonisation (§5g), resolve remaining @Stephan inline comments (§7), Methods λ-sweep + joint-σ paragraphs, Gibbs correction (theory.tex:101, §8 item 3).
- MRM compliance pass (re `reference_mrm_guidelines`): ≤5000 body words, structured 250-word abstract, 3–6 keywords, Data Availability + code SHA, IRB statement, ORCID, references ≤6-authors-then-et-al, **LLM-disclosure policy** (no AI-generated text/figures without editor permission — figures are code-generated from data, prose is Patrick's; confirm framing).

### 11.7 Progress 2026-06-04 (figure fan-out — FIRST PASS DONE, pending Patrick review + central .tex rewire)

Built `src/.../visualization/paper_style.py` (shared fonts/colours/D-labels/legend helpers, pinned to Fig 1). All figure scripts now import it. **New version files written (old versions left intact; .tex NOT yet rewired — done centrally after review):**
- **Fig 1** `fig1_v4` regenerated — CV annotation removed (edited `generate_paper_figures.py::fig_spectra_combined`). ✅ verified.
- **Fig 2** `fig2_v3` — D=20 re-added (magenta "irrelevant"), NUTS explicit, single-bin curves switched to RAW single-feature ROC in max orientation (so weak bins hug chance, not the LOOCV sub-diagonal artifact). D=20 hugs chance (AUC 0.52/0.55, dips ≤0.1 below diag mid-range — honest). **Caption must state single-bin curves are raw ROC (not LOOCV); Table 1 single-bin AUCs change vs LOOCV values.**
- **Fig 3** `fig3_v4` — transposed to PZ-left/TZ-right columns, NUTS-top/MAP-bottom rows. ✅
- **Fig 4** TWO versions `fig4_std_v4` + `fig4_raw_v4` — decluttered (removed grey CV strip, significance stars, CV sub-plots). Raw lifts middle bins; r(w,∂ADC/∂R) PZ −0.61 raw vs −0.79 std. **Patrick picks at review.**
- **Fig 5** `fig5_v5` — two panels (emergence GGG1-vs-≥2 | aggressiveness GGG2-vs-≥3), normal grey baseline both, blue→purple→dark-red palette. Minor: legend "n=21" glyph + panels a touch short.
- **Fig 6** `fig6_v2` — 4 quadrants (PZ/TZ × sorted-predictions/CI-width-violins), violins not boxes, light=correct/dark=misclassified, no black rings. TZ misclassified n=5 (sparse, Welch p=0.118 ns). **Caption: distinguish per-bin σ vs propagated P(tumor) CI; boundary-widening partly logit-link geometry (ρ −0.94 prob → −0.29 logit; ratio 2.4× → 1.3×).**
- **Fig 7 = Fisher/CRLB** `fig_fisher_v2` (PROMOTED) — 3-bar panel b (unconstrained CRLB / van-Trees Bayesian CRLB / NUTS std) at SNR=303, matrix kept, no angled labels. **⚠️ van-Trees derivation PENDING SANDY** (Gaussian-approx of HalfNormal prior precision λ=0.1; route assumptions). theory.tex:61–62 "100–8000×" must reframe to match.
- **Fig 8 = simulation battery** `fig8_v4` — ✅ DONE. New script `scripts/fig8_battery.py` (real pipeline: NUTSSampler nuts.yaml + compute_map_spectrum λ=1e-3 + ridge.yaml; `fig8_validation.py` untouched). 5×2: 9 GTs at SNR=303 (normal/tumour/inverse/bimodal/uniform/single-peak D=1.0/δ@0.75/log-normal narrow+broad) as box plots + green-× MAP + black truth, R̂ in each subtitle (**max R̂ 1.006, all converged**), panel (j) = σ̂ vs true σ=1/SNR. **Caption note:** concentrated truths (δ, narrow log-normal) over-estimate σ ~1.3–1.5× (misfit-as-noise; F6-consistent). Cache `results/simulation/fig8_battery.npz` + `fig8_battery_nc/`; replot cheaply via `uv run python scripts/fig8_battery.py`.
- **Fig 9** `fig9_v2` — subplot G deleted, score maps → PuOr (not red/blue), uncertainty maps → viridis (not violet→white), colourbars on every quantitative panel. ✅
- **SI directional** `fig_directions_v3` (DEMOTED) — 2×4 of 4 patients × 2 ROIs (9675/9322/9283/10203), new direction colours + black trace-avg, no corner CV annotation, consistent style. All 8 ROIs decoded. MAP-only (outside the 149-ROI gold set).

**Agent IDs (resumable via fresh Agent, SendMessage unavailable):** fig2 a765dc32a03e38ebd, fig3 af37ea696c5def85b, fig4 a339f3862514aebe9, fig5 af61519c97b449071, fig6 aa1ad08fcc943b330, fig9 a5531ffad55245b84, SI-dir a92864b3c2acb97d7, fisher acfd1d1edf4875f63, fig8(bg) a1004376c5856cb31.

**NEXT (after Patrick's figure review):** (1) central .tex rewire — move Fisher block supporting.tex→figures.tex (pos 7), directional figures.tex→supporting.tex, swap fig8 include, repoint all `\includegraphics` to new version files; semantic labels auto-renumber. (2) Manuscript text: theory CRLB van-Trees reframe, Fig 2 raw-ROC caption clause, Fig 4 caption (post std/raw pick), Fig 6 uncertainty caption, Results 4-pillar restructure, abstract/conclusion, Gibbs correction (theory.tex:101). (3) Table 1 reconcile (single-bin AUC semantics changed) + classification waterproofing. (4) MRM compliance pass.

### 11.8 Round-2 refinements 2026-06-05 (Patrick's per-figure style pass)

- **Fig 1** `fig1_v4` — fliers DELETED (showfliers=False); legend label → "NUTS (posterior mean)" (it's per-ROI posterior means, cohort-distributed — NOT pooled full posteriors); spacing recipe `subplots_adjust(top=0.88,hspace=0.34,…)` + legend y=0.965 (legend↔row1 ≈ row1↔row2). **This recipe is the cross-figure spacing standard.**
- **Fig 2** `fig2_v3` — legend: removed title row, ALL parens, ":irrelevant", "chance" entry; concise labels (ADC / Spectral, 8 bins / Spectral, 2 bins / D=0.25 / D=3.0 / D=2.0 / D=20 / Other single bins), width≈panels. Added **D=2.0 (goldenrod #b8860b)** as 2nd weak bin (worst non-D=20 by aggregate; PZ AUC 0.587, TZ 0.792 — asymmetric, flag). NUTS/LOOCV-LR/raw-orientation qualifiers now belong in CAPTION.
- **Fig 3** `fig3_v4` — spacing matched to Fig 1 (top=0.86,hspace=0.31,legend y=0.975; 2-line titles need slightly more room).
- **Fig 4** `fig4_std_v4`/`fig4_raw_v4` — grey axvspan highlights on D=0.25/3.0 removed (plain white). **std-vs-raw still PENDING Patrick (deeper classification discussion).**
- **Fig 5** `fig5_v5` — single top legend (Normal once: Normal/GGG1/GGG≥2/GGG2/GGG≥3); grey + 4 distinct reds (GGG1 #fc9272, GGG2 #fb6a4a, GGG≥2 #cb181d, GGG≥3 #67000d — GGG1≠GGG2); y-label → "spectral fraction $R_j$".
- **Fig 6** `fig6_v2` — **dropped the CI-width violin row**; now 2×1 full-width (PZ top/TZ bottom), pale correct (#a6cee3/#fb9a99) vs bold misclassified (#1f77b4/#d62728), no boundary legend entry. **CI-width comparison (2.4× wider misclassified; boundary-widening partly logit geometry) MUST now go in caption/text.**
- **Fig 7 directional** `fig_directions_v4` — relaid to 4×2 (patients=rows, **PZ left / TZ right**, tissue per-cell label; P3 reversed). **Saved as fig_directions_v4 NOT fig7_v4** (Fig 7 = Fisher in confirmed scheme) — Patrick asked for fig7_v4; flag the naming/demotion (does he still want directional in SI?).
- **Fig 9** `fig9_v2` — H,I uncertainty → **plasma** (purple→yellow, higher contrast); E,F scores → **BrBG** (teal↔brown, avoids reserved red/blue/purple/yellow), parens removed from titles; colourbar labels enlarged.
- **Fig 8** `fig8_v5` — REBUILDING (background, decided 2026-06-05): **4 spectra × 2 SNR grid** (normal/tumour/bimodal/δ at SNR≈75 low | 600 high, rows=spectra, cols=SNR) + **σ̂-vs-true-σ calibration panel** (numeric true-σ x-axis, dot-free violins over ~12 realizations, identity line). Real pipeline, R̂ in subtitles, no Gibbs. Script `scripts/fig8_battery.py`, cache `results/simulation/fig8_battery_snr.npz`. **NUTS-poor-recovery-on-concentrated-truths resolved as a framing point:** it's the F8 identifiability limit (collinear intermediate bins) + F6 σ-misfit, NOT a NUTS weakness — Gibbs samples the same diffuse posterior and can't beat it; defer the fair-fight.

**figures.tex wiring DONE 2026-06-05 for the approved figures** (includes repointed: fig2_v3, fig3_v4, fig4_std_v4 [provisional], fig5_v5, fig6_v2, fig9_v2; fig1_v4 same. Captions fixed to match new content: Fig 1 CV line removed + "per-ROI posterior mean, cohort spread" clarified; Fig 2 raw-ROC + D=20/D=2.0 shown; Fig 3 rows↔cols transposed; Fig 4 star/CV-strip removed; Fig 5 two-panel emergence|aggressiveness; Fig 6 bottom row dropped, 2.4× CI-width + logit-geometry folded into caption prose; Fig 9 panel G removed). **STILL DEFERRED (need Fig 8 v5 final + Sandy + std/raw decision):** the RENUMBERING (move Fisher block supporting.tex→figures.tex pos 7 + van-Trees caption + theory.tex:61-62 reframe; move directional block →supporting.tex; swap Fig 8 include→v5 + battery caption), Fig 4 caption finalize, Results 4-pillar restructure, abstract/conclusion, Gibbs correction, Table 1 reconcile + **classification waterproofing** (std-vs-raw / raw-feature-vs-logistic / in-vs-out-sample — Patrick's "in-depth classification discussion"; do as a dedicated analysis), MRM compliance.

**Directional naming flag:** Patrick asked to name the directional figure "fig7_v4"; saved as `fig_directions_v4` because Fig 7 = Fisher in the confirmed scheme. **CONFIRMED 2026-06-05: directional → SI, keep fig_directions_v4.**

### 11.9 Round-3 refinements 2026-06-05 + NEXT-SESSION handoff

**Done this round (Patrick's 2nd detailed style pass):**
- **Fig 1** `fig1_v4` — legend gap widened (top=0.85, legend y=0.975).
- **Fig 2** `fig2_v3` — added **D=1.5 (indigo #3949ab)** as the TZ-weakest single bin, in both panels + legend (zone-labeled "weak in PZ/TZ/both"). Confirms zone-dependence: D=2.0 weak only in PZ (0.587/0.792), D=1.5 weak only in TZ (0.833/0.545), D=20 weak both. Legend now 9 entries (3×3).
- **Fig 5** `fig5_v5` — TWO colour families (LEFT warm: GGG1 #fdae6b, GGG≥2 #cb181d; RIGHT cool: GGG2 #9e9ac8, GGG≥3 #54278f; Normal grey both) + one-line subtitles ("Tumour emergence (GGG 1 vs ≥2)" / "Aggressiveness (GGG 2 vs ≥3)"), top legend above subtitles.
- **Fig 6** `fig6_v2` — contrast maximized (pale pastel correct vs saturated+large+black-edge misclassified), case-count subtitles, single x-label, fonts harmonized to 17. **DATA CHECK (Patrick asked): latest .nc (149×8000 draws), same Fig-2 pipeline, 15/149 misclassified = MINIMUM (C=1.0 ties; C-sweep 0.01→29,0.1→16,0.3→15,1→15,3→16). Uncertainty pattern GENUINE not a bug: ρ(dist,CIw)=−0.94; zero narrow-CI-near-boundary cases; wide-CI-far points = real broad posteriors; normalization sound.** TZ misclassified n=5 → CI-width separation only a trend in TZ (p=0.118), robust pooled (p=0.003).
- **Fisher = Fig 7** `fig_fisher_v2` — single TOP legend (consolidated incl. SNR floor lines, fixes panel-c overflow), bigger panels, one-line titles, "Bayesian-CRLB / NUTS gap" label kept on (b), per-bar number labels REMOVED → go in caption. **Gap factors for the caption (Bayesian-CRLB/NUTS per bin D=0.25..20): 17×, 42×, 74×, 50×, 27×, 14×, 6×, 2×.** Note: script keeps its own title=15 rcParams (overrides paper_style 17) — presentation-tuned, left as-is.
- **Fig 8** `fig8_v6` — script `fig8_battery.py` rebuilt: SNR widened to **50 / 1000** (extremes; low because pixel-wise SNR is low), **5×2 layout** (rows 1–4 spectra recovery × SNR cols; row 5 = σ̂ **box plots** per SNR, distinct colour, n-realizations noted), top-legend-only. **Re-sampling in BACKGROUND (Bash b2ri5430z) — VERIFY fig8_v6.png exists + looks right next session** (the rework agent stalled mid-sample; I relaunched the run).

**NEXT SESSION — start fresh, context was full. Priority order:**

1. ~~VERIFY Fig 8 v6 built~~ **DONE + verified 2026-06-05** (`fig8_v6.png` looks great: 5×2, SNR 50 wide / 1000 tight, box-plot noise rows split by SNR with δ above true-σ line, top legend only, max R̂ 1.017; cached so replot via `uv run python scripts/fig8_battery.py`). REMAINING: wire into `figures.tex` — Fig 8 include `fig8_v5`→`fig8_v6` + tweak caption (SNR 50/1000, box-plot noise row split by SNR; δ over-estimates σ ~1.4–1.5×).

2. **CLASSIFICATION WATERPROOFING** (Patrick's "in-depth classification discussion"; gates Figs 2/4/9 + Table 1). Do as a careful analysis, NOT fanned out. Questions:
   - **std vs raw LR coefficients** (Fig 4 `fig4_std_v4` vs `fig4_raw_v4`) — which is the honest representation? std shrinks middle bins (÷σ_j); raw lets them track ADC-sensitivity. Decide + wire the chosen one into figures.tex (currently provisional std) + finalize Fig 4 caption.
   - **single-raw-fraction vs logistic classifier** + **in- vs out-of-sample fairness vs ADC** — Fig 2 single-bin curves are now RAW max-orientation ROC (in-sample) while ADC/8-bin/2-bin are LOOCV (out-of-sample). Is that a fair mixed presentation? Reconcile.
   - **Table 1 reconcile** — single-bin AUC semantics changed (raw vs LOOCV); + the D=2.0/D=1.5 additions. Make Table 1 consistent with Fig 2.
   - **Fig 9** — confirm B,C = fractions, E,F score = which feature/classifier; consistency with the above.
   - Relevant code: `biomarkers/recompute.py`, `biomarkers/mc_classification.py`, `scripts/classifier_comparison.py`, `scripts/two_feature_lr_vs_adc.py`, `scripts/fig2_roc_detection.py`, `scripts/fig4_lr_coefs_and_sensitivity.py`. Frozen CSVs in `results/biomarkers/`.

3. **RENUMBERING .tex block-moves** (directional→SI confirmed; Fisher→Fig 7 confirmed): move directional block figures.tex→supporting.tex (repoint fig_directions_v4, rewrite caption for the 4×2 / 4-patient layout); move Fisher block supporting.tex→figures.tex at position 7 (after Fig 6, before Fig 8), repoint fig_fisher_v2, write caption INCLUDING the 8 gap factors above + "van-Trees pending Sandy"; remove the "Fisher lives in supplement" comment; **reframe theory.tex:61–62** CRLB "100–8000×" → van-Trees framing (%TODO Sandy). Semantic labels auto-renumber.

4. **Fig 6 quantification** — OPEN (Patrick wants to quantify uncertainty-as-biomarker). Recommendation to put to him: lead with **misclassified-vs-correct CI-width** (genuine: 2.4× prob / 1.3× logit, pooled p=0.003) as the biomarker signal; the **boundary-distance** effect is LARGELY logit-link geometry (ρ −0.94→−0.29 in logit space) so do NOT headline it as a biomarker. If a visual is wanted, ONE pooled correct-vs-misclassified CI-width panel (avoids TZ n=5); else keep the 2 prediction panels + report in caption/text.

5. Then the big prose pass: Results 4-pillar restructure, abstract/conclusion, Gibbs correction (theory.tex:101), MRM compliance.

### 11.10 Session 2026-06-05 (cont.) — Fig 8 wire + renumbering + prose harmonization

**Patrick decisions this session:** (1) **Fig 4 = STANDARDIZED** — `fig4_std_v4` stays wired; raw dropped (its caption already describes std, r=−0.79 PZ/−0.88 TZ; no change needed). (2) **Next-focus = prose harmonization** (over classification-waterproofing / MRM-compliance).

**DONE:**
- **Fig 8 wired** `fig8_v5`→`fig8_v6` in figures.tex; caption rewritten (SNR 50/1000 columns, per-SNR σ̂ box-plot bottom row, R̂≤1.02, δ over-estimates σ ~40–50%).
- **RENUMBERING COMPLETE** (§11.1 scheme): directional block moved figures.tex→supporting.tex = now **Fig S1** (repointed `fig_directions_v4`, caption rewritten for the 4×2 / 4-patient layout); Fisher block moved supporting.tex→figures.tex **position 7 = Fig 7** (repointed `fig_fisher_v2`, new caption: SNR=303 three-bar CRLB + the 8 Bayesian-CRLB/NUTS gap factors 17/42/74/50/27/14/6/2×). Stale "Fisher lives in SI" comment removed. Main = **9 figs** (spectra, roc, adc_discriminant, sensitivity, spectrum_ggg, uncertainty, **fisher**, **validation**, pixelwise) + Table 1 = 10 (cap ✓). Verified: every `\ref` resolves, no undefined refs.
- **CRLB van-Trees reframe** in BOTH load-bearing spots — theory.tex:61–63 and discussion.tex:41: "100–8000×" single ratio → two-mechanism decomposition (prior ~2 orders; constraint a further 2–74×). `% TODO(Sandy)` at both + in the Fig 7 figures.tex comment. **Still pending Sandy's derivation validation.**
- **GIBBS correction already done** — theory.tex:102 reads "truncated-normal full conditionals … conjugate inverse-gamma update for σ² … targets the same posterior; we use NUTS because … mix extremely slowly." The old "not conjugate" error (§8 item 3 / prior #5) is gone. No action needed.
- **PROSE HARMONIZATION:** fixed stale Fig 1 CV reference (results.tex identifiability subsection cited the removed Fig 1 CV annotation → repointed to Fig 4 shading + SI atlas); **added Fig 7 (Fisher) + Fig 8 (validation) callouts** to the Results identifiability subsection — **Fig 8 was previously UNCITED in the main body** (only in supporting.tex comments), a figure-callout violation, now fixed and tied to the F8 data-limit story. Abstract CI overclaim fixed (was "r≈−0.98, CI within [−0.99,−0.96] both estimators" — false for TZ-MAP whose CI reaches −0.93 → now "r≈−0.97 to −0.98, |r|>0.95"). Conclusion: F4b sensitivity-alignment softened ("moderate but real, near-perfect under heavy reg is a smoothing artifact"); CRLB line aligned to van-Trees. Results subsections already map to the 4 pillars — **no reorder needed**.

**⚠️ NEW CRITICAL FINDING — WORD COUNTS OVER MRM CAPS** (texcount not installed; python prose-proxy, math/equations stripped):
- **Abstract 407 / cap 250** → ~160 over.
- **Body 6053 / cap 5000** → ~1050 over: intro 431, theory 1227, methods 1052, results 1390, **discussion 1739**, conclusion 214. Fattest = discussion + theory.
- This is the **#1 submission blocker for Saturday.** Trimming is the MRM-compliance pass (Patrick deferred it this round). A tightened ~250-word abstract was drafted in-session for his approval before overwriting `abstract.tex`.

**REMAINING for Sat 2026-06-06:** (1) **word-count trim** abstract→250 + body→5000 (critical path); (2) **classification waterproofing** (Figs 2/4/9 + Table 1 raw-single-bin in-sample vs LOOCV out-of-sample fairness — Patrick's flagged #2, still open, do carefully not fanned out); (3) Fig 6 quantification decision (§11.9 #4); (4) remaining MRM compliance (data-availability + code SHA, ORCID, LLM disclosure, refs ≤6-authors-then-et-al, 3–6 keywords); (5) **Sandy** van-Trees validation (TODO markers in place); (6) **Overleaf sync** — all this session's edits are repo-only, not yet pushed.

### 11.11 Session 2026-06-05 (cont.) — JMRI evaluation + prose harmonization + word-trim

**JMRI vs MRM (Patrick weighed JMRI for faster decision, 22 vs 33 days) → DECISION: STAY MRM.** Pulled both author guidelines (Wiley live pages + MRM PDF are 402-blocked to the crawler; JMRI specs from the official ISMRM-hosted instructions PDF, cross-checked with a 2024–25 search). Specs saved to `reference_jmri_guidelines` memory. Rationale to stay: the paper is methods/theory-heavy (natural MRM fit); JMRI needs a **4000-word** body (vs MRM ~5000), a **JMRI-specific 9-part structured abstract** (Background/Purpose/Study Type/Population/Field Strength&Sequence/Assessment/Statistical Tests/Results/Data Conclusion) **+ Evidence Level + Technical Efficacy Stage**, **double-blind anonymization**, STARD, and a prescribed Discussion structure — a heavier lift than finishing MRM, and it would push the methods into SI. JMRI stays a fast fallback if MRM rejects (would be Evidence Level 3 / Technical Efficacy Stage 2).

**PROSE PASS DONE (harmonization + MRM word-trim):**
- **Abstract** rewritten + trimmed **407→266 words** (proxy; MRM cap 250), Purpose/Methods/Results/Conclusion. Fixed `|r|` overclaim (was "$r\approx-0.98$, CI within $[-0.99,-0.96]$ both estimators"; TZ-MAP CI actually reaches $-0.93$ → now "$r\approx-0.97$ to $-0.98$, $|r|>0.95$").
- **Body trimmed 6053→5216 words** (proxy; ~14%), all findings/numbers/citations preserved. Per-section now: intro 407, theory 1064, methods 953, results 1200, discussion 1378, conclusion 214. Five trim rounds (redundancy/verbosity/tangents removed; dropped the Conlin-smearing aside and the descriptive intermediate-bin GGG sentence).
- **Harmonization:** stale Fig 1 CV reference fixed; **Fig 7 (Fisher) + Fig 8 (validation) callouts added** to Results (Fig 8 was previously UNCITED in the main body); conclusion F4b-softened + CRLB→van-Trees; discussion opening F4b-softened (bin-level ADC-sensitivity alignment "moderate", the near-unity value a smoothing artifact). Results subsections already map to the 4 pillars (no reorder).
- **CAVEAT:** word counts are a python prose-proxy (math/equations stripped). The authoritative **texcount** (Overleaf) may differ — real body likely still slightly over 5000. **Final fine-tune against Overleaf texcount**; MRM "exceptional cases" clause permits minor overage.

**✅ RESOLVED 2026-06-05 — stale MAP fractions in Results corrected.** Verified against `features.csv` (recompute.py `RIDGE_STRENGTH=1e-3`): the Results tissue-spectra **NUTS** values were all correct, but every **MAP** value was a stale-low λ=0.1 leftover (never updated after the re-tune + F9 solver fix). Corrected to tuned-MAP cohort means — PZ tumor/normal D=0.25: 18.5/7.2% (was 12.0/2.5); PZ tumor/normal D=3.0: 18.9/40.3% (was 16.9/23.5); TZ tumor D=0.25: 26.4% (was 19.5). PZ-normal D=3.0 is now 40.3% ≈ NUTS 47.5%, matching Fig 1 (which loads `features.csv` at λ=1e-3 — confirmed `generate_paper_figures.py:63`) and F1. Removed the now-false "MAP smears free-water across 1.5–3.0" sentence; added a brief "MAP and NUTS agree closely on the outer compartments" note. No other stale MAP percentages remain (the Discussion "≈35%" is the intentional λ=0.1-artifact characterization).

**REMAINING for Sat 2026-06-06:** (1) ~~verify/fix flagged Results MAP numbers~~ ✅ DONE 2026-06-05; (2) **final word-count check on Overleaf texcount** + trim last ~200 if needed; (3) **classification waterproofing** (Figs 2/4/9 + Table 1 — Patrick's #2, still open); (4) MRM compliance (data-availability + code SHA already in methods.tex; ORCID, LLM disclosure, keywords); (5) **Sandy** van-Trees validation (TODO markers); (6) **Overleaf sync** — all repo-only.

### 11.12 Session 2026-06-05 (cont.) — classification waterproofing

**Pushed to origin/main:** `2749e3a` (prose pass) + `055df4a` (MAP-fraction fix). The two caption fixes below are committed-pending-push (Patrick reviewing Overleaf in parallel).

**Classification waterproofing (Patrick's #2) — ANALYZED; substantively SOUND.** Grounded in `recompute.py::run_classification` + `mc_classification.py` + `fig2_roc_detection.py`/`fig9_pixelwise.py` + a single-bin verification against `features.csv`.
- **Provenance:** auc_table.csv rows — ADC (raw rank) = in-sample `roc_auc_score` max-orientation (NO CV); every other row (ADC-LR, Full LR, 2-feat) = `loocv_auc` (out-of-sample). Table 1 values match auc_table.csv C=1.0 exactly.
- **Fair comparison = all-LOOCV** (ADC-LR vs spectral-LR): PZ ADC-LR 0.940 vs NUTS-2feat 0.933 / NUTS-8feat 0.926; TZ ADC-LR 0.964 vs MAP-2feat 0.965 / NUTS-2feat 0.937. CIs overlap → thesis (spectral ≈ ADC) holds under the fair, apples-to-apples comparison. ADC raw-rank (in-sample 0.951/0.979) is the conservative, hardest-to-beat reference.
- **Single-bin raw (in-sample max-orient) vs LOOCV-LR (out-of-sample), C=1 — verified:** informative features (ADC, D=0.25, D=3.0, mid bins) have raw ≈ LOOCV (gap ≤0.03), so the mixed Fig 2 presentation is fine; weak/empty bins have LOOCV-LR dipping BELOW chance (D=20 PZ 0.224 / TZ **0.000**; D=2.0 PZ 0.489; D=1.5 TZ 0.417) — the held-out LR cannot ORIENT a near-noise feature, NOT inverse signal. Raw max-orientation (≥0.5) is the honest descriptor → Fig 2's raw single-bin curves are justified (rationale already in fig2 docstring + methods.tex + Fig 2 caption). Zone-dependence: D=2.0 weak in PZ only (TZ 0.792), D=1.5 weak in TZ only (PZ 0.833), D=20 weak in both.
- **Fig 9** E,F per-voxel score = same LR (C=1, 2-bin {0.25,3.0} / 8-bin) + StandardScaler — consistent with Fig 2/Table 1. B,C = fractions. (PZ classifier still PROVISIONAL pending patient-8640 zone, §8.)
- **Two caption fixes applied (pending push):** (a) **Table 1** — opening falsely claimed "Entries are leave-one-out cross-validated AUCs" (the ADC raw-rank row is in-sample); reworded to carve it out + state the fair head-to-head is ADC-LR vs spectral. (b) **Fig 2** — caption named only "two weakest bins (D=2.0, D=20)" but the plot shows THREE (added D=1.5 indigo in round 3) and "near chance" was wrong for D=2.0 in TZ (0.792); reframed the three as zone-dependent. **No code/CSV changes** — purely caption accuracy.
- **Verdict: reviewer-defensible.** Optional nicety: a Results clause noting the fair comparison is ADC-LR vs spectral-LR (Methods already flags ADC raw-rank as in-sample "for reference").
