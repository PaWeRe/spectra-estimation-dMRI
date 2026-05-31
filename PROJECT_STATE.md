# PROJECT_STATE — MRM submission

**Single source of truth for the manuscript state. Read this file first every session. `notes/archive/MEETING_PREP_2026-05-25.md` is the archived Q&A from the 2026-05-25 coauthor meeting.**

- **Last update:** 2026-05-31 — figure-scoping session: pillars + full main/supplementary figure plan reworked (see §6).
- **Target submission:** next week (was 2026-05-31). Today's goal: finalized draft for Sandy + Stephan review.
- **Status:** Figure scope locked (§6). Fig 3 + Fig 5 done and wired into the manuscript. Reworking remaining figures sequentially (Fig 1 → Fig 4 → ...), then manuscript text. Two-camps literature review running in background (→ `notes/lit_review_two_camps.md`). Old Fisher/CRLB Fig 8 dissolved into supplementary; new main Fig 8 = method-validation (spectrum recovery + joint noise inference).
- **Locked claims** (no longer under revision):
  - F1 — Tuned MAP (λ=1e-3) recovers log-normal spectra ≥0.98 fraction, matching NUTS. Confirmed with corrected solver.
  - F4b — ADC-sensitivity ≈ inverted-LR-coef r ≈ −0.98 elegance was a regulariser artifact (drops to −0.80 at tuned MAP and NUTS). Comparison kept in revised form (per Stephan/Sandy 2026-05-25 meeting).
  - F8 — Intermediate bins are data-limited, not prior-limited.
  - F9 (NEW) — MAP ridge branch in prob_model.py was projecting unconstrained Gaussian MAP onto non-negative orthant rather than solving constrained QP. Bug active on 58% of ROIs, p99 bin diff 0.37. Fixed 2026-05-26 to NNLS on augmented [U; √λI] system. Cohort re-fitted; ROI-scalar ADC-vs-discriminant r ≈ −0.97 robust.

**The manuscript narrative is reshaped around what NUTS uniquely contributes:** per-bin posterior uncertainty (with F6 coverage caveat). Tuned MAP is the point-estimate workhorse.

---

## 1. Locked one-line thesis (under revision, see §4)

> **Bayesian spectral decomposition with per-bin posterior uncertainty resolves the compartment-volume mechanism underlying ADC's clinical success.** At the ROI level (N=149), the spectrum is diagnostically equivalent to ADC for tumor detection and grading. The spectrum's added value is (a) per-bin posterior uncertainty (well-calibrated on smooth ground truths; over-confident on δ-spectra — see F6), (b) a model-free fixed grid that — distinct from prior free-floating biexponential fits — expresses tissue change as a *fraction shift at canonical D values* rather than as a drift in fitted D, and (c) an explicit decomposition of the detection axis (outer bins) from the grading axis (intermediate + lumen bins) in the univariate per-bin profile.

**Title under consideration (already in repo, may need refresh):** "Why ADC Works: Bayesian Spectral Decomposition of Prostate Multi-b Diffusion MRI"

**Scope honesty.** All quantitative results are at ROI level (149 ROIs). Pixel-wise heatmap (Fig 9) is MAP-only feasibility demo on one slice from one patient — *not* a delivered result. Per-voxel multi-compartment is already done by HM-MRI/VERDICT/rVERDICT/RSI/LWI with parametric constraints; **our novelty is model-free grid + Bayesian uncertainty + axis separation at the ROI level**, not per-voxel mapping.

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
| 2 | REWORK | ROC: PZ + TZ detection only (GGG ROC dropped). Legend on top, one row. |
| 3 | ✅ DONE | ADC vs discriminant, 2×2 (PZ/TZ × NUTS/MAP), shared axes, legend top, no title. `fig3_v3`. |
| 4 | REWORK (centrepiece) | Per-bin standardised LR coef (bars) + −∂ADC/∂R (charcoal line) on a SINGLE normalised axis (no dual-axis offset); bars COLOURED by identifiability (purple sequential). 2 panels (PZ, TZ). Story: outer bins = high weight + ADC-aligned + well-identified; intermediate bins = where LR/ADC diverge AND poorly identified — yet Fig 5 shows they still move with grade. Complement to Fig 3 (Fig3="≈ADC"; Fig4=per-bin anatomy + where it breaks). Drop raw-vs-standardised duality (standardised only). |
| 5 | ✅ DONE | Spectrum by GGG (GGG=1 vs ≥2). `fig5_v2`, width 0.85\textwidth. Possible extension: 2nd panel GGG≤2 vs ≥3 (matches one-row ROC). |
| 6 | REWORK | Uncertainty-aware classifier. Restyle (kill rejected-ISMRM leftovers) + recompute: propagate MCMC samples through LR → probability + CI; misclassified ↑ uncertainty. |
| 7 | REWORK (MAIN) | Direction independence (trace-averaging validation). 1 representative patient + aggregate per-direction stats in text. Needs Stephan's 10-patient tarball. |
| 8 | NEW (MAIN) | Method validation, 3-panel row: (a) recovery of expected spectrum — truth vs NUTS±90%CI vs tuned-MAP; (b) contrasting GT (concentrated/bimodal); (c) joint noise (σ) posterior over realisations vs true σ. Takeaways: reconstruct realistic spectrum, two methods, joint Bayesian noise inference. **ADD GIBBS as a 3rd method** (next session, see §8): show NUTS > Gibbs (Gibbs fails — too correlated / poor mixing); frames Gibbs as the first exploratory method that didn't work. Use `src/spectra_estimation_dmri/inference/gibbs.py`. (Old Fisher/CRLB Fig 8 DISSOLVED → supp.) |
| 9 | REWORK (MAIN) | Pixelwise feasibility demo. Drop subplot D. Multi-method MAP/NUTS/ADC + discriminant + uncertainty. Anonymise patient ID. |
| Table 1 | keep | AUC table. |

**Supplementary:**
- S1 — ⚠️ **REWORK NEEDED (next session — Patrick not satisfied, see §8 deferred items).** Current draft = `scripts/figS1_all_roi_spectra.py` → multi-page `figS1_all_roi_spectra.pdf` (15 pages, 2 cols, Gleason-score+GGG titles), embedded via `\includepdf` (needs `\usepackage{pdfpages}`). Issues: revert to box-plots-of-posterior; top legend cut off + no y-axis label (PDF layout broken); reconsider patient numbering for traceability. Low priority — after the main figures.
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

**Done this session (cont.):**
- Two-camps literature review complete → `notes/lit_review_two_camps.md` (web-verified DOIs/PMIDs). **Reconciliation = detection vs grading:** fair PI-RADS ADC is hard to beat for DETECTION (Camp A — Fennessy & Maier 2023 Eur J Radiol is *Stephan's own* paper; He et al. 2025 = `assets/s00261...` is the cleanest fair head-to-head, ADC = MC for detection); cellular-compartment metrics (VERDICT fIC, DKI, MC restricted fractions) beat ADC for GRADING (Camp B), because grading signal lives in intracellular/intermediate diffusivities. **KEY CORRECTION:** the RSIrs detection gap (ADC AUC 0.48–0.54) is from an automated whole-gland *minimum* ADC vs radiologist-localized lesion ADC — NOT high-b vs PI-RADS b-values (all headline Camp B papers used b≤1000). Lead the Discussion reconciliation with Maier 2023.
- Fig 1 rebuilt as `fig1_v4` (cohort box plots, MAP green-hatched / NUTS orange-solid, grey mean-CV annotation, PZ/TZ titles, legend top). Wired into figures.tex (caption rewritten, "ridge smoothing" softened to λ-dependent).

**TODO — near-final literature deep-dive (one of the LAST steps, before submission):**
- Extend the lit review beyond medical MR: general signal-reconstruction / inverse-problem literature + ML approaches (variational inference, amortized / learned inference for spectral or parameter estimation). Goal: position our fully-Bayesian *joint spectrum + noise* inference against the wider methods landscape. Do once main text + final figures are locked.

**Figure reworks:** ~~Fig 1 (fig1_v4)~~ ✅, ~~supp S1 all-ROI spectra~~ ✅ → **Fig 4 (NEXT SESSION)** → Fig 6 → Fig 2 → Fig 7 → Fig 8(new) → Fig 9 → supplementary S2–S6.

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

*2. Simulation/validation figure (main Fig 8) — ADD GIBBS as a third method:*
- Compare Gibbs vs NUTS vs tuned-MAP. Use the existing Gibbs code `src/spectra_estimation_dmri/inference/gibbs.py` (also in the old "code copy" repo). Show **NUTS outperforms Gibbs**: Gibbs fails because the chains are **too correlated** (extremely slow mixing, does not reliably converge to the target). Frames Gibbs as the first exploratory method that didn't work → motivates gradient-based NUTS/HMC. (Promotes the parked VI/Gibbs note in §9 to a main-figure + paragraph.)

*3. Manuscript CORRECTION (Gibbs) — current text is WRONG:*
- The manuscript claims there is **no conjugate prior / no closed-form solution** (used to justify NUTS). This is **incorrect** — Sandy implemented Gibbs with **closed-form full conditionals** (univariate truncated normals are conjugate); Patrick ran Gibbs for months. The real problem was **mixing** (very slow convergence, not necessarily to the right posterior), NOT lack of a closed form.
- Find the claim (likely `theory.tex` / `methods.tex`) and correct it; add a short paragraph on the Gibbs→NUTS progression, tied to the Fig 8 Gibbs comparison.

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
