# PROJECT_STATE — MRM submission

**Single source of truth for the manuscript state. Read this file first every session. `MEETING_PREP_2026-05-25.md` is the archived Q&A from the 2026-05-25 coauthor meeting.**

- **Last update:** 2026-05-26 after coauthor meeting (Sandy + Stephan) and MAP solver fix.
- **Target submission:** 2026-05-31 (Sunday).
- **Status:** Narrative locked. MAP solver bug found by Sandy in meeting, fixed today, cohort + simulation re-run; downstream story preserved. Three figure agents in flight regenerating Fig 1, Fig 3, Fig 4, Fig 8. Eq. 5 rewritten as constrained QP. CRLB write-up ready to email Sandy.
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

## 6. Figure TODO list

**Working assumption (subject to meeting):** rebuild figures around NUTS-for-uncertainty + MAP-tuned-for-point-estimates dual narrative, not NUTS-vs-MAP-divergence narrative.

| Fig | Status | What it shows | File |
|---|---|---|---|
| **1** | **v3 DONE 2026-05-26** | 8 representative ROIs (2 per zone×class), tuned MAP @ λ=1e-3 (NNLS-correct) vs NUTS bars side-by-side, CV-coloured. Big fonts. | `paper/figures/fig1_v3.{png,pdf}` |
| **2** | PENDING | ROC: PZ + TZ only (GGG ROC dropped, N=29 too small). 3 curves per panel (ADC, MAP, NUTS). Legends outside. Big fonts. | `paper/figures/fig_roc.{png,pdf}` (rebuild) |
| **3** | **v3 DONE 2026-05-26** | ADC vs spectral discriminant scatter, 2×2 grid (PZ/TZ × NUTS/tuned-MAP). r ≈ −0.97 to −0.98 robust across method × zone × λ. | `paper/figures/fig3_v3.{png,pdf}` |
| **4** | IN FLIGHT (agent) | Replaces old vector-correlation figure. 2-panel: LR coefficient profile per bin (raw + standardised, PZ + TZ) + comparison to ∂ADC/∂R. Sensitivity-vector framing kept (Stephan + Sandy meeting) but no longer headline. | `paper/figures/fig4_v1.{png,pdf}` |
| **5** | NEEDS PROMOTION | spectrum_by_ggg figure created in-meeting 2026-05-25. Overlay of Normal + GGG=1 + GGG≥2 spectra. Histology interpretation: lacunae loss + glandular preservation in GGG=1, full collapse in GGG≥2. Replaces old AUC-based GGG content. **STYLE PASS needed** to match Fig 1/2/3 fonts. | `results/biomarkers/spectrum_by_ggg.{png,pdf}` → move to `paper/figures/fig5_v1.*` |
| **6** | PENDING | Per-ROI classifier uncertainty figure. Layout change: 2-on-top, 1-on-bottom. Legends outside. Bigger misclassified crosses with non-light colour. | `paper/figures/fig_uncertainty.{png,pdf}` (rebuild) |
| **7** | NEEDS REWORK | Directional dependence. Use 1 representative patient from Stephan's tarball (all 10 patients) as figure + aggregate per-direction-variance stats in Results text. Clarify dots-vs-lines (Stephan's email). Possibly drop MAP if tuned MAP ≈ NUTS so visual is cleaner. | `paper/figures/fig_directions.{png,pdf}` (rebuild) |
| **8** | IN FLIGHT (agent) | NEW: 2-panel main figure. (a) MAP-vs-NUTS recovery curves on simulated GTs as a function of λ; log-normal GTs highlighted. (b) Bayesian-CRLB-corrected 3-bar comparison per D-bin (unconstrained vs Bayesian CRLB vs NUTS empirical), SNR=303. Replaces old Fisher figure. | `paper/figures/fig8_v1.{png,pdf}` |
| **9** | PENDING | Pixelwise heatmap. Drop subplot D (feature importance). 3×3 (or 4×3) layout: MAP, NUTS, ADC + spectral-discriminant + uncertainty variants. Anonymise patient ID. | `paper/figures/fig_pixelwise_v2.{png,pdf}` (rebuild) |

**Figure-count check:** 9 main figs. MRM 10-figure cap OK.

**Supplementary:**
- S1 — Posterior diagnostics (trace plots overlaid for representative ROIs, R-hat). Rebuild — current arviz-default version too small/illegible.
- S2 — Spectrum recovery for individual simulated GTs (NUTS + tuned MAP overlay, truth=black). Restyle per Stephan TODO.
- S3 — λ-sweep MSE/fraction-recovered detail (the smaller plot that complements main Fig 8 panel a).

**Outstanding cosmetic items (consistent across all figures, Stephan-perfectionist):**
- Global font sizes: xtick/ytick/axis labels = 17, title = 15, legend = 15. (Set via mpl.rcParams.)
- Legends OUT of plot panels (right side of grid or below).
- Colour palette: NUTS = orange, tuned MAP = green, ADC = grey/black, tumor points = red, normal points = blue. CV bar colours: green CV<0.4, yellow 0.4–0.6, orange 0.6–0.8, red >0.8.
- 300 dpi PNG + PDF for every figure.
- Anonymise patient IDs anywhere they appear (Fig 9 currently the only candidate).

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

## 8. Open actions for the remaining 5 days

**TODAY DONE (2026-05-26):**
- MAP solver fix in prob_model.py + recompute.py (F9).
- Cohort re-fit at λ=1e-3 (features.csv, auc_table.csv, adc_discriminant.csv etc. all regenerated).
- Simulation re-run with corrected solver (F1 preserved).
- CRLB analysis (F10) — `notes/CRLB_NOTE_FOR_SANDY.md`.
- Eq. 5 rewritten in theory.tex.
- Fig 1 v3, Fig 3 v3 regenerated. Fig 4, Fig 8 in flight.
- Spectrum_by_ggg sent to Stephan for his grant.

**TOMORROW (Wed 5/27):**
- P1. Confirm Fig 4 + Fig 8 agent outputs. Style pass to match Fig 1/3.
- P2. Promote spectrum_by_ggg to `paper/figures/fig5_v1.png` with style pass.
- P3. Edit `scripts/generate_paper_figures.py:fig_roc` — drop GGG subplot, move legends outside, keep PZ + TZ.
- P4. Rebuild Fig 6 (uncertainty): 2-on-top + 1-on-bottom, legends out, brighter misclassification colours, bigger fonts.
- P5. **Send package to Stephan + Sandy:** updated figures + CRLB note + new Eq. 5. Ask for review by end of week.

**Thu 5/28:**
- Fig 7 rework with all 10 patients from Stephan's tarball. 1 patient as figure + aggregate stats in Results.
- Fig 9 pixelwise: drop subplot D, multi-method 3×3.
- Supplementary trace plots S1.

**Fri 5/29:**
- All writing: Abstract + Conclusion rewrite (Task #12), Methods MAP-tuning paragraph (#13), Discussion histology speculation (#14), update Eq. 5 surrounding prose. Incorporate Sandy's Eq. 5 review if received.
- Address remaining 7 @Stephan inline comments.

**Sat 5/30:** Final review pass. Run latex build. Address any leftover.

**Sun 5/31:** Submit.

**Blocked / dependencies:**
- Eq. 5 final wording — waiting on Sandy's review of draft in theory.tex (Patrick will email today). Draft is good enough to submit as-is if Sandy is silent.
- Fig 7 — needs Patrick to load Stephan's full 10-patient tarball, not the 1-patient subset previously used.

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
