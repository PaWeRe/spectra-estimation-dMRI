# Manuscript Revision Plan — 3rd Pass

_Consolidated from `notes_manuscript_2nd_pass.txt` (1st/2nd pass) and the freshly
recorded 3rd-pass walkthrough of every section, table and figure._

_Date: 2026-05-13 (drafted before next working session)._

Conventions in this doc:

- **[P1]** = must resolve before next co-author circulation (correctness or core-claim risk).
- **[P2]** = important for final submission quality (style, clarity, missing context).
- **[P3]** = nice-to-have / supplementary / explicit decision needed about scope.
- **[CUT?]** = candidate to drop or move to supplement.
- **[ASK]** = explicit open question for Sandy/Stephan (or for me to research before asking them).
- _Italics in this doc_ = my critical commentary on the raw note, not a directive.

The single guiding principle for this round (per the 2nd-pass note already in
the repo): _these are sanity checks, not a prompt to reformulate everything._
Most of the manuscript is in good shape; we are hardening, not rewriting.

---

## 0. Executive priority list (do these first)

These are the items where the current manuscript is at risk of being wrong, not
just unclear. They should be triaged before any cosmetic figure work.

1. **[P1] Logistic-regression-vs-individual-feature paradox — RESOLVED
   (mechanism identified, fix pending).** Empirical check on the current
   `features.csv` (149 ROIs) confirms the asymmetry and pins it to a
   methodology mismatch in `scripts/generate_paper_figures.py:213–253`:
   per-component AUCs in Fig 2 are computed **in-sample** with
   `roc_auc_score(y, df["map_{d}"].values)` over the full PZ / TZ / GGG
   subset; the 8-feature LR AUCs on the same panel are computed
   **out-of-fold via LOOCV** (`loocv_roc(...)`).
   For a single feature with no learnable parameters this asymmetry is
   harmless (in-sample AUC equals LOOCV AUC since the held-out prediction
   is just the raw value), but for an 8-dim LR it isn't: the LR
   in-sample AUC almost exactly matches the best single bin (PZ
   0.949 vs 0.947), but LOOCV LR drops to 0.919 because the extra 7
   coefficients overfit at N = 81.

   Verified numbers (MAP features, current `features.csv`):

   | | PZ tumor | TZ tumor | GGG |
   | --- | ---: | ---: | ---: |
   | Best single MAP fraction (in-sample) | **D=0.50 → 0.947** | **D=0.50 → 0.972** | **D=0.25 → 0.817** |
   | MAP 8-feat LR — LOOCV @ C=1.0 | 0.919 | 0.946 | 0.733 |
   | MAP 8-feat LR — LOOCV @ C=0.1 | 0.938 | 0.950 | — |
   | MAP 8-feat LR — **in-sample @ C=1.0** | **0.949** | **0.973** | **0.861** |
   | ADC raw rank (in-sample, 1 feat) | 0.951 | 0.979 | 0.811 |

   The pre-LOOCV bullets in the original P1 (scaler / regularizer / leakage
   / non-identifiable bins) are not the cause: the scaler is fit inside the
   fold (`mc_classification.py:169–175`), and the regularization grid
   `C ∈ {0.1, 1, 10}` does not close the in-sample vs LOOCV gap.

   _Implications beyond fixing the figure:_

   a. **Manuscript text fix (results.tex L27).** "D = 0.25 was the
      strongest single predictor (PZ: 0.88, TZ: 0.91)" is doubly wrong:
      the numbers are stale (D=0.25 gives 0.90 / 0.95 on current
      `features.csv`, not 0.88 / 0.91) AND the strongest single bin in
      both zones is actually **D = 0.50** at 0.947 / 0.972. Either rewrite
      around D=0.50 (and accept that the figure's strongest bin is one we
      also called poorly identifiable — see Exec #3) or report the
      strongest _well-identified_ bin (D=0.25 or D=3.0, which is also a
      narrative-coherent choice) and explain the cherry-pick honestly.

   b. **Narrative consequence (connects to Exec #2 and #3).** In-sample,
      the 8-feature LR barely exceeds the best single bin
      (PZ: 0.949 vs 0.947; TZ: 0.973 vs 0.972). That means most of the
      discriminative information lives in 1–2 spectral coordinates, not
      in the joint 8-vector. The "optimal multi-compartment spectral
      classifier" framing should be softened (already flagged in Exec #2),
      and the section that motivates the LR over a single-fraction
      classifier needs an honest one-line statement of this.

   c. **Figure fix options for Fig 2 (pick one):**
      - **(A) Report LOOCV for everything.** For single features this
        equals the current in-sample number, so the per-component
        AUCs don't change; the difference is in the legend wording
        ("AUC, LOOCV"). The LR comparison then sits on the same axis
        without an asymmetry footnote. _My recommendation._
      - **(B) Drop per-component lines from Fig 2** and rely on the
        identifiability figure (Fig 1 CV annotations) plus Fig 4
        (MAP↔NUTS) to communicate per-bin behaviour. The ROC panel
        becomes pure LR vs ADC vs ADC-LR. Cleaner figure, but loses
        the "which bin is doing the work" visual hook.
      - **(C) Keep both and add a caption sentence.** Cheapest, but
        introduces the methodology asymmetry into the paper rather
        than out of it.
2. **[P1] "Optimal multi-compartment spectral classifier" claim (abstract +
   discussion).** _Calling our LOOCV-LR fit "optimal" is too strong._ Optimal
   over what hypothesis class? With this sample size (149 ROIs, 40 tumor) we
   should soften to "a fitted multi-compartment spectral classifier" or
   "the data-driven linear combination of spectral features," and reserve
   "near-optimal-in-class" language for the spectral-discriminant ↔ ADC
   comparison only.
3. **[P1] Identifiability vs. classifier weight — narrowed.** Confirmed
   empirically: the 0.50 bin gives the **highest** in-sample per-component
   AUC in both PZ (0.947) and TZ (0.972) tumor detection, and the 0.75 bin
   is also high. Per Exec #1 these are in-sample numbers, not LOOCV — but
   with single features that's mathematically the same as LOOCV, so they
   are honest. The problem is the **interpretive collision** with the
   Results §"Per-Component Identifiability" claim that D = 0.50–1.00 have
   posterior CV > 0.80 ("uncertainty comparable to the estimate itself").
   Resolve by:
   - Inspecting the LR coefficient vector for each task (still needed —
     does the LR _also_ weight D=0.50 heavily, or does it lean on
     well-identified bins instead?).
   - Re-running the LR with the non-identifiable bins (D=0.50, 0.75,
     1.00, 1.50) zeroed/merged. If LOOCV AUC stays ≥ 0.92 → the
     well-identified bins (D=0.25, 2.0, 3.0, 20.0) carry the signal and
     the figure's high single-bin AUC at D=0.50 is a coincidence of
     bin-spread driven by prior. If it drops → the prior is genuinely
     informative and we should say so explicitly.
   - The honest reading of the joint Exec #1 + #3 evidence: a high
     in-sample AUC on a high-CV bin is _expected_ when the bin's
     posterior mean is correlated with disease state even though each
     individual ROI's posterior is wide. The point estimate _separates
     groups_ even when each estimate is _individually uncertain_. We
     should say that in one sentence in Results rather than pretending
     the tension doesn't exist.
4. **[P1] SNR sanity check.** NUTS median σ across 149 ROIs gives SNR ≈ 303;
   old Gibbs-era numbers were 400–600. Either the old number was wrong (Gibbs
   used a closed-form SNR estimator with fixed voxel count, σ not jointly
   inferred) or the new one is wrong (NUTS half-Cauchy prior pulling σ up).
   Resolve by:
   - Re-deriving the old Gibbs-era SNR formula from
     `/Users/PWR/Documents/Professional/Papers/Paper3/code copy/spectra-estimation-dMRI/`
     and comparing on the same ROI.
   - Cross-checking with Langkilde 2018 SNR table (PDF in `assets/`).
   - Reporting the distribution of σ_NUTS, not just the median, and whether
     the half-Cauchy prior is informative on it.
5. **[P1] MAP equation (Eq. 5) audit.** The 2nd-pass note already flags this:
   verify `max(·, 0)` truncation, λ inclusion, and σ handling (since σ is
   assumed known for MAP, what value do we use and how does it enter U?). Do
   this by reading `models/prob_model.py` and `inference/map.py` line-by-line
   against the equation in `paper/sections/theory.tex`.
6. **[P1] Prior consistency between MAP and NUTS.** Eq. 5 uses a Gaussian
   prior on R, Eq. 6 uses a Half-Normal. _This is not necessarily wrong_ (MAP
   in log-space with a non-negativity projection can be the mode of a
   truncated Gaussian; Half-Normal is the "right" continuous analog under
   non-negativity). But the manuscript should either (a) use the same prior
   in both equations and just say "MAP = mode, NUTS = full posterior," or (b)
   explain why the two formulations are equivalent in practice. Pick one.
7. **[P1] Fisher / CRLB discretization claim.** _If our bin grid is **not**
   actually placed using the ΔD ∝ D^(3/2) spacing law we derive, we should
   either change the grid or remove the claim that the grid is informed by
   the law._ Both are defensible; mismatched theory and code is not. Check
   `configs/` for the bin definition vs the derivation in `theory.tex`.
8. **[P1] Pixel- and direction-data provenance.** Methods currently say the
   pixel-wise demo is on "an independent patient dataset." That is wrong: the
   pixel slice comes from the same BWH cohort as the ROIs (just a single
   axial slice from patient `8640-sl6-bin`).
   The directional figure was regenerated 2026-05-10 (commits 27f446b →
   b504bbf) from a per-direction `.dat` ROI file in Stephan's recent tarball
   `diff_spectrum_3_directions.tar.gz` (specifically `9283-Series12-Slice6-
   {Normal,Tumor}PZ.dat`). [ASK Sandy/Stephan] — whether patient 9283 (and
   the other tarball patients: 10203, 8804, 8805, 8864, 9322, 9675) belong
   to the same Langkilde 2018 BWH cohort as the 56 ROIs cannot be determined
   from the repo: BWH ROIs are stored under pseudonyms `new01..new56` with
   no mapping back to raw study IDs. The acquisition protocol in the .dat
   headers matches Langkilde 2018 exactly, so "same cohort, raw ID before
   pseudonymization" is plausible but not proven. _Do not assert "different
   dataset" until confirmed._
   Independent of provenance, two text fixes are required regardless:
     - Methods §"Pixel-wise and Direction-wise" (methods.tex L65) currently
       claims both pixel and direction data come from "one patient dataset
       acquired outside the patient cohort." This conflates two different
       patients (8640 for pixel, 9283 for direction) and asserts an outside-
       cohort status that isn't established for either.
     - Fig 7 caption (`fig:directions`, figures.tex L167–177) still says
       "representative prostate voxels from the supplementary patient" — but
       the current figure is per-ROI mean decays, not per-voxel, and from a
       different patient than the pixel demo. Caption needs a full rewrite
       after we settle the provenance question.

Everything below is conditional on the above being resolved (or explicitly
deferred).

---

## 1. Abstract

- **[P1] "Optimal multi-compartment spectral classifier."** See Exec #2.
  Soften the wording.
- **[P2] "Identifying a systematic MAP underestimation of the acinar-lumen
  fraction."** Confirm this is still the direction of the bias after the
  Session 8 numbers (re-read `results/biomarkers/map_nuts_comparison.csv`).
  If MAP _over_-estimates or the sign flips for some bins, rewrite. _This is
  cheap and high-leverage to verify before circulating._
- **[P2] "Per-voxel uncertainty for pixel-wise tissue characterisation."** True
  but understates the contribution — uncertainty is _also_ available per ROI
  (we use it in Fig 6 / classification). Worth one extra clause: "…and
  per-ROI uncertainty quantification for downstream classification."
- **[P3] "Fully Bayesian No-U-Turn Sampler."** _Slight category mix._ NUTS is
  an MCMC algorithm; "fully Bayesian" describes the model (joint posterior
  over R and σ with proper priors), not the sampler. Split: "We perform fully
  Bayesian inference using the No-U-Turn Sampler (NUTS), an adaptive
  Hamiltonian Monte Carlo method."
- **[P3] Keywords.** Add "Markov chain Monte Carlo" and "Hamiltonian Monte
  Carlo." _Skip "machine learning" — it's marketing here; LR + MCMC is not
  what MRM readers will search "ML" for, and it dilutes the precise
  positioning._

---

## 2. Introduction

- **[P2] ADC historical context.** Add 1–2 sentences (with citations) on why
  ADC is the de facto standard and whether anyone has questioned its
  optimality. The 3rd-pass concern is real: _it's suspicious that we're the
  first to try to explain why ADC works._ It's more likely (a) it's been
  done in adjacent fields (NMR relaxometry, IVIM) and we should cite that,
  or (b) the community treats ADC as a phenomenological tool and never asked
  the question. Either framing strengthens the paper; the current silence
  is the weakest option. Action: spend ≤2 hours on a focused literature
  pass before next session.
- **[P2] Neural-net / deep-learning approaches to multi-compartment diffusion.**
  Worth a single citation (e.g., learned IVIM/NODDI, score-based
  reconstruction). _Don't oversell — most of these are reconstruction, not
  Bayesian spectral decomposition with calibrated uncertainty._
- **[P3] "Non-negative least squares and regularised regression."** Note from
  pass: these can overlap. Reword to either "non-negative least squares
  (NNLS) and its regularised variants" or list them as a sequence of
  increasingly constrained estimators. Trivial fix.
- **[P1] "We demonstrate the framework at the pixel level on an independent
  patient dataset…"** Same factual error as Exec #8. The pixel demo is on
  patient `8640` from the BWH source data — we have not established that
  this patient is _outside_ the Langkilde 2018 cohort, only that they are
  not part of the 149 ROIs (since 8640 has no `.json` ROI signal entry).
  Reword to: "We additionally demonstrate the framework at the pixel level
  on a single axial slice from a representative patient in the same source
  acquisition." Pending confirmation per Exec #8.

---

## 3. Theory

(Order follows section flow.)

- **[P3] "Two fundamentally different estimation strategies exist."** _Worth
  citing Prange/Song et al. on inverse-Laplace style approaches if we can
  find a clean reference._ Decision needed: do we also want to demonstrate
  the alternative on our data? My recommendation: _no_, scope creep — cite,
  contrast philosophically, move on.
- **[P2] Why normalize by b=0?** Add one explanatory sentence; verify in
  `data/loaders.py` and `models/prob_model.py` that we _actually_ do this
  consistently across MAP, NUTS, and ADC. The note flags an uncertainty
  here; we should be certain.
- **[P1] Fisher information scaling law derivation.** Re-derive on paper and
  cross-check against the rendered equation. Then settle whether the bin
  grid is consistent with the derivation (see Exec #7).
- **[P1] Unconstrained CRLB computation.** If we claim to "beat" the CRLB
  by adding priors, the unconstrained CRLB must be correctly computed and
  we must be explicit that the comparison is informative-prior vs.
  noninformative. Otherwise the claim is hollow.
- **[P2] MAP Eq. 5 — σ entry into U.** Add a sentence: "Because σ is treated
  as known under the MAP formulation, it enters via the noise-weighted
  design matrix Ũ = U/σ; we use the residual-based plug-in estimate σ̂
  obtained from [exact procedure]." Then confirm this matches the code.
- **[P2] MAP vs NUTS prior wording.** See Exec #6. The line _"smooth gradients
  for HMC, and its variance 1/λ"_ should pick up the √ if Eq. 6 uses √λ —
  cross-check.
- **[P2] "Direct link between MAP and Bayesian" callout for radiology readers.**
  _I lean: keep the existing wording, do not add the "MAP is the mode of a
  truncated multivariate normal" sentence._ It's accurate but introduces
  vocabulary (truncated MVN) we then don't use elsewhere. Reserve for a
  footnote if anyone asks.
- **[P3] Italic "rather than fixed a priori" phrasing.** _It's fine. Keep._
  HalfCauchy on σ is a prior, yes; the italic just emphasizes we are not
  fixing σ, which is one of our methodological points.
- **[P3] HalfCauchy choice (carryover from 2nd-pass notes).** Add one sentence
  explaining why (Gelman 2006 weakly-informative scale prior; heavy tail
  permits unexpectedly large σ without dominating posterior). _Don't
  re-litigate alternatives in the main text._ A supplementary sensitivity
  to Inverse-Gamma vs HalfNormal vs HalfCauchy would be ideal but is [P3].
- **[P3] Conjugate pair / analytic solution to truncated MVN.** _Mention only
  if it fits in a clause; don't open this rabbit hole in the main text._
- **[P3] Why NUTS over VI / amortized inference.** I'd add 2 sentences in
  theory, _not_ a comparison plot. Something like: "Approximate inference
  methods (mean-field VI, normalizing flows, amortized neural posteriors)
  are attractive computationally but tend to underestimate posterior
  variance in tightly correlated, partially identified problems like ours;
  we therefore use NUTS, an asymptotically exact sampler, as our reference
  inference." This handles the 2nd-pass concern about motivating NUTS
  without adding a Gibbs-vs-NUTS-vs-VI section. The longer history goes in
  the discussion's "future directions," not theory (see §6).
- **[P2] Confirm ADC is computed via log-linear fit.** Trivial verification
  task; the manuscript states it, so the code must match. Check
  `biomarkers/adc_baseline.py`.
- **[P1] LR sensitivity-vector implementation.** _This is the engine of the
  ADC-spectral-discriminant equivalence claim, the centerpiece of the paper._
  Cross-check `biomarkers/mc_classification.py` against the equation in
  theory.tex. Verify normalization, scaling, and the per-feature vs
  full-feature interpretation. _Also flagged in Exec #1._

---

## 4. Methods

- **[P1] Pixel + direction data provenance.** See Exec #8. Single most
  important Methods fix. Direction figure is now per-ROI mean from patient
  9283 (Stephan tarball); pixel figure is from patient 8640. Whether either
  belongs to the BWH cohort is an open [ASK] item, not a known fact.
- **[P2] How the pixel maps are computed.** Spell out explicitly: are LR
  coefficients fit once on all ROIs and then applied per-voxel? Or
  separately per zone/task? Per the note, this is currently ambiguous to
  the reader. One paragraph adds clarity at no risk.
- **[P2] 500 MC samples for pixel-wise uncertainty — justify or revisit.**
  Either show that the discriminant histogram is stable at 500 (probably
  true) or bump to a round 1000. Cheap.
- **[P2] Bootstrap CI for ROI-level ADC↔discriminant correlation.** Add a
  half-sentence explaining the procedure ("non-parametric percentile
  bootstrap over ROI pairs, B = 10,000"). This is standard, just spell it
  out. _Separately_, the 3rd-pass note raises whether the bootstrap CI is
  the right uncertainty quantity here at all, given that we have MCMC
  posterior samples — see §5 / Exec re: carrying uncertainty through.
- **[P2] Hanley-McNeil method.** One sentence: "We compare paired AUCs using
  the Hanley-McNeil method, which accounts for correlated test statistics
  computed on the same subjects." Don't justify exhaustively.
- **[P2] LR coefficients fit on entire dataset vs the per-ROI ADC sensitivity
  vector.** _The note is right to flag this asymmetry._ The honest framing:
  the LR coefficients are estimated once at the population level (so that
  we have a single "discriminant" to interpret), while the ADC sensitivity
  vector is a derivative of a per-ROI single-number summary. Add a clarifying
  sentence; this is not a bug but it does need to be stated.
- **[P2] "Spectral discriminant" — define term on first use.** The word
  "discriminant" appears in figures and text but is not crisply defined.
  One sentence: "We refer to the population-level LR linear combination of
  spectral fractions as the **spectral discriminant**; it maps an 8-dim
  spectrum to a single tumor-likelihood score."
- **[P3] Carrying MCMC uncertainty into the classifier (the original ISMRM
  2025 idea).** _This is the most interesting methodological idea Patrick
  raises across both notes._ Concretely: instead of a single point estimate
  per ROI, run 2000 NUTS draws through the LR discriminant and report a
  full distribution of tumor probabilities per ROI. We have a figure that
  hints at this (uncertainty higher for misclassified ROIs). Decision:
  - **Conservative (recommended for this submission):** Keep the current
    framing; mention the per-ROI discriminant uncertainty in §Results
    where Fig 6 already lives. Don't rebuild the classifier.
  - **Aggressive:** Add identifiability features (per-bin CV from NUTS) to
    the LR feature set and re-run AUC. Could legitimately boost performance
    and would justify NUTS over MAP. _But this changes the central result
    table at a late stage of the paper._
  My recommendation: _conservative for this paper_, explicitly flag the
  aggressive variant in Future Directions as a follow-up.

---

## 5. Results

- **[P1] LR-vs-individual-feature anomaly.** Exec #1 — mechanism is the
  in-sample-vs-LOOCV asymmetry, not a code bug. Results.tex L27 also has
  stale single-component AUC numbers (0.88 / 0.91) that must be replaced
  with current `features.csv` numbers (D=0.25 gives 0.90 / 0.95;
  D=0.50 is strongest at 0.947 / 0.972). Decide single-bin to spotlight
  and rewrite L26–27 accordingly. Connect to Exec #2 (don't call the
  8-LR "optimal") and Exec #3 (the strongest single bin is the
  poorly-identified one).
- **[P2] MAP-NUTS correlation (r ≈ 0.985–0.994) is a double-edged sword.**
  3rd-pass note correctly identifies this: if MAP discriminant ≈ NUTS
  discriminant, why bother with NUTS? Options:
  - Lean on the σ-inference contribution (NUTS gives calibrated σ, MAP
    doesn't).
  - Lean on the per-ROI uncertainty (NUTS gives it, MAP doesn't).
  - Show NUTS-derived identifiability changes the classifier (aggressive
    variant from §4).
  Pick one or two; do not retreat to all three or the message blurs.
- **[P2] "LR coefficients become unstable" (r = −0.788 at PZ tumor).** Replace
  "unstable" with something measurable: "the LR coefficient vector flips
  sign for the partially identified mid-diffusivity bins between MAP and
  NUTS inputs" (or whatever the actual mechanism is — verify in
  `auc_table.csv` / coefficient logs).
- **[P2] "Can be confidently interpreted as free-water diffusion in the
  acinar lumen."** Repetition flag from the note. Check this phrasing
  doesn't appear in 3+ places; consolidate to one.
- **[P1] SNR result statement.** Tied to Exec #4. If we report median 303
  and the Gibbs-era number is meaningfully different, either:
  - Resolve and report one number with provenance.
  - Report both side-by-side with an explanation.
  Don't paper over the discrepancy.
- **[P3] Gibbs-era closed-form σ.** Sandy's applied-math derivation. _For
  this paper: omit from main text._ One sentence in supplement is enough.
  The point is that we now jointly infer σ; we don't need to defend that
  choice in detail.
- **[CUT?] p-value justification in the ADC vs spectral-discriminant
  section.** Note is right — drop the "we are not using p-values because…"
  paragraph. Just report effect sizes and CIs and move on.
- **[P2] Conciseness pass on Results.** The note repeatedly observes we
  re-state "mid-bins are not identifiable" in three places. Pass through
  the section and consolidate to one canonical statement (probably in the
  spectra subsection), reference back elsewhere.

---

## 6. Discussion

- **[P2] ADC perception in the clinical community.** Add references on how
  radiologists currently view ADC (use rather than challenge it). _Tone-wise,
  the right framing is "ADC is already pretty good, here's why, and here's
  what spectral analysis adds" — exactly what the note proposes. Don't
  argue against ADC; explain it._
- **[P3] "Three areas of added value."** If after resolving Exec #1+#3 we
  find a meaningful classification improvement for intermediate-GGG
  distinctions, add it as a fourth area. If not, keep the current three
  and don't oversell.
- **[P1 wording / P2 content] "Does not affect tumor–normal discrimination"
  for the 3.0 bin.** The note is correct — this is internally inconsistent
  with claiming the spectral discriminant uses the full spectrum. Resolve
  in one of two ways:
  - If the LR coefficient on 3.0 is effectively zero for tumor–normal,
    state that explicitly and discuss why the other bins carry the signal.
  - If it's not zero, soften the claim to "is not the dominant contributor
    to tumor–normal discrimination, though it is informative for
    physiological interpretation."
- **[P2] Colin et al. comparison.** _I don't trust this paragraph until I
  re-read Colin._ Action: pull the Colin reference, read the relevant
  section, confirm what we are saying about their work is accurate. Also
  scan for more recent (post-2020) multi-compartment prostate work to cite
  alongside.
- **[CUT?] CRLB / priors recap in Discussion.** The note flags this as
  redundant with Methods/Theory. Agreed — trim to one paragraph that
  _interprets_ rather than _re-derives_.
- **[P2] Pip-installable package / Zenodo angle (from 2nd-pass notes).** One
  to two sentences in Future Directions. Already drafted by Patrick in the
  2nd-pass notes; just lift verbatim. _Don't overclaim — explicitly say
  "intended to grow into" a package._

---

## 7. Figures (in current numerical order)

General style fixes that apply to multiple figures (apply once, don't
re-litigate per figure):

- **Color-coded MAP/NUTS or normal/tumor legend** sits as a row above the
  subplots (Fig 1 pattern). Apply to Figs 3, 4, 5, 6.
- **Yellowish in-axes legend boxes** (current p-value/correlation overlays):
  remove. Move the relevant number to the subplot title or the caption.
- **Subplot sizing**: where there are 3 subplots, use 2-on-top + 1-below
  (Fig 1 layout). Apply to Figs 5 and 6.
- **Sanity check for repetition**: each figure must have a 1-sentence
  justification in the caption ("this figure shows X _because_ Y, and is
  not redundant with Fig Z"). If we can't write that sentence, the figure
  is a candidate for cut.

### Fig 1 — MAP vs NUTS spectra (reference style)
- No changes. This is our style template.

### Fig 2 — ROC / AUC by zone and task
- **[P1] LR-vs-individual-feature anomaly — mechanism known, fix pending.**
  See Exec #1: per-component lines are in-sample, LR lines are LOOCV.
  Pick figure-fix option A / B / C from Exec #1 before regenerating.
- **[P1] Diagonal line in 0.5/0.25 ROC curve (PZ tumor subplot).** ROC
  curves should be staircase (horizontal/vertical only) for finite samples.
  A diagonal suggests interpolation or a bug. Investigate.
- **[P2] AUC legend occluding the ROC curves.** Move outside the axes or
  shrink. Larger fonts (matches recent fig_roc commits — see commit
  17d107d).
- **[P2] AUC ≈ 0.95 for 0.5 bin in PZ tumor while the 0.5 bin is
  non-identifiable.** Confirmed: D=0.50 is the strongest per-component
  AUC (PZ 0.947, TZ 0.972) and yet has cohort-averaged posterior CV >
  0.80. Tied to Exec #3. Resolution path is in Exec #3, not in the
  figure itself.

### Fig 3 — ADC vs spectral discriminant scatter
- **[P2]** Unify normal/tumor legend at the top.
- **[P2]** Drop yellow in-axes box; move correlation + p-value into title.
- **[P3]** Investigate the 2 disagreement points per zone (PZ + TZ).
  _My recommendation: skip the deep dive for this submission._ With ~149
  ROIs and 2 outliers per zone, anything we say will be n=4 storytelling.
  Mention in the limitations as "occasional ROIs where ADC and the
  spectral discriminant diverge; investigation deferred to larger cohorts."
- **[P3]** Add NUTS to this figure? _My recommendation: no, keep MAP-only.
  The MAP-NUTS correlation is already shown in Fig 4; mixing it in here
  doubles the visual load without adding a new claim._
- **[P3]** GGG binary tasks. _Skip for this submission._ With our GGG
  counts the figure would be underpowered, and it pulls the paper away
  from the tumor-vs-normal narrative.

### Fig 4 — MAP↔NUTS comparison
- **[P2]** Unify legend; drop yellow box.
- **[P2]** Color palette: pick something that doesn't collide with the
  MAP/NUTS color encoding from Fig 1. _Suggestion: keep MAP/NUTS as the
  global color axis everywhere; never use the same hues for normal/tumor.
  That probably means normal/tumor as fill vs outline, or marker shape._
- **[P2] Repetition check.** What does Fig 4 say that Fig 3 doesn't? Write
  the one-sentence justification. _If the answer is "Fig 3 shows ADC vs
  spectral discriminant, Fig 4 shows MAP vs NUTS coefficient vectors,"
  then both are needed but the captions must make that distinction loud._

### Fig 5 — (likely spectra by group / robustness?)
- **[P2]** Apply global style (top color legend, no yellow box, 2+1 layout
  to match Fig 1 sizing).
- **[P2] Justification one-liner.** Write it; if it overlaps too heavily
  with Fig 1 or Fig 4, consider cutting.

### Fig 6 — Per-ROI classifier uncertainty
- **[P2]** Style: top legend, no yellow box, 2+1 layout.
- **[P2] Add MAP and ADC predictions per ROI (numbered) to show that
  uncertainty-flagged misclassifications also fail under the simpler
  methods.** _This is a strong addition if it can be done without visual
  clutter — it directly answers "why bother with NUTS uncertainty?"_
  Caveat: don't overload the figure; if it gets messy, put the
  cross-method confusion table in supplementary and just reference it.

### Fig 7 — (check; probably ADC / discriminant correlation per cohort)
- _No specific 3rd-pass note. Run the same repetition / style check._

### Fig 8 — Fisher information / correlation matrix
- **[CUT?]** _Strong candidate for supplementary._ The note's instinct is
  right: Fisher information is theoretically nice but we are not actually
  using the spacing law to define the bin grid (see Exec #7). The
  correlation matrix is interesting in isolation but not load-bearing for
  the main story.
- **[P3]** If kept in main text: cut to one panel (the standard-deviation
  reduction vs. unregularized), drop the full correlation matrix. The
  noise-floor / component-decay threshold is informative but already
  carried by the SNR discussion in Methods/Results.
- **Decision needed:** keep one panel in main / move all to supplement /
  cut entirely. _My vote: one panel in main (the σ-reduction message),
  rest to supplement._

### Fig 9 — Pixel-wise heatmaps
- **[P2]** Subplot sizing/ratios to match Fig 1.
- **[P2 / decision]** Drop subplot D. _Agreed with the note: D rehashes
  ROI-level content from Fig 4/6 and breaks the heatmap framing of Fig 9.
  Move to supplementary or delete._
- **[P3]** Add NUTS heatmap as a comparator. _Tradeoff: more informative
  but more compute-heavy and visually denser. My vote: keep MAP-only in
  main, NUTS heatmap in supplement; mention in caption that the supplement
  has the NUTS version._
- **[P3] Interpretation.** Add 1 sentence connecting uncertainty pattern
  to lesion aggressiveness / boundary regions. Don't claim causality.

### Direction-encoding figure (Fig 7, `fig:directions`)
- **[P1] Provenance.** Current figure is per-ROI mean spectra from
  `9283-Series12-Slice6-{Normal,Tumor}PZ.dat` in Stephan's tarball. Whether
  patient 9283 sits inside or outside the Langkilde 2018 BWH cohort cannot
  be answered from the repo (raw IDs vs `new01..new56` pseudonyms). [ASK
  Sandy/Stephan] before finalising the figure caption + Methods §"Pixel-wise
  and Direction-wise Spectral Estimation."
- **[P1] Caption is stale.** Caption (figures.tex L167–177) still claims
  "representative prostate voxels from the supplementary patient" while the
  underlying figure is now ROI-level from a different patient than the pixel
  demo. Full caption rewrite after provenance is confirmed.
- **[P2]** Same style pass as the rest.

### S2 — Simulation
- **[P2]** Restore the ISMRM-era blue-lines + star-init + median-overlay
  style. The current version doesn't read well.
- **[P3]** SNR posterior across realisations (from rejected ISMRM 2025
  abstract) → add to supplement as a noise-recovery validation figure.
  _This directly addresses the SNR sanity check in Exec #4 — same figure,
  different framing._
- **[P3]** Trace plots and posterior diff plots: keep one canonical trace
  (e.g., R_0.25 and σ) and a single posterior diff. Don't carry all of
  them — concision over completeness in the supplement.

---

## 8. Tables

### Table 1 — Classification performance
- **[P3] GGG-specific AUC table.** The note flags the old
  `025+1:2+ 1:250_ggg_stat_analysis_combined.csv` showed
  configurations beating ADC on intermediate GGG distinctions. _Decision
  needed:_
  - **Option A (conservative, recommended):** Stay tumor-vs-normal. Mention
    intermediate-GGG as future work in Discussion.
  - **Option B (interesting but risky):** Add a small supplementary table
    with GGG-binary AUCs and a heavy caveat about sample size. Don't put
    it in the main text.
  My vote: Option B. The intermediate-GGG ADC-failure story is genuinely
  interesting and aligns with published literature; keeping it in
  supplement protects us from overclaiming while preserving the hook.
- **[P1] After resolving Exec #1**, regenerate the main AUC table from
  `recompute.py`. Don't hand-edit numbers.

---

## 9. Items moved out of scope (carryover from 2nd-pass)

These are explicitly _not_ blocking the next submission, but worth noting
so we don't lose them:

- **Gibbs / VI history paragraph in Methods.** _My recommendation: do not
  add to main text._ The 2nd-pass note (line 7) acknowledges this isn't
  the main story. A short paragraph in Discussion → Future Directions
  saying "we explored Gibbs sampling and variational inference; NUTS gave
  the best convergence properties for this problem" is sufficient.
  Reserve the trace-plot comparisons for supplementary _only if a reviewer
  asks._
- **Variational inference revival.** Explicit future work item, not a
  current paper deliverable. Mention in Discussion alongside the
  pip-installable-package angle.
- **Identifiability features in the classifier (aggressive variant).** See
  §4. Future-work flag for now.

---

## 10. Repo / data sharing (already in 2nd-pass notes; carry forward)

- Confirm `signal_decays.json` is BWH IRB / DUA clean before public release.
- Decide on Zenodo deposit + DOI for the methods-section code-availability
  sentence.
- README still TODO: install (`uv sync`), reproduce features.csv
  (`uv run python -m spectra_estimation_dmri.biomarkers.recompute`), regenerate
  figures, IRB note.
- Pip-installable trajectory → Discussion / Future Directions, do not
  overclaim.

---

## 11. Open questions to raise with Sandy / Stephan

Group these so we can batch them into a single message rather than 10
separate threads.

**Core methodology (Sandy):**
1. Eq. 5 prior choice and σ handling — does Sandy still endorse the current
   formulation after the joint σ inference change?
2. Fisher information / CRLB derivation — is the ΔD ∝ D^(3/2) spacing law a
   _description_ of optimal grids or a _prescription_ we should be following?
3. Old Gibbs-era closed-form SNR estimator vs. joint NUTS σ — is there a
   conceptual reason to prefer one over the other? Or is the change a clear
   methodological improvement?

**Clinical positioning (Stephan):**
4. ADC current clinical perception — is ADC viewed as "solved" or "good
   enough but limited"? Affects Intro and Discussion framing.
5. Intermediate-GGG ADC failure literature — does Stephan know the standard
   references? Affects Table 1 Option A vs B decision.
6. Pixel-data and direction-data cohort identification — confirm provenance
   for both before we put it in Methods. Specifically:
   - Pixel demo: patient 8640 (slice 6). Is this one of the 56 Langkilde
     2018 BWH patients (and if so, which `newXX` pseudonym), or a separate
     acquisition?
   - Direction figure: patient 9283 + the rest of Stephan's recent tarball
     (10203, 8804, 8805, 8864, 9322, 9675). Same cohort as the 149 ROIs?

**Both:**
7. Is the MAP↔NUTS near-equivalence a feature or a bug for the paper's
   narrative? Their preference will guide Exec #2 wording.

---

## 12. Suggested order of operations for the next session

1. Resolve Exec #1 (LR vs individual feature). Until this is fixed, the
   AUC table, Fig 2, and several Results paragraphs are in limbo.
2. Resolve Exec #4 (SNR sanity check). Cheap and unblocks Results §SNR.
3. Resolve Exec #5+#6 (Eq. 5 audit + prior consistency). Theory section
   text edits follow naturally.
4. Resolve Exec #7+#8 (Fisher discretization claim + pixel data
   provenance). Methods text edits follow.
5. Cosmetic figure pass (top legends, no yellow boxes, 2+1 layout).
6. Section-by-section text edits in order: Methods → Results → Discussion
   → Intro → Abstract.
7. Final consistency / repetition pass.

---

_End of 3rd-pass plan. Source materials: `notes_manuscript_2nd_pass.txt`
(retained alongside this file) and recorded 3rd-pass walkthrough._
