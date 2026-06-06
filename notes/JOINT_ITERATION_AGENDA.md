# JOINT ITERATION AGENDA — final MRM draft

**Created 2026-06-06 to prep the next session.** Patrick will bring (a) additional notes and (b) the MRM author guidelines; goal = produce the **final draft of the entire manuscript**.

**State:** merged + pushed (`ac1576c` on origin/main = Overleaf). Patrick's full annotation pass (**87 inline `(@patrick: …)` notes** across all sections) + Claude's figure renumbering / CRLB van-Trees reframe / MRM word-trim / MAP-fraction fix / classification-waterproofing caption fixes are all unified.

⚠️ **The 87 notes RENDER in the compiled PDF** (parenthetical text, not `%` comments) — every one must be resolved and removed before submission. Counts: theory 23, results 25, methods 22, abstract 9, discussion 5, intro 2, conclusion 1.

---

## 0. RESOLVE FIRST — manuscript restructuring (foundational; everything else depends on it)

Patrick's proposed narrative reorder (from the long results.tex note on uncertainty-aware classification):
- **Block A — How we derive the spectra & what is identifiable:** Fig 1 (cohort spectra) + Fig 7 (Fisher/identifiability) + Fig 8 (simulation validation).
- **Block B — PZ/TZ detection results:** Fig 2 (ROC) + Fig 9 (pixel-wise heatmap, exploratory) + Fig 6 (uncertainty-aware classifier).
- **Block C — "Why ADC works" post-analysis (mechanism, not performance):** Fig 3 (ADC vs discriminant) + Fig 4 (sensitivity).
- **Block D — Gleason-grading add-on:** Fig 5 (spectrum by GGG).

Current order: 1, 2, 3, 4, 5, 6, 7=Fisher, 8=validation, 9=pixelwise. **Proposed:** 1, 7, 8 | 2, 9, 6 | 3, 4 | 5.

→ This **supersedes/revisits the §11.1 renumbering** (Fisher→7, validation→8). Decide the figure + section order FIRST, then renumber + rewire once (semantic `\label`s auto-update). Also fold in: **reduce subsections → paragraphs** (Stephan's ask), and clearly demarcate **Results vs Discussion territory** (Patrick: Discussion currently regurgitates Results).

---

## 1. Themed action items (distilled from the 87 notes)

### A. Narrative balance — don't over-center the "2-bin collapse"
- Leverage the **full spectrum** more. The good/bad binary is too sharp: D=2.0 is reasonably constrained (Fig 4) and varies with grade (Fig 5); intermediate-bin CVs differ (PZ vs TZ; 0.5/1.0). Be more nuanced than "two bins good, rest bad."
- **Detection-vs-grading "two axes"**: Patrick worries it's too bold (middle bins show little *detection* signal). Needs literature support + Stephan; confirm whether the detection-vs-grading lit (`notes/lit_review_two_camps.md`) actually made it into the manuscript.
- Surface the *other* contributions beyond the collapse: the spectrum estimation itself, the uncertainty-aware classifier (a new biomarker), the GGG grading story.

### B. Concision & redundancy (Patrick: aim WELL under the cap — short is a plus)
- CV defined 3× (theory/methods/results) → once. Many parenthetical restatements flagged "redundant." Conclusion vs Discussion overlap. Results-vs-Discussion regurgitation (esp. Discussion opening).
- **Cut the historic λ=0.1 "regularizer smearing"** from the main text (Patrick + agreed: at most one SI figure; it's a historic hurdle, not a finding). Touches results ADC-sensitivity para, discussion MAP/NUTS para, theory.
- Reduce subsections → paragraphs.
- Body ≈ 5,200 words (proxy) vs MRM cap 5,000 — concision serves both goals.

### C. Framing & tone — several "too negative," reframe positively
- **MAP/NUTS**: "MAP = fast point estimate; NUTS = exploratory full-posterior tool with free uncertainty," NOT "MAP misleadingly precise" / "Bayesian gain is *not* better point estimates."
- "cannot resolve δ" → more nuanced.
- **0.8 ADC-sensitivity correlation**: Stephan says 0.8 is good — don't present it negatively.
- Conclusion "so hard to improve upon" → make it the *sensational* finding (collapse to two bins uniquely captured by ADC: D=0.25 ↑ while D=3.0 ↓ in one motion) + cite people struggling to beat ADC.
- F4b "ADC implicitly does X": Patrick worries it's too mysterious given the collapse explanation (theory.tex:117) — reconcile the "implicit alignment" wording with "it's just the two outer bins."

### D. Citations & literature pass (the deferred near-final lit deep-dive)
- citations?? markers: abstract Purpose (compartment-volume models), intro, zone-training rationale, biological/histology refs for the GGG interpretation, detection-vs-grading lit, **"ADC hard to beat" lit** (people struggling to outperform ADC), **uncertainty-quantification-in-MRI lit**.
- Also: position the **joint noise+spectra inference** novelty against the wider inverse-problem/ML literature (PROJECT_STATE §8 TODO).

### E. Methodology clarity
- Introduce **MAP** abbreviation properly for non-stats readers (decide: abstract or intro first-use).
- **NUTS-vs-MAP framing**: NUTS is a *sampler*, MAP a probabilistic *estimate* — comparing them (and labeling Fig 1 axes) is apples-to-oranges. Use "fully Bayesian (NUTS-sampled) posterior" vs "MAP point estimate."
- LOOCV abbreviation once; **explain DeLong test** (what/why) and **Welch** (what); fix vague "pipeline" wording.
- **Interval/test consistency**: Results uses bootstrap CIs, but Fig 6 reports a Welch *p*-value — unify the convention (or justify the mix).

### F. Novelty advertising (don't bury the methods)
- **Joint inference of noise σ + spectra** (possibly novel in spectral reconstruction — check lit).
- **ADC sensitivity analysis** (Stephan flagged as cool/novel).
- **Uncertainty-as-biomarker** (propagated P(tumor) CI; misclassified-vs-correct; distance-to-boundary) — frame as a new, under-explored biomarker for image-guided analysis.

### G. Factual checks / verifications (some pre-resolved in §3)
- **Pixel-wise patient**: in-cohort or "outside cohort"? Patrick thinks in-cohort (matters: LR trained on cohort ROIs). → Stephan + provenance (PROJECT_STATE §8: patient 8640 not recoverable from repo).
- **Directional count**: Methods says "13 ROIs across 7 patients" but `fig_directions_v4` shows 4 patients × 2 ROIs = 8; Patrick recalls 3 patients/6 ROIs → reconcile the true count.
- **SNR** median 303 / IQR 176–478 / range 25–1548 — re-verify (source = NUTS posterior σ; not a column in features.csv).
- **"10,000 resamples"** for the ADC–discriminant correlation CIs — locate the source (AUC CIs confirmed = 2000).
- **split-R̂**: confirm Fig 8 R̂ computed with the cited Vehtari-2021 split method.
- **Normalization-after-optimization** (MAP & NUTS): does order matter, like the earlier MAP-projection issue?
- **GGG**: 7 GGG=0 + 4 ungraded = 11 tumor ROIs not shown — why aren't the GGG=0 in the normal split?
- **Fig 6 2.4×**: re-check with the geometric LR correction (logit-space → 1.3×); emphasize the *two* phenomena (misclassified-vs-correct AND distance-to-boundary).
- **inverse-gamma σ Gibbs update** (theory.tex:101 / :94): Patrick unsure it was used (he used the voxel-count SNR formula) → reconcile with the actual sampler; ties to Sandy.

### H. Math / Sandy items (theory.tex, 23 notes)
- Retrace Eq. 2 + the Fisher derivation; the **D^3/2 rule** — does our grid actually follow it? (the 2.0-bin placement concern, which would affect the Fig 5 grading interpretation). Augmented-system notation + active-set algorithm (cvxopt?) explanation; is K defined? **van-Trees Bayesian-CRLB derivation** (the pending Sandy validation — why is NUTS so much tighter than the Bayesian CRLB?). Equation/notation correctness pass with Sandy.

### I. Figures & Table 1
- **Update the figure-justification table** (Patrick's ask — §2 below has the current version; revise after restructure).
- Possible **new SI figure**: λ-sweep (MSE / mass-recovery) to justify λ=1e-3 + show the reference-λ=0.1 smearing (call 0.1 the "reference λ"). One or two panels max.
- **Table 1**: Patrick wants single-feature rows shown, and to revisit the early multi-feature combos that "beat" ADC (why were they artifacts? — ties to waterproofing §11.12).
- **std-vs-raw Fig 4 = standardized** (decided this session) — confirm Fig 4 caption matches.

---

## 2. Figure-justification table (CURRENT numbering — revise after restructure)

| Fig | Label | One-line purpose |
|---|---|---|
| 1 | fig:spectra | Cohort tumor/normal × PZ/TZ spectra; MAP ≈ NUTS at tuned λ. |
| 2 | fig:roc | Detection collapses onto the two outer bins; spectral ≈ ADC. |
| 3 | fig:adc_discriminant | ADC ≈ spectral discriminant score at ROI level (\|r\|>0.95). |
| 4 | fig:sensitivity | Per-bin discriminant anatomy: two-bin detector aligned with ADC sensitivity. |
| 5 | fig:spectrum_ggg | Spectrum shifts with Gleason grade (detection vs grading bins). |
| 6 | fig:uncertainty | Posterior → P(tumor) with credible interval; uncertainty flags errors. |
| 7 | fig:fisher | Fisher/CRLB: why intermediate bins are unidentifiable (van-Trees). |
| 8 | fig:validation | Simulation recovery battery across spectrum shape × SNR + joint σ. |
| 9 | fig:pixelwise | Pixel-wise feasibility + per-voxel uncertainty (exploratory). |
| T1 | tab:auc | Detection AUCs: spectral matches ADC; 2 outer fractions suffice. |
| S1 | fig:directions | Direction independence validates trace averaging. |
| S-atlas | — | All-ROI posterior spectra atlas. |

---

## 3. Pre-resolved this session (so we don't re-litigate)
- **NUTS params are NOT default** (`nuts.yaml`: 2000 draws / 200 tune / 4 chains / target_accept 0.95 / init=map). PyMC defaults are 1000/1000/0.8/jitter — so tune=200 is *low*, 0.95 is deliberate careful sampling; report as specified, not "default."
- **AUC bootstrap = 2000 resamples** (`bootstrap_auc_ci n_boot=2000`), matching Methods. (Correlation-CI "10,000" still to be located — §1G.)
- **Raw-vs-LR Table 1 / Fig 2 confusion** → already addressed (PROJECT_STATE §11.12 + the two caption fixes): fair comparison = ADC-LR vs spectral-LR (all LOOCV); single-bin curves are raw because LOOCV-LR sub-diagonal dips on weak bins are orientation artifacts, not inverse signal.
- **MAP-fraction stale numbers in Results** → fixed (§11.11): tuned-MAP cohort means; MAP ≈ NUTS.
- **std-vs-raw Fig 4** → standardized (Patrick decision).
