# Inline comment inventory + figure map + MRM compliance

Parsed 2026-06-07 from `paper/sections/*.tex`. ~55 inline `(@patrick…)` notes, grouped
by theme so repetition is visible. Tags: ✂️ cut/concise · 🔧 mechanical · 🔢 verify code/numbers · 👥 ask co-author · 🧠 conceptual decision · 📚 citation.

## Figure / table map (in-paper number ↔ label)

| # | label | content | Patrick's notes call it |
|---|---|---|---|
| Fig 1 | `spectra` | tumor/normal mean spectra | fig1 |
| Fig 2 | `roc` | detection ROC / AUCs | fig2 |
| Fig 3 | `adc_discriminant` | ADC vs discriminant correlation | fig3 |
| Fig 4 | `sensitivity` | per-bin ADC sensitivity vs LR weights | fig4 |
| Fig 5 | `spectrum_ggg` | spectra by Gleason group | fig5 |
| Fig 6 | `uncertainty` | uncertainty-aware classification | fig6 |
| Fig 7 | `fisher` | Fisher matrix / CRLB | fig7 |
| Fig 8 | `validation` | simulation recovery | fig8 |
| Fig 9 | `pixelwise` | pixel heatmaps | fig9 |
| Table 1 | `auc` | AUC table | Table1 |
| Fig S1 | `directions` | direction-wise check | supplementary |

**9 figures + 1 table = 10 = MRM hard limit. No room to add to main text.**

## Themed comments

### 1. ✂️ Conciseness / redundancy / fewer subsections (most repeated, ~14)
- discussion:2 — whole Discussion weak/redundant/shallow; wants razor-sharp points 🧠
- discussion:5 — first para regurgitates Results; mark Results vs Discussion boundary
- discussion:10 — last sentence redundant
- discussion:17 — redundant with Results; also too pessimistic (tone)
- discussion:21 — "historic artifact / old smearing" para weak; drop old smearing
- conclusion:2 — redundant with Discussion; unclear what belongs where
- results:12 — fewer subsections → paragraphs (Stephan); cuts words
- results:13 — CV definition repeated
- results:28 — C-robustness sentence redundant
- results:31 — reduce subsections
- results:65,69 — pixelwise paragraph very redundant; keep only Fig-9-unique points
- theory:78 — "no measure of reliability" redundant w/ prior sentence
- methods:51 — ADC-sensitivity para redundant
- abstract:6 — is CV mention necessary in abstract?
- (general note) repetitive across files; be concise; under wordcount is fine

### 2. 🧠 Don't over-index on the 2-bin collapse; use the full spectrum
- results:9 — describe intermediate bins more (link to Fig 5 / grade)
- results:18 — more nuanced; D=2.0 reasonably constrained; TZ/PZ diffs; "cannot resolve" too strong
- results:25 — individual-bin pitch too pessimistic; several intermediate bins have decent raw AUC; detection-vs-grading axes; literature
- results:44 — interpret Fig 4 in more detail; Stephan: r=0.8 IS good; drop regularizer-smearing scope
- abstract:10 — "two axes" too bold given middle bins inactive in detection
- discussion:5 — broaden beyond the single "central finding"

### 3. 🧠/🔧 Tone — stop framing the Bayesian contribution negatively
- abstract:13 — "gain is calibrated uncertainty, NOT better point estimates" too negative; reframe MAP=fast / NUTS=exploratory + free uncertainty
- results:60 — ends on "MAP can't produce these intervals" — too negative a closer
- discussion:17 — over-dwells on limitations/small cohort
- results:44 — Stephan: 0.8 correlation is good, don't present it negatively

### 4. 🧠 Uncertainty-aware classifier (THE hard conceptual item)
- results:57 — (also the big restructure comment) uncertainty subsection placement
- results:58 — geometric LR effect vs posterior; TWO phenomena wanted (misclass vs class; distance from boundary); p-value vs bootstrap inconsistency; "Welch"?
- results:59 — deep dive into misclassified ROIs; check labels; match metadata/GGG
- results:60 — capability framing too negative
- discussion:26-28 — the 2.4× / 1.3× / "genuine component" claim ⚠️ **conflicts with memory `project_uncertainty_investigation`: logit-ratio CI includes 1** → this sentence overstates the data (integrity fix)
- discussion:9 — uncertainty as a new biomarker; find supporting literature

### 5. 🧠 Manuscript restructure / figure reorder (BIG)
- results:57 — full reorg proposal (4 thematic blocks). Proposed figure order:
  - Block A "how spectra are derived & why they make sense": Fig 1, Fig 7, Fig 8
  - Block B "TZ/PZ LR classification": Fig 2, Fig 9 (exploratory heatmap), Fig 6 (uncertainty)
  - Block C "why ADC works (post-hoc analysis)": Fig 3, Fig 4
  - Block D "GGG add-on": Fig 5
- discussion:5 — reopen overall structure / contribution balance

### 6. 🧠+👥 Fig 7 (Fisher) + CRLB explanation
- theory:30 — condition-number repetition (κ(U) vs κ(F))
- theory:55 — physical range sanity check
- theory:56 / discussion:36 / figures:191 — ⚠️ **TODO(Sandy): validate van-Trees Bayesian-CRLB derivation** (gates Fig 7b numbers)
- theory:57 — citations for CRLB methods; why λ=0.1; want to read code; why NUTS ≪ Bayesian CRLB
- theory:58 — "single unconstrained-vs-observed ratio" wording unclear
- discussion:37 — factor decomposition (2 orders + 2–74×) is vague; make clear
- (Patrick's notes) detailed Fig 7 layout: drop panel a; 2-up + 1-centered; factors color-coded above bars for both comparisons; SNR labels inside; concise titles, no parentheses; fix y-axis

### 7. 🔢 Code / number verification (clean batch — run recompute.py)
- theory:17 — recompute κ(U)
- theory:21 — re-derive Fisher Eq 2
- theory:38 — check Appendix derivation
- theory:115 — sanity-check sensitivity code + reference spectrum
- methods:24 — NUTS params (2000/200/4/0.95) default?
- methods:25 — split vs non-split R̂
- methods:31 — verify ADC recipe (b≤1000, 5 points)
- methods:53 — 10k resamples actually used?
- results:5 — SNR numbers (median 303 etc.) still accurate
- methods:23,26 — normalize-after-optimization order (MAP & NUTS)

### 8. 🔢/🔧 Stale methodology text
- methods:36 — "Gleason grade classification" via AUC no longer done (we use spectral shifts) — stale
- methods:37 — Fig 2 now TZ/PZ only
- methods:42 — classification description (raw vs LR? single features in table? combos that beat ADC?) — revisit
- methods:7 — "specificity check" on 8 benign/HGPIN — did we actually run a test?
- results:51 — why 11 tumor ROIs (GGG=0 / ungraded) excluded from GGG; should they be normal?
- results:24 — raw vs LR confusing; reviewer may ask why 2-bin > 8-bin LR

### 9. 👥 Pixel-wise "outside cohort" factual question (recurs 3×, → Stephan)
- intro:19, methods:58, results:65 — is the pixelwise patient really OUTSIDE the cohort? (LR is trained on this cohort's ROIs)
- methods:58 — typo "b=0-"; "500 draws through discriminant" rigor; PZ-coeff rationale; mention ROI-trained→pixel-applied limitation
- methods:60 — direction count: 3 patients/6 ROIs vs stated "13 ROIs/7 patients"

### 10. 🔧 Abbreviation / notation hygiene
- abstract:7 — MAP not expanded in abstract; LOOCV abbreviation
- intro:17 — where to introduce MAP abbreviation
- methods:40 — introduce LOOCV once
- theory:71 — is K introduced? (8-dim)
- theory:73 — augmented-system notation; which active-set solver (cvxopt?)
- theory:62 — show MAP-for-multivariate-Gaussian form?
- abstract:7 — "MAP vs NUTS not a perfect comparison" (probabilistic term vs sampler) — framing

### 11. 📚 Citations / literature
- abstract:4 — citations for compartment-volume models
- methods:38 — cite zonal-difference claim
- theory:104 — Stephan to check (citing his prior work, kuczera2023)
- (Discussion) histology refs; uncertainty-quantification refs; detection/grading-axis lit
- methods:24 — cite PyMC
- (notes) "cite Obsidian"; references review as final item

## Co-author routing
- **Stephan (clinical/physics):** pixelwise-cohort question; histology interpretation; SNR↔free-water claim; GGG-count framing; his kuczera2023 self-cite; subsection advice.
- **Sandy (methodology):** ⚠️ van-Trees Bayesian-CRLB derivation (gates Fig 7); MAP math/notation; Gibbs inverse-gamma vs the SNR formula actually used; (optional) ESS comparison.

## MRM compliance findings (objective)

### Hard blockers (required / currently missing)
1. **IRB / ethics statement — MISSING.** Add to Methods (approval for retrospective human data).
2. **Data Availability Statement — not a real DAS.** Replace bare GitHub URL with formal DAS: repo link + DOI + SHA-1 hash. Optionally request Code Review in cover letter.
3. **Word count.** `\wordcount{\todo{count}}` unfilled; body ~4.8–5.4k (est.) — at/over the 5000 limit. Abstract ~250+ (cap 250; "<200" placeholder is wrong).

### Quick objective fixes
4. **Keywords: 7 → ≤6.**
5. **Abstract:** passive voice / no first person ("We use" → passive); expand MAP; remove formulae ($D=0.25$); structured format ✓.
6. **Title bug:** `\title{Magnetic Resonance in Medicine}` placeholder; real title only on title page.
7. **"Fig." → "Figure"** (discussion:47, figures:271); figure sub-parts uppercase A/B/C (text uses a/b/c).
8. **Portal (not manuscript):** ORCID required; cover letter states type=Research Article + cites overlapping prior ISMRM abstracts (rejected ISMRM PDFs in assets/).
9. **LLM policy:** AI only for spelling/grammar; prose stays the authors'.

### Tailwind
- MRM explicitly welcomes honest/negative results. The "ADC is hard to beat + mechanistic why" framing qualifies — lean in.
