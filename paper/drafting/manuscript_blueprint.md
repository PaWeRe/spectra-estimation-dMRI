# Manuscript blueprint (content + structure, no prose)

The "abstract version of the manuscript." We settle the **thesis**, the **architecture**, and
**every big claim** here. Once this is locked we generate prose from it. Status tags on every
decision/claim: 🟥 OPEN · 🟨 LEANING · 🟩 LOCKED.

---

## 0. Fixed constraints (MRM — non-negotiable)
- Research Article: body ≤5000 words; **≤10 figures+tables** (now at 10 — none free for main text).
- Sections: Introduction → (Theory, optional) → Methods → Results → Discussion → Conclusion.
- Structured abstract ≤250 words, passive voice, no formulae/citations.
- Required: IRB/ethics statement (Methods); **Data Availability Statement** (replaces inline GitHub
  paragraph) with repo link + DOI + SHA-1 hash.
- Figures numbered in **order of first mention** (current numbering is non-compliant → renumber).

## Figure / table inventory
| label | content | current Fig # | order-of-first-mention |
|---|---|---|---|
| `fisher` | Fisher matrix / CRLB | 7 | **1st** (theory:33) |
| `spectra` | tumor/normal mean spectra | 1 | 2nd (results:7) |
| `sensitivity` | ADC sensitivity vs LR weights | 4 | 3rd (results:13) |
| `validation` | simulation recovery | 8 | 4th (results:18) |
| `roc` | detection ROC / AUCs | 2 | 5th (results:22) |
| `spectrum_ggg` | spectra by Gleason group | 5 | 6th (results:26) |
| `adc_discriminant` | ADC vs discriminant corr | 3 | 7th (results:32) |
| `uncertainty` | uncertainty-aware classification | 6 | 8th (results:57) |
| `pixelwise` | pixel heatmaps | 9 | 9th (results:67) |
| `auc` (Table 1) | AUC table | T1 | — |
| `directions` (Fig S1) | direction-wise check | S1 | supplementary |

---

## DECISION 0 — Thesis / contribution framing 🟥 OPEN
*Everything else hangs off this. "The story needs to sit" = settle this first.*

Patrick's recurring discomfort (Theme 2; abstract:13; discussion:5): the draft over-centers the
single "ADC ≈ optimal discriminant" finding and under-sells the rest (the estimation+identifiability
framework, the uncertainty-aware classifier, the Gleason-grading axis).

Candidate framings:
- **(A) Single spine — "Why ADC works."** Mechanism is THE thesis; everything supports it.
  *Pro:* sharp, quotable, matches current title. *Con:* the parts Patrick values (identifiability,
  uncertainty, grading) become subordinate; feels like a one-result paper.
- **(B) Multi-contribution framework.** Thesis = a Bayesian spectral-decomposition *framework* that
  (i) quantifies what is recoverable (identifiability + CRLB), (ii) **mechanistically explains ADC's
  near-optimality**, (iii) yields per-prediction uncertainty, (iv) exposes a grading axis.
  "Why ADC works" stays the most quotable single result and can stay in the title, positioned as a
  *consequence* of the framework. *Pro:* balances the contributions, matches the work done.
  *Con:* needs a crisp 1-sentence thesis or it reads as a grab-bag; word budget pressure.
- **(C) Hybrid lead.** Lead and conclude on "why ADC works," but explicitly frame identifiability +
  uncertainty as the *methodological* contribution that makes the mechanistic claim trustworthy.

**Claude's lean: (B), executed tightly.** It's what the work actually is, it resolves Patrick's
discomfort, and MRM rewards honest scope.

**RESOLVED 2026-06-07 (via Patrick's 8-step arc):** effectively (B/C). Thesis = a Bayesian framework
that estimates a hard-to-recover spectrum, finds it diagnostically equivalent to ADC, and explains
why; grading + uncertainty + pixel are honest **explorations/demonstrations** (the tail), not
load-bearing claims. **Abstract / Intro / Conclusion must be built on the CORE (steps 1–6), not the tail.**

### What the Bayesian/NUTS layer actually buys (honest accounting, 2026-06-07)
The "Bayesian = better biomarker / uncertainty-aware classifier" angle is **DEAD** (U3 triangulated negative).
Drop "uncertainty-aware" from framing/title. What legitimately survives — NUTS is a *recoverability &
validation* tool, not a classifier upgrade:
- **Joint noise inference (σ per ROI)** → the cohort SNR characterization (median 303) and feeds the
  Fisher/CRLB analysis. MAP can't give this. (Candidate novelty — verify against literature.)
- **Per-ROI identifiability (posterior std/CV)** — a DIFFERENT axis from cohort-discriminability. The
  D=0.50 case (CV 0.81 but AUC 0.80) is the proof: a compartment can shift systematically across the
  cohort yet be unreliable in any single patient. This is an *interpretation-integrity* function (which
  spectral features a reader/clinician may believe per-patient), not a prediction tool.
- **NUTS validates fast MAP** (S8, r>0.98) → you'd deploy MAP; NUTS certifies it on the recoverable parts.
**Implication:** the honest paper is "what's recoverable + why ADC works," MAP-deployed, NUTS-certified.
The Bayesian layer is thinner than hoped but load-bearing for recoverability — not decorative, not the headline.
→ Title/framing decision for Patrick (this is now a Decision-0-level call).

## DECISION 1 — Section & figure arc 🟨 DEFINED (Patrick's 8-step, 2026-06-07)
Superseded the earlier 4-block sketch. See "Results / narrative arc — WORKING" below for the ordered
detective arc + per-step feedback. New figure order — **1** Fisher, **2** spectra, **3** simulation,
**4** ROC+Table 1, **5** ADC↔discriminant, **6** sensitivity, **7** GGG, **8** uncertainty, **9** pixel
— is compliant with order-of-first-mention. *Open seam:* Fisher sits at the Theory↔Results boundary
(derivation in Theory; empirical cohort SNR + posterior-std discussed at the start of Results).

---

## Big statements to PRESSURE-TEST (before any prose)
| # | claim | where | status | note |
|---|---|---|---|---|
| S1a | **Fig 3** ADC↔discriminant **ROI-score** corr \|r\|≈0.97–0.98 | results, disc, concl, abs | 🟩 | VERIFIED (adc_discriminant.csv): PZ/TZ × MAP/NUTS × C∈{1,10} all −0.96…−0.98. The genuinely robust result. discriminant = LR decision score w·R+b (recompute.py:508). |
| S1b | **Fig 4** ADC-sensitivity↔LR-weight **per-bin** alignment | results, disc | 🟩 | VERIFIED (adc_sensitivity.csv): Pearson −0.78…−0.88 (MAP & NUTS). **KEEP PEARSON** (its gap from 1 is informative). **NO λ / no "artifact" framing** — the λ=0.1 sweep is a historic misuse of MAP, purge entirely (see Global cleanup). recompute.py:547. |
| S2 | Detection collapses onto 2 outer bins (0.25, 3.0) | abs, results | 🟩🟨 | true & central FOR DETECTION (2-feat AUC ≈ full-8, auc_table.csv). Detection-specific; does NOT erase middle-bin structure → bridge to S6/S9. |
| S3 | Identifiability: which bins are well/poorly identified by CV | results, concl | 🟨 | PARTLY WRONG as written: identifiability.csv → D_20 CV=0.36, **D_2.0 CV=0.52** (NOT >0.8); only D_0.5/0.75/1.0 are >0.8. CV inflated for small-mean bins (Stephan: D_0.5 mean=0.038). Identifiability ≠ usefulness → S10. |
| S4 | Uncertainty-aware classifier flags errors (2.4× / 1.3× logit) | results, disc | ❌ REFUTED | FULL TEST (149 ROIs, 15 errs): prob-width 2.41× reproduces manuscript (0.389 vs 0.162 ✓) BUT it's PURE GEOMETRY — logit-width only 1.28× (MW **p=0.25**, ns); controlled coef +0.20, **p=0.42, CI [−0.28,+0.67]**; error-AUC logit_w=0.59 vs dist=0.82. **Claim does not survive. discussion.tex:28 "genuine component" MUST be cut.** |
| S5 | Bayesian CRLB / van-Trees factor decomposition (2 orders + 2–74×) | theory, disc, Fig7b | 🟥 | **derivation unvalidated by Sandy.** Gates Fig 7. |
| S6 | Detection-axis vs grading-axis dissociation | abs, results, disc | ⏸ PARKED | Patrick needs more time + Stephan + literature before any claim. Reference only: partial_corr_ggg.csv (n=29) hints at beyond-ADC grade signal in D_0.50 (partial ρ=0.42, p=.026, CI [0.007,0.68]) — exploratory, NOT a result. Build nothing on this yet. |
| S7 | Pixelwise patient is "outside the cohort" | intro, methods, results | 🟥 | likely false (same cohort) → Stephan. Affects the "independent dataset" claim. |
| S8 | MAP ≈ NUTS (discriminant r>0.98) | results, disc | 🟩 | solid per memory. No λ value in the framing. |
| S9 | Fig 4 Pearson-gap: residual lives in intermediate bins (STATE, don't interpret) | results(Fig4) | ⏸ 🟨 | per-bin detection AUCs COMPUTED (see evidence log): middle bins reach 0.80–0.85 (NUTS) — not chance — BUT add ~nothing beyond the 2 outer bins (2-feat≈8-feat, S2). Decent standalone AUC ≠ independent contribution. **Do NOT frame Fig 4 as a "bridge" yet** — Patrick to write provable statements first. |
| S10 | **CV is not a clean proxy for predictive usefulness** | results, disc | 🟨 | D_0.50 CV=0.81 (poorly identified) yet strongest beyond-ADC grade signal (S6). Poor identifiability ≠ useless. CV inflated by small mean (Stephan). Test: per-bin CV vs per-bin AUC / grade-ρ. |
| S11 | **U1: LR gives confidence "by construction" — but NOT unique to the spectrum** | results(Fig6), disc | 🟩(logic) | Univariate LR on ADC is defensible (ADC-LR 0.94 ≈ ADC-raw 0.95, auc_table). So the sigmoid/geometric confidence interval is available to ADC too → **decouple from "why spectrum".** Spectrum's only added-value route = posterior (S12/S13). |
| S12 | **U3: posterior propagated through the discriminant adds NO independent error signal** | results(Fig6), disc | ✅ RESOLVED (negative) | logit-width vs error: coef +0.20, **p=0.42, CI includes 0**, AUC 0.59; distance owns it (AUC 0.82). Confirms the mechanism: detection rides on low-variance bins → little epistemic uncertainty to propagate. Honest negative result; reinforces the spine. |
| S13 | **U3-alt: whole-spectrum posterior spread as a reliability flag** (NEW) | results(Fig6), disc | 🟨 | TESTED (features.csv, n=149, 15 errors): misclassified have higher total spread (mean 0.49 vs 0.41, MW **p=0.011**, error-AUC 0.70). BUT confounded w/ distance-to-boundary (Spearman −0.57); controlling for distance, spread coef +0.69→**+0.28** (distance −1.30 dominates, dist-AUC 0.82). **Weak residual; controlled-coef significance UNVERIFIED** (15 errors → need bootstrap p/CI). Honest reading: high-spread↔near-boundary may be the SAME "ambiguous spectrum". Not a rescue; an honest weak signal. |

### Follow-ups from pressure-testing
- **S1 figures — DECIDED (Patrick, 2026-06-07):** keep Fig 3 & Fig 4 **separate** (10 items total is fine). Fig 3 = phenomenon (ranking identical). **Fig 4 = mechanism AND the bridge: keep PEARSON, not Spearman** — its gap below 1 is the middle-bin structure that opens the grading story (S9). Swapping to Spearman would just duplicate Fig 3's ranking message.
- **S1 prose fixes:** (a) "robust across C=0.1 to 50" → unsupported; recompute only sweeps C∈{0.1,1,10}. Change to "0.1–10" or re-run to 50. (b) TZ NUTS r: text −0.979, actual −0.981. (c) Define "spectral discriminant" in one clause at first use (no equation).

### The uncertainty story, decomposed (S11/S12/S13) — keep these three SEPARATE in Fig 6
1. **U1 = geometric/sigmoid confidence (S11).** Distance-to-boundary → P(1−P) interval. Real and tangible, but available to ANY classifier incl. ADC-via-LR → not a spectrum justification. *Can still be shown* as "every score comes with a confidence interval."
2. **U2 = per-bin posterior spread → identifiability (S3/S10).** The CV story. Re-examine CV as a metric; identifiability ≠ usefulness.
3. **U3 = posterior propagated to the prediction (S12/S13).** ❌ **RESOLVED NEGATIVE — triangulated 3 ways (2026-06-07).** (i) Through the discriminant: no independent error signal (p=0.42, AUC 0.59). (ii) Whole-spectrum spread (S13): weak univariate (AUC 0.70) but dies under distance control. (iii) **Uncertainty-as-features (the fair test): adding per-bin σ to the detection LR → ΔAUC PZ −0.001 [−0.021,+0.017], TZ −0.010 [−0.036,+0.008] = zero.** (stds-only AUC 0.83 is a heteroscedastic echo of the means, NOT independent info.) The 2.4× prob-space effect is entirely the sigmoid geometry (U1). **Posterior uncertainty has no downstream diagnostic value, full stop.**
- **RESOLVED → (b):** the honest uncertainty story = **U1** (a calibrated confidence interval per prediction — geometric, available to ADC-via-LR too) + **U2** (per-bin identifiability: which compartments to trust). The posterior does NOT add an independent error-flag. Clean negative result that REINFORCES the spine (detection collapses to well-determined bins → nothing to propagate). MRM welcomes it.
- **Fig 6 — DECIDED (Patrick, 2026-06-07, framing A):** reframe as **"Prediction confidence"** (a classifier property — point-estimate / geometric — NOT Bayesian, shared by ADC & spectral classifiers), **relocate** next to the ROC/Table 1 classification block, **keep in main** (earns its place as the honest disposal of the uncertainty question). Supplementary = fallback if a slot is needed.
  - ✅ Discussion uncertainty paragraph rewritten (discussion.tex; false "genuine component" cut, overclaim removed, tied to spine).
  - ⏳ Results subsection reframe drafted ("Prediction confidence") — awaiting Patrick's nod to commit.
  - ⏳ Physical relocation + Fig 6 caption update → during the figure-renumber/reorder pass.
  - ✅ **recompute.py wiring DONE (2026-06-07):** `uncertainty_propagation()` + `load_nuts_draws()` added to `biomarkers/recompute.py` (canonical, runs inside `recompute_all`, writes `results/biomarkers/uncertainty_propagation.csv` + `_per_roi.csv`). Reproduces every cited number exactly — prob CI-width ratio **2.41×** (0.389 vs 0.162, MW p=1.2e-4); logit ratio **1.27×** (MW **p=0.26**, ns); controlled logistic `error ~ z(logit_width)+z(dist)`: width coef **+0.23, boot p=0.45, CI [−0.54,+0.76]** (includes 0), err-AUC width **0.59** vs distance **0.82**; uncertainty-as-features ΔAUC **PZ −0.001 [−0.021,+0.017] / TZ −0.010 [−0.036,+0.008]**; stds-only AUC 0.83/0.82; whole-spectrum-spread err-AUC **0.70** (S13). (Bootstrap controlled-coef CI is wider than the earlier in-session [−0.28,+0.67] — that was a Wald CI; bootstrap is the project convention, same conclusion. No cited manuscript number is affected — the text cites only ratios + "doesn't predict once distance controlled", all confirmed.) Frozen gold CSVs (features/auc_table/…) regenerated byte-identical (headline AUCs reproduce PROJECT_STATE §2 to 4 dp).

### Figure reorder + caption fixes committed (2026-06-07)
- ✅ **figures.tex reordered to arc** (verified: body mention order == block order == arc). New numbering: **1** Fisher, **2** spectra, **3** validation, **4** ROC, **5** ADC↔discriminant, **6** sensitivity, **7** GGG, **8** uncertainty, **9** pixel-wise. (includegraphics filenames still carry old numbers — harmless; rename later if desired.)
- ✅ Two body cross-refs fixed so mention order matches: identifiability paragraph no longer points to the sensitivity fig for CV (→ supplementary atlas; sensitivity caption is self-contained); tumor-detection grading pointer de-\ref'd ("presented below").
- ✅ **Uncertainty caption (Fig 8) reframed** — geometric not posterior; "genuine residual"/"MAP can't produce" removed. **Numbers unchanged** (2.4× stands; it's real, just geometric).
- ✅ **Sensitivity caption (Fig 6)** — λ=0.1 smearing-artifact line + grading-bridge line removed (aligns with body).
- ⚠️ **Patrick's-notes figure remap:** his "Fig 7 (Fisher)" → now **Fig 1**; "Fig 8 too big (validation)" → now **Fig 3**; "Fig 6 smaller (uncertainty)" → now **Fig 8**.
- ✅ **Fisher (Fig 1) reworked (Patrick: keep subplot a):** `generate_fisher_figure.py` → 2+1 layout (matrix + CRLB bars top, decay curves centered below); improvement factors drawn ON the bars for both steps (gray prior-gain unc→bay, orange constraint-gain bay→NUTS up to 74×); SNR identities labelled inline in panel (c); concise titles, "gap" banner removed, bigger matched fonts. Caption updated to reference on-bar factors + accurate combined gain ("up to three orders of magnitude"). Regenerated fig_fisher_v2.{pdf,png}; verified visually. ⚠️ panel (b) van-Trees numbers still **Sandy-gated**.
- ⏳ **Still outstanding from notes:** Fig 3 (validation) & Fig 8 (uncertainty) "too big / caption cut off" sizing; figN_*.pdf filenames still carry old numbers (cosmetic).

### Theory cleanup committed (2026-06-07)
- ✅ All 18 inline comments removed. Verified: **κ(U)=2.78×10⁵, κ(F)=7.75×10¹⁰** (both match); κ(F) reframed as κ(U)² (kills the redundancy). Fisher eq F=UᵀU/σ² confirmed standard. Sensitivity code confirmed (earlier).
- ✅ Softened overclaims: "ADC implicitly performs near-optimal discrimination" → testable hypothesis ("ranking by ADC approximates the optimal classifier; tested in Results"); "frequently lies outside orthant" → "can". Clarified confusing bits: the "ratio replaced" sentence, "standardized" (= fit on standardized fractions), "physical range" (= [0,1]). Reduced "ill-conditioning" overuse.
- ✅ Made generic the unverified specifics: Gibbs σ² "inverse-gamma update" → "conjugate full conditionals" (Patrick wasn't sure he used it); active-set solver kept generic.
- ⚠️ **Still gated/flagged:** van-Trees Bayesian-CRLB (theory:57 + Fig 7b; "2 orders" + "2–74×") remains **Sandy-gated** (%TODO kept). The toy-model ΔD∝D^{3/2} derivation lost its "see Appendix" pointer → needs a **Supporting Information** home if kept (MRM discourages appendices). `kuczera2023` cite → Stephan to confirm. Active-set solver could be named (cvxopt?) if desired.

### Compliance block committed (2026-06-07)
- ✅ **Abstract** rewritten: passive voice (no first person), no formulae, MAP/NUTS/ROI expanded, two-axis grading claim dropped (parked), uncertainty reframed as recoverability, Bayesian contribution = identifiability. **247 words** (≤250 ✓). Joint-noise-inference advertised per Patrick's note.
- ✅ **IRB statement** added to Methods cohort subsection (IRB-approved protocol + de-identified). ⚠️ Patrick to confirm exact IRB/protocol/consent wording matches BWH/Langkilde.
- ✅ **Data Availability Statement** — inline GitHub paragraph removed from Methods; formal DAS section added to main.tex with repo link + `\todo{Zenodo DOI}` + `\todo{SHA-1}` (Patrick to mint Zenodo DOI + paste commit hash before submission).
- ✅ **Keywords 7→6** (dropped "uncertainty estimation"). ✅ **Title** placeholder fixed. ✅ Abstract word count set (247); **body count still `\todo` — run texcount on Overleaf**. ✅ last "Fig."→"Figure" (figures.tex:271).
- ⚠️ **DeLong citation now orphaned** + repo README/data to refresh before submission (per Patrick's note).

### Manuscript edits committed (2026-06-07)
- ✅ **Uncertainty chunk:** `results.tex` "Uncertainty-aware Classification" → **"Prediction confidence"** (honest reframe: classifier-geometric U1 + negative U3; comments stripped). `discussion.tex` uncertainty paragraph rewritten (false "genuine component" cut; tied to spine).
- ✅ **Why-ADC-works chunk:** `results.tex` "ADC and the spectral discriminant" (Spearman-led: NUTS ρ=−0.979 PZ / −0.977 TZ; Pearson kept; C-range fixed 0.1→10; TZ Pearson −0.981; discriminant defined; bootstrap-vs-p justification trimmed) + "ADC sensitivity analysis" (Pearson kept, 0.8 framed positively, λ-artifact purged, no grading-bridge claim).
- ✅ **λ-artifact cleanup:** "smoothing artifact" clauses removed from `discussion.tex` (central-finding para) and `conclusion.tex`.
- ✅ **Redundancy + subsection pass (Results↔Discussion):** D1 central-finding restatement compressed to 1 sentence; D2 uncertainty mechanics trimmed (now only in Results); D3 Fisher restatement trimmed; D4 PI-RADS b=0 tangent cut; the defensive intermediate-coefficient sentence compressed. Results subsections **8→6**: merged Spectra+Identifiability → "Diffusivity spectra and identifiability"; merged ADC-discriminant+sensitivity → "Why ADC works". (Classification+Prediction-confidence merge deferred to reorder — needs relocation.)
- ✅ **Classification block + per-bin AUCs (2026-06-07):** added per-bin single-feature raw-rank AUC computation to `recompute.py` (canonical) → regenerated `auc_table.csv` (existing rows reproduce exactly). Table 1 now includes the 8 individual NUTS fractions with bootstrap CIs. Results "Classification Performance" → **"Tumor detection"**: raw-vs-LR clarified, the 2-feat-≥-8-feat puzzle explained (finite-sample noise from 6 uninformative fractions in the held-out pipeline), intermediate bins nuanced (several reach ~0.85 standalone but add nothing combined), C-robustness folded in, comments stripped.
  - ✅ **DeLong DROPPED (Patrick):** removed from Table 1 caption AND `methods.tex` (paired test was never actually reported; paper uses bootstrap CIs throughout). → `delong1988comparing` is now an **unused citation** — remove in the references pass.
  - ✅ **Methods "Classification" rewritten:** detection-only (PZ/TZ; dropped the stale GGG-classification task), LOOCV introduced once, method-only (findings live in Results), comments stripped. **Gap when grading unparked:** Methods has no description of the grade-stratified spectral-shift analysis (Fig 5) method.
- ⏳ **Still open:** ~~recompute.py uncertainty wiring~~ ✅ DONE 2026-06-07 (see §"Fig 6 — DECIDED"); Fig 3 annotation → Spearman; Fig 6 relocation + caption (figure pass); methods.tex "ADC sensitivity analysis" subsection still has comments (10k-resamples check); the **grading paragraph (disc 9–14) + Results GGG subsection + Methods grading-method** stay PARKED (overlap + the to-soften "detection vs grading axis" claim) until grading is unparked; the section-top "razor-sharp Discussion" ambition is only partly met (regurgitation removed; deeper interpretation still possible).

### Computed evidence log
- **Per-bin single-feature detection AUC** (raw-rank, sign-adjusted; features.csv, 2026-06-07). For Table 1.
  | bin | PZ NUTS | PZ MAP | TZ NUTS | TZ MAP |
  |---|---|---|---|---|
  | 0.25 | 0.88 | 0.89 | 0.91 | 0.94 |
  | 0.50 | 0.80 | 0.60 | 0.85 | 0.67 |
  | 0.75 | 0.81 | 0.67 | 0.85 | 0.67 |
  | 1.00 | 0.83 | 0.78 | 0.83 | 0.68 |
  | 1.50 | 0.83 | 0.77 | 0.55 | 0.51 |
  | 2.00 | 0.59 | 0.66 | 0.79 | 0.84 |
  | 3.00 | 0.92 | 0.92 | 0.90 | 0.94 |
  | 20.00 | 0.52 | 0.56 | 0.55 | 0.55 |
  | ADC | 0.95 | — | 0.98 | — |
  Reading: middle bins are NOT chance (0.80–0.85 NUTS), but add ~0 beyond the 2 outer bins (S2). NUTS > MAP on ill-identified bins. *For Table 1, recompute with LOOCV-LR + bootstrap CIs to match the other rows.*

### Global cleanup tasks (manuscript-wide, do when editing each section)
- **Purge ONLY the λ=0.1 "0.98 artifact" narrative** (DECIDED: option a) — i.e. presenting the discriminant/sensitivity correlation as regularization-dependent, or the 0.98-under-heavy-reg value. Spots to scrub: results.tex:44, discussion.tex:21, theory.tex:57. Fig (sensitivity) reports Pearson ≈0.8 (MAP & NUTS), no λ.
- **KEEP & HIGHLIGHT the tuned λ (=1e-3):** important POSITIVE methods point — we tuned λ on simulation data (Fig: simulation, step 3) specifically to AVOID regularizer smearing. "Smearing" stays in the paper as the *motivation for tuning*, NOT as a finding about the correlation.

---

## Section-by-section blueprint (TO FILL once Decisions 0 & 1 set)
*Template per section: **Job** (1 sentence) · **Main points** (bullets) · **Figures/Tables** ·
**Relationships** (what it sets up / pays off) · **Open Qs**.*

### Abstract — *fill last (derived from body)*
### Introduction — TBD
### Theory — TBD (signal model · identifiability/Fisher/CRLB · MAP · Bayesian · ADC functional)
### Methods — TBD (+ IRB statement; + Data Availability Statement replacing GitHub para)
### Results / narrative arc — WORKING (Patrick's 8-step, 2026-06-07)
Detective structure: motivate hard problem → show we solve it → validate → try to use it → discover
it ≈ ADC → explain why → explore the rest. ✓ = Claude agrees · ⚠️ = honest caution.

1. **Fisher** (Fig 1) — *why estimate a discrete spectrum & why it's hard.* Subplots: ill-conditioning,
   why this discretization, cohort-SNR motivation. Introduce inference methods; tease "NUTS undercuts
   theoretical std" → next paragraph.
   ⚠️ teaser rests on **S5 (Sandy's van-Trees derivation, UNVALIDATED)** — don't lead the paper on it
   until signed off. Also a forward-reference (NUTS result before NUTS shown to work). The Fisher-corr
   subplot you wanted to cut may earn its place back here as "why intermediate bins unresolvable."
2. **Spectra** (Fig 2) — *here's the estimated spectrum per zone/tissue.* Results before validation.
   ✓ engaging, BUT carry the identifiability caveat from Fig 1 into this figure (pre-empt "artifacts?").
3. **Simulation** (Fig 3) — *how we tuned λ (on sim, to avoid smearing) & validated both methods; NUTS
   gives free per-bin uncertainty.* ✓ this is where tuned-λ lives as a POSITIVE point.
4. **ROC + Table 1** (Fig 4) — *can the spectrum classify? LR optimal combo + per-bin + ADC → can't beat
   ADC.* Mention controversy (claims ADC is impoverished vs works struggling to beat it — needs cites).
   ⚠️ frame per-bin AUCs (the 0.8s) as NOT independent (ride tumor axis; 2-feat≈8-feat) or reviewer
   concludes middle bins work.
5. **ADC↔discriminant** (Fig 5) — *deeper: rank-correlation over cohort → near-perfect anti-corr. How?*
   ✓ report **Spearman** here (ranking message; distinguishes from Fig 6 Pearson). CSV currently Pearson → add Spearman.
6. **Sensitivity + Pearson** (Fig 6) — *mechanism: ADC sensitivity vector vs classifier weights → outer
   bins dominate & align with ADC; intermediate contribute little.* Keep Pearson.
7. **Spectrum by GGG** (Fig 7) — *are intermediate bins garbage? No — identifiability ≠ useless.* 
   ✓ AGREE: **abandon the "second grading axis" claim** (n=29 + collinearity → can't separate real signal
   from discretization smearing; reviewer magnet). Present as EXPLORATORY spectrum interpretation, no
   usefulness claim. Consequence: **scrub "detection axis vs grading axis" from Abstract + Intro.**
8. **Uncertainty (Fig 8) → Pixel (Fig 9)** — tail; both small points.
   - Uncertainty needs the S11/S12/S13 deep-dive FIRST (does U3 survive in logit space?). ⚠️ if not,
     consider Fig 8 → supplementary to tighten the tail.
   - Pixel last as per-voxel demo. ⚠️ verify uncertainty heat maps against the per-voxel SNR-inflation
     of D=0.25 (discussion.tex:48) before trusting them.

**OVERALL:** strong core (1–6), exploratory tail (7–8). This arc IS the answer to Decision 0. Keep the
Abstract/Intro/Conclusion anchored on the core; let the tail be honestly exploratory.
### Discussion — TBD (must be interpretation, not a 2nd Results — Theme 1)
### Conclusion — TBD (no overlap with Discussion)
### Data Availability Statement — NEW (MRM): repo link + DOI + SHA-1 hash
### Acknowledgments — funding (NIH P41EB028741, R01CA241817) [keep]
