# Manuscript Revision Plan — 4th Pass (Path A' Locked)

_Stub overlay on `notes_manuscript_3rd_pass.md` (2026-05-13). Most of
that doc's Exec items have been resolved or rendered obsolete by work
done 2026-05-14 → 2026-05-17. Use this file as the current spine; refer
back to the 3rd-pass doc for items not addressed here._

_Drafted: 2026-05-17 evening, after Path A' lock._

---

## 0. New framing in three sentences

1. **Mechanism first.** ADC's clinical success is explained by a compartment-volume mechanism (Chatterjee 2015): gland epithelium / lumen volume fractions shift with Gleason, and ADC's monoexponential fit projects those compartments onto a single scalar. Our Bayesian spectral decomposition makes that mechanism quantitative at the ROI level with calibrated per-bin uncertainty.

2. **Two genuinely novel empirical findings.** (a) Fully Bayesian inference with per-bin posterior uncertainty on prostate diffusion data, validated by simulation against MAP smearing artifact. (b) Detection-vs-grading axis separation in spectral space: detection lives at outer bins (D=0.25, D=3.0), grading at intermediate bins (D=0.50, D=2.0). Mapped onto Bourne 2018 compartments.

3. **Diagnostic equivalence as a confirming check, not a headline.** Triangulated across partial-Spearman, LR+DeLong, and ADC variants (B-range and DKI), no spectral feature meaningfully lifts AUC over std-ADC at N=29 ROIs for grading. We report this as a *prediction of the mechanism*, with a three-reason explanation in Discussion (biological collinearity, estimator efficiency floor, label-resolution bottleneck).

**Scope honesty.** All quantitative results are at the ROI level (149 ROIs). The pixel-wise heatmap (Fig 9) is a MAP-only feasibility demonstration on one slice from one patient, not a results figure. Full Bayesian per-voxel inference is future work pending computational improvements. Existing per-voxel multi-compartment methods (HM-MRI, VERDICT, rVERDICT, RSI, LWI) already deliver per-voxel maps with parametric constraints; **per-voxel multi-compartment is not our novelty.** Our novelty is the model-free grid + Bayesian uncertainty + axis separation, all at ROI level.

The MAP smearing finding and the bias-heatmap simulation justify NUTS-over-MAP as a methodological necessity. Without them the paper has no spine.

---

## 1. Status of 3rd-pass Exec items

| Exec # | Issue | Status | Note |
|---|---|---|---|
| 1 | LR-vs-individual-feature paradox | **RESOLVED** | Simulation 2026-05-16 confirmed MAP smearing artifact. Switch to NUTS 2-feat {D=0.25, D=3.00} dissolves the paradox. No more 8-feat-MAP-LR headline; no in-sample/LOOCV asymmetry |
| 2 | "Optimal multi-compartment classifier" overclaim | **RESOLVED** | Path A' framing is honest about equivalence |
| 3 | Identifiability vs classifier weight collision | **RESOLVED** | Same root cause as #1 |
| 4 | SNR sanity check | **RESOLVED** | Investigation A 2026-05-16: data overrides HalfCauchy; σ_NUTS is honest |
| 5 | MAP Eq. 5 audit (truncation, λ, σ) | **DEMOTED — still pending** | MAP now methods-comparison only; audit still needed for the comparison panel but not load-bearing |
| 6 | Prior consistency MAP↔NUTS | **DEMOTED — still pending** | Same reason as #5 |
| 7 | Fisher / CRLB discretization claim | **DEMOTED** | Fig 8 → supplementary. Bin grid is not derived from CRLB; theory and code remain decoupled. Honest decoupling in supplement |
| 8 | Pixel / direction data provenance | **STILL OPEN — [ASK Sandy/Stephan]** | Highest-priority unresolved blocker for Methods text |

**Two new Path A' Exec items:**

| Exec # | Issue | Status |
|---|---|---|
| 4P1 | §10h re-scoped: 8-feat NUTS-LR coefficient projection onto D_vec for both tumor-vs-normal and GGG≥3, with cos(w_T, w_G) to quantify axis separation | OPEN — first task next session |
| 4P2 | Three-reason thesis paragraph for "why averaging suffices" (biological collinearity, estimator efficiency floor, label-resolution bottleneck) | OPEN — drafted in [[project_pathA_prime_locked_2026-05-17]], needs to land in Discussion |

---

## 2. Section-by-section delta vs 3rd-pass plan

Only sections that change. Items not listed here inherit the 3rd-pass plan as-is.

### Abstract — REWRITE FROM SCRATCH

Old plan: soften "optimal multi-compartment classifier." Add MCMC keywords.

Path A' rewrite, in order:
1. **Background.** ADC is the de facto prostate dMRI biomarker. Compartment-volume models (Chatterjee 2015, Bourne 2018) explain *why* it works but are not directly measured per voxel.
2. **Methods.** Fully Bayesian spectral decomposition on an 8-bin grid with HalfCauchy noise prior, NUTS inference, validated by simulation against MAP smearing.
3. **Results.** (a) 2-feature spectral classifier matches ADC for tumor detection (PZ AUC 0.933 vs 0.951, TZ 0.937 vs 0.979). (b) Spectral grid resolves detection-axis and grading-axis components, orthogonal in our data. (c) At N=29 with valid GGG, spectrum and ADC are diagnostically equivalent for grading.
4. **Conclusion.** Bayesian spectral decomposition provides per-voxel compartment uncertainty and mechanistically explains ADC's clinical success. Diagnostic improvement over ADC awaits finer-grained reference labels.

### Introduction — INSERT CHATTERJEE FRAMEWORK UPFRONT, HONESTLY

3rd-pass plan items still apply (ADC historical context, DL/NN citation, NNLS phrasing fix). Add:
- Open the second paragraph with the Chatterjee 2015 compartment-volume mechanism explicitly. Frame the question: "Can we recover those compartments from in-vivo dMRI in a model-free Bayesian framework with calibrated uncertainty per parameter?"
- Acknowledge existing per-voxel multi-compartment methods (HM-MRI Chatterjee 2017, VERDICT, rVERDICT, RSI, LWI Sabouri 2017). Explicitly position our contribution as model-free + Bayesian uncertainty quantification at the ROI level, distinct from the parametric per-voxel methods these papers deliver.
- Biopsy-replacement long-term motivation: move to Future Directions, NOT the Intro lede. We do not deliver per-voxel uncertainty at scale in this paper; motivating the paper around a goal we don't reach invites a reviewer's "where is it?" The one-sentence version that fits in the Intro is the per-voxel uncertainty *aspiration* as motivation for why Bayesian-with-uncertainty matters even at ROI level (calibration must be sound at any spatial scale).

### Theory — DEMOTE FISHER

3rd-pass items #5, #6 still apply (MAP Eq audit, prior consistency). For Fisher / CRLB: keep the derivation in a methods-supplement, remove "the grid is informed by ΔD ∝ D^(3/2)" claim from main theory. Honest decoupling.

### Methods — ADD SIMULATION PARAGRAPH

3rd-pass item #8 (pixel/direction provenance) still the highest-priority Methods blocker.

**New paragraph**: simulation methodology used to verify MAP smearing. ~150 words. Points at `results/simulation/bias_heatmap.png` (now a main-text figure, not supplementary).

**New paragraph**: ADC variants. Briefly describe what was tested (b ≤ 1000 std, ext1500/2000, all-15-b, high-b only, DKI D+K) and which was used as the primary reference. Cite results table in Results.

Cut/condense: bootstrap-CI explanation (still needed, but one sentence not a paragraph), Hanley-McNeil (one sentence).

### Results — FULL REWRITE AROUND FOUR PILLARS

Old plan had AUC table + LR-vs-feature explanation + MAP-NUTS correlation. Path A' rewrite:

**Pillar 1: Tumor detection.** 2-feat NUTS-LR {D=0.25, D=3.00} matches ADC. AUC table replaces Session 8 numbers with the same numbers but NUTS-primary, MAP-as-comparison. ADC reference is std-ADC (b≤1000) with one-line acknowledgement that other variants tested in supplement give equivalent results.

**Pillar 2: Detection-vs-grading axis separation.** New subsection. The §10f finding: detection at outer bins, grading at intermediate bins, orthogonal in spectrum space. Map to Bourne 2018 compartments. This is the cleanest empirical result and probably needs its own figure (see §3 below).

**Pillar 3: Grading is diagnostically equivalent.** New subsection. Continuous-GGG Spearman ρ for μ_D=0.50 (+0.565) vs ADC (−0.546), both Bonferroni-significant. Partial Spearman ρ(μ_D=0.50, GGG | ADC) = +0.42, p=0.026 uncorrected (does not survive Bonferroni). LR+DeLong: no AUC lift. ADC-variants sweep: no fairer ADC reference exists. Report all four tests honestly in one subsection with a triangulating table.

**Pillar 4: Methodology validation.** MAP smearing simulation + per-bin posterior CV + bias-heatmap figure. NUTS-vs-MAP correlation table. Frame as "MAP is convenient, NUTS is necessary; here is why" — one paragraph, not three.

Cut: "LR coefficients become unstable" wording (resolved). Stale single-bin numbers (rewrite from current features.csv). Repeated "mid-bins are not identifiable" — say once.

### Discussion — NEW THESIS

Old plan: three areas of added value; Colin comparison; ADC perception.

Path A' Discussion structure:
1. **Opener (4-point novelty list).** Bayesian per-bin uncertainty, model-free 8-bin grid recovering 3-compartment story, MAP smearing caveat, detection-vs-grading axis separation. Each gets one citation-supported sentence.
2. **Why ADC succeeds — the three-reason thesis.** Biological collinearity (Chatterjee 2015 ρ=−0.78), estimator efficiency floor (spec_M1 underperforms direct ADC by ΔAUC ≈ 0.23), label-resolution bottleneck (GGG is coarse). New paragraph; this is the interpretive centerpiece.
3. **What the spectrum adds that ADC cannot — at the ROI level.** Per-bin posterior uncertainty (calibrated σ via joint inference, distinct from existing parametric multi-compartment methods that report point estimates only), model-free decomposition without fixed-D compartment assumptions, explicit decomposition into orthogonal detection/grading axes. Distinguish honestly from HM-MRI / VERDICT / rVERDICT / RSI which already provide parametric per-voxel maps — our model-free grid + Bayesian uncertainty is the differentiator, not per-voxel mapping per se.
4. **DKI kurtosis as the parametric analog.** DKI_K ρ=+0.476 vs GGG aligns with the spectrum's intermediate-bin grading signal. Both index "deviation from monoexponential." Brief paragraph; positions us within the DKI literature without competing with it.
5. **NUTS coverage caveat** (from [[project_nuts_coverage_caveat]]). One paragraph.
6. **Limitations.** Selection bias (§10c, radiologist-drawn ROIs on ADC-dominated mpMRI), N=29 underpowered for grading subset analyses, no whole-mount histology, NUTS coverage on δ-spectra.
7. **Future directions.** Computationally tractable Bayesian per-voxel inference (NUTS at voxel scale is currently prohibitive; VI or amortized neural posteriors are candidates). Finer-grained reference labels (whole-mount %compartments, MIB-1) to unlock the spectrum's resolution advantage. Image-guided biopsy correlation. Long-term: biopsy-replacement clinical translation. pip-installable package, Zenodo deposit.

Cut: Colin paragraph as currently drafted (re-read first; may keep). p-value justification paragraph (drop). CRLB-priors recap (trim to one interpretive sentence).

---

## 3. Figure deltas vs 3rd-pass plan

| Fig | 3rd-pass status | Path A' delta |
|---|---|---|
| 1 — MAP vs NUTS spectra | Reference style | No change |
| 2 — ROC | Pick A/B/C for in-sample/LOOCV fix | **Option A confirmed.** Rebuild on NUTS 2-feat classifier as primary; MAP-LR as comparison panel only |
| 3 — ADC vs spectral discriminant | Style fixes | Rebuild with NUTS discriminant. **Most narrative-load-bearing figure now** |
| 4 — MAP↔NUTS comparison | Repetition justification | **Justification is built-in**: MAP smearing demonstration. Keep |
| 5 — (unclear) | Style pass | Confirm what this is. May be subsumed by new axis-separation panel |
| 6 — per-ROI classifier uncertainty | Style; add MAP/ADC overlays | Keep; under Path A' the NUTS-uncertainty story is stronger |
| 7 — directions | Caption + provenance | Still pending [ASK] |
| 8 — Fisher | One panel or cut | **Cut to supplementary entirely.** Path A' doesn't lean on CRLB-derived grid |
| 9 — pixel-wise heatmaps | Drop subplot D | Keep as a **feasibility demonstration only**, label as such in caption. MAP-based; single patient, single slice. Frame explicitly as proof-of-concept that the framework extends spatially, not as a delivered per-voxel result. Future-work flag |

**Two new figures Path A' needs:**

- **F-new-1 — Bias heatmap (already exists at `results/simulation/bias_heatmap.png`).** Promote from supplementary to main text. Methods-comparison panel justifying NUTS over MAP.

- **F-new-2 — Detection-vs-grading axis separation.** Small 2-panel figure. (a) Tumor-vs-normal LR coefficient profile across 8 bins. (b) Continuous-GGG Spearman ρ profile across 8 bins. Side-by-side shows the orthogonality. Could optionally include 3-panel with cos similarity / dot product as text overlay. **Highest-leverage new figure to add.**

10-figure budget check (MRM limit): with F-new-1 added, Fig 8 cut, F-new-2 added → still at the limit. Either trim Fig 5 if it's a redundant simulation panel, or push Fig 4 (MAP↔NUTS) to supplement.

---

## 4. Tables

Old Table 1 (Classification performance) becomes the Path A' AUC table. Single table, structure:

| Method | PZ tumor-vs-normal | TZ tumor-vs-normal | GGG≥3 (N=29) |
|---|---|---|---|
| ADC (std) | 0.951 / 0.940 LR | 0.979 / 0.964 LR | 0.778 LR |
| NUTS 2-feat {0.25, 3.0} | 0.933 | 0.937 | 0.744 |
| NUTS 8-feat | 0.926 | 0.923 | 0.772 |
| MAP 8-feat (comparison) | 0.919 | 0.946 | 0.733 |

Optional supplementary Table S1: ADC variants (std, ext1500, ext2000, full, high-b, midrange, DKI_D, DKI_K, spec_M1) × detection AUC × grading ρ. Show that std-ADC is a fair upper bound for the ADC reference. Source: `results/biomarkers/adc_variants_summary.csv`.

---

## 5. Items moved out of scope under Path A'

- **Identifiability features in the classifier (aggressive variant from §4 of 3rd-pass).** Future-work item. Not needed for Path A' submission.
- **VI / amortized inference comparison.** Theory §6 one-paragraph mention only. Not a results figure.
- **Continuous-spectrum (GP prior).** Parked indefinitely.
- **Pixel-wise per-voxel classification AUC.** Future work — no whole-mount histology to validate.
- **TCIA cross-dataset validation.** SKIP — extended b-range has no public counterpart.

---

## 6. Items requiring [ASK] before drafting

Carry-over from 3rd-pass §11, refined by Path A':

**Sandy:**
1. MAP Eq. 5 prior choice and σ handling — still endorsed after joint σ inference? (DEMOTED but unresolved)
2. Fisher / CRLB derivation — does Sandy agree to move Fig 8 to supplement?

**Stephan:**
3. **Pixel + direction provenance** (3rd-pass Exec #8). Still the highest-priority [ASK]. Both pixel patient 8640 and direction patient 9283 + tarball siblings need cohort identification.
4. ADC current clinical perception — is ADC viewed as "solved" or "good enough but limited"? Affects Path A' framing emphasis.
5. Reaction to Path A' framing — does "spectrum mechanistically explains ADC, diagnostically equivalent at this N" land well with the clinical reader, or do we need a different lede?

**Both:**
6. Is the four-pillar Results structure (detection / axis-separation / grading-equivalence / methodology) the right shape, or do they prefer a different ordering?

---

## 7. Suggested next-session order of operations

1. **Run re-scoped §10h** (Exec 4P1): 8-feat NUTS-LR coefficient decomposition onto D_vec for tumor-vs-normal and GGG≥3. ~30-45 min. New script `scripts/lr_coef_decomp.py`. Output to `results/biomarkers/lr_coef_decomp.csv`.
2. **Draft Path A' abstract from scratch.** Patrick edits.
3. **Decide [ASK] batch** — single message to Sandy/Stephan covering items 1-6 above.
4. **Decide figure budget** — keep Fig 4 in main or push to supplement, given F-new-1 and F-new-2 additions.
5. **Begin Results rewrite** (Methods+Theory text edits can wait until [ASK] returns).

---

## 8. What is *not* in scope this session

- Section-by-section text edits to `paper/sections/*.tex`. Not until next session.
- Figure regeneration. Not until [ASK] returns on Fig 7.
- Per-patient supplement script. Parallel-track; can wait.
- Pip-installable / Zenodo. Discussion-Future-Directions only.

---

_End of 4th-pass stub. Source: this session's work (partial_corr_ggg, two_feature_lr_vs_adc, adc_variants_sweep) + Path A' lock memory [[project_pathA_prime_locked_2026-05-17]]. Refer to `notes_manuscript_3rd_pass.md` for items not addressed here._
