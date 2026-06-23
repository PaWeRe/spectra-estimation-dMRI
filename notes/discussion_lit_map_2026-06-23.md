# Discussion + Literature Planning Map — 2026-06-23

**Purpose.** Patrick's steer (last rework session): *before writing any Discussion prose*, build (1) a figure-anchored outline of the Discussion's main points tied to the literature, and (2) a literature map of all ~29 candidate works — what each does, how it relates to our story, where it's cited, and how influential — so Patrick can go into each reference himself, weigh it, and we never cite a paper we haven't understood or place out of context. **No head-on rebuttals yet; map first.**

**The MRM framing to keep in mind (Patrick).** We set out to *beat* ADC and came back with an *explanation for why ADC is good enough for detection* (at least in this cohort). The Discussion must make that case credibly to MRM: ADC is not an impoverished signal; it already captures the dominant detection axis, and the genuine added value (grading, identifiability, per-voxel uncertainty) lives elsewhere.

**Citation counts now verified** (OpenAlex `cited_by_count`, a conservative floor — typically 10–30% below Google Scholar). See the weight-ranked table in **B0** below; the per-group tables keep my qualitative tiers (⏳), which B0's real numbers supersede where they differ. Patrick to sanity-check influence himself — esp. Langkilde (load-bearing but OpenAlex likely undercounts it).

---

## PART A — Discussion main-text point map (figure-anchored)

Order Patrick wants: **Fig 6 (why ADC works) → Fig 7 (tentative grading) → Fig 8 (uncertainty for free + null) → Fig 9 (pixel-wise) → Limitations → Future.** Each block = the bigger points for the *main text*; figure legends get trimmed to "what you see" (Part E).

### A0. Lead-in / central claim (1 short ¶)
- ADC, though a single scalar, is a **near-optimal spectral discriminant for detection**: across zones and estimators it ranks ROIs almost identically to the optimal 8-component classifier (Fig 5).
- Thesis sentence (honest framing): the decomposition does **not obsolete ADC — it explains it**; multi-component value lies in interpretability, identifiability, and grading, not detection.
- *Literature touch:* He 2025 as the reconciling anchor (advanced models ≈ ADC for detection, better for grading) — but introduce it gently here, develop in A1/A2.

### A1. Fig 6 — WHY ADC works (THE CORE — must be rock-solid, in-depth, non-repetitive)
**This is the paragraph Patrick flagged as still too vague/repetitive. Its job is the MECHANISM — distinct from Theory (identifiability *tool*/Fisher), Methods (*how*), and Results (the correlation *number*). Do not re-derive Fisher here.**

The argument, in rigorous steps:
1. **Both summaries are (near-)linear functionals of the same 8-bin spectrum.** ADC from a monoexponential fit over b≤1000 is, to first order, a smooth monotonic-in-D weighting of the fractions (effectively a signal-weighted mean diffusivity); the spectral discriminant is literally a linear predictor `Σ wⱼ Rⱼ`. Two linear readouts of one vector.
2. **The detection-relevant spectral variation across the cohort is effectively one-dimensional.** Tumor-vs-normal change concentrates in the two outer bins (D=0.25 ↑, D=3.0 ↓), which co-move. So the cohort's R-vectors vary predominantly along a single "tumorness" axis (restricted-up / free-down).
3. **Therefore any two linear functionals that both project onto that dominant axis must rank ROIs near-identically** → r ≈ −0.98. It is ~0.98 and not exactly 1 only because of small, noisy intermediate-bin contributions.
4. **"Redundancy of the buckets" — make this precise.** The intermediate bins (D=0.5–1.5) are (a) collinear / poorly identified (high posterior CV), and (b) carry little *independent* tumor–normal contrast beyond the outer pair. So they add **no new axis of detection variation** — they are redundant *given* the outer two. This is why 8-feature ≈ 2-feature ≈ ADC, and why the mirror ablation (inner-6 only) drops to AUC ~0.78–0.82: the inner bins are individually informative *because they co-vary with the dominant axis*, but contribute nothing once the outer pair is present. **Redundancy ≠ uselessness** (keep this distinction explicit — it was a reviewer-confusion point).
5. **Sensitivity × contrast resolution (the Fig 6 visual).** ADC's sensitivity vector ∂ADC/∂Rⱼ is −1.59 at D=0.25, +0.85 at D=3.0, and is *not* zero in the middle — ADC stays sensitive there. But discriminative power = sensitivity **×** contrast; in the middle the contrast is small and the bins are noisy, so ADC's sensitivity is "spent" on a low-contrast, ill-determined direction that moves neither ranking. The LR coefficients and −∂ADC/∂Rⱼ align (r=−0.79 PZ, −0.88 TZ), carried entirely by the two outer bins (only D=0.25 & 3.0 have bootstrap CIs excluding zero).
6. **(OPTIONAL, decide with Patrick) one airtight supporting analysis:** report the effective rank / % variance on PC1 of the cohort spectra, and show ADC and the discriminant both ≈ the PC1 score. Would make "one-dimensional detection axis" a measured fact, not an assertion. Small, read-only; flag as candidate — may be scope creep for a "last pass."

*Literature ties (engage, don't attack):* He 2025 (detection ≈ ADC) as corroboration; VERDICT/INNOVATE (Singh 2022, Johnston 2019), RSI (Yamin 2016), HM-MRI (Chatterjee 2018/2022) as the works claiming detection gains — position our result as: in *our* cohort/acquisition, the extra compartments are redundant for detection; where others see gains may reflect acquisition (diffusion times, b-range), cohort, or that the gain was really in grading. **Patrick to weigh which of these to cite after reviewing them.**

### A2. Fig 7 — Tentative grading (the genuine frontier)
*(Note: this content currently lives in rendered discussion.tex ¶3, ref'ing `fig:spectrum_ggg` — keep it, make it a clear figure-anchored block. Patrick noted he wants Fig 7 clearly present.)*
- **Grade-dependent shifts:** D=3.0 (lumen) collapses at tumor onset and saturates (detection, not grade); D=0.25 (restricted/cellular) rises monotonically with grade; D=2.0 (glandular-epithelial) preserved in low-grade then depleted at high grade.
- **Histology interpretation (candidate, not validated):** lumen replacement by solid tissue; increasing cell-packing density; glandular architecture loss. Maps onto where ADC is *least* sensitive → the plausible locus of any grading value beyond ADC.
- **Honesty:** n=29, uneven subgroups (down to n=8), wide bands → "signs of life," not a statistical claim; needs whole-mount histology + larger cohort.
- *Literature ties:* He 2025 / Johnston 2019 (VERDICT) / Palombo 2023 (rVERDICT) all locate gains in **grading** — directly supports "grading is the frontier." Chatterjee 2015 (gland-component volumes ↔ Gleason) supports the histology mapping. ADC-grading-limitation set (Yan 2024, Rosenkrantz 2015, Donati 2014, Rozenberg 2016) supports "ADC compresses grade."

### A3. Fig 8 — Uncertainty: "confidence for free" + the null propagation result
- **Prediction confidence for free (the usable, honest point):** the logistic classifier returns a per-ROI P(tumor), and misclassifications cluster at the boundary → distance-to-boundary triages the least-reliable calls. A *new, useful way to present cancer likelihood* (graded confidence, not yes/no) — but it is a **standard property of the logistic model**, available from the point estimate, shared by ADC- and spectrum-based classifiers. **NOT a Bayesian-posterior contribution, NOT a methodological novelty** (lit: calibration / selective prediction / conformal are standard — don't overclaim).
- **NULL result (must be stated plainly):** propagating the full NUTS posterior through the classifier adds **no** detection value beyond the point estimate. The 2.4× wider CI for misclassified ROIs is **logistic geometry** (sigmoid slope P(1−P) largest at boundary), not the spectral posterior — it falls to ~1.3× and is NS in logit space, and CI width does not predict misclassification once distance-to-boundary is controlled. Posterior-SD-as-features → ΔAUC ≈ 0.
- **Nuance (accept, don't fully explain):** posterior adds little partly because outer-bin CV is already low; D=3.0 CV is high yet still doesn't propagate to a classification signal.
- *Literature ties:* minimal/none. If we want one citation for "confidence/calibration is standard," keep it to a single well-known reference — TBD, low priority. Do not pad.

### A4. Fig 9 — Pixel-wise feasibility + per-voxel uncertainty maps
- **Feasibility/appearance:** low-D (D=0.25) fraction map ≈ ADC; D=3.0 map ≈ its negative; per-voxel discriminant reproduces ADC (r=−0.97). Extends the ROI detector to voxels.
- **Unique contribution:** the **per-voxel uncertainty map** — 8-bin detector 1.8× more uncertain than 2-bin (it weights the unidentifiable middle); ADC/point-estimate methods cannot provide this.
- **Caveat (honest):** single-voxel SNR inflates the restricted (D=0.25) fraction → ROI-trained detector skews tumor-like → reinforces reporting at ROI level; faithful per-voxel needs explicit noise modeling or spatial regularization. Exploratory; the uncertainty doesn't (yet) propagate to better detection — accept for now, but it does flag which buckets are reliable.
- **Direction independence — ONE sentence + re-reference `fig:directions`** (currently orphaned!): per-direction CV lowest for the dominant D=0.25 bin (mean 9%) across 13 ROIs/7 patients → trace averaging is appropriate. Sanity check only; do not over-emphasize.

### A5. Limitations
- **Circularity + F12:** tumor ROIs placed on mpMRI (incl. DWI/ADC) → ADC-conspicuous lesions overrepresented; **only Gleason (biopsy) is free of this.** Add F12: the detection positives are radiological mpMRI targets — **11 of 40 not biopsy-confirmed cancer** (7 benign, 4 ungraded) — so detection contrast is partly against the radiologist's own ADC-informed reading. Reinforces circularity.
- n=29 graded tumors → no reliable AUC comparison for grading (ΔAUC<0.10 indistinguishable).
- Fixed diffusivity bins (continuous spectrum could reveal more, at more ill-posedness).
- Single fixed TE/diffusion-time (newer gradient hardware could change SNR + compartment weights — Zhu 2024, Elsaid 2026).
- Pixel-wise = single supplementary patient.

### A6. Future
- Other organs/tumors (breast, liver, brain) where multi-b DWI is acquired.
- Amortized inference (NN posterior approx) → ms/voxel real-time.
- Fisher framework → joint b-value-protocol + grid optimization.
- T2 integration → multiparametric spectral biomarkers (LWI parallel — Sabouri 2017).

---

## PART B — Literature map (full candidate set, ~29 works)

Legend: **Where** = current section(s) cited (✗ = not currently cited). **PDF** = in `assets/`? **Verdict** = keep-as-is / wire-in / add-new / delete / fetch. Tier ⏳ = my estimate pending metrics agent.

### B0. Citation weight (verified — ranked by influence)
OpenAlex counts (conservative floor; Google Scholar runs higher). Use to weigh which works are worth citing/engaging.

| Paper | ~Cites | Tier | Role in our story |
|---|---:|---|---|
| Hanley & McNeil 1982 (AUC/ROC) | 21,800 | LANDMARK | canonical AUC ref (wire-in or drop) |
| Le Bihan 1988 (IVIM) | ~8,000 | LANDMARK | origin of multi-compartment diffusion |
| Hoffman & Gelman 2014 (NUTS) | ~5,000 | LANDMARK | our sampler |
| Weinreb 2016 (PI-RADS v2) | 2,935 | LANDMARK | PI-RADS predecessor (consider drop) |
| Turkbey 2019 (PI-RADS v2.1) | 2,400 | LANDMARK | ADC clinical standard |
| Donati 2014 (ADC histogram → aggressiveness) | 306 | INFLUENTIAL | **strongest ADC-grading ref** |
| Panagiotaki 2015 (VERDICT) | 182 | INFLUENTIAL | multi-compartment prostate origin |
| Chatterjee 2015 (gland components ↔ Gleason) | 175 | INFLUENTIAL | **supports our histology mapping (A2)** |
| Mulkern 2006 (biexponential) | 124 | INFLUENTIAL | prior prostate multi-compartment |
| Chatterjee 2018 (HM-MRI feasibility) | 118 | INFLUENTIAL | detection-tension (self-caveats SNR/time) |
| Prange & Song 2009 (MC T2 inversion) | 109 | INFLUENTIAL | our methodological template |
| Johnston 2019 (VERDICT vs ADC, grade) | 90 | INFLUENTIAL | grading gain → "frontier" |
| Sabouri 2017 (luminal water imaging) | 83 | INFLUENTIAL | orthogonal microstructure (Future) |
| Rosenkrantz 2015 (whole-lesion ADC, %G4) | 81 | INFLUENTIAL | ADC-grading |
| Brunsing 2017 (RSI review) | 79 | INFLUENTIAL | RSI camp (review) |
| Rozenberg 2016 (ADC histogram, upgrading) | 77 | INFLUENTIAL | ADC-grading (currently uncited) |
| Manetta 2019 (ADC ↔ GS review) | 56 | STANDARD | grading context |
| Conlin 2021 (multicompartment opt.) | 50 | STANDARD | advanced-model context |
| Jalnefjord 2019 (b-value/CRLB ill-cond.) | 46 | STANDARD | anchors ill-conditioning claim |
| Chatterjee 2022 (HM-MRI validation) | 42 | STANDARD | validated follow-up to 2018 |
| Palombo 2023 (rVERDICT) | 37 | STANDARD | grading gain |
| **Singh 2022 (INNOVATE VERDICT)** | 36 | STANDARD | **strongest detection-tension — but only ~36 cites** |
| Langkilde 2018 (our protocol source) | 28* | STANDARD | *OpenAlex undercount; verify — load-bearing* |
| Yamin 2016 (RSI cellularity) | 27 | STANDARD | RSI camp, voxel grading |
| Maier 2022 (b-weighting level) | 17 | RECENT | b-value rationale |
| Fennessy 2023 (quant. dMRI review) | 13 | RECENT | general context |
| Yan 2024 (ADC → GS grading) | 11 | RECENT | ADC-grading-limit |
| Kuczera 2023 (reproducible ADC) | 9 | RECENT | ADC estimation context |
| **He 2025 (advanced models)** | ~0 | NEW | **our ally — cite as recent corroboration, not authority** |
| Wells 2022 (ISMRM abstract) | — | foundational | **direct predecessor (must cite)** |

**Weighting takeaways for the framing:**
- The "beats-ADC for **detection**" works are *solid but not overwhelming*: Singh/INNOVATE ~36, Johnston ~90, Chatterjee-2018 ~118, VERDICT-origin ~182. We are **not** contradicting a thousand-cite consensus — useful: our claim is defensible without being reckless.
- The **ADC-grading-limitation** literature is well-cited (Donati 306, Rosenkrantz 81, Rozenberg 77) → firm ground for "ADC compresses grade; grading is the frontier."
- **He 2025 (our ally) is brand new (~0 cites)** → cite as recent corroboration, lean on it for *framing*, not authority.
- **Langkilde 2018** (closest prior + our protocol) shows only ~28 in OpenAlex — almost certainly undercounted; verify manually since it carries weight.

### Group 1 — Foundational / our method's lineage
| Key | Cite (year, venue) | What it does | Relation to us | Where | PDF | Tier ⏳ | Verdict |
|---|---|---|---|---|---|---|---|
| `wells2022estimation` | Wells, Maier, Westin 2022, ISMRM abstract | Bayesian/Gibbs inverse-Laplace diffusivity-spectrum estimation on prostate decays | **Direct predecessor — this paper builds on it** | ✗ | ISMRM-2022-abstract.pdf | foundational (abstract→low cites) | **WIRE IN** (Intro ¶3 + Methods) |
| `prange2009quantifying` | Prange & Song 2009, J Magn Reson | Monte-Carlo uncertainty for NMR T2 spectra | Methodological template (MC inversion + uncertainty) | intro, disc | – | INFLUENTIAL | keep |
| `langkilde2018evaluation` | Langkilde 2018, MRM | Fitting models for extended-b prostate DWI | Source of our 15-b protocol + model context | intro, meth, disc | – | INFLUENTIAL | keep |
| `lebihan1988separation` | Le Bihan 1988, Radiology | IVIM (diffusion+perfusion separation) | Origin of multi-compartment diffusion idea | intro | – | LANDMARK | keep |
| `mulkern2006biexponential` | Mulkern 2006, MRI | Biexponential prostate diffusion over extended b | Prior multi-compartment prostate fit | intro | – | INFLUENTIAL | keep |

### Group 2 — "Beats-ADC" / advanced-model group (the tension + the ally)
| Key | Cite | What it does | Relation to us | Where | PDF | Tier ⏳ | Verdict |
|---|---|---|---|---|---|---|---|
| `he2025improved` | He 2025, Abdom Radiol | Mono vs FROC vs multi-compartment: **detection ≈ ADC (0.92 vs 0.91); grading better** | **THE ALLY** — corroborates our exact thesis | ✗ | s00261-024-04684-z.pdf | RECENT | **ADD** (A0/A1/A2) |
| `singh2022innovate` | Singh 2022 (INNOVATE), Radiology | VERDICT fIC > ADC+PSAD for **csPCa detection** (AUC 0.96) | **Strongest detection tension** — engage carefully | ✗ | **NOT HELD** | INFLUENTIAL | ADD *(Patrick: fetch?)* |
| `johnston2019verdict` | Johnston 2019, Radiology | VERDICT fIC > ADC for **grade differentiation** (ADC NS, p=.26) | Grading gain; supports "frontier = grading" | ✗ | VERDICT.pdf | INFLUENTIAL | ADD (A2) |
| `palombo2023rverdict` | Palombo 2023, Sci Rep | rVERDICT separates 3+3/3+4/≥4+3 > VERDICT & ADC | Grading gain | ✗ | rVERDICT.pdf | RECENT | ADD (A2) |
| `chatterjee2018hybrid` | Chatterjee **2018**, Radiology *(file mislabeled 2017)* | HM-MRI epithelium fraction detects cancer (AUC 0.99) > ADC; flags SNR/time confound | Detection-tension **but** self-caveats acquisition | ✗ | chatterjee2017feasibilityRadiology.pdf | INFLUENTIAL | ADD (A1) — note their confound |
| `chatterjee2022validation` | Chatterjee **2022**, Radiology *(file mislabeled 2018)* | Histologic validation of HM-MRI fractions (AUC 0.94–0.96) | Validated follow-up to 2018 | ✗ | chatterjee2018Radiology.pdf | STANDARD | ADD or cite-with-2018 |
| `chatterjee2015changes` | Chatterjee 2015, Radiology | Ex-vivo gland-component volumes ↔ Gleason & ADC > cellularity | **Supports our histology mapping (A2)** | ✗ | Chatterjeeetal2015Radiologyprint.pdf | INFLUENTIAL | ADD (A2 mechanism) |
| `yamin2016voxel` | Yamin 2016, Clin Cancer Res | RSI cellularity index ↔ Gleason at voxel level | RSI camp; voxel grading | ✗ | Yamin2016RSI.pdf | STANDARD | ADD *(weigh)* |
| `brunsing2017restriction` | Brunsing 2017, JMRI | RSI review — improved conspicuity/detection | RSI camp (review) | ✗ | Brunsing2017JMRI.pdf | STANDARD | optional |
| `sabouri2017luminal` | Sabouri **2017**, Radiology *(file mislabeled 2015)* | Luminal Water Imaging (T2) detects PCa (AUC 0.97/0.98), ↔ Gleason | Orthogonal microstructure; LWF ↔ our lumen bins (A4/A6) | ✗ | sabouri2015Radiology.pdf | INFLUENTIAL | optional (Future/parallel) |
| `panagiotaki2014microstructural` | Panagiotaki **2015**, Invest Radiol | VERDICT original | Multi-compartment prostate origin | intro | VERDICT.pdf(diff) | INFLUENTIAL | keep *(fix key/year)* |
| `conlin2021improved` | Conlin 2021, JMRI | Optimized multicompartment signal models (RSI lineage) | Advanced-model context | intro | – | INFLUENTIAL | keep |

### Group 3 — ADC ↔ Gleason grade (supports "ADC compresses grade")
| Key | Cite | What it does | Relation to us | Where | PDF | Tier ⏳ | Verdict |
|---|---|---|---|---|---|---|---|
| `manetta2019correlation` | Manetta 2019, Gland Surg | ADC↔Gleason review (multicentre) | Grading context | intro | – | STANDARD | keep *(verify it's not a meeting suppl)* |
| `yan2024value` | Yan 2024, Insights Imaging | ADC has limited Gleason-predictive value; grade overlap | Strongest "ADC limited for grading" | intro | – | RECENT | keep |
| `rosenkrantz2015whole` | Rosenkrantz 2015, JMRI | Whole-lesion ADC ↔ %Gleason-4 in GS7 | Grading context | intro | – | INFLUENTIAL | keep |
| `donati2014prostate` | Donati 2014, Radiology | Whole-lesion ADC histogram ↔ aggressiveness | Grading context | intro | – | INFLUENTIAL | keep |
| `rozenberg2016whole` | Rozenberg 2016, AJR | ADC histogram/texture predicts GS upgrading in 3+4 | Grading context | ✗ | – | STANDARD | **WIRE IN or delete** |

### Group 4 — Acquisition / hardware / b-value
| Key | Cite | What it does | Relation to us | Where | PDF | Tier ⏳ | Verdict |
|---|---|---|---|---|---|---|---|
| `maier2022choice` | Maier 2022, JMRI | Does b-weighting level matter | b-value rationale | meth | – | STANDARD | keep |
| `kuczera2023truly` | Kuczera 2023, MRM | Reproducible uniform ADC from multi-b | ADC estimation context | theory | – | RECENT | keep |
| `jalnefjord2019optimization` | Jalnefjord 2019, MRM | b-value scheme optimization; multi-exp CRLB ill-conditioning | Anchors our ill-conditioning claim | disc | – | STANDARD | keep |
| `fennessy2023quantitative` | Fennessy & Maier 2023, Eur J Radiol | Quantitative prostate DWI review | General context | intro | – | RECENT | keep |
| `zhu2024human` | Zhu 2024, MRM | Ultrahigh-gradient prostate MRI feasibility | Future (TE/TD limits) | disc | – | RECENT | keep |
| `elsaid2026nonlinear` | Elsaid 2026, MRM | Nonlinear gradient insert for prostate diffusion | Future (TE/TD limits) | disc | – | RECENT(new) | keep |

### Group 5 — Guidelines / stats / software (tool citations — keep minimal, no weighting needed)
| Key | Cite | Role | Where | Verdict |
|---|---|---|---|---|
| `turkbey2019prostate` | PI-RADS v2.1 | ADC clinical-standard ref | intro, meth | keep |
| `weinreb2016pirads` | PI-RADS v2 | PI-RADS predecessor | intro | **consider drop** (we standardized on v2.1; MRM "no bulk cites") |
| `hoffman2014nuts` | NUTS | sampler | intro, theory | keep |
| `neal2011mcmc` | HMC | sampler | theory | keep *(fix to @incollection)* |
| `gelman2006prior` | priors for variance params | half-Cauchy prior | theory | keep |
| `vehtari2021rank` | improved R̂ | convergence | meth | keep |
| `efron1993bootstrap` | bootstrap | CIs | meth | keep |
| `hanley1982meaning` | AUC/ROC meaning | AUC definition | ✗ | **WIRE IN or delete** (canonical AUC ref; DeLong gone) |
| `abrilpla2023pymc` / `kumar2019arviz` / `virtanen2020scipy` | PyMC / ArviZ / SciPy | software | meth | keep |

### Group 6 — DELETE (uncited + obsolete/wrong)
| Key | Why delete |
|---|---|
| `delong1988comparing` | DeLong test dropped from analysis; uncited. |
| `aksitciris2019accelerated` | Uncited; acceleration method not referenced. |
| `betancourt2017conceptual` | Uncited HMC tutorial; neal2011 + hoffman2014 already cover HMC. |
| `maier2004diffusion` | **Wrong paper (brain tumors) + live `\todo` + uncited.** Comment out pending Stephan's intended JMRI 4-compartment prostate paper. |

---

## PART C — Papers still MISSING (Patrick to fetch into `assets/` for close reading)
1. **Singh 2022, INNOVATE** (Radiology 305:623–630, doi:10.1148/radiol.212536) — the strongest *detection*-tension paper (VERDICT fIC > ADC+PSAD, AUC 0.96). Headline verified; PDF not held. Fetch if we want exact per-zone/per-comparison numbers and to be sure we represent it fairly. **Highest priority to fetch.**
2. **The intended `maier2004diffusion` replacement** — Stephan's "JMRI 4-coefficient diffusion-restriction prostate paper." Current entry is a brain-tumor stand-in. Need the actual paper to decide keep/replace/delete.
3. *(Optional)* He 2025 issue number confirmation (vol 50, pp 4235–4248, doi:10.1007/s00261-024-04684-z verified; issue inferred).

**Also flag for Patrick's influence check:** the agent could not exhaustively verify every author on Singh/Johnston/Palombo (first ~5–10 + all numbers verified). Verify author lists before final.

---

## PART D — Bibliography hygiene action list (`paper/references.bib`)
1. **Delete:** `delong1988comparing`, `aksitciris2019accelerated`, `betancourt2017conceptual`.
2. **Comment out pending Stephan:** `maier2004diffusion` (+ its `\todo`).
3. **Add (verified BibTeX in agent report):** `he2025improved`, `singh2022innovate`, `johnston2019verdict`, `palombo2023rverdict`, `chatterjee2018hybrid`, `chatterjee2022validation`, `chatterjee2015changes`, `yamin2016voxel`, `brunsing2017restriction`, `sabouri2017luminal` — **only those Patrick approves after reading.**
4. **Wire into text (currently uncited):** `wells2022estimation` (must), `rozenberg2016whole`, `hanley1982meaning`.
5. **Format fixes:** `neal2011mcmc` → `@incollection` (book: Handbook of MCMC, Chapman & Hall/CRC; editor/publisher); `panagiotaki2014microstructural` key/year mismatch (entry is 2015) — rename key or accept cosmetic; `manetta2019` verify it's a full article not a meeting supplement; add DOIs to primary clinical refs; consider dropping `weinreb2016pirads` (v2) now we cite v2.1.
6. **Thin the Introduction** — it currently carries **14 citations** (bulk-citation pattern MRM warns against). Target: keep the essential framing cites, push the rest to Discussion where they're engaged.

---

## PART E — Figure-legend trim plan (Patrick: shorten Fig 6–9 legends; move interpretation → main text)
Same treatment as Theory/Methods/Results legends: legend = **what you see**; interpretation/key numbers → Discussion body.
- **Fig 6 (sensitivity):** current legend states the mechanism conclusion ("two-compartment detector," the r=−0.79/−0.88, the divergence interpretation). → Trim to: what the bars/diamonds/colors are. Move the alignment numbers + "redundancy not different quantity" to A1.
- **Fig 7 (GGG):** current legend carries all the bin numbers (0.39→0.21→0.16 etc.) + "carries detection not grade." → Trim to: what's plotted (groups, baseline, bands). Move trajectories + interpretation to A2.
- **Fig 8 (uncertainty):** current legend states the 2.4×, the logistic-geometry explanation, the triage conclusion. → Trim to: what's plotted (sorted P(tumor), CI, color = correct/mis). Move 2.4×/1.3×/null + "confidence for free" to A3.
- **Fig 9 (pixel-wise):** current legend carries the r values + 1.8× + SNR caveat. → Trim to: panel descriptions (A–I). Move r=−0.97, 1.8×, calibration caveat to A4.
- (Captions don't count toward the 5000-word body limit, but Patrick wants the paper's house style consistent.)

---

## Immediate next actions (after Patrick reviews this map)
1. Patrick reviews Part B, weighs influence, marks keep/cite/drop; fetches Part C papers.
2. Patrick confirms the A1 "why" framing (esp. the optional PC1 analysis) and the Fig 6→9 order.
3. Then: write Discussion prose (A0–A6) → trim legends (Part E) → integrate approved citations + thin Intro → execute bib hygiene (Part D).
4. Then submission roadmap: Conclusion → Abstract → SI legends → DAS/GitHub/Zenodo/SHA-1 → cover letter → co-authors.
