# Literature Review: The Two Camps in Prostate Diffusion-MRI (MC vs ADC)

**Prepared:** 2026-05-31 (web-verified via WebSearch/WebFetch + local PDFs in `assets/`).
**For:** Paper 3 Discussion — reconciling the "advanced diffusion model ≈ ADC" vs "advanced > ADC" literatures, in support of the thesis that **ADC works for prostate cancer DETECTION because the diffusivity spectrum collapses onto two well-identified, co-moving outer compartments (very-restricted ~0.25 and free-water ~3.0 µm²/ms), whereas GRADING signal shifts to poorly-identifiable intermediate/lumen diffusivities.**

Citation counts are Semantic Scholar as of 2026-05-31 (rough, lag the literature). DOIs/PMIDs verified.

---

## Camp A — Advanced / multi-compartment ≈ ADC (no real AUC gain for detection)

| Paper | Year | Journal | DOI / PMID | Cohort | Model vs ADC | ADC reference (b-values) | Key result | ~Cites |
|---|---|---|---|---|---|---|---|---|
| **He Y, Qi X, Zhou M-X, et al.** "Improved differentiation of prostate cancer using advanced diffusion models: mono-exponential, FROC, and multi-compartment." | 2025 (online 2/2025) | Abdominal Radiology | 10.1007/s00261-024-04684-z | 224 men (129 BPH, 95 PCa; GS6=14, GS7=56, GS≥8=25) | Mono (ADC), FROC (D,β,µ), 4-compartment MC (D fixed 0.52/1.9/3.0/30 ×10⁻³; F1–F4) | **PI-RADS-fair**: ADC fit on b = 50, 500, 1000; b=0 excluded to suppress perfusion | **DETECTION (BPH vs PCa): ADC AUC 0.91 = F1 (MC) 0.92 = D (FROC) 0.91 — no gain.** MC/FROC help only for **GRADING**: low-vs-int and int-vs-high (MC F2, C1+F2 AUC 0.73–0.76; "ADC alone proved inadequate"). | 0 (new) |
| **Fennessy FM & Maier SE.** "Quantitative Diffusion MRI in Prostate Cancer: What We Can Measure and How It Improves Clinical Assessment." | 2023 | Eur J Radiol | 10.1016/j.ejrad.2023.111066; PMID 37651828 | Review | IVIM, DKI, VERDICT, RSI vs ADC | Recommends standardized 2-point ADC, e.g. b≈100 & 900 | **"More accurate signal-decay description does not imply higher sensitivity in tissue differentiation." ADC, the least-noisy parameter, is uniquely sensitive; tumor/normal ADC *ratio* is robust to b-value choice.** (Note: Maier = our co-author Stephan.) This is the paper the prompt mislabeled "Quigley & Mitchell 2023 / PMC10623580". | 12 |
| **Yi Si J, et al.** "Diagnostic Performance of Monoexponential DWI Versus Diffusion Kurtosis Imaging in Prostate Cancer: Systematic Review & Meta-Analysis." | 2018 | AJR Am J Roentgenol | PMID 29812977 (211(2):358) | 5 studies / 463 patients | DKI (D, K) vs ADC | Per-study (mostly b≤1000–2000) | **Pooled AUC ADC 0.93 = DKI 0.93; sens/spec comparable.** "We do not recommend including DKI in routine clinical assessment of PCa for the moment." | 34 |
| **Rajabi P, et al.** "Unveiling the diagnostic potential of DKI and IVIM for detecting/characterizing prostate cancer: a meta-analysis." | 2024 | Abdom Radiol | 10.1007/s00261-024-04454-x | 27 studies | DKI, IVIM vs ADC | Per-study | Detection sens/spec 0.85/0.81; **IVIM & DKI "comparable diagnostic performance with each other as well as ADC"** for grading PCa aggressiveness. | 6 |
| **Hectors SJ, et al.** "Advanced DWI Modeling for Prostate Cancer Characterization: Correlation with Quantitative Histopathologic Tumor Tissue Composition." | 2018 | Radiology | PMID 29117481 | Hypothesis-generating | Mono/stretched/DTI/kurtosis vs histology | b≤2000 | ADC & advanced params all **negatively correlate with cellular/cytoplasmic fraction, positively with stroma** — i.e. they measure the *same* underlying cellularity axis. (Mechanistic support for redundancy.) | ~120 |

**Camp A pattern:** When ADC is computed fairly (PI-RADS b≤1000, radiologist-localized lesion), advanced models do **not** beat it for *detection*. The (modest) advantage that survives is for *grading* (Gleason stratification), exactly where He 2025 localizes it.

---

## Camp B — Advanced / multi-compartment > ADC

| Paper | Year | Journal | DOI / PMID | Cohort | Model vs ADC | ADC reference (b-values) | Key result | ~Cites |
|---|---|---|---|---|---|---|---|---|
| **Singh S, Rogers HJ, Kanber B, … Panagiotaki E, Punwani S.** "Avoiding Unnecessary Biopsy after mpMRI with VERDICT Analysis: The INNOVATE Study." | 2022 | Radiology | 10.1148/radiol.212536 | 165 biopsy-naïve men | VERDICT fIC vs ADC vs PSAD | **PI-RADS-fair**: clinical mpMRI ADC, b = 0, 150, 500, 1000 | **fIC AUC 0.96 vs ADC 0.85 vs PSAD 0.74** for csPCa. *Genuine* gap on a fair ADC — but confounded: ADC from 3 scanners, VERDICT from 1 (interscanner variability acknowledged). | 33 |
| **Johnston EW, … Punwani S, Panagiotaki E.** "VERDICT MRI for Prostate Cancer: Intracellular Volume Fraction vs ADC." | 2019 | Radiology | 10.1148/radiol.2019181749; PMID 30938627 | 70 (42 biopsied) | VERDICT fIC vs ADC | **PI-RADS-fair**: clinical ADC b = 0,150,500,1000 (VERDICT b up to 3000 separate) | **GRADING: fIC AUC 0.93 (GS≥3+4 vs benign/3+3); ADC AUC 0.85 but ADC means NOT significant (P=0.26) while fIC P=0.002.** fIC tracks epithelial/intracellular compartment. | 81 |
| **Palombo M, Valindria V, Singh S, … Panagiotaki E.** "Joint estimation of relaxation & diffusion tissue parameters … relaxation-VERDICT (rVERDICT)." | 2023 | Scientific Reports | 10.1038/s41598-023-30182-1 | 44 men | rVERDICT fIC vs VERDICT vs ADC | mpMRI ADC | rVERDICT **fIC discriminates GS3+3 vs 3+4 (P=.003) and 3+4 vs ≥4+3 (P=.040), outperforming classic VERDICT and ADC** — i.e. grading, again in the cellular compartment. | 30 |
| **Zhong AY, Digma LA, Conlin CC, … Dale AM, Seibert TM.** "Automated Patient-level Prostate Cancer Detection with Quantitative Diffusion MRI" (RSIrs). | 2022 | Eur Urol Open Sci | 10.1016/j.euros.2022.11.009; PMID 36601040 | 151 | RSIrs vs ADC vs PI-RADS | **b-fair but method-UNFAIR**: ADC from clinical b=0,1000 (and RSI b=0,500,1000); but used **automated whole-prostate MINIMUM ADC**, no radiologist lesion localization | **RSIrs AUC 0.78 vs ADC 0.48 vs PI-RADS 0.77.** Authors explicitly flag the auto min-ADC ≠ clinical lesion-ADC as the cause of ADC's collapse. | 24 |
| **Rojo Domingo M, … Seibert TM, et al.** "Restriction Spectrum Imaging as a quantitative biomarker for prostate cancer with reliable PPV." | 2025 | J Urology | 10.1097/JU.0000000000004611 | 7 centers; **n=877 biopsy-naïve** | RSIrs vs ADC vs PI-RADS | Same auto **minimum-ADC** paradigm, b=0,1000 | **RSIrs AUC 0.73 vs ADC 0.54 vs PI-RADS 0.75; RSIrs ≈ PI-RADS (P=.31), > ADC (P<.01).** Largest RSIrs series; same ~0.54 weak-automated-ADC. | 1 (new) |
| (Foundational) **Yamin G, … Karow DS, et al.** RSI prostate (in `assets/Yamin2016RSI.pdf`); **Conlin CC** RSI multi-compartment modeling. | 2016+ | J Urol / Radiology | — | — | RSI vs ADC | high-b RSI acquisition | RSI cellularity map outperforms ADC for conspicuity/detection. | high |

**Camp B pattern:** Two distinct families. (1) **VERDICT/rVERDICT** report a *real* fIC>ADC gap on a **fair PI-RADS ADC** — but the win is concentrated in **grading** (Gleason stratification), where ADC means are frankly non-significant. (2) **RSIrs (Seibert group)** reports huge detection gaps (ADC 0.48–0.54) but these come from an **automated whole-prostate minimum-ADC** with no radiologist localization — an unfair *operationalization* of ADC, not unfair b-values.

---

## The ADC-reference disambiguation

The prompt's hypothesis was that Camp B's advantage comes from comparing against an unfair **high-b / non-PI-RADS** ADC. **The evidence only partly supports this — and points to a different, sharper culprit.**

1. **b-value fairness is NOT the main issue.** The headline Camp-B papers all used **PI-RADS-compliant ADC (b≤1000)**:
   - VERDICT INNOVATE & Johnston 2019: clinical ADC b = 0,150,500,1000.
   - RSIrs (Zhong 2022, Rojo Domingo 2025): ADC b = 0,1000 (and a b=0,500,1000 sensitivity check, *no* difference, P=0.24).
   So the gap is **not** explained by high-b-only ADC.

2. **The real unfairness in RSIrs is the ADC *operationalization*, not its b-values.** Both Seibert-group papers use a **fully automated whole-prostate-minimum ADC** (no radiologist identifying the lesion first). The authors themselves state this "differs from clinical practice, where an expert radiologist typically identifies a suspicious lesion." A whole-gland min-ADC is dominated by BPH nodules / benign restricted foci → ADC AUC collapses to 0.48–0.54, far below the ~0.80–0.90 typical of lesion-localized ADC (cf. He 2025 ADC 0.91, Johnston ADC 0.85). **This is the single most important fairness caveat to foreground.** RSIrs ≈ PI-RADS (radiologist) but RSIrs ≫ automated-min-ADC — an apples-to-oranges detection comparison.

3. **VERDICT's gap is real but is a GRADING gap, with a minor scanner confound.** On a fair lesion-ADC, fIC (0.93) still beats ADC (0.85) for GS≥3+4, and ADC means do not even separate GS3+3 from 3+4 (P=0.26). The residual confound is interscanner (ADC pooled across 3 scanners vs VERDICT on 1) — a hardware artifact, again not a b-value one.

**Net:** Camp B's *detection* advantage largely evaporates against a properly localized PI-RADS ADC (consistent with Camp A). Camp B's *grading* advantage is robust and lives in the cellular/intracellular compartment.

---

## Grading vs detection: which compartments carry the signal

Strong, convergent support for the paper's central claim that **grading signal sits in intermediate/cellular diffusivities that ADC under-weights**:

- **He 2025 (Camp A's own data).** ADC = MC = FROC for **detection** (BPH vs PCa, ~0.91). MC/FROC add value **only** for **grading** (low-vs-int, int-vs-high), via the restricted/intracellular fractions (F1 rises BPH→GS≥8: 0.081→0.145; F1F2 0.063→0.105), while "ADC alone proved inadequate." The discriminative grading parameters are *fractions of inner/intermediate compartments*, not the bulk ADC.
- **VERDICT fIC** maps specifically to the **intracellular / epithelial** compartment (fIC ↔ epithelium; extracellular-extravascular ↔ stroma). fIC separates Gleason grades where ADC cannot (Johnston 2019; Palombo 2023 rVERDICT). Cellularity = fIC/R³.
- **DKI kurtosis K** correlates moderately–strongly with Gleason (r ≈ 0.42–0.73 across studies; Wang/AJR 2018 K90 ↔ GS) — a *grading* signal — yet adds little for *detection* over ADC (Yi Si 2018 meta-analysis AUC 0.93=0.93). K probes non-Gaussian restriction = the intermediate/cellular regime.
- **Hectors 2018** (histology correlation): all diffusion params track the cellularity↔stroma axis; the cellular/cytoplasmic fraction is where the grade-relevant microstructure lives.

**Interpretation for our spectrum model:** detection is a two-compartment problem (very-restricted ~0.25 ↑ and free-water ~3.0 ↓ move together as cellularity rises → ADC captures it in one scalar). Grading requires resolving the **intermediate/lumen compartments (~0.5–2.0 µm²/ms)** — precisely the bins our identifiability analysis flags as poorly determined. So the literature's "grading needs MC, detection doesn't" maps cleanly onto our identifiability geometry.

---

## Reconciliation synthesis

The two camps are not in genuine conflict once the task (detection vs grading) and the ADC operationalization are held fixed. For **detection**, a properly lesion-localized, PI-RADS-compliant ADC (b≤1000) is hard to beat — Camp A is right, and Camp B's large detection gaps (RSIrs ADC AUC 0.48–0.54) are an artifact of an automated whole-prostate *minimum*-ADC rather than unfair b-values. For **grading**, advanced cellular-compartment metrics (VERDICT/rVERDICT fIC, DKI K, MC restricted fractions) genuinely outperform ADC — Camp B is right *there* — because Gleason information lives in the intermediate/intracellular diffusivities that bulk ADC averages away. Our paper reconciles this by showing the spectrum **collapses onto two co-moving outer compartments for detection (hence one scalar suffices)** but **carries grading signal in the poorly-identifiable intermediate bins** — explaining both why ADC "wins" at detection and why MC "wins" at grading, while warning that the grading-relevant compartments are exactly the ill-posed ones.

**Foreground in Discussion:**
- *Camp A anchor:* **Fennessy & Maier 2023** (our co-author; "more accurate decay ≠ better differentiation") + **He 2025** (clean head-to-head: ADC=MC for detection, MC>ADC only for grading, on a fair ADC).
- *Camp B anchor:* **Johnston/Singh VERDICT (Radiology 2019; INNOVATE 2022)** — real fIC>ADC for grading on a fair PI-RADS ADC, with fIC = intracellular/epithelial compartment.
- *Disambiguation citation:* **Zhong 2022 / Rojo Domingo 2025 RSIrs** — cite to make the automated-min-ADC-vs-localized-ADC fairness point explicit.

**2023–2025 papers we must not miss:** He et al. 2025 (Abdom Radiol, the cleanest fair-ADC head-to-head — our single most important new citation), Palombo rVERDICT 2023 (grading in the cellular compartment), Fennessy & Maier 2023 (Stephan's own review — Camp A spine), and the 877-patient RSIrs Rojo Domingo 2025 (largest detection series, latest, the min-ADC caveat at scale).
