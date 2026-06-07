# Literature-grounded novelty & accept-case assessment (2026-06-06)

**Why this file exists.** Patrick asked, honestly: *why should this paper be accepted?* — is "why ADC works" a question the community is burning to answer, or a rationalization for not beating a simple scalar? This is the web-verified answer (three parallel research agents, 2026-06-06), used to lock the manuscript spine + title + citation strategy. Complements `notes/lit_review_two_camps.md` (the detection-vs-grading reconciliation).

Citations marked [VERIFIED] = author/year/venue/DOI-PMID confirmed against PubMed/PMC/publisher. Quotes marked **[QUOTE-FLAG]** still need verbatim confirmation from the PDF before we quote them.

---

## The honest accept-case (the reframe that makes the paper land)

The paper is **not** "we beat ADC" (we don't) and **not** "why ADC works" standing alone (demand is latent; the flat detection version is attackable). The strongest, defensible, reviewer-facing reason to accept is a **measurement-theory contribution**:

> Multi-compartment diffusion models for prostate are proliferating (VERDICT, RSI, IVIM, DKI, biexponential, FROC, multi-exponential NNLS). The field has (a) repeatedly observed they don't beat a *fair* ADC for detection, and (b) warned *qualitatively* that multiexponential prostate fits may extract unsupported parameters (**Mulkern, Balasubramanian & Maier 2017** — our co-author). We give the **first rigorous, prostate-specific identifiability analysis** — Fisher/CRLB **and** a full Bayesian posterior in agreement — showing this clinical acquisition supports only ~2 identifiable diffusivity compartments; that those two are exactly what ADC already captures (so ADC's near-optimality for detection is *explained*, not coincidental); and that the grading-relevant intermediate compartments are precisely the ill-posed ones. The full posterior additionally buys what point estimates can't: honest per-compartment uncertainty and a propagated per-prediction confidence flag.

This hits MRM's stated value criteria head-on: it **resolves an ongoing debate** (the two-camps detection-vs-grading dispute) and **saves replication effort** (tells the field which compartments are fittable) — the exact "negative results we want" clause in the guidelines. "We made spectral decomposition happen AND bounded what it can recover" is not a contradiction — that rigor *is* the contribution.

---

## Angle-by-angle verdicts

### "Why ADC works" — DEMAND: latent, not loud
- **Bourne R, Panagiotaki E. "Limitations and Prospects for Diffusion-Weighted MRI of the Prostate." Diagnostics. 2016;6(2):21. doi 10.3390/diagnostics6020021. PMID 27240408** [VERIFIED]. States the puzzle in print: ADC is "the simplest possible description… a very poor solution to the inverse problem of assessing tissue structure" YET "performs well compared with other MRI contrast methods for prostate cancer detection and grading." → **This is our wedge.** We sharpen their implicit tension into an explicit, answered question. Be honest: the field mostly keeps building complex models rather than asking *why* ADC wins — frame as "an unexamined tension," not "a question everyone is asking."
- **Avoid as demand-support:** Sigmund & Rosenkrantz 2019 "Occam's Razor" (Radiology, doi 10.1148/radiol.2019190371) — title sounds pro-parsimony but content argues VERDICT fIC *beats* ADC; would undercut us.

### "Why ADC works" — NOVELTY of the explanation: HIGH
- No rigorous (identifiability / information-theoretic) explanation of ADC's prostate detection performance exists. Prior statements are **loose verbal correlation asides**, not mechanisms:
  - **He Y et al. 2025, Abdom Radiol, doi 10.1007/s00261-024-04684-z, PMID 39964371** [VERIFIED]: "ADC exhibited strong correlations with D, μ, F1, F1F2 (0.97, 0.90, −0.95, −0.96)… This correlation may explain… the limited improvements in AUC." One-sentence aside. NB the paper's own thesis is that advanced models *improve grading* — cite accurately.
  - **Fennessy & Maier 2023, Eur J Radiol 167:111066, PMID 37651828** [VERIFIED] (our co-author): "more accurate decay description ≠ better differentiation." **[QUOTE-FLAG]** exact sentence not fetch-confirmed — pull from `assets/`.
  - **Hectors SJ et al. 2018, Radiology 286(3):918–928, doi 10.1148/radiol.2017170904, PMID 29117481** [VERIFIED]: *most* diffusion metrics track the same cellularity↔stroma axis (more nuanced than "all the same axis").
  - **Chatterjee A et al. 2018, Radiology 287(3), doi 10.1148/radiol.2018171130, PMID 29393821** [VERIFIED] (HM-MRI): epithelium/stroma/lumen fractional volumes track Gleason — second histology anchor.

### "Why ADC works" — the DETECTION-scoping risk (must address head-on)
- Flat "ADC hard to beat for **detection**" is the weakest framing because **RSI/RSIrs (Seibert group)** claims to beat ADC for detection. **BUT** our `lit_review_two_camps.md` already disambiguated this: the RSIrs gap (ADC AUC 0.48–0.54) is an artifact of *automated whole-gland minimum-ADC*, not a fair lesion-localized PI-RADS ADC — and not a b-value issue. → We can defend "fair PI-RADS ADC is hard to beat for detection," but we MUST engage RSI explicitly (don't ignore it). Anchor "hard to beat": **Si Y & Liu R-B 2018, AJR 211(2):358–368, doi 10.2214/AJR.17.18934, PMID 29812977** [VERIFIED] — pooled AUC ADC 0.93 = DKI 0.93, "we do not recommend including DKI in routine clinical assessment… for the moment."

### Identifiability / ill-posedness — PHENOMENON is prior art; our PROSTATE quantification (Fisher+posterior) is novel
- Established, but all via fit-landscape topology / empirical bias, all **brain**, none Fisher/CRLB+Bayesian, none prostate:
  - **Istratov & Vyvenko 1999, Rev Sci Instrum 70(2):1233, doi 10.1063/1.1149581** [VERIFIED] — multiexponential analysis is fundamentally ill-posed (mathematical bedrock).
  - **Jelescu IO et al. 2016, NMR Biomed 29(1):33–47, doi 10.1002/nbm.3450, PMID 26615981** [VERIFIED] — two local minima; bias/poor precision even when oversampled.
  - **Novikov DS, Kiselev VG, Jespersen SN 2018, MRM 79(6):3172–3193, doi 10.1002/mrm.27101, PMID 29493816** [VERIFIED] — "flatness/shallowness of the fit objective"; limited information content.
  - **Lampinen B et al. 2019, Hum Brain Mapp 40(8):2529–2545, doi 10.1002/hbm.24542** [VERIFIED] — must constrain #compartments to data DOF "to avoid degeneracy."
  - **Novikov DS et al. 2019, NMR Biomed 32(4):e3998, doi 10.1002/nbm.3998, PMID 30321478** [VERIFIED] — "resolving its hidden degeneracies."
- **CRLB+MCMC in diffusion already exists:** **Alexander DC 2008, MRM 60(2):439–448, doi 10.1002/mrm.21646, PMID 18666109** [VERIFIED] — pairs CRLB optimization with MCMC posterior, but for *acquisition design*, not counting identifiable prostate compartments. Cite as "we are not first to use this machinery."
- **Prostate-specific prior warning (cite prominently — co-author):** **Mulkern RV, Balasubramanian M, Maier SE 2017, JMRI 45(5):1545–1547, doi 10.1002/jmri.25485** [VERIFIED] — "On the perils of multiexponential fitting of diffusion MR data." Qualitative; we quantify. **[QUOTE-FLAG]** paywalled, verify wording.
- **DISCONFIRMING — engage directly:** **Conlin CC et al. 2021, JMRI 53(2):628–639, doi 10.1002/jmri.27393** [VERIFIED] — BIC favors a **4-compartment** prostate model. Our rebuttal (a sharp, publishable point): BIC model-*selection* (penalized population fit) ≠ per-ROI parameter *identifiability* (posterior CV, CRLB); a model can lower BIC while individual compartment fractions stay undetermined. Also recent NNLS prostate work claims 3 resolvable peaks — point estimates from regularized inversion *always* show structure; our posterior CV shows the intermediate fraction is unconstrained.

### NMR relaxometry analogy + our own prior abstract (overlap policy!)
- **Whittall KP & MacKay AL 1989, J Magn Reson 84(1):134–152** [VERIFIED] — regularized-NNLS T2 distributions (relaxometry analogue).
- **Prange M, Song YQ 2009, J Magn Reson 196(1):54–60, doi 10.1016/j.jmr.2008.10.008, PMID 18952474** [VERIFIED] — Monte-Carlo uncertainty on NMR T2 spectra; **closest methodological precedent** to our posterior. Cite as direct lineage.
- **⚠️ OVERLAP — must disclose:** our own **ISMRM 2022 abstract (Wells, Maier, Westin), "Bayesian Estimation of Diffusivity Spectra: Application to Prostate Diffusion MRI"** (`assets/ISMRM-2022-abstract.pdf`) already published the Bayesian-spectrum method on the same BWH data + same bins. → The journal paper's novelty CANNOT rest on the spectrum method; it rests on (i) Fisher/CRLB identifiability, (ii) posterior-CV quantification, (iii) "why ADC works," (iv) downstream uncertainty-aware classifier. **MRM policy: cite the abstract in references AND mention in the cover letter** (ISMRM abstract reuse is free but must be cited).

### Joint noise + signal inference — NOT NOVEL (do not headline)
- **Sjölund J et al. 2018, NeuroImage 175:272–285, doi 10.1016/j.neuroimage.2018.03.059, PMID 29604453** [VERIFIED] — already marginalizes the noise variance (inverse-Gamma prior → multivariate-t posterior) in *linear-model dMRI*. Decisive. Also **Behrens TEJ et al. 2003, MRM 50(5):1077–1088, doi 10.1002/mrm.10609, PMID 14587019** (BEDPOSTX) and **Orton MR et al. 2014, MRM 71(1):411–420, PMID 23408505** (Bayesian IVIM). → **Reframe:** "fully Bayesian noise treatment (no plug-in σ) within a multi-compartment spectral model in prostate," NOT "first to infer noise jointly." (Answers Patrick's abstract.tex:6 note: don't over-advertise.)

### UQ demand — STRONGLY supported (great motivation, not a novelty claim)
- **Faghani S et al. 2023, Radiology 308(2):e222217, doi 10.1148/radiol.222217** [VERIFIED] — explicit call for UQ in radiologic DL. **[QUOTE-FLAG]**.
- **Alves N et al. 2023, Radiology 308(3):e230275, doi 10.1148/radiol.230275, PMID 37724961** [VERIFIED] — prediction-variability UQ flags reduced AI performance, **includes prostate MRI**. Most on-point.
- **Lambert B et al. 2024, Artif Intell Med (arXiv 2210.03736)** — review-level "UQ needed for clinical deployment." **[QUOTE-FLAG]**.

### Uncertainty propagated to the decision / as a confidence flag — PARTIALLY NOVEL
- Generic pattern is taken: **Ghesu FC et al. 2021, Med Image Anal 68:101855, doi 10.1016/j.media.2020.101855, PMID 33260116** (uncertainty-based rejection improves AUC) and **Tanno R et al. 2021, NeuroImage 225:117366, doi 10.1016/j.neuroimage.2020.117366** (dMRI uncertainty propagated to derived scalars, flags failures — but DL enhancement, not biophysical-posterior→diagnosis). → **Claim, qualified:** "to our knowledge, first to propagate a biophysical multi-compartment posterior into a prostate ROI cancer classifier and use predictive uncertainty as a confidence flag." Patrick: still exploratory; the two posterior-uncertainty comparisons (misclassified-vs-correct; near-vs-far-from-boundary) need the math re-cleaned (logit-geometry subtraction) before any solid claim.

---

## Critical action items this assessment forces
1. **Cite prominently: Mulkern/Balasubramanian/Maier 2017** (co-author prior warning we quantify) + **Bourne & Panagiotaki 2016** (the stated tension we answer).
2. **Disclose the ISMRM 2022 abstract** in references + cover letter; position novelty on the new layers (identifiability + why-ADC + UQ-classifier), not the spectrum method.
3. **Engage RSI and Conlin-2021-BIC head-on** in Discussion (identifiability ≠ model-selection; localized vs automated-min ADC).
4. **Do NOT claim "joint noise inference" as novel** — reframe (Sjölund 2018). **Do NOT claim "uncertainty-aware classification" broadly novel** — qualify "to our knowledge, in this biophysical-to-diagnosis setting" (Ghesu/Tanno).
5. **Add UQ-demand citations** (Faghani 2023, Alves 2023) to motivate the uncertainty contribution.
6. **Verify all [QUOTE-FLAG] verbatim quotes** from PDFs before quoting; lock RSI/VERDICT DOIs in the citation pass.
7. Frame the GGG-grading and uncertainty-classifier results as **feasibility/exploratory** (MRM wants that stated explicitly).
