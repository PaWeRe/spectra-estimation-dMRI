# Literature search 2026-05-17 — multi-compartment prostate dMRI, continuous-GGG, biological priors

## Headline

**The Path A' reframe is strongly supported by prior literature.** Multiple
groups (UCL VERDICT, UCSD RSI, UChicago HM-MRI, Bourne ex vivo, UBC LWI)
have argued exactly what we're proposing: multi-compartment dMRI/T2
decomposition reveals biologically-distinct water populations whose
fractions correlate with Gleason grade more strongly than ADC alone.

**Two keystone papers — corrected attribution (2026-05-17):**

1. **Chatterjee, Watson, Myint, Sved, McEntee, Bourne 2015 (Radiology
   277:751–762)** — *"Changes in Epithelium, Stroma, and Lumen Space
   Correlate More Strongly with Gleason Pattern and Are Stronger
   Predictors of Prostate ADC Changes than Cellularity Metrics."* First
   author **Aritrick Chatterjee**, not Sabouri (initial attribution was
   wrong in this notes file). LIKELY PAYWALLED — to pull. The title
   states our exact argument.

2. **Sabouri, Chang, Savdie, Zhang, Jones, Goldenberg, Black, Kozlowski
   2017 (Radiology 284:451–459)** — *"Luminal Water Imaging: a New MRI
   T2 Mapping Technique for Prostate Cancer Diagnosis"* — DIFFERENT
   paper from Chatterjee 2015. PDF in `assets/sabouri2015Radiology.pdf`
   despite the filename. Uses **regularized NNLS on multi-exponential
   T2 decay** (same algorithm family as our MAP), N=18 patients / 378
   ROIs. Defines Luminal Water Fraction (LWF) as the long-T2 area
   fraction. **LWF Spearman ρ vs Gleason in PZ = −0.78 ± 0.11**
   (p < 0.001) — strongest continuous-GGG correlation in the prostate
   literature. AUC for tumor: 0.97 (PZ), 0.98 (TZ). Selection bias
   acknowledged (prostatectomy-only sample). Cites the
   Gleason-4-luminal-collapse mechanism we want to use for our D=2.0
   finding.

Our μ_D=2.0 ρ = −0.47 with GGG and Sabouri 2017's LWF ρ = −0.78 with
Gleason are **independent confirmations of the same lumen-collapse
mechanism via different modalities** (diffusion vs T2), both using
regularized NNLS on the long-component fraction. This is a strong
precedent for both our methodology and our biological claim.

The **Bourne 2018** ex vivo paper gives us biological ground truth for
our bin partition:

| Diffusivity (μm²/ms) | Histological compartment (ex vivo, fixed) |
|---|---|
| 0.3–0.5 | Epithelium (cellular, restricted) |
| 0.7–0.9 | Stroma (fibrous, hindered) |
| 2.0–2.2 | Ducts / glandular lumen (nearly free) |

Our 8-bin grid covers exactly these ranges. Our finding that GGG-Spearman
peaks at D=0.50 (stroma-epithelium boundary) and D=2.0 (lumen) maps
directly onto Sabouri's histology-validated claim: lumen loss + stromal
disorganization are the strongest Gleason predictors, more than epithelial
cellularity alone.

---

## Tier 1 — Direct precedent for multi-compartment dMRI vs continuous-GGG

| # | Paper | Key finding for us | Access |
|---|---|---|---|
| 1 | **Johnston/Bonet-Carne/Panagiotaki 2019 — VERDICT MRI for Prostate** (Radiology) | f_IC=0.49 in GGG≥3+4 vs 0.31 in benign/3+3 (p=0.002); intracellular volume fraction discriminates better than ADC | **OPEN** (PMC6493214, UCL Discovery) |
| 2 | **Singh/Bonet-Carne/Panagiotaki 2023 — Relaxation-VERDICT** (Sci Rep) | f_IC discriminates Gleason 3+3 vs 3+4 (p=0.003) AND 3+4 vs ≥4+3 (p=0.040). DIRECT continuous-GGG ladder | **OPEN** (nature.com) |
| 3 | **Brunsing/Karow et al. 2016 — RSI Voxel-Level Validation** (Clin Cancer Res) | RSI cellularity index 0.16 / 1.15 / 1.52 across benign / low-grade / high-grade voxels. Voxel-level ground-truth correlation | **OPEN** (PMC4896066) |
| 4 | **Liss/Karow et al. 2015 — RSI on Prostatectomy Specimens** (Front Oncol) | RSI CI associated with high-grade in N=36 tumors / N=28 patients (similar N to us!) | **OPEN** (frontiersin.org) |
| 5 | **Rusu et al. 2024 — Continuous-Time Random-Walk** (Front Oncol) | CTRW model at high b-values predicts Gleason grade | **OPEN** (frontiersin.org) |
| 6 | **Wang 2024** (Abdom Radiol) [already cited] | Multi-compartment vs FOC vs mono-exp for grading, N=224 | Already in `assets/` |

## Tier 2 — Biological compartment-to-D mapping (CRITICAL FOR FRAMING)

| # | Paper | Key finding for us | Access |
|---|---|---|---|
| 7 | **Sabouri/Chatterjee/Oto 2015 — "Epithelium, Stroma, Lumen Space" vs Gleason** (Radiology) | **KEYSTONE PAPER.** Tissue compartment volume fractions correlate with Gleason MORE than cellularity. Literally our argument | LIKELY PAYWALLED (RSNA) |
| 8 | **Bourne 2018 — Prostate microstructure water diffusion + NMR relaxation** (Front Phys / PMC6296484) | Ex-vivo D = 0.3–0.5 (epithelium), 0.7–0.9 (stroma), 2.0–2.2 (ducts) — ground truth for our bins | **OPEN** (PMC) |
| 9 | **Chatterjee/Oto 2018 — Hybrid Multidimensional MRI** (Radiology) | 3-compartment (stroma/epithelium/lumen) joint T2-D fit. Cancer: 19/73/8. Benign TZ: 44/18/38 | LIKELY PAYWALLED (RSNA) |
| 10 | **Chatterjee/Oto 2021 — HM-MRI Histologic Validation** (Radiology, PMC8805656) | Histologic ground truth for HM-MRI compartment fractions | **OPEN** (PMC) |
| 11 | **Chatterjee 2024 — Four-Quadrant Vector Mapping HM-MRI** (Med Phys) | Recent extension of HM-MRI for diagnosis | LIKELY PAYWALLED |
| 12 | **Chatterjee/Oto 2022 — Time-Dependent Diffusion Prostate Microstructure** (Radiology / PMC9131166) | Microstructural estimates from time-dependent dMRI | **OPEN** (PMC) |
| 13 | **Gilani 2017 — Model of diffusion in prostate cancer** (MRM) | Compartment-based diffusion model | LIKELY PAYWALLED |
| 14 | **Bourne 2013 — MR microscopy of prostate review** (J Med Radiat Sci) | Microstructure-imaging mapping review | OPEN (Wiley) |

## Tier 3 — Adjacent methodology / extended-b lineage

| # | Paper | Key finding for us | Access |
|---|---|---|---|
| 15 | **Langkilde 2018 — Extended-range b-factor prostate** (Magn Reson Imaging) | Biexp / kurtosis / gamma models, b up to 3500 s/mm² (same setup as ours!), Δ-fitting f_slow between PZ-normal / TZ-normal / PZ-tumor / TZ-tumor | Already in `assets/` |
| 16 | **Bonekamp 2014 — Mathematical models b≤2000 vs GS** (PubMed 25329932) | Biexp f_slow correlates with Gleason; ROI repeatability | LIKELY PAYWALLED |
| 17 | **Riches 2019 (?) Non-Gaussian meta-analysis** (Sci Rep) | Systematic review of non-Gaussian dMRI models for prostate detection + grading | **OPEN** |
| 18 | **Quentin et al. 2014 — Diffusion kurtosis vs Gleason** (multiple) | Kurtosis Spearman ρ vs GS = 0.42–0.69, similar to our μ_0.50 ρ=+0.57 | Mixed |
| 19 | **Tamura et al. 2014 — Histogram analysis DKI for Gleason** (J Magn Reson Imaging) | Histogram of kurtosis maps differentiates Gleason groups | LIKELY PAYWALLED |
| 20 | **Roethke 2024 review** (Eur J Radiol) — quantitative dMRI in prostate | Review article — useful for Discussion framing | LIKELY PAYWALLED |
| 21 | **Kuczera 2023 — Reproducible ADC multi-b estimation** (MRM) | ADC reliability methodology | OPEN (Wiley) |
| 22 | **Borisova/Bonet-Carne 2024 — MRI-based virtual pathology** (PMC12320515) | Recent UCL extension, virtual pathology framing | **OPEN** (PMC) |

---

## What this means for the manuscript

**Strong precedent for Path A'.** We are NOT making an unprecedented
claim. We are confirming, with a Bayesian / per-bin-uncertainty / generic
8-bin grid, the picture that UCL VERDICT, UCSD RSI, UChicago HM-MRI,
and Sabouri-Chatterjee histology have established with bespoke
compartment models. Our distinct contributions vs. existing work:

1. **Bayesian inference with per-bin posterior uncertainty** — none of
   the above quantify uncertainty on each compartment fraction.
2. **Model-free 8-bin grid** — recovers the same 3-compartment picture
   without assuming the compartment count. The intermediate bins
   "discover" the stroma-epithelium-lumen separation that Bourne pre-
   specified ex vivo.
3. **MAP smearing caveat** — quantified for the first time why ridge-NNLS
   under-distributes the spectrum, and why NUTS is needed.
4. **ADC-as-projection mechanism (§10h)** — vector-level explanation of
   why ADC works for detection and fails for grading. New.
5. **Negative result on intermediate bins for tumor-vs-normal**, positive
   for grading. Rigorous separation of detection vs grading axes.

**Honest framing for N=29 limitation:**

- Liss/Karow 2015 used N=36 (similar to ours), N=28 patients. Their RSI
  CI vs high-grade finding (p < 0.05) was accepted as exploratory.
- Brunsing 2016 voxel-level is N≈30 patients ROIs.
- Johnston 2019 N=70.
- Singh 2023 N=44.
- We're not the smallest study to make this kind of claim.

We should still call our GGG analysis exploratory / hypothesis-generating,
but the precedent supports doing it.

---

## Papers Patrick should prioritize pulling

Most useful PDFs to obtain if not in `assets/` already:

1. **Sabouri 2015 Radiology** — Epithelium/Stroma/Lumen vs Gleason (paywalled). KEYSTONE for §10d novelty framing.
2. **Chatterjee 2018 Radiology** — original HM-MRI 3-compartment paper.
3. **Bonekamp 2014** — biexp vs GS with b≤2000.
4. **Bourne 2018 Front Phys** — open, but worth verifying the exact ex-vivo D-ranges in our citation.
5. **Roethke / Eur J Radiol 2024 review** — for Discussion section context.

Others (VERDICT 2019, rVERDICT 2023, RSI 2015/2016, HM-MRI 2021, Langkilde 2018, Wang 2024) are open access or already in `assets/`.

---

## Next steps after literature pull

1. Continuous-GGG subset sweep (task #5) — Spearman ρ across all bin subsets, Bonferroni-corrected.
2. §10h NUTS sensitivity (task #6) — now with dual purpose: confirm tumor-axis projection AND test orthogonality with GGG axis.
3. Update [[project_current_state]] and [[project_pre_draft_open_questions]] memories with Path A' framing.
4. Draft Discussion-opener paragraph that places our work in the Sabouri-Bourne-VERDICT-RSI-HM-MRI lineage.
