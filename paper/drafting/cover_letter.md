# Cover letter — MRM submission

*Draft 2026-08-08. Paste into the submission portal's cover-letter field (or upload as
PDF). Sections marked **[DECIDE]** need a call from Patrick before sending.*

---

Dear Editors,

Please consider the enclosed **Original Research Article**, "Why ADC works: Bayesian
spectral decomposition of prostate multi-b diffusion MRI," for publication in *Magnetic
Resonance in Medicine*.

**What the paper does.** Extended-b-value prostate DWI is routinely summarized by a single
scalar, the apparent diffusion coefficient, and a decade of richer compartment models has
repeatedly failed to beat it for lesion detection while showing gains for grading. Rather
than proposing another model, we ask why one scalar is so hard to beat. Using fully
Bayesian spectral decomposition of 149 regions of interest from 56 patients, we show
(i) which diffusivity compartments a 15-b-value acquisition can actually resolve, with
per-component posterior uncertainty and a Fisher-information/Cramér–Rao account of the
ill-conditioning; and (ii) that tumor–normal variation collapses onto a single
restricted-to-free-water axis, so that ADC is a near-optimal spectral discriminant —
anti-correlated with the optimal eight-component classifier at |r| > 0.97. The
contribution is explanatory: the decomposition does not obsolete ADC, it accounts for it,
and it localizes where added contrast (rather than added model complexity) would have to
come from.

We believe this fits MRM's scope as original methodological work with a directly clinical
readout, and that the result is of practical interest to the many groups developing
multi-compartment diffusion biomarkers: it identifies the acquisition-side reason such
models tie with ADC on detection in this regime, and it does so with an explicit
identifiability analysis rather than an empirical comparison alone.

**Young Investigator Award.** I wish to enter this manuscript in the **ISMRM Young
Investigator Award** competition (I.I. Rabi Award / Prince-Meaney Award) as first author.
I will separately register the paper with the ISMRM at http://www.ismrm.org/YIA.
**[DECIDE — eligibility]** *YIA requires the first author to be a trainee scientist or
clinician; confirm against the current criteria at ismrm.org/YIA given that Patrick's
listed affiliation is "presently an independent researcher" (the work was carried out
at Brigham and Women's Hospital / Harvard Medical School). If the criteria are not met,
delete this paragraph.*

**Code Review request.** We would like to request an **RRSG Code Review** of the
software accompanying this submission.
- *Two-sentence summary:* The paper decomposes multi-b prostate diffusion signal decays
  into an eight-component diffusivity spectrum using both a constrained-ridge MAP
  estimator and full Bayesian (NUTS) inference, and compares spectral classifiers with
  ADC for region-of-interest tumor detection. All reported numbers, tables, and figures
  are generated from the deposited region-of-interest signal decays by the accompanying
  code.
- *Code description:* Python 3.11, dependencies and virtual environment managed with
  `uv`; PyMC/ArviZ for posterior sampling, SciPy for the constrained least-squares MAP
  solve, scikit-learn for the leave-one-out classification, Hydra for configuration,
  Matplotlib for figures. A single entry point (`uv run python -m
  spectra_estimation_dmri.biomarkers.recompute`) regenerates every quantitative result
  in the manuscript from the deposited data; per-figure scripts live in `scripts/`.
- *Link:* the repository URL, Zenodo software DOI, and the commit SHA-1 are given in the
  Data Availability Statement in the manuscript.

**Data availability.** The region-of-interest signal decays and tissue/Gleason labels are
openly deposited on Zenodo (DOI 10.5281/zenodo.20787155). They contain no
patient-identifying information and derive from a cohort acquired under an
IRB-approved protocol, previously described by Langkilde et al. (*Magn Reson Med*
2018;79:2346–2358).

**Prior presentation.** The spectral estimator this work builds on was presented as a
conference abstract at ISMRM 2022 (Wells WM, Maier SE, Westin C-F, "Estimation of
Diffusivity Spectra: Application to Prostate Diffusion MRI"), and is cited as such in the
manuscript. A related abstract submitted to ISMRM 2025 was not accepted and was not
published. No part of this work has been published in a peer-reviewed journal, and the
manuscript is not under consideration elsewhere.

**Conflicts of interest.** The authors declare no competing interests relevant to this
work. *(A separate conflict-of-interest document is uploaded as required.)*
**[DECIDE]** *confirm for Wells and Maier before submitting.*

**[DECIDE — use of language models]** *Include the following paragraph if you want the
disclosure on record. Recommendation: include it. The public repository linked in the
Data Availability Statement contains commit trailers recording assistant-aided editing,
so a reviewer can discover the tool use; a short, factual, up-front statement is the
lower-risk option, and MRM explicitly permits AI tools for language editing.*

> In preparing this manuscript a large language model (Claude, Anthropic) was used as a
> writing and editing aid — for language and structure, for consistency checking of
> reported numbers against the analysis outputs, and for code review of the analysis
> scripts. No text, figure, or result was accepted without verification against the
> underlying data by the authors; the study design, analysis, interpretation, and all
> scientific claims are the authors' own. No AI-generated content is presented as data or
> results.

Thank you for considering our submission.

Sincerely,

Patrick Remerscheid
(on behalf of all authors)
patrick.remerscheid@gmail.com
ORCID: **[DECIDE — insert; ORCID iD is required for submission]**
