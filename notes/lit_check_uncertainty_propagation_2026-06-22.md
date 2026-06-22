# Lit check — uncertainty propagation + classifier confidence (2026-06-22)

For the Discussion paragraphs on the **uncertainty figure** (current Fig 8) and **pixel-wise figure** (current Fig 9). Background agent synthesis; sourcing caveats at bottom.

## Q1 — Propagating fit-parameter posterior uncertainty into a downstream classifier
**Verdict: PARTLY NOVEL → lead with it as a NULL result.**
The general idea (propagate estimation uncertainty into a downstream task) exists, and Bayesian UQ of diffusion-MRI fits is an active small field. But no prior work takes the *full Bayesian posterior over fitted spectral/microstructural parameters*, pushes every draw through a *trained probabilistic classifier* to get a posterior over P(tumor), and **tests whether that adds diagnostic value** (with a null finding) — especially not in prostate/quantitative dMRI.

Closest precedents (and how they differ):
- **Mehta et al. 2022, IEEE TMI 41(2):360–373** — "Propagating Uncertainty Across Cascaded Medical Imaging Tasks." Strongest precedent for the *idea*; but DL *predictive* uncertainty (not a fit posterior), downstream = seg/detect/regress, and finding is *positive*. (PMID 34543193)
- **Mehta et al. 2023, MICCAI UNSURE** — propagation/attribution of uncertainty in imaging pipelines. Same family, DL not fit-posterior.
- **IVIM UQ framework 2025, arXiv:2508.04588** — quantifies IVIM fit uncertainty, proposes *flagging unreliable estimates*; stops at parameter-level UQ, never into a classifier.
- **SBI for dMRI posteriors (Jallais/Karaman-type), 2024, Med Image Anal / Imaging Neuroscience** — full posteriors of MC dMRI params propagated into FA/tractography, not a classifier; goal is uncertainty mapping/robustness.
- **Bayesian IVIM (Orton lineage)** — uses posterior *summaries* as features; no full-posterior-through-classifier test.
- **Prostate spectral/DBSI/RSI classifiers** (spectral diffusion feasibility PMC11645486; DBSI+NN biorxiv 2021.03.22.436514; 4-comp RSI medRxiv 2020.07.25.20162172) — all classify on **point estimates**; none carry posterior uncertainty into the classifier. (Confirm in full before citing as "none.")

**How to phrase:** "to our knowledge, propagating a full fit-parameter posterior through a downstream tumor classifier and testing its incremental diagnostic value has not been reported in prostate (or quantitative) diffusion MRI; the nearest precedents propagate a different kind of uncertainty, target a different downstream task, or stop at parameter-level UQ. Our **null finding is the informative contribution** — the point estimate already captures the discriminative signal; posterior width does not flag misclassifications here." Tie to the nuance Patrick raised: outer bins have low CV anyway, and even the high-CV 3.0 bin's uncertainty does not propagate to a useful classification signal.

## Q2 — "Prediction confidence for free" from a probabilistic classifier
**Verdict: STANDARD — do NOT claim novelty.** Calibrated predicted probability + distance-to-boundary as confidence is textbook and routine in prostate-MRI radiomics.
- Standard terms: **calibration / reliability diagram**; **decision-curve analysis (DCA)** (Vickers & Elkin 2006; Vickers et al. 2021); **selective prediction / reject option / abstention** (Chow 1970 → modern selective classification); **conformal prediction** (distribution-free; already in prostate — histopath flagging; DL prostate-MRI seg UQ PMID 39613981). Honesty caveat: "Pitfalls of Conformal Predictions for Medical Image Classification" 2023 (MICCAI UNSURE, arXiv:2506.18162) — guarantees break under distribution shift.
**How to phrase:** present as a *usable property we make explicit*, not a methodological novelty: "Because the classifier is probabilistic, each ROI gets a calibrated cancer-likelihood rather than a binary label; predictions near the boundary are automatically least confident, connecting to the established selective-prediction/abstention framework (formally, conformal prediction). We note this as a usable property, not a novelty." If we want weight, add a calibration curve and/or DCA.

## Positioning the two findings (honest split)
- **Q1 = novel null result** (lead with it; the posterior adds no detection value).
- **Q2 = explicitly standard** (concede; convenient clinical property of the probabilistic classifier).
This split pre-empts reviewer pushback on overclaiming.

## Sourcing caveats
Verified via abstracts/full text: Mehta 2022 TMI, IVIM-UQ 2025 preprint, SBI-dMRI papers, prostate radiomics/DCA/conformal items. The prostate spectral/DBSI/RSI "classify on point estimates" claim was from result snippets, not full reads — confirm before citing as "none propagate uncertainty." No paper matching our exact null finding was found (supports but does not prove novelty).
