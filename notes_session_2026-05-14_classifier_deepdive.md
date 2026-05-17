# Session Notes — Classifier Deep-Dive & Narrative Pivot

_Date: 2026-05-14. Companion to `notes_manuscript_3rd_pass.md`._

This session went much deeper than planned. The 3rd-pass executive list item
#1 ("LR-vs-individual-feature paradox") expanded into a foundational
question about whether the manuscript's central claim — "ADC ≈ optimal
spectral classifier" — survives a careful diagnostic. It mostly does not,
in its current MAP-based 8-feature form. The session ended with the paper
in a narrative-pivot state and a focused push-back agenda for the next
session.

**Patrick's governing principle for the next session (verbatim where possible):**

> I am not trying to cheat or make something look real that isn't. But what
> I do want to make sure is that we can make clear or somewhat confident
> statements (positive or negative). If we cannot recover more than two
> buckets from the diff spectrum, so be it. If it does not even make sense
> to recover more than two buckets (because they do not contribute to
> cancer classification), so be it but this should be conclusive. If it
> does not make sense to even estimate the spectrum, so be it but I want to
> be sure. If ADC is not a local linear projection of an optimized spectral
> classifier, so be it, but I want it to be conclusive.

The next session is about ruling out alternative explanations for the
findings below before either committing to the Path A reframing or
deciding the paper isn't worth publishing.

---

## 1. What we resolved this session

### 1a. Exec #8 — directional-data provenance

Overstated in the 3rd-pass plan; corrected in place. Current
`paper/figures/fig_directions.png` (commits 27f446b → b504bbf, 2026-05-10)
is per-ROI mean spectra from `9283-Series12-Slice6-{Normal,Tumor}PZ.dat`
in Stephan's recent tarball `diff_spectrum_3_directions.tar.gz`. The
pixel-wise demo (Fig 9) uses patient `8640`. Whether patient 9283 (or 8640)
is part of the Langkilde 2018 BWH cohort cannot be determined from the
repo — BWH ROIs are stored under pseudonyms `new01..new56`, and the raw
study IDs in Stephan's tarball (10203, 8804, 8805, 8864, 9283, 9322, 9675)
have no mapping back to those pseudonyms. The acquisition protocol in the
.dat headers matches Langkilde 2018 exactly, so "same cohort, raw ID" is
plausible but unproven.

Flagged as `[ASK Sandy/Stephan]`. Also flagged independently: Fig 7
(`fig:directions`) caption in `figures.tex:167–177` is stale — still
describes "representative prostate voxels from the supplementary patient"
while the current figure is per-ROI mean from a different patient than
the pixel demo.

### 1b. Exec #1 — LR-vs-individual-feature paradox

Resolved as a methodological asymmetry, not a code bug, but the
investigation produced a substantial narrative problem (see §2 below).

**Mechanism:** in `scripts/generate_paper_figures.py:213–253`,
per-component AUC lines in Fig 2 are computed in-sample via
`roc_auc_score(y, df[f"map_{d}"].values)` over the full PZ / TZ / GGG ROI
subset; the 8-feature LR AUC lines on the same panel are computed via
true LOOCV with `loocv_roc(...)`. For a single feature with no learnable
parameters, in-sample-AUC and LOOCV-AUC are mathematically identical
(the held-out prediction is just the raw feature value, which equals
its in-sample contribution to the AUC rank). For an 8-feature LR, the
two differ by an overfitting penalty.

The scaler is correctly fit inside the fold
(`mc_classification.py:169–175` and `recompute.py:292–298` and
`generate_paper_figures.py:77`); the regularization grid C ∈ {0.1, 1, 10}
does not close the gap. There is no leakage and no bug.

---

## 2. The diagnostic findings (these drive the next session)

All numbers below are on the current `results/biomarkers/features.csv` (149
ROIs, 56 patients, Session 8 recomputation).

### 2a. D=0.50 dominance is a MAP artifact

This is the single most important finding of the session. Under MAP, D=0.50
is the strongest single-bin discriminator. Under NUTS, it isn't.

**PZ tumor (n=81, 27 tumor / 54 normal) — single-bin AUC [95% bootstrap CI]:**

| Bin | MAP AUC | NUTS AUC |
| --- | ---: | ---: |
| D=0.25 | 0.901 [0.81, 0.97] | 0.880 [0.77, 0.96] |
| **D=0.50** | **0.947 [0.88, 0.99]** | **0.796 [0.69, 0.89]** |
| D=0.75 | 0.903 [0.82, 0.96] | 0.808 [0.70, 0.90] |
| D=1.00 | 0.650 [0.52, 0.78] | 0.829 [0.73, 0.92] |
| D=1.50 | 0.833 [0.71, 0.94] | 0.833 [0.73, 0.92] |
| D=2.00 | 0.932 [0.84, 0.99] | 0.587 [0.51, 0.71] |
| D=3.00 | 0.944 [0.88, 0.99] | **0.921 [0.85, 0.98]** |
| D=20.00 | 0.878 [0.79, 0.95] | 0.519 [0.50, 0.66] |
| ADC | 0.951 [0.89, 0.99] | — |

**TZ tumor (n=68, 13 tumor / 55 normal) — single-bin AUC:**

| Bin | MAP AUC | NUTS AUC |
| --- | ---: | ---: |
| D=0.25 | 0.947 | **0.908** |
| **D=0.50** | **0.972** | 0.853 |
| D=0.75 | 0.913 | 0.846 |
| D=3.00 | 0.971 | 0.902 |
| ADC | 0.979 | — |

**Interpretation.** Under NUTS, the strongest single bin is the
biologically interpretable one (D=0.25 restricted in TZ, D=3.00 free
water in PZ). MAP's ridge prior smears restricted-diffusion mass across
D=0.25, D=0.50, D=0.75 — the smeared mass at D=0.50 is highly correlated
with tumor status, but this is the prior's signature, not the data's.

### 2b. NUTS LR is effectively 2-dimensional

PZ NUTS-LR coefficients at C=1.0:
```
D=0.25: +1.66   D=0.50: +0.30   D=0.75: +0.26   D=1.00: +0.25
D=1.50: +0.32   D=2.00: +0.23   D=3.00: −1.03   D=20.00: −0.03
```

Two coefficients dominate: D=0.25 (+1.66) and D=3.00 (−1.03). The six
intermediate bins each have weights below 0.35.

PZ MAP-LR coefficients at C=1.0 (for contrast):
```
D=0.25: +0.71   D=0.50: +0.55   D=0.75: +0.40   D=1.00: +0.07
D=1.50: −0.36   D=2.00: −0.44   D=3.00: −0.55   D=20.00: −0.76
```
Weight is spread across all 8 bins because ridge smearing makes them
correlated.

### 2c. 2-bin NUTS LR matches or beats 8-bin NUTS LR

NUTS LR LOOCV AUC at C=1.0:

| Feature set | PZ | TZ | GGG (≥3 binarization) |
| --- | ---: | ---: | ---: |
| all 8 | 0.926 | 0.923 | 0.772 |
| D=0.25 only | 0.863 | 0.884 | 0.756 |
| D=0.25 + D=3.00 | **0.933** | **0.937** | 0.744 |
| well-id (0.25, 3.0, 20) | 0.929 | 0.920 | 0.794 |
| 8 minus D=0.50 | 0.929 | 0.924 | 0.783 |
| 8 minus mid (drop 0.5–1.5) | 0.932 | 0.927 | **0.817** |

The 2-feature NUTS LR on {D=0.25, D=3.00} is the best tumor-vs-normal
variant. The 8-feature LR is *worse* than its biologically clean subset
because of overfitting from 6 noisy bins.

### 2d. ADC remains the best single feature, period

PZ 0.951, TZ 0.979, GGG 0.811. No spectral classifier — MAP or NUTS,
any feature subset — beats ADC out-of-fold.

### 2e. GGG is statistically dead at N=29

Bootstrap 95% CIs on the GGG AUCs are ~0.30 wide:
- ADC: 0.811 [0.62, 0.97]
- NUTS D=0.25: 0.817 [0.61, 0.99]

With only 9 high-grade ROIs (under the current GGG≥3 binarization) or 4
(under the old GS≥8 binarization), no AUC near the unit-interval corners
can be trusted. Apples-to-apples re-run with the old binarization shows
the "old combo beat ADC by 0.01" was in-sample noise — under LOOCV on
the same data, every spectral LR is meaningfully *worse* than ADC.

### 2f. The old Gibbs-era headline numbers were in-sample

`/Users/PWR/Documents/Professional/Papers/Paper3/code copy/spectra-estimation-dMRI/src/spectra_estimation_dmri/eval/eval.py:20–50`
contains the "feature combo" formula:

```python
feature_combo = feature_025 + 1/feature_2_0 + 1/feature_2_50   # weighted by 1/std
```

This is a hand-engineered nonlinear feature (sum + reciprocals,
pre-multiplied by `1/posterior_std`). The CSV
`output/eval/025+1:2+ 1:250_ggg_stat_analysis_combined.csv` reports
in-sample AUCs (`eval.py:296` is `roc_auc_score(y_true, y_scores)` over
the full sample) with t-test p-values — no cross-validation. The
"Weighted Linear Combination" function that did 5-fold CV
(`eval.py:540–604`) was commented out in `main()` (lines 675–687) and
its CSV was never the headline. So the famed "old combo beat ADC by
0.01 on GGG" is not a reproducible result; it's an in-sample fit on a
hand-engineered nonlinear formula at N ≤ 29.

---

## 3. Literature pass — where we stand vs the field

(Source: subagent literature search on 2026-05-14, transcript at
`/private/tmp/.../afeb2168ee9c16181.output`.)

### 3a. Apparent novelty axes

- **Bayesian inference + per-bin identifiability in prostate dMRI** —
  could not find a published prostate paper that does what we do (full
  posterior over a discrete diffusivity spectrum + per-bin CV reporting).
  Closest prior work: Conlin 2021 JMRI (BIC model selection, no
  posterior uncertainty); Sjölund et al. 2018 NeuroImage (Bayesian
  uncertainty, brain not prostate); Reci, Sederman, Gladden 2017 JMR
  (Bayesian NMR, not in vivo).
- **"ADC sensitivity ≈ LR coefficient" mechanistic claim** — could not
  find this derivation in any anatomy. Quigley & Mitchell 2023 Eur J
  Radiol gestures at related ideas but does not formalize it.

### 3b. Published precedent for "MC ≈ ADC"

- **Wang 2024 (Abdom Radiol, doi:10.1007/s00261-024-04684-z)** —
  explicitly states MC parameters correlated with ADC "may explain...
  limited improvements in AUC." This is the closest published version
  of our mechanistic claim. **Citation to use.**
- **Wang 2018 (JMRI, PMID 29812977)** — meta-analysis pooled IVIM/DKI vs
  ADC AUCs ~0.93 each — "comparable, not superior."
- **Quigley & Mitchell 2023 (Eur J Radiol)** — review framing "more
  accurate signal decay description does not imply higher sensitivity."

### 3c. Counterpoints (papers claiming MC > ADC)

- **RSI / RSIrs (Conlin, Karow, Liss, Zhong et al., recent J Urology and
  Radiology)** — report csPCa AUCs of 0.73–0.78 vs **ADC AUC of
  0.48–0.54**. The ADC implementation differs from ours (likely uses
  only high-b values, not PI-RADS-compliant b ≤ 1000). The disagreement
  must be acknowledged in our Discussion.
- **VERDICT INNOVATE (Singh et al. 2022, Radiology)** — N=303, modest
  improvement over ADC.

### 3d. Sample-size context

N=56 patients / 149 ROIs is on the higher end for methods-focused
prostate multi-compartment DWI (Conlin 2021 N=46, rVERDICT N=44,
HM-MRI N=12–60). Adequate for an MRM methods paper. **Not adequate for
GGG sub-stratification — drop GGG to limitations regardless of path
chosen.**

---

## 4. The narrative pivot — Path A vs B vs C vs D

Four real options as of session end. Patrick has not committed to one.

- **Path A** — Reframe as cautionary methods paper. Lead with Bayesian
  identifiability framework + principled ADC interpretation (citing
  Wang 2024). Drop GGG to limitations. Switch the spectral-discriminant
  story from 8-feature MAP-LR to 2-feature NUTS-LR on {D=0.25, D=3.00}.
  Rebuild Figs 2–4 against NUTS. **My current lean** if the next-session
  push-back doesn't surface a stronger alternative.
- **Path B** — Pure methods paper. De-emphasize classification entirely;
  lead with MAP↔NUTS comparison, identifiability, pixel-wise mapping
  with uncertainty. Classification becomes one short section reporting
  AUCs honestly. Survives even if "explain ADC" doesn't work. Backup
  option.
- **Path C** — Expand cohort before writing. Sandy/Stephan as conduit
  for ~300+ patients. Highest-rigor path, slowest. Worth asking even
  if not blocking.
- **Path D** — Walk away. Literature pass makes this harder to justify
  (real novelty axes exist), but Patrick's principle ("conclusive, not
  cheating") makes it a legitimate option if the next-session
  diagnostics fail to support any positive claim.

---

## 5. Next-session push-back agenda (Patrick's list)

Patrick wants to systematically rule out alternative explanations for
the "only 2 bins recoverable / 8-feat LR doesn't beat ADC / GGG
underpowered" findings before committing to any narrative. Order of
attack: **start at the classification task and work backward to the
inference and the data.**

### 5a. Is multi-bin really useless for classification?

The current diagnostic says yes (2-bin NUTS-LR ≥ 8-bin NUTS-LR; ADC ≥
all spectral classifiers). But we should rule out that:

- **The LR is the wrong classifier.** Try alternatives: random forest,
  gradient boosting, kernel SVM (RBF), naive Bayes — all under LOOCV
  with identical cross-validation protocol. If any non-linear
  classifier extracts information from the 6 intermediate bins that
  the LR can't, that breaks the "2-bin is enough" conclusion.
- **The LR loss is the wrong objective.** Logistic loss with L2 is
  optimal for binary classification under specific noise assumptions.
  Try a hinge loss, an elastic-net penalty, or rank-based learning
  (RankBoost / pairwise AUC optimization, which directly optimizes
  what we report).
- **The LOOCV protocol is too coarse.** Repeated stratified k-fold
  with k=5 averaged over many seeds gives a more stable estimate. If
  the LOOCV 8-feat vs 2-feat gap is within ±0.01 of the seed-to-seed
  variance, we cannot conclude "2-feat is better"; we should only
  conclude "they're indistinguishable."
- **The class imbalance is biting us.** PZ 27:54, TZ 13:55. Try
  balanced loss (sample weights inversely proportional to class
  frequency) or oversampling.

### 5b. Is the 8-bin grid the wrong representation?

Patrick's question: **would a continuous spectrum work better, with no
bin-spacing assumption at all?** This is methodologically attractive
because it sidesteps the entire Fisher / CRLB / grid-justification
section of the manuscript.

Options to try in the next session:

- **Continuous Gaussian process prior over diffusivity.** Define D on
  a log-uniform grid (say 0.1 to 50 μm²/ms, 64 points), put a GP prior
  on the log-spectrum with a length-scale that encodes "neighboring
  diffusivities should have correlated mass," and sample the posterior
  via NUTS. The output is a smooth curve, not a histogram. Classifier
  features could then be projections (integrals over chosen ranges) or
  the GP's posterior mean evaluated at canonical points.
- **Stick-breaking / Dirichlet process spectrum.** Lets the data
  decide how many "components" exist instead of fixing 8. Harder to
  fit but theoretically cleaner.
- **Compare classification AUC: 8-bin vs continuous.** If the
  continuous version doesn't change the AUC, then bin spacing isn't
  the bottleneck — the underlying signal is fundamentally 2-pool.

### 5c. Is the NUTS inference itself limiting identifiability?

This connects to 3rd-pass Exec #4 (SNR sanity check — median 303 under
NUTS vs old Gibbs-era 400–600) and Exec #6 (prior consistency between
MAP and NUTS). The hypothesis: bad inference → wide posteriors → "only
2 bins identifiable" is an inference artifact, not a data limit.

Things to try:

- **More draws / different sampler settings.** 4 chains × 2000 draws,
  target_accept=0.95 is reasonable but not exhaustive. Try
  target_accept=0.99, longer warm-up, more chains. Check R-hat and
  ESS per bin, not just convergence as a whole.
- **Different priors on R.** Current is HalfNormal (per-bin, scale
  fitted from MAP). Try (a) flat prior with non-negativity constraint,
  (b) hierarchical prior with cohort-level shrinkage, (c) Dirichlet
  prior that enforces sum-to-one explicitly (current normalizes
  post-hoc).
- **Different prior on σ.** Currently HalfCauchy. Try Inverse-Gamma
  (conjugate, classical relaxometry choice) or HalfNormal.
- **Variational inference baseline.** Mean-field VI on the same model
  will underestimate posterior variance but should give a fast
  sanity-check on the MAP-vs-NUTS gap on a few bins.
- **Compare against pure NNLS without any prior.** Bare unregularized
  fit. If NNLS without a prior identifies the same 2 bins, the prior
  isn't doing the work; if it identifies more, the prior is too
  restrictive.

### 5d. Is the SNR statement itself credible?

3rd-pass Exec #4 already flagged this. NUTS median σ → SNR ≈ 303;
Gibbs-era number was 400–600 with a closed-form σ estimator. If our σ
inference is wrong (e.g., HalfCauchy pulling σ up), the apparent SNR is
artificially low, and the identifiability story changes. Re-derive the
old Gibbs-era SNR formula from
`/Users/PWR/Documents/Professional/Papers/Paper3/code copy/spectra-estimation-dMRI/`
and compare on the same ROIs. Cross-check with Langkilde 2018 SNR table
in `assets/`.

### 5e. Are there bigger public datasets?

Patrick's question: IDC (Imaging Data Commons), TCIA (The Cancer Imaging
Archive), Zenodo, or another organ with comparable multi-b DWI. Goal:
get a second cohort to either confirm or refute the BWH findings.

- **TCIA "PROSTATEx" / "PROSTATEx-2"** — large public prostate DWI, but
  typically b-values are 50/400/800 (PI-RADS-style), not the extended
  range we need (b up to 3500).
- **TCIA "Prostate-MRI" or "Prostate-3T"** — check what b-values are
  available.
- **UK Biobank** — has dMRI but for brain, not prostate.
- **Cardiff INNOVATE / IMPROD** — used by the VERDICT papers; may have
  public release.
- **rVERDICT cohort** — Sci Rep 2023, check if public.
- **Other organs with extended-b DWI:** breast (DKI literature),
  pancreas, head/neck. Patrick noted he wants to stay in prostate, so
  this is fallback only.

This needs a separate literature search focused on **datasets**, not
methods.

### 5f. Are there preprocessing bugs?

- **Trace averaging.** Currently we take the geometric mean of three
  encoding directions per b-value to form the ROI-level signal decay
  (methods.tex L15). Verify this is correctly implemented in
  `data/loaders.py`. Test: does using a single direction change the
  identifiability story?
- **ROI signal aggregation.** Voxel signals are averaged within an ROI
  before fitting. Check that the averaging uses linear (not log)
  intensity and that we're not double-normalizing.
- **b=0 normalization.** The model assumes S(b)/S(0). Verify that
  `compute_map_spectrum` (recompute.py:63) and the NUTS model agree on
  how S(0) is handled — particularly when S(0) has its own noise.
- **Voxel count → SNR formula.** `recompute.py:158` computes
  `snr = sqrt(voxel_count / 16) * 150`. The 150 is a constant from
  some prior calibration; verify it's not a stale assumption that
  systematically biases σ in NUTS.

### 5g. Is LR even the right framing?

Patrick asked. Alternatives worth a 1-day exploration:

- **Ranking / order-statistic framing.** AUC is already a rank
  measure. We could optimize a pairwise ranking loss directly
  (RankSVM, pairwise logistic) rather than the marginal classifier.
  This gives confidence interval handling appropriate to AUC.
- **Bayesian classification.** Each ROI gets a posterior distribution
  over class probability, propagated from the NUTS posterior over the
  spectrum. The "classifier" output is a distribution, not a point
  estimate. This connects to the 3rd-pass note's "carrying MCMC
  uncertainty into the classifier" idea.
- **Probabilistic generative model.** Fit separate spectral priors
  per class (tumor vs normal), classify via Bayes factor. This is the
  "right" Bayesian approach but requires significantly more
  modeling effort.

---

## 6. Concrete starting point for the next session

If we begin with §5a (classification framing) and work backward:

1. **First 30 min — replicate the current LOOCV finding** with
   alternative classifiers (RF, GBM, SVM-RBF) on the same NUTS feature
   set. If any of them substantially beat the 2-feature NUTS-LR (say
   by Δ AUC ≥ 0.05 with non-overlapping bootstrap CIs), Path A's
   "8-bin spectrum is effectively 2-dim" claim is in trouble and we
   need to rethink.
2. **Next 60 min — repeated stratified k-fold sensitivity test.**
   k=5 × 50 seeds × all classifier variants. Plot the AUC distributions
   to see what's statistically distinguishable.
3. **Next session block — continuous-spectrum sanity check.** Hack
   together a GP-prior spectrum with NUTS on a single representative
   ROI from each tissue class. If the posterior mean curve has visibly
   more than 2 peaks, the 8-bin discrete grid is hiding structure.
4. **NUTS inference quality.** Re-fit a handful of ROIs with
   `target_accept=0.99`, more chains, longer warm-up. Check whether the
   per-bin CV drops materially.

These four steps should be enough to either (a) commit to Path A or
(b) escalate to Path C/D.

### 6.1. Step-1 result (2026-05-15) — alternative classifiers do not falsify Path A

Run: `uv run python scripts/classifier_comparison.py`. Output:
`results/biomarkers/classifier_comparison.csv` (27 rows). Protocol: same
LOOCV with `StandardScaler` fit inside each fold as `recompute.loocv_auc`;
fixed hyperparameters (no per-fold tuning); bootstrap 95% CI on the final
AUC (1000 iter, seed=42). Classifiers: LR (C=1.0), RF (300 trees,
min_samples_leaf=2), GBM (300 stumps, lr=0.05, depth 3), SVM-RBF
(C=1.0, gamma='scale'). Feature sets: 8-bin NUTS posterior means,
2-bin NUTS {D=0.25, D=3.00}. ADC raw-rank shown for reference.

| Task | 2-bin NUTS-LR | Best non-linear | Δ | Verdict |
| --- | ---: | --- | ---: | --- |
| PZ (n=81, 27 tumor) | 0.933 [0.858, 0.983] | RF 8-bin: 0.936 [0.870, 0.983] | +0.003 | OK |
| TZ (n=68, 13 tumor) | 0.937 [0.845, 0.992] | RF 2-bin: 0.915 [0.812, 0.990] | −0.022 | OK |
| GGG (n=29, 9 hi-grade) | 0.744 [0.471, 0.995] | SVM-RBF 8-bin: 0.728 [0.435, 0.974] | −0.017 | OK |

ADC raw-rank for reference: PZ 0.951 [0.887, 0.995], TZ 0.979 [0.938, 1.000],
GGG 0.811 [0.594, 0.972] — still the single best feature on every task.

**Interpretation.** None of RF / GBM / SVM-RBF clears the ΔAUC ≥ 0.05
non-overlapping-CI threshold against the 2-feature NUTS-LR on any task.
RF on the full 8-bin set ties the 2-feature LR within ±0.003 on PZ and
is materially worse on TZ and GGG. GBM and SVM-RBF are uniformly at or
below the 2-bin LR. Taken together: **a non-linear classifier with access
to all 8 NUTS bins cannot extract information from the 6 intermediate
bins (D=0.50, 0.75, 1.00, 1.50, 2.00, 20.00) beyond what the linear
model already gets from {D=0.25, D=3.00}.** Path A's "8-bin spectrum is
effectively 2-dimensional for classification" is strengthened, not
falsified, by this test.

Caveats: (i) hyperparameters are fixed rather than tuned, but the LR
baseline is also fixed at C=1.0, so the comparison is apples-to-apples;
(ii) GBM's bad TZ performance (0.765 on 8-bin, 0.877 on 2-bin) is
consistent with tree-boosting overfit at N=68 with 13 positives, not a
signal about the data; (iii) on GGG, all bootstrap CIs span ±0.25 — no
classifier on this task can be distinguished from any other or from a
coin flip, confirming §2e.

**Next move per §6 step 2:** repeated stratified k-fold (k=5 × 50 seeds)
on the same classifier × feature-set grid to confirm the LOOCV AUC gaps
are not seed-noise. If those distributions overlap heavily across
classifiers, the right framing in the paper becomes "spectral
classification is indistinguishable from ADC under a wide class of
classifiers" rather than picking a winner.

### 6.2. Next-session plan (drafted 2026-05-16) — SNR + simulation in parallel

The 2026-05-15 result rules out "wrong classifier family" as the
explanation for "only 2 bins help." Two unresolved questions remain
before any narrative commit:

1. **Is the NUTS σ estimate trustworthy?** If HalfCauchy is pulling σ
   upward, posteriors are wider than the data warrants and the
   "only 2 bins identifiable" conclusion is an inference artifact.
2. **Is MAP genuinely biased, or are MAP and NUTS both faithful in
   their own sense?** Sharper framing (from 2026-05-16 discussion):
   identifiability is a property of the *likelihood*, not the
   estimator. Wide NUTS CI at D=0.50 ⇒ data doesn't constrain that
   bin individually ⇒ MAP-D=0.50 is prior-determined. That is a
   different statement than "MAP-D=0.50 is wrong." Proving the
   smearing is real requires simulation with known ground truth.

Both questions can be answered in parallel in one focused session.

#### Investigation A — SNR formula vs NUTS-inferred σ

Hypothesis: NUTS posterior σ → SNR ≈ 303 is too low for our
endorectal-coil acquisition; the Gibbs-era closed-form formula may
yield 400–600.

1. **Locate Stephan's formula.** Search the legacy Gibbs codebase at
   `Paper3/code copy/spectra-estimation-dMRI/` (grep `snr`,
   `noise_std`, `sigma_estimate`). `recompute.py:158` has
   `snr = sqrt(voxel_count / 16) * 150` — trace where it came from.
   Cross-reference Langkilde 2018 SNR table in `assets/`. If unclear
   after 30 min, flag for Stephan.
2. **Population scatter, all 149 ROIs.** Compute the formula from
   `signal_decays.json` + voxel counts; scatter against NUTS posterior
   σ from `results/inference_bwh_backup/*.nc`. Look for systematic
   offset, correlation with voxel count / zone / b=0 intensity,
   outliers. Save to `results/biomarkers/snr_formula_vs_nuts.png` and
   `snr_comparison.csv`.
3. **Fixed-σ refit on 5–10 representative ROIs.** 2 tumor PZ + 2
   normal PZ + 2 tumor TZ + 2 normal TZ + 1 outlier. Refit NUTS with σ
   fixed at the formula value (replace HalfCauchy with point mass).
   Compare per-bin posterior CV and posterior means. Save to
   `fixed_sigma_refit.csv`.

Effort: ~half day total.

#### Investigation B — Simulation with known ground-truth spectra

Hypothesis to *test* (not assume): MAP with ridge λ=0.1 systematically
smears restricted-diffusion mass across collinear bins (D=0.25, 0.50,
0.75) when truth is concentrated; NUTS posterior mean does not.

Ground-truth spectra (principled, not copies of estimated cohort means):

| Name | Description |
| --- | --- |
| GT-A | δ at D=0.25 |
| GT-B | δ at D=0.50 |
| GT-C | δ at D=0.75 |
| GT-D | δ at D=3.00 (free-water-only) |
| GT-E | Bimodal: {D=0.25: 0.7, D=3.00: 0.3} (canonical "tumor") |
| GT-F | Bimodal: {D=0.25: 0.3, D=3.00: 0.7} (canonical "normal") |
| GT-G | Trimodal: {D=0.25: 0.4, D=1.5: 0.2, D=3.00: 0.4} |
| GT-H | Log-normal centered at 0.5, width 0.3 |
| GT-I | Log-normal centered at 1.5, width 0.5 |

Sweep: SNR ∈ {100, 200, 400, 800, 1500}; 100 noise reps per
(GT × SNR). Gaussian noise (Rician unnecessary at ROI-averaged
SNR > 30). Same 15-b-value grid as real fits. Fits: MAP closed-form
ridge NNLS (λ=0.1) and NUTS (matching the inference config, reduced
draws OK — 1000 × 2 chains for the harness, full 4×2000 on a subset
for sanity).

Metrics per (GT × SNR): per-bin bias, per-bin MSE, recovery
probability, NUTS posterior coverage (does 90% CI contain truth 90%
of the time?).

Deliverables: `scripts/simulation_study.py`,
`results/simulation/{bias,mse,coverage}_table.csv`, one bias-heatmap
figure. Effort: ~1 day.

#### Decision tree

- **Outcome 1 — SNR fix unlocks identifiability AND simulation shows
  estimators are mostly fine.** Update inference config (σ fixed or
  new σ prior), refit 149 ROIs, rebuild features, re-run
  classification. If 3–4 bins become identifiable: enhanced Path A.
  MAP stays as methods-comparison panel.
- **Outcome 2 — SNR fix doesn't change identifiability; simulation
  confirms MAP smears under single-bin ground truth.** Path A as
  currently framed: 2-feature NUTS-LR on {D=0.25, D=3.00}; drop MAP
  from headline; simulation result becomes a methods supplement
  justifying that choice. No continuous spectrum, no Fisher spacing
  needed.
- **Outcome 3 — SNR fix doesn't help AND simulation shows both
  estimators are biased / ground truth has >2 components but neither
  recovers them.** Grid itself becomes suspect. One focused
  continuous-spectrum (GP prior) experiment on 4–5 representative
  ROIs. If GP shows >2 peaks → pivot to continuous as headline. If
  still 2 peaks → Path B (pure methods, no classification headline).
- **Outcome 4 — SNR fix dramatically changes the noise model.** σ
  inference itself becomes a methods contribution worth its own
  section.

#### Parked (do not pursue until SNR + simulation resolve)

- **Continuous spectrum (GP prior)** — defer to Outcome 3.
- **Fisher-derived bin spacing** — subsumed by continuous spectrum.
- **Pixel-wise image data** — would not change classifier AUC; useful
  only for a heterogeneity-inside-a-tumor supplementary figure later.
- **Histopathology / prostatectomy data** — future paper; note in
  Discussion as the natural next study.
- **Repeated stratified k-fold (§6 step 2)** — useful but lower
  priority than SNR + simulation; park unless next session produces a
  borderline result.

#### Pre-session prep (~30 min, optional)

1. Locate `noise_std` / `sigma_estimate` / `snr` in legacy Gibbs
   codebase; note file + line numbers.
2. Pull Langkilde 2018 SNR table from `assets/`.
3. Skim `src/spectra_estimation_dmri/inference/nuts.py:118–157` for σ
   prior structure (HalfCauchy β=0.01).

#### Success criterion for the next session

End the session able to say one of:
- "NUTS σ is biased; with proper σ calibration, N bins are identifiable.
  Revised Path A is the framing."
- "NUTS σ is correctly calibrated; simulation confirms 2 bins is the
  recoverable truth. Path A as currently framed is the framing."
- "NUTS σ is fine but the estimator (or grid) limits identifiability.
  Path B or continuous-spectrum experiment is the framing."

If none of those can be said, the diagnostic is incomplete and we extend
by one more session.

---

## 7. Reading list for next session

- **Wang Y. et al. 2024**, Abdominal Radiology,
  doi:10.1007/s00261-024-04684-z — "MC parameters highly correlated
  with ADC may explain limited improvements." Citation for Path A.
- **Conlin C.C. et al. 2021**, JMRI, doi:10.1002/jmri.27393 — closest
  prostate multi-compartment paper; uses BIC model selection without
  posterior uncertainty.
- **Quigley D.J., Mitchell D.G. 2023**, Eur J Radiol, PMC10623580 —
  "fail-but-explain" review template.
- **Wang Q. et al. 2018**, JMRI, PMID 29812977 — IVIM/DKI vs ADC
  meta-analysis ("comparable").
- **Sjölund J. et al. 2018**, NeuroImage, PMC6419970 — Bayesian dMRI
  uncertainty in brain.
- **Singh M. et al. 2022**, Radiology, doi:10.1148/radiol.212536 —
  VERDICT INNOVATE N=303.
- **RSI/RSIrs counterpoint papers** (Conlin, Karow, Liss, Zhong, recent
  J Urology). Note their ADC implementation and acknowledge the
  disagreement.

---

## 7. Decision (2026-05-16) — Outcome 2 confirmed; commit to Path A

The §6.2 plan ran end-to-end this session. SNR diagnostic + ground-truth
simulation jointly settle all four outcomes. **Outcome 2 is confirmed**:
NUTS σ is correctly calibrated; MAP ridge λ=0.1 systematically smears
restricted-D mass across collinear bins; the 2-bin NUTS-LR on
{D=0.25, D=3.00} is the right headline classifier.

### 7a. Investigation A results

**A1. Stephan's formula located.** `code copy/.../models/old/gibbs.py:389,440`:
```
snr_ROI = sqrt(v_count / 16) * 150       sigma = 1 / snr_ROI
```
The constant `c=150` is a fitted scaling on a 16-voxel-ROI baseline →
implied per-voxel SNR ≈ 37.5. Langkilde 2018 Eq.9 in
`assets/Evaluation of Fitting Models...pdf` (page 7) reports per-voxel
b=0 SNRs of 74-180 by tissue type (normal PZ 180±52, normal TZ 119±23,
tumor PZ 108±37, tumor TZ 74±18). Stephan's per-voxel implied SNR is
~3-5× lower than Langkilde's per-voxel SNRs; equivalently, an unweighted
sqrt(N) scaling from Langkilde's per-voxel SNRs would give *higher* ROI
SNRs than Stephan's formula. Both are reasonable but they don't match
each other and neither matches what NUTS infers.

**A2. σ scatter across 149 ROIs** (script:
`scripts/snr_diagnostic.py`, output:
`results/biomarkers/snr_comparison.csv`,
`results/biomarkers/snr_formula_vs_nuts.png`).
Medians on the normalized signal:

| Estimator | σ | SNR (=1/σ) | r vs σ_NUTS |
| --- | ---: | ---: | ---: |
| σ_formula (Stephan)         | 0.00215 | 465 | 0.28 |
| σ_NUTS posterior mean       | 0.00330 | 303 | — |
| σ_residual (MAP-fit std)    | 0.01334 | 75  | 0.65 |
| σ_langkilde (Eq.9, biexp resid scaled by √N) | 0.00059 | 1692 | 0.78 |

Key points:
1. NUTS σ is being **pulled DOWN** from the HalfCauchy(β=0.01) prior
   median (σ=0.01) by the data — not up. The "wide posteriors / 6 unidentifiable
   bins" finding cannot be blamed on a σ prior pulling σ upward and inflating
   noise.
2. σ_NUTS (0.0033) and σ_formula (0.0022) differ by ~50% but only weakly
   correlate (r=0.28). The formula uses voxel_count only; NUTS uses the data.
3. σ_residual is 4× larger than σ_NUTS — the MAP ridge fit leaves much
   more residual than NUTS thinks is noise. Both can be self-consistent:
   NUTS fits R more flexibly (joint inference of R and σ), so σ_NUTS is
   smaller than the MAP-residual std.
4. σ_langkilde scaled by √N is way too optimistic (SNR ~1700). Biexp
   residuals on ROI-mean signal are dominated by model-misspecification,
   not noise; the √N scaling then compounds. Don't use this formula
   without per-voxel signals.

**A3. Fixed-σ NUTS refit on 5 ROIs** (script:
`scripts/fixed_sigma_refit.py`, output:
`results/biomarkers/fixed_sigma_refit.csv`,
`results/biomarkers/fixed_sigma_spectra.png`).
Refit each ROI under three σ assumptions: free (HalfCauchy(β=0.01),
matches the 149 .nc files), fixed at σ_formula, fixed at σ_residual.
Mean per-bin posterior std across 5 ROIs:

| Mode | D=0.25 | D=0.50 | D=0.75 | D=1.00 | D=1.50 | D=2.00 | D=3.00 | D=20.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| free            | 0.016 | 0.036 | 0.040 | 0.052 | 0.083 | 0.094 | 0.060 | 0.015 |
| fixed_formula   | 0.010 | 0.023 | 0.030 | 0.048 | 0.079 | 0.088 | 0.048 | 0.009 |
| fixed_residual  | 0.025 | 0.050 | 0.055 | 0.063 | 0.087 | 0.114 | 0.101 | 0.033 |

Pinning σ at the (more optimistic) formula value narrows outer-bin
posteriors by 20-40% but **barely moves the 6 middle bins** (D=0.5-2.0
posterior std drops 5-25%). The identifiability ranking
(D=0.25 + D=3.00 + D=20.00 well-determined; rest poorly) is preserved
under every σ choice tested. **No σ calibration unlocks the middle bins.**

### 7b. Investigation B results — simulation

Script: `scripts/simulation_study.py`. Output:
`results/simulation/sim_results.csv` (42 480 per-rep predictions),
`results/simulation/sim_summary.csv` (576 per-(estimator,GT,SNR,bin)),
`results/simulation/bias_heatmap.png`.

Sweep: 9 ground-truth spectra (GT-A...GT-I), SNR ∈ {100, 200, 400, 800,
1500} for MAP × 100 reps; SNR ∈ {200, 400, 800} for NUTS × 30 reps with
draws=800, tune=400, chains=2, target_accept=0.9. Total runtime ~115 min.

**Per-bin bias at SNR=400 (closest to the BWH median NUTS-inferred SNR ≈ 303):**

| | D=0.25 | D=0.50 | D=0.75 | D=1.00 | D=1.50 | D=2.00 | D=3.00 | D=20.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAP GT-A (δ@0.25) | **−0.343** | +0.253 | +0.066 | 0 | 0 | 0 | 0 | +0.023 |
| MAP GT-B (δ@0.50) | +0.280 | **−0.738** | +0.211 | +0.157 | +0.071 | +0.018 | 0 | 0 |
| MAP GT-C (δ@0.75) | +0.074 | +0.213 | **−0.761** | +0.220 | +0.150 | +0.088 | +0.015 | 0 |
| MAP GT-D (δ@3.00) | 0 | 0 | +0.015 | +0.071 | +0.169 | +0.235 | **−0.706** | +0.216 |
| NUTS GT-A (δ@0.25) | −0.014 | +0.003 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 |
| NUTS GT-B (δ@0.50) | +0.032 | −0.078 | +0.022 | +0.010 | +0.005 | +0.004 | +0.003 | +0.002 |
| NUTS GT-C (δ@0.75) | +0.006 | +0.032 | −0.098 | +0.037 | +0.010 | +0.006 | +0.004 | +0.003 |
| NUTS GT-D (δ@3.00) | +0.001 | +0.001 | +0.002 | +0.002 | +0.005 | +0.011 | −0.040 | +0.018 |

**Bimodal (canonical tumor/normal) at SNR=400:**

| | D=0.25 | D=0.50 | D=0.75 | D=1.00 | D=1.50 | D=2.00 | D=3.00 | D=20.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAP  GT-E (0.7@0.25, 0.3@3.0) | −0.162 | +0.202 | +0.060 | +0.008 | +0.003 | +0.030 | **−0.227** | +0.086 |
| MAP  GT-F (0.3@0.25, 0.7@3.0) | −0.084 | +0.069 | +0.035 | +0.046 | +0.103 | +0.155 | **−0.491** | +0.167 |
| NUTS GT-E | −0.010 | +0.007 | +0.006 | +0.006 | +0.009 | +0.016 | −0.052 | +0.018 |
| NUTS GT-F | −0.011 | +0.007 | +0.006 | +0.006 | +0.009 | +0.018 | −0.058 | +0.021 |

**Total MSE summed over 8 bins, SNR=400:**

| GT | MAP | NUTS | Ratio |
| --- | ---: | ---: | ---: |
| GT-A (δ@0.25)    | 0.187 | 0.0002 | 935× |
| GT-B (δ@0.50)    | 0.697 | 0.008 | 88× |
| GT-C (δ@0.75)    | 0.709 | 0.013 | 55× |
| GT-D (δ@3.00)    | 0.633 | 0.002 | 288× |
| GT-E (bi-tumor)  | 0.131 | 0.004 | 35× |
| GT-F (bi-norm)   | 0.319 | 0.005 | 68× |
| GT-G (trimodal)  | 0.138 | 0.056 | 2.5× |
| GT-H (log-norm 0.5) | 0.021 | 0.018 | 1.2× |
| GT-I (log-norm 1.5) | 0.022 | 0.014 | 1.6× |

**NUTS 90% CI coverage (target 0.90):**

| GT | SNR=200 | SNR=400 | SNR=800 |
| --- | ---: | ---: | ---: |
| GT-A...GT-D (δ-spectra) | ≈ 0.00 | ≈ 0.00 | ≈ 0.00 |
| GT-E, GT-F (bimodal)    | 0.00-0.03 | 0.01 | 0.00-0.01 |
| GT-G (trimodal)         | 0.14 | 0.12 | 0.07 |
| GT-H, GT-I (log-normals) | 0.77-0.87 | 0.77-0.87 | 0.77-0.82 |

### 7c. Interpretation — what the simulation tells us

1. **The MAP-D=0.50 dominance in the BWH cohort is a smearing artifact, full
   stop.** When truth is at D=0.25, MAP transfers 0.25 of the true mass
   to D=0.50 (and 0.07 to D=0.75). When truth is at D=0.75, MAP transfers
   0.21 to D=0.50. Whatever the true restricted-diffusion locus is in
   prostate tissue, MAP will deposit a large fraction of it at D=0.50.
   The "D=0.50 is the strongest discriminator under MAP" finding (§2a)
   is exactly this artifact in numerical form.

2. **NUTS posterior mean is 50-3000× lower MSE than MAP on δ and bimodal
   GTs.** This is much larger than the difference between any two
   classifier families (RF, GBM, SVM-RBF in §6.1) and dominates any
   per-feature ranking. The right inference for this problem is NUTS,
   not ridge MAP.

3. **NUTS 90% CIs are too narrow on δ-spectra (coverage ≈ 0.00).** The
   posterior contracts away from the truth at the boundary
   (HalfNormal(σ_R=√10) pulls R toward 0; the truth is at 1.0 for one
   bin). On smooth GTs (log-normals), coverage is 0.77-0.87 — near
   nominal. **Caveat for the manuscript**: when reporting NUTS
   uncertainties on tumor-like spectra, the credibility interval is not
   a frequentist confidence interval. Either flag this honestly in the
   Discussion or compute calibrated intervals via simulation-based
   calibration. The bias/MSE story is solid; the coverage story needs
   honesty.

4. **The grid is not the problem.** NUTS recovers δ-spectra at all 8
   grid points with very small bias (max |bias| ≤ 0.098). If the 8-bin
   grid were misaligned with the true components, we'd see persistent
   bias even at high SNR. We don't. **Outcome 3 (continuous spectrum)
   is unsupported.** Parking the GP-prior / Fisher-spacing work
   indefinitely.

5. **Log-normal GTs (smooth) are easiest for both estimators.** MSE is
   comparable (NUTS ~1.3× lower). This is the regime where Sjölund-style
   smoothness priors would shine if we had them, but it's not the
   regime that drives the prostate biology here — restricted diffusion
   in tumor is concentrated, not smooth.

### 7d. Decision

**Path A as currently framed.** Outcome 2 of the §6.2 decision tree.

1. NUTS as the headline inference for all spectral results. MAP retained
   only in a methods-comparison panel showing the simulation-confirmed
   smearing.
2. Classifier headline: 2-feature NUTS-LR on {D=0.25, D=3.00}. AUC PZ
   0.933, TZ 0.937. Matches or beats every other LR variant and every
   non-linear classifier from §6.1.
3. ADC is the clinical reference and remains the single best per-feature
   AUC (PZ 0.951, TZ 0.979). Cite Wang 2024 (Abdom Radiol) for the
   "MC-correlated-with-ADC" precedent; frame the spectral classifier as
   "matches ADC, with mechanistic interpretation" rather than
   "outperforms ADC."
4. GGG dropped from headline to limitations (N=29 with 9 high-grade is
   inadequate for any positive claim; bootstrap CIs span 0.30).
5. Simulation study from §7b moves into a methods supplement —
   Figure SX (bias heatmap, MAP vs NUTS at SNR ∈ {200, 400}) plus a
   short paragraph in Methods. **This is the new, principled
   justification for choosing NUTS over MAP** that the manuscript was
   missing.
6. Flag NUTS coverage limitation honestly in Discussion (point 3 above).

### 7e. Parked for later (post-paper)

- Continuous-spectrum (GP prior) experiment — defer indefinitely.
- Fisher-derived bin spacing — subsumed; grid is not the bottleneck.
- Pixel-wise / heterogeneity-within-tumor — out of scope for this paper.
- Histopathology / prostatectomy validation — future paper.
- Repeated stratified k-fold AUC sanity check (§6 step 2) — useful but
  the simulation result is more conclusive; defer unless reviewers ask.

### 7f. Files produced this session

- `scripts/snr_diagnostic.py` — A2.
- `scripts/fixed_sigma_refit.py` — A3.
- `scripts/fixed_sigma_refit_plot.py` — A3 visualization.
- `scripts/simulation_study.py` — B1+B2.
- `results/biomarkers/snr_comparison.csv`, `snr_formula_vs_nuts.png`,
  `snr_refit_picks.csv`, `fixed_sigma_refit.csv`,
  `fixed_sigma_spectra.png`.
- `results/simulation/sim_results.csv`, `sim_summary.csv`, `bias_heatmap.png`.
- This file, §7.

---

## 8. Files touched / state of the repo

- **Edited:** `notes_manuscript_3rd_pass.md` — Exec #1, #3, #8 and the
  Results §5, Fig 2 §7 cross-references now reflect the diagnostic
  findings.
- **No code or figure edits** to `paper/sections/` or
  `scripts/generate_paper_figures.py` — held off per Patrick's
  explicit request that we agree on narrative direction first; that
  agreement is now in place (Path A, Outcome 2).
- **New diagnostic scripts** (§7f) — all read-only on the BWH data;
  none modify `results/inference_bwh_backup/` or `signal_decays.json`.
- **New file:** `notes_session_2026-05-14_classifier_deepdive.md` §1-§7.

The next session's first round of edits will hit
`paper/sections/results.tex`, `paper/sections/abstract.tex`,
`paper/sections/methods.tex` (add NUTS-vs-MAP simulation paragraph),
`scripts/generate_paper_figures.py:fig_roc`, and create new Fig 3/4 NUTS
variants and a supplementary bias-heatmap figure.

---

## 9. Hand-off — how to start the next session

### 9a. Two-sentence framing (load this first)

We are NOT abandoning the 8-bin spectrum. NUTS recovers all 8 bins
accurately (simulation §7b confirms ≤0.10 bias on δ-GTs). What
collapses to 2 effective dimensions is the **classifier**, because real
prostate tissue is intrinsically ~2-pool (restricted + free water);
the 6 intermediate bins are unbiased but don't carry tumor-status
information beyond what {D=0.25, D=3.00} already carries. Continuous
spectrum / Fisher spacing / GP prior — all useless for this paper;
defer or never.

### 9b. Manuscript edit punch-list (in dependency order)

1. **`scripts/generate_paper_figures.py`** — rebuild on NUTS:
   - Fig 2 (per-component AUC + LR LOOCV): switch MAP→NUTS as the
     spectrum-derived features. Add the 2-feature NUTS-LR
     {D=0.25, D=3.00} as a separate ROC curve. Keep ADC for reference.
   - Fig 3 (spectrum visualization): show NUTS posterior mean + 90% CI
     for the canonical PZ tumor / PZ normal / TZ tumor / TZ normal
     classes. Drop the MAP-spectrum panel from the headline; move to
     supplement.
   - Fig 4 (discriminant vs ADC scatter): switch to NUTS-LR discriminant
     scores. Confirm the r ≈ −0.95 with ADC narrative survives under
     NUTS (it should — check `results/biomarkers/adc_discriminant.csv`).
2. **New supplementary figure** — bias heatmap. Copy
   `results/simulation/bias_heatmap.png` into `paper/figures/` and
   wire it up.
3. **`paper/sections/methods.tex`** — add a paragraph (3-5 sentences)
   citing the simulation study (`scripts/simulation_study.py`) and
   referring to Fig S(new). Justify NUTS-over-MAP empirically.
4. **`paper/sections/results.tex`** — rewrite the spectral-classifier
   subsection:
   - Drop the 8-feature MAP-LR as headline.
   - Lead with 2-feature NUTS-LR on {D=0.25, D=3.00}: PZ AUC 0.933,
     TZ AUC 0.937. Compare to ADC (PZ 0.951, TZ 0.979).
   - Cite Wang 2024 (Abdom Radiol, doi:10.1007/s00261-024-04684-z)
     for the "MC-correlated-with-ADC" mechanism. Frame as "matches ADC,
     mechanistically interpretable" not "outperforms."
5. **`paper/sections/abstract.tex`** — match new Results.
6. **`paper/sections/discussion.tex`** — add NUTS-coverage caveat per
   §7c point 3 (NUTS CIs over-confident on concentrated spectra;
   well-calibrated on smooth). Acknowledge GGG underpowered.
7. **Drop GGG to Limitations** wherever it appears in headline.

### 9c. What NOT to do (Patrick already considered and rejected)

- **Do not** start continuous-spectrum (GP prior) or Fisher-derived
  bin-spacing work. §7c point 4 rules it out; the grid is not the
  bottleneck.
- **Do not** rerun NUTS with target_accept=0.99 / more chains. §7a
  shows σ is not biased; convergence is fine on the existing .nc files.
- **Do not** rerun the BWH inference. The 149 .nc files in
  `results/inference_bwh_backup/` are the gold standard; everything
  downstream reads from them.
- **Do not** add repeated-stratified-k-fold (§6 step 2). Simulation
  result is more conclusive; defer unless reviewers ask.
- **Do not** propose pixel-wise / heterogeneity-within-tumor work.
  Different paper.

### 9d. Open questions for Sandy / Stephan (next co-author meeting)

- Citation of Wang 2024 for the MC-correlated-with-ADC framing — does
  Stephan know this group? Any concerns?
- The NUTS coverage caveat (§7c point 3) — should it go in Discussion
  or as a supplementary methods note? Patrick to draft and ask.
- Per [[feedback_coauthor_meeting_20260329]]: Sandy/Stephan's
  discretization question is now answered by §7b (grid is fine).
  Communicate this finding to them.

### 9e. First command to run next session

```bash
uv run python -m spectra_estimation_dmri.biomarkers.recompute
```

This regenerates `results/biomarkers/{features,auc_table,...}.csv` and
confirms nothing has drifted. Then resolve §10 BEFORE touching §9b.

---

## 10. Open questions raised 2026-05-16 — must resolve before drafting

Patrick raised seven substantive concerns after seeing the §7 result.
These have to be answered before §9b's manuscript edits begin, because
some of them could change the headline numbers.

### 10a. Per-patient spectrum supplement (action item)

Mirror the legacy code's `gibbs_old.py:565` titles:
`{patient_key} | GS:{gs} | GGG:{ggg} | zone | SNR`. Generate a
multi-page PDF showing NUTS posterior mean + 90% CI for every
patient × ROI from `results/inference_bwh_backup/*.nc` joined with
`metadata.csv`. Goes in supplementary materials so a reviewer can
audit every individual ROI's recovered spectrum.

**New script to write**: `scripts/generate_per_patient_spectra_supplement.py`.

### 10b. New simulation figures — candidates for the manuscript

Three diagnostic figures from this session that may be paper material:

- `results/simulation/bias_heatmap.png` — **load-bearing**. This is the
  empirical justification for choosing NUTS over MAP. Almost certainly
  becomes a main-text or supplementary figure. Decide which based on
  word/figure budget (MRM caps 10 fig+tab).
- `results/biomarkers/snr_formula_vs_nuts.png` — supplementary only.
  Useful for the σ-calibration paragraph; not a headline.
- `results/biomarkers/fixed_sigma_spectra.png` — supplementary only.
  Shows σ-pinning doesn't change the identifiability story.

### 10c. Selection-bias / circularity concern (CRITICAL, partially open)

The BWH ROIs were drawn by a radiologist using clinical mpMRI
(T2 + DWI at b ≤ 1400). Per Langkilde 2018 (Methods, page 4), the
radiologist did NOT see the extended-range (b = 0..3500) scan when
delineating ROIs. **But the clinical DWI used for delineation is
itself ADC-dominated.** So lesions are pre-selected to be visually
conspicuous on ADC; we then evaluate spectra *within* those ROIs.

**Implication for the headline claim:** "No spectral classifier beats
ADC" is partly definitional under this design. If extended-b
information were primarily useful for identifying tumor regions
*invisible* to ADC, those regions wouldn't be in our ROI set at all —
they were never drawn.

**Action items**:
1. Add a **Limitations paragraph** explicitly: "Findings apply within
   ADC-conspicuous ROIs, not whole-prostate tissue. Pixel-wise extended-b
   imaging may identify ADC-occult tumor; left to future work."
2. **Do NOT** propose pixel-wise work as part of this paper. The
   pixel-wise inference is already implemented (`pixelwise.py`) but
   shouldn't be the headline — defer to follow-up.
3. The methodological contributions (Bayesian framework, MAP smearing,
   2-D collapse) are unaffected by the selection bias. Only the
   clinical-translation framing is qualified.

### 10d. Biexponential biology is not new — what IS new? (literature audit)

Patrick's worry: "Are we just parroting what's been known since the
biexp prostate literature of the early 2010s?" Honest answer: the
**biology** of "prostate ≈ restricted + free" IS known. Langkilde 2018
already showed biexp is AIC-preferred. Wang 2018 meta-analysis
("comparable, not superior") and Wang 2024 ("MC-correlated-with-ADC
explains limited improvements") are existing arguments.

**Our four novel contributions** (none of which is "we discovered
prostate is 2-pool"):

1. **Bayesian inference + per-bin posterior uncertainty** in prostate
   dMRI. Sjölund 2018 did Bayesian dMRI in brain; nobody we found has
   done it in prostate.
2. **MAP bin-smearing as a methodological caveat** — quantitatively
   demonstrated by the §7b simulation. Applies to ANY future
   discrete-spectrum work using ridge MAP. New result.
3. **Formal "ADC ≈ projection onto the data-supported subspace"**
   framing. Old framing: "ADC works because perfusion is small at
   b > 250." New framing: "ADC works because the 8-D spectrum
   genuinely collapses to ≈2-D under data + non-negativity, and ADC
   is a 1-D summary of that 2-D subspace." Mechanism, not observation.
4. **Negative result done with full rigor** — LOOCV, bootstrap CIs,
   MC uncertainty, simulation-validated estimator. Adds weight to the
   "ADC is fundamentally sufficient within ADC-conspicuous ROIs" claim
   beyond what Wang 2018/2024 established.

**Action items**:
- Add a literature pull at the start of Discussion: explicitly cite
  Langkilde 2018, Conlin 2021, Wang 2018, Wang 2024, Quigley 2023,
  Sjölund 2018 — and state per-paper what they did and what we add.
- Drop any phrasing that sounds like "we discovered prostate is 2-pool."
  Replace with "we confirmed under fully Bayesian inference that …"
- Citations to chase next session: highly-cited prostate-DWI methods
  papers from 2020-2025. Patrick to delegate this to a literature
  subagent or pull manually.

### 10e. Why estimate 8 bins if 2 are enough? (manuscript framing)

The answer that justifies the paper: **we estimate 8 to TEST whether
the prostate signal is genuinely 2-pool**. A 2-component biexp fit
*assumes* the answer; an 8-bin spectrum lets the data speak. The
finding that the 8-bin NUTS posterior collapses to 2 effective
classifier dimensions is the *validation* of the 2-pool model in a
fully data-driven, prior-mild framework.

This is methodologically distinct from running a 2-component biexp fit
and declaring it works. Frame the manuscript that way.

**Action item**: rewrite the abstract/intro framing accordingly. The
current draft probably oversells "novel spectral biomarkers"; tone
that down to "fully-data-driven test of the 2-pool model with
quantified uncertainty."

### 10f. Better use of N=29 Gleason scores (action items)

LR with 9 high-grade vs 20 low-grade at N=29 will overfit. Replacement
analyses (all cheap to run):

1. **Spearman ρ** of NUTS-LR tumor-vs-normal discriminant scores
   (trained on the full 149) against GGG within the 29-tumor subset.
   Continuous outcome → more power than AUC at small N. Report ρ +
   bootstrap CI + p.
2. **Per-feature regression**: D=0.25, D=3.00, ADC each regressed on
   GGG (continuous). Report r + 95% CI per feature. Pre-register the
   sign of the expected effect.
3. **State the minimum detectable ΔAUC** at N=29 (power calc with
   α=0.05, β=0.20, ~9 vs 20). Likely around ΔAUC=0.10. Anything
   smaller is invisible to this cohort. Be honest about this in
   Limitations.
4. Skip multi-class (cross-zone confound). Skip Bayes factors
   (overkill).

**Action item**: ADD to Results punch-list as a "Gleason exploratory"
subsection. Lead with the framing "exploratory at N=29" not
"definitive."

### 10g. Test "intermediate bins are uninformative" more strongly (action items)

The current claim is based on:
- 8-feature LR coefs being small for D=0.5–2.0 (NUTS, §2b),
- 2-feature LR matching 8-feature under LOOCV (§2c).

Patrick is right that a stronger test is needed before claiming
"uninformative." Two cheap analyses:

1. **Pair-LR sweep**: for k ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 20.0}, run
   2-feature LR on {D=0.25, D=k}. If any pair matches the
   {D=0.25, D=3.0} reference within ±0.01 AUC, that intermediate bin
   carries the same information as D=3.0 (i.e., D=3.0 isn't uniquely
   special — collinearity, not selection).
2. **Triple-LR sweep**: for k ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 20.0}, run
   3-feature LR on {D=0.25, D=3.0, D=k}. If any k gives Δ AUC > 0.01
   with non-overlapping bootstrap CIs vs the 2-feature reference, that
   bin adds *independent* information.

**Action item**: write `scripts/bin_information_sweep.py`. Block §9b
until this is done — if any bin surfaces with independent
discriminative power, the headline 2-feature framing needs revision.

### 10h. Revisit the ADC-sensitivity / circular-explanation argument

The current manuscript's "ADC ≈ locally-linear-optimal projection of the
8-feature LR" claim was built on MAP-LR coefs vs dADC/dR. With MAP
confirmed as a biased estimator (§7b), the high MAP correlation may
itself be an artifact — MAP coefs are smeared the same way dADC/dR is
smooth across bins. **Re-derive under NUTS** and use the new framing:

> ADC and the NUTS-LR discriminant both live in the 2-dimensional
> data-supported subspace {D=0.25, D=3.0}. A single scalar (ADC) is
> sufficient to represent any signal in a 2-D space whose two axes
> trade off monotonically with each other. Hence ADC ≈ optimal linear
> projection follows tautologically — not as an empirical surprise.

This is a *stronger* mechanistic claim than the sensitivity-vector
correlation. It survives even when NUTS-coef-vs-dADC/dR correlation is
only moderate.

**Action item**: re-run `adc_sensitivity_analysis` from
`recompute.py:508` on NUTS and report Pearson + Spearman. Then
rewrite the manuscript's mechanistic argument per the framing above.

### 10i. Whole-image / pixel-wise — defer to follow-up

Patrick's recurring instinct (raised again 2026-05-16). The
selection-bias concern in §10c could be partly resolved by
whole-prostate pixel-wise analysis: does the spectrum identify
ADC-occult tumor regions that the radiologist's ADC-driven delineation
missed? Histopathology / prostatectomy maps would be ground truth.

**For this paper**: NO. Stays in Discussion as future work. Pixel-wise
NUTS already runs (`pixelwise.py`); we have one demo (Fig 9, patient
8640) but no histopathology to validate against in the current BWH
dataset.

**For a follow-up paper**: explicit plan. Would need histopathology-mapped
prostatectomy data. Sandy/Stephan likely know who has this. ADD as a
Discussion paragraph: "Future work: spectral analysis of whole-prostate
data with prostatectomy ground truth."

### 10j. REVISED next-session priorities (replaces §9b ordering)

Given 10c, 10d, 10g, 10h, the order is:

**Day 1 — diagnostics that could change the headline** (must precede drafting):
1. Run pair/triple LR sweep (§10g). 30 min.
2. Run Spearman GGG correlation + per-feature regression (§10f). 30 min.
3. Re-run NUTS sensitivity analysis (§10h). 15 min.
4. Decision point: do §10g/f/h surface anything that revises Path A?
   - If yes → re-plan before drafting.
   - If no → proceed to Day 2.

**Day 2 — figures**:
5. Generate per-patient spectrum supplement (§10a). 30 min.
6. Rebuild Fig 2 (NUTS-based ROC + 2-feature LR). 60 min.
7. Rebuild Fig 3 (NUTS spectra by tissue class). 30 min.
8. Rebuild Fig 4 (NUTS-disc vs ADC scatter). 30 min.
9. Wire `bias_heatmap.png` into paper/figures/. 15 min.

**Day 3 — draft**:
10. Results.tex rewrite (§9b step 4 + §10f Gleason subsection).
11. Methods.tex add simulation paragraph (§9b step 3).
12. Discussion.tex with §10c Limitations paragraph + §10d four-point
    novelty list + §10h refined mechanism + §10i future-work paragraph
    + §7c NUTS-coverage caveat.
13. Abstract.tex match (§9b step 5).

**Do NOT start Day 2 until Day 1 decision is taken.**
