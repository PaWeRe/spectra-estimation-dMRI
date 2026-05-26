# Session notes — 2026-05-22

_Wrapping context for the next session, which will start fresh with Stephan's
2026-05-22 email response as the primary input. Read this file first, then
the memory entries [[project_stephan_response_2026-05-22]] and
[[project_next_session_entry_point]]._

---

## 1. Exec 4P1 — done

`scripts/lr_coef_decomp.py` written and run. Outputs:
- `results/biomarkers/lr_coef_decomp.csv` (5 rows, per-task)
- `results/biomarkers/lr_coef_decomp_cross.csv` (2 rows, cross-task)

**Finding:** The LR-coefficient version of the §10f axis-separation claim is
**not clean at N=29**.

- Tissue-restricted (D≤3) `cos(w, D_vec)` cosines straddle zero (CIs include 0)
  for both tasks → LR discriminant is *not* a clean first-moment projection.
  Mildly supports Path A' (spectrum carries info beyond E[D]).
- `cos(w_T, w_G)` is **modest-positive ~+0.3 with wide CIs**, not orthogonal.
  The LR-weight version of axis separation does **not** support the
  "axes-orthogonal" claim. Wide CIs attributable to bin collinearity + N=29.
- D=20 free-water bin dominates the "full" cosines numerically — always strip
  D=20 when reporting these.

**Implication for manuscript:** Lead axis separation with the §10f per-bin
*univariate Spearman ρ profile* (clean), not with LR-weight cosines. F-new-2
caption must include "consistent with but not Bonferroni-significant at N=29."
Don't headline orthogonality from LR weights — the data don't strongly support
it.

Full details: [[project_exec_4P1_lr_coef_decomp_2026-05-17]].

---

## 2. Mid-session framing crisis — methods-only "Alive" reframe proposed

Patrick raised the question: given that the empirical clinical findings keep
shrinking (Path A' triangulating tests in May, Exec 4P1 today), is the paper
worth publishing at all? He named a hard constraint: **1 week to submit
something or he walks away from publication.**

Claude proposed a methods-only reframe — title approximately *"Calibrated
Bayesian spectrum estimation for the dMRI inverse Laplace problem — and why
NNLS/MAP misleads"* — with the MAP-smearing simulation as headline, and the
prostate dataset as a validation case (not the lede). Argument: the
methodology has audience in the broader inverse-Laplace dMRI literature
(T2 spectra, RSI, IVIM, microstructure), the simulation is clean and load-
bearing, and the 1-week deadline is more achievable with that scope.

Side-by-side comparison preserved below for the next session.

### Side-by-side: 4th-pass (Path A') vs 1-week Alive (methods-only)

| Section | **4th-pass Path A' clinical-mechanism** | **1-week Alive methods-only** |
|---|---|---|
| One-line thesis | "Bayesian spectral decomposition explains ADC's compartment mechanism; diagnostically equivalent at this N; spectrum adds per-bin uncertainty + axis separation." | "Calibrated Bayesian spectrum estimation for the dMRI inverse Laplace problem — and why NNLS/MAP misleads." |
| Headline result | Detection-vs-grading axis separation (Pillar 2) + ADC-equivalence as triangulated finding | MAP-smearing simulation + bias heatmap |
| Primary audience | Prostate cancer clinical readers (MRM) | Methodists doing T2/RSI/IVIM/microstructure inversion |
| Abstract | Rewrite around 3 pillars — mechanism, axis separation, ADC-equivalence | Method-focused: inverse problem → NUTS → MAP smearing → prostate validation |
| Intro | Chatterjee compartment mechanism; position vs HM-MRI/VERDICT/rVERDICT/RSI/LWI | Motivate inverse Laplace in dMRI broadly (T2, RSI, IVIM, microstructure); prostate as demonstration |
| Theory | Demote Fisher to supplement; audit MAP Eq 5 + prior consistency | Forward model + Bayesian formulation + NUTS. Fisher cut. No MAP Eq audit needed |
| Methods | Sim + ADC-variants + provenance + bootstrap | Sim methodology + NUTS + prostate dataset. Provenance only if Fig 9 stays in main |
| Results | Four pillars: detection / axis-separation / grading-equivalence / methodology | Linear: sim MAP-smearing → bias heatmap → prostate demonstration → ADC-equivalence (one paragraph) |
| Discussion | 4-point novelty + three-reason thesis + what spectrum adds + DKI parallel + NUTS coverage + limitations + future | 1-sentence ADC-equivalence; connection to T2/RSI/IVIM lit; NUTS coverage; limitations; future |
| Figures (main) | ~9-10 | 5-6 |
| Tables (main) | AUC table + supp ADC-variants | One AUC table |
| [ASK] batch | 6 items | 1-2 (provenance only) |
| Write time | 2-3+ weeks | ~1 week if reframe first |

---

## 3. Stephan's response (2026-05-22) — changes the math

Stephan's email response **endorsed Path A'** and added a new differentiator
Claude had not articulated:

> "This biexponential approach, using predefined diffusivities at 0.25 and
> 3.00 is fundamentally different from what we and others have been doing in
> the past, i.e., free floating diffusivities and fractions. With the old
> approach we always got higher than 0.25 slow diffusivity and lower than
> 3.00 fast diffusivity. Of course, these are the intermediate diffusivities
> getting accommodated in the solution. The change in fraction was evident
> previously, but now it is even more evident."

The novelty angle: prior free-floating biexponential fits in prostate dMRI
absorbed compartment mixtures into *intermediate-D drift*; our fixed-grid
Bayesian decomposition forces the model to express change as *fraction
shift at canonical D values*, making the compartment-volume story directly
readable. This is a new framing differentiating us from IVIM-style and
free-spectrum approaches.

He also endorsed the "why ADC works" framing as a real contribution
("nobody has revealed why ADC performs so well and I think our approach
shows this in quite a solid way"), and dismissed the circularity concern
on the grounds that clinicians use ADC as a contrast map, not as a
quantitative biomarker.

He wants: a meeting (flexible time-wise), ROI-level directional dependence
results (re-asking whether Patrick extracted from Dropbox), and "soon
submission."

Full text and analysis: [[project_stephan_response_2026-05-22]].

---

## 4. Open framing decision for next session

Three realistic paths:

**(a) Commit to Path A' clinical-mechanism** as Stephan prefers, with
the new "fixed-grid vs free-floating biexponential" differentiator
incorporated. Honest write time: 1.5-2 weeks. Would need to negotiate
deadline.

**(b) Hybrid:** Path A' framing for Intro / Abstract / Discussion, but
compress Results to the linear methodology-spine structure from the
1-week Alive plan. Keep F-new-2 (axis separation) but with honest
underpowered caveat per Exec 4P1. Closer to 1 week. Less ambitious than
full Path A' but preserves Stephan's preferred angle.

**(c) Hold path decision until coauthor meeting**, then sprint. Risk:
shorter writing window after meeting.

**Recommendation for next session:** raise (a) vs (b) directly at the
meeting and let Stephan weigh in before committing prose.

### What stays cut regardless of path

- ADC-variants sweep as main-text content (supplementary at most).
- Biopsy-replacement Intro motivation.
- LR-coefficient axis-separation framing (today's Exec 4P1 result — wide CIs).

### What needs to be reinstated if we move from Alive back toward Path A'

- **Three-reason thesis** in Discussion (Stephan endorses "why ADC works").
- "What spectrum adds beyond ADC" paragraph.
- Possibly the DKI kurtosis parallel (Stephan didn't comment on it; one
  sentence at most).

---

## 5. Open actions

- [ ] Schedule meeting with Stephan (flexible, his side). Patrick to propose
      times.
- [ ] Did Patrick extract the ROI-level directional dependence data from
      Dropbox? Stephan re-asked. Resolve before meeting.
- [ ] No response yet from Sandy on the 2026-05-16 email. Following up?
- [ ] Lit gap: any recent (2023-2025) prostate-DWI methods papers we've
      missed? Asked in original email, neither coauthor has answered.

---

## 6. Hard deadline reminder

Patrick's stated stance on 2026-05-22:

> "I have 1 week left to submit something, if by then we don't have an
> ok paper I will stop this work and focus on other things."

Effective walk-away date: **~2026-05-29**.

This is a real constraint, not aspirational. Plan the writing accordingly.

---

_End of 2026-05-22 wrap. See [[project_next_session_entry_point]] for the
recommended start of the next session._
