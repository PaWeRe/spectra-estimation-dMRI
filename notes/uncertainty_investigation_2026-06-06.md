# Investigation: does the NUTS posterior uncertainty carry classification signal?

**Opened 2026-06-06 (Patrick).** Triggered by a conclusive-but-disappointing result during the joint iteration: the propagated classification uncertainty does **not** add a reliable error-flag beyond what the point estimate already gives. Patrick wants to investigate rigorously before changing the manuscript framing and Fig 7. **Start a fresh session from this note.**

---

## 0. Why this matters

The whole rationale for the fully Bayesian NUTS sampler (vs the fast MAP point estimate) has two pillars:
1. **Identifiability** — per-component posterior σ quantifies which compartments are recoverable (Fisher + posterior agreement). **This is the paper's spine and is NOT in question.**
2. **Uncertainty as a downstream signal** — the hope that propagated posterior uncertainty helps disease classification (a new biomarker). **This is what the result below undercuts.**

So the finding weakens **Block C** (exploratory uncertainty biomarker), not the spine. But it's important: if uncertainty has *no* classification value, the "uncertainty-aware classifier" (Fig 7) should be reframed as a capability demo or demoted, and the NUTS-over-MAP argument leans entirely on identifiability + joint noise inference.

---

## 1. The finding (evidence so far)

Pipeline (Fig 7, `scripts/fig6_uncertainty_classifier.py`): for each ROI, push all 8000 NUTS draws of its spectrum through the LOOCV logistic-regression detector → a posterior over P(tumor). Two candidate signals were tested: (a) misclassified vs correct interval width; (b) interval width vs distance to the decision boundary.

**The geometry confound.** The probability-interval width = (spread of the linear predictor) × sigmoid slope $P(1-P)$, which is maximal at $P=0.5$. So probability-space width is inflated near the boundary purely by geometry. The geometry-free measure is the **logit-space spread** (`z_std` in the script).

**Numbers (pooled n=149, 15 misclassified):**

| Quantity | Probability space | Logit space (geometry-free) |
|---|---|---|
| ρ(distance-to-boundary, width) | −0.94 | −0.29 |
| misclassified / correct width ratio | 2.41× | 1.27× |
| patient-level bootstrap 95% CI on ratio | [1.69, 3.40] | **[0.94, 1.67] — includes 1** |

**Rejection curve (defer most-uncertain X%, accuracy on retained):**

| keep | by probability CI width | by logit spread | random |
|---|---|---|---|
| 100% | 0.899 | 0.899 | 0.899 |
| 70% | 0.962 | 0.904 | 0.899 |
| 50% | 0.973 | 0.905 | 0.901 |

**Interpretation.** Deferring by probability-CI-width helps (0.90→0.97) — but ρ(dist, prob-width)=−0.94 means that ordering ≈ "defer cases near P=0.5", i.e. the point estimate's own distance-to-boundary, which a plain MAP-LR gives **for free, without NUTS**. The genuinely-Bayesian part (logit spread) gives a **flat** rejection curve and a misclassified/correct ratio whose patient-level CI **includes 1**. The earlier Welch p=0.003 was ROI-level and overstated significance.

**Bottom line so far:** in this cohort, the propagated posterior interval adds no statistically reliable error-flagging power beyond the point estimate's distance-to-boundary.

---

## 2. Alternative explanations to rule out FIRST (conclusive, not cheating)

Do not conclude "uncertainty has no signal" until these are excluded:

1. **The propagation channel is too narrow.** The LR is trained on posterior *means*, so uncertainty only enters via the spread of P(tumor). The uncertainty signal might live in the **per-bin posterior σ themselves**, never reaching the classifier. → Test B below (uncertainty-as-features) is the fair test.
2. **Underpowered.** 149 ROIs, 15 errors; the logit CI [0.94,1.67] is wide. We can only say "not demonstrated at this N," not "no effect." Quantify the power.
3. **Miscalibration.** F6: NUTS is over-confident on concentrated spectra. If the propagated intervals are not calibrated, they cannot flag errors by construction. → Test A.
4. **Confound with acquisition quality.** If uncertainty just tracks SNR / tissue type / zone, it is an image-quality meter, not a disease biomarker (still useful, different framing). → Test F.
5. **Wrong target.** Uncertainty may not predict *detection* errors but may track *grade*, or out-of-distribution voxels (pixelwise, low SNR). → Tests E, D.

---

## 3. Rigorous investigation plan (prioritized)

**The precise question:** does the NUTS posterior provide classification-relevant information beyond the MAP point estimate *and its own distance-to-boundary confidence*? Null H0 = "it does not." Current evidence fails to reject H0 for the propagated-interval route.

**Test A — Calibration / coverage of the propagated probability.**
Reliability diagram + expected calibration error (ECE) of P(tumor); and does the 90% credible interval on P(tumor) contain the true label ~90% of the time? If miscalibrated, the uncertainty is unreliable by construction. (No ground-truth spectra for real ROIs, so calibrate against the *label*, not the spectrum.)

**Test B — Uncertainty-as-features (the fair test of independent signal).**
Add per-bin posterior std/CV (`nuts_std_D_*` in `features.csv`) as explicit features to the detection LR alongside the means. Does AUC improve vs means-only? Proper LOOCV, patient-level bootstrap CI on ΔAUC, both zones. (Partially done: F2 `bin_information_sweep.py` found σ_D=0.25 in TZ marginally helps Δ+0.010, CI overlapping — revisit exhaustively and rigorously.) If even this fails, the "no independent signal" conclusion is robust.

**Test C — Selective prediction done properly, head-to-head.**
Three rejection criteria on the SAME held-out predictions: (i) point-estimate |P−0.5| (the free baseline), (ii) propagated prob-CI-width, (iii) logit spread. Compute the area under the accuracy-rejection curve for each, with patient-level bootstrap CIs on the differences. The bar is "beat (i)". (i) vs (ii) should be ~equal (ρ=−0.94); the real test is whether (iii) adds anything.

**Test D — A task that SHOULD need uncertainty.** The pixelwise tumor-skew (F12): single-voxel low SNR inflates the restricted fraction. Does per-voxel posterior uncertainty flag the unreliable (low-SNR) voxels that the point estimate gets wrong? This is where uncertainty is most likely to earn its keep.

**Test E — Does uncertainty predict grade or other labels?** CI-width / logit-spread vs GGG (Spearman); does uncertainty separate the confidently-misclassified low-grade tumors?

**Test F — Confounds.** Correlate CI-width and logit-spread with SNR, zone, tissue type. Decompose how much "uncertainty" is just SNR.

**Test G — Power / sample-size.** Given the observed logit effect (1.27×), what N would be needed to detect it at 80% power? Frames the honest "not demonstrated vs absent" statement.

---

## 4. If the result holds (no reliable classification signal) — framing options

- **Spine unaffected.** NUTS is justified by identifiability + joint noise inference + honest spectral error bars (scientific reporting), independent of any classification-uncertainty biomarker. State this explicitly.
- **Fig 7 options:** (1) keep as an honest *capability* demo (calibrated P(tumor) + interval — something MAP cannot produce), explicitly *not* a validated biomarker; (2) demote to Supplementary and reclaim a main-figure slot (λ-sweep, or single-feature AUC panel); (3) replace with Test-D pixelwise-uncertainty-flags-low-SNR if that works.
- **Coherent narrative:** "only 2 compartments are identifiable → MAP≈NUTS on them → propagated classification uncertainty adds little" is a *consistent* story that reinforces the identifiability thesis, not a contradiction.

---

## 5. Code / data pointers

- **Propagation + diagnostics:** `scripts/fig6_uncertainty_classifier.py` (`propagate_zone` returns p_point, p_lo/hi, p_std, **z_std** [logit spread], ci_width, dist, correct; `diagnostics` already prints the logit decomposition).
- **Per-bin posterior σ:** `results/biomarkers/features.csv` cols `nuts_std_D_*` (means in `nuts_D_*`).
- **Uncertainty-as-features prior attempt:** `scripts/bin_information_sweep.py` → `results/biomarkers/bin_information_sweep.csv` (F2).
- **Posterior draws (gold standard):** `results/inference_bwh_backup/*.nc`.
- **Rejection-curve + patient-bootstrap snippet used 2026-06-06** (reproduce the §1 numbers): imports `fig6_uncertainty_classifier` as a module, calls `load_draws` + `propagate_zone`, then sorts by `z_std`/`ci_width` for the rejection curve and resamples `patient` for the ratio CI. (Re-derive; ~1-2 min to reload .nc.)

---

## 6. Manuscript status pending this investigation
- Results + Fig 7 caption + Discussion edited 2026-06-06 to the **honest interim**: report the patient-bootstrap CIs (logit ratio 1.27× [0.94,1.67]), frame the propagated uncertainty as a **capability, not a validated biomarker**, Welch p removed. A `\note` marks Fig 7 / Block C as pending this investigation.
- **Do not submit the uncertainty claim as a biomarker** until A–C resolve.
