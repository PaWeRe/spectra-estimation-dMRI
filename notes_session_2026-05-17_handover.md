# Session handover — 2026-05-17

## TL;DR

§10g (intermediate-bin GGG sweep) and §10f (continuous-GGG sweep) both ran.
Path A headline (2-feat NUTS-LR {μ_D=0.25, μ_D=3.00} for tumor-vs-normal)
holds. A *new* clean Path A' result emerged: **μ_D=0.50 carries continuous-
GGG signal with Spearman ρ = +0.565 (Bonferroni-significant, pooled N=29,
p = 0.0014)** — biologically interpretable as the epithelium-stroma boundary
where Gleason 4/5 microstructural disorganization happens.

**But ADC has ρ = −0.546 for the same task.** ADC and spectral features
have similar GGG correlation magnitudes. The "ADC is blind to grading"
hypothesis is *empirically false* — Chatterjee 2015 explicitly explains
why: ADC is driven by the same compartment-volume changes (epi↑, lumen↓)
that drive Gleason grade.

**The Path A' framing must shift**: from "spectrum beats ADC for grading"
to "spectrum decomposes the compartment volumes that ADC is averaging."
Diagnostic AUC may be similar; spectrum offers biological interpretability
and per-voxel compartment labels that ADC cannot.

**The key remaining test is partial correlation** — does the spectrum
carry grading info ADC misses? This is the first thing to run in the
next session.

---

## What was done this session

### §10g — Bin-information sweep (tumor-vs-normal)

- Script: `scripts/bin_information_sweep.py`
- Output: `results/biomarkers/bin_information_sweep.csv` (1782 rows)
- Verdict: **No subset beats reference {μ_D=0.25, μ_D=3.0} with CI
  separation in any zone.** Path A holds.
- Three findings to land in manuscript:
  1. Intermediate-only LR collapses by ΔAUC ≈ −0.13 in every zone — direct
     test that intermediate bins carry no independent tumor-vs-normal signal.
  2. TZ triple {0.25, 0.50, 3.00} gives non-CI-significant +0.014 lift —
     acknowledge but don't elevate.
  3. σ_D=0.25 marginally improves TZ tumor-vs-normal (Δ +0.010, CI overlap)
     — brief Discussion mention.

Memory: [[project_sec10g_closed]] (already written).

### §10f — Continuous-GGG sweep (NEW SESSION FOCUS)

- Script: `scripts/ggg_continuous_sweep.py`
- Output: `results/biomarkers/ggg_continuous_sweep.csv` (597 rows)
- Sample: 29 tumor ROIs with valid GGG (PZ 21, TZ 8, low=20 hi=9).

**Pooled (N=29) Bonferroni-significant findings (α=0.05/16=0.0031):**

| Feature | ρ vs continuous GGG | 95% CI | p |
|---|---:|:---:|---:|
| **μ_D=0.50** | **+0.565** | [+0.287, +0.757] | **0.0014** |
| **σ_D=0.50** | **+0.566** | [+0.320, +0.747] | **0.0014** |
| ADC (ref) | −0.546 | [−0.799, −0.200] | 0.0022 |

Below Bonferroni but strong: μ_D=0.75 +0.512, σ_D=0.25 +0.499, μ_D=2.0
−0.474, μ_D=0.25 +0.437.

**PZ (N=21)**: μ_D=0.50 ρ=+0.576 p=0.006 (best single, below Bonferroni
due to smaller N). ADC PZ ρ=−0.480.

**TZ (N=8)**: noise-dominated, CIs touch ±1.0. Don't interpret.

**No multi-feature subset reliably beats single μ_D=0.50** — LR at N=29
overfits and destroys signal.

### Detection vs grading axis separation

| Bin | Tumor-vs-Normal AUC role | Continuous-GGG ρ |
|---|---|---:|
| D=0.25 | high (in 2-feat headline) | +0.44 (subordinate) |
| **D=0.50** | low (n.s. in sweep) | **+0.57 (Bonferroni)** |
| **D=2.00** | low (n.s. in sweep) | **−0.47** |
| D=3.00 | high (in 2-feat headline) | −0.16 (n.s.) |

Detection and grading live on approximately orthogonal axes in spectrum
space. This is the cleanest empirical result of the session.

### Literature pull and synthesis

- All key precedent PDFs now in `assets/`:
  - **Chatterjee 2015 Radiology** (the keystone) — `Chatterjeeetal2015Radiologyprint.pdf`. Spearman: epithelium ρ=+0.898, lumen ρ=−0.912, stroma ρ=−0.651 vs Gleason pattern (n=553 ex-vivo subimages). **Conclusion: gland compartment volumes (not cellularity) drive ADC variation.** This is the keystone for our framing.
  - **Chatterjee 2017/2018 feasibility** — `chatterjee2017feasibilityRadiology.pdf`. Original HM-MRI 3-compartment in-vivo paper.
  - **Chatterjee 2022 validation** (already read this session) — `chatterjee2018Radiology.pdf` (misnamed). N=25, validates HM-MRI with whole-mount histology.
  - **VERDICT 2019** — `VERDICT.pdf`. f_IC=0.49 (≥GGG3+4) vs 0.31 (benign/GGG3+3), p=0.002.
  - **rVERDICT 2023** — `rVERDICT.pdf`. f_IC discriminates Gleason 3+3 vs 3+4 (p=0.003) and 3+4 vs ≥4+3 (p=0.040). N=44 with biopsy.
  - **Sabouri 2017 LWI** — `sabouri2015Radiology.pdf` (misnamed). T2-NNLS LWF Spearman ρ=−0.78 vs Gleason in PZ. Uses regularized NNLS (same algorithm family as our MAP).
  - **Brunsing 2017 RSI** — `Brunsing2017JMRI.pdf`. (Not read in detail this session.)
  - **Yamin 2016 RSI** — `Yamin2016RSI.pdf`. Voxel-level RSI CI vs Gleason. (Not read in detail.)

- Full annotated list: `notes_session_2026-05-17_literature.md`

### Path A' framing — REVISED based on session findings

Old Path A': "Spectrum beats ADC for grading because of intermediate-bin signal."
**New Path A' (after this session): "Spectrum decomposes the compartment
volumes that ADC averages over, providing biological interpretability per voxel."**

Supporting framework:
- **Chatterjee 2015 mechanism**: ADC is driven by gland compartment
  volumes (epi/stroma/lumen), not cellularity. Spectrum measures those
  compartments directly.
- **Bourne 2018 bin-to-compartment mapping**: D=0.3–0.5 epithelium, 0.7–0.9
  stroma, 2.0–2.2 lumen. Matches our 8-bin grid.
- **Our empirical finding**: detection axis (D=0.25 ≈ epi, D=3.0 ≈ free)
  is orthogonal to grading axis (D=0.50 ≈ epi-stroma boundary, D=2.0 ≈
  lumen). ADC compresses both axes into one scalar.

The "ADC ≈ projection" claim from Path A becomes mechanistically richer:
ADC projects the *multi-compartment* signal onto the dominant scalar
axis. Spectrum keeps the compartments separated.

---

## What still needs to run

### IMMEDIATE NEXT (the key test)

**Partial Spearman correlation**: ρ(μ_D=0.50, GGG | ADC)
and ρ(μ_D=2.00, GGG | ADC). If either is still significant, spectrum
carries grading info ADC misses.

Trivial code, ~10 lines:
```python
from scipy import stats
# After loading tumor-ROIs-with-GGG subset:
for col in ["nuts_D_0.50", "nuts_D_2.00", "nuts_D_0.25", "nuts_D_3.00"]:
    # Compute partial correlation via residuals
    # 1) regress feature on ADC, take residuals
    # 2) regress GGG on ADC, take residuals
    # 3) Spearman of the two residuals = partial correlation
    feat = sub[col].values; adc = sub["adc"].values; ggg = sub["ggg"].values
    # ... linear regression residuals, then spearmanr
```

Or use `pingouin.partial_corr` if available.

### §10h — NUTS sensitivity vs ADC, extended (pending, task #6)

Repurposed dual purpose:
1. Confirm tumor-vs-normal LR coefs ≈ ADC sensitivity vector (vector-level)
2. Test that grading-axis LR coefs (trained on GGG≥3 or continuous-GGG
   discriminant) are *orthogonal* to ADC sensitivity vector

This becomes the methodological backbone of the new Path A' Discussion
mechanism paragraph.

### Methodological alternatives Patrick wants to explore

Priority order (raised 2026-05-17 end-of-session):
1. **Partial correlation** (above) — first
2. **2-feature LR test**: AUC for GGG≥3 with {ADC} vs {ADC + μ_D=0.50, μ_D=2.0}, DeLong
3. **Alternative ADC computations**: high-b ADC (b ≥ 1000), all-b mono-exp, DKI-corrected. Fair-comparison upper bound for ADC.
4. **Per-patient averaging**: 56 patients → 56 datapoints. Reduces ROI noise.
5. **Compartment-style derived features**: define "restricted_frac" = μ_0.25 + μ_0.50; "luminal_frac" = μ_2.0 + μ_3.0; test against ADC.
6. **Spectral entropy / number-of-modes per ROI**: single scalar capturing heterogeneity that ADC structurally cannot. Tests whether shape-of-spectrum (not just mean) correlates with GGG.

### What ADC fundamentally loses (structural framing — Patrick 2026-05-17 end-of-session)

ADC = ∫ R(D) · D dD is the first moment of the diffusivity distribution.
Averaging discards:

- **Compartment identity** — same ADC from many (epi/stroma/lumen) volume-fraction combinations.
- **Distribution shape / mode count** — bimodal vs unimodal R(D) with same mean give same ADC. Same-ADC ROI could be a mix of GGG3-character + GGG4-character regions, averaged into "intermediate."
- **Spectral entropy / heterogeneity** — broadness vs concentration of the spectrum.
- **Tail behavior** — small fractions at extreme D (free water, microcysts, perfusion contamination) average in but lose identity.

**The partial correlation test IS the empirical version of this question.**
ρ(μ_D=0.50, GGG | ADC) asks: *among ROIs with similar ADC, does μ_D=0.50
still track grade?* If yes → spectrum identifies grade-relevant
compartmental shifts that average out in ADC. That is what averaging
fundamentally loses.

§10h sensitivity analysis stays but with broader claim:
- ADC sensitivity vector ≈ tumor-vs-normal LR direction (confirms scalar projection).
- ADC sensitivity vector ⊥ GGG-LR direction (vector evidence that the spectrum's grading axis is what ADC cannot reach).

### Biopsy-replacement clinical framing (new — Patrick 2026-05-17)

Long-term clinical motivation worth adding to Introduction:

- **Liver precedent**: HCC diagnosed image-only via LI-RADS because biopsy seeding/bleeding risk outweighs benefit.
- **Prostate biopsy problems**: 1–2% serious complication rate (sepsis/bleeding); 12-core sampling covers ~1% of prostate tissue; 30–50% of GGG1 biopsies upgrade on prostatectomy due to sampling error.
- **Frame**: long-term goal is image-based grading that complements/reduces biopsy. Mechanistic spatially-resolved compartment imaging is the path. This paper contributes a Bayesian, model-free, uncertainty-quantified step toward that goal — explicitly motivates **why per-voxel uncertainty matters** (biopsy-replacement scenarios need calibrated uncertainty per region).
- **Do NOT claim to solve it.** Just motivate.

### Heatmap-by-Gleason figure (raised Patrick 2026-05-17)

**Verdict: supplementary figure, NOT main. Structured as box plots, NOT cherry-picked examples.**

Works:
- Box plots of bin fractions stratified by GGG (1–5), N=8/12/5/2/2 per group, overlaid individual points (29 dots total). Defensible because we show trend, not adjacent-group separation.
- Average spectrum per GGG group (5 spectra overlaid) — shows D=0.50 and D=2.0 shifts.
- One representative NUTS posterior heatmap per GGG level (5 panels) showing per-bin uncertainty + grade trend. Illustrative.

Does NOT work:
- Cherry-picked "look how different GGG5 looks!" panels — reviewers will flag.
- Spatial per-voxel heatmaps across grades — no whole-mount histology to validate.

**TCIA cross-dataset validation: SKIP.** Public prostate dMRI uses clinical b-value sets (max 1400). Our 15-b extended (up to 3500) has no public counterpart. Spectral inference doesn't reduce cleanly to 5 b-values. Cross-dataset validation = months of work, next-paper material.

### Manuscript drafting still blocked on

- §10h (above)
- §10c (selection bias) — Limitations paragraph; no new analysis needed
- §10d (novelty framing) — Discussion-opener; rewrite with new findings
- §10e (why 8 bins) — Methods/Intro; tie to Bourne 2018 compartments
- §10a (per-patient spectrum supplement) — new script, can parallelize

---

## Files written this session

- `scripts/bin_information_sweep.py`
- `scripts/ggg_continuous_sweep.py`
- `results/biomarkers/bin_information_sweep.csv`
- `results/biomarkers/ggg_continuous_sweep.csv`
- `notes_session_2026-05-17_literature.md`
- `notes_session_2026-05-17_handover.md` (this file)

Memory updates:
- `project_sec10g_closed.md` (new)
- `project_pre_draft_open_questions.md` (§10g struck off)
- `MEMORY.md` (index updated)

## Open task list (next session)

- #6 in_progress: §10h NUTS sensitivity (now dual-purpose for orthogonality)
- New: partial correlation test (highest priority, run first)
- New: alternative ADC computations test
- New: per-patient averaging test
- New: compartment-style derived features test

---

## Key memory pointers for next session

- [[project_current_state]] — outdated; needs rewrite with Path A' framing
- [[project_pre_draft_open_questions]] — §10g struck; §10f mostly resolved (continuous-GGG done) but partial-correlation question remains
- [[project_sec10g_closed]] — §10g closure rationale
- [[project_nuts_coverage_caveat]] — Discussion caveat
- [[reference_mrm_guidelines]] — page budget
- [[project_llm_policy_mrm]] — disclosure plan
- [[user_patrick]] — Patrick's preferences and working style
- [[feedback_conclusive_not_cheating]] — drove the §10g and §10f rigor

## Honest summary of where we stand

**The spectrum approach is more nuanced than originally framed.** It's not
"better than ADC" in a simple AUC sense at our N. But it IS:

1. A *direct biological measurement* of compartment volumes that ADC
   averages over (Chatterjee 2015 mechanism).
2. *Mechanistically separable* into detection (outer bins) and grading
   (intermediate + D=2.0) axes — orthogonal in our data.
3. *Quantifying uncertainty per compartment* — no other method does this.
4. Matching the published lineage (VERDICT, RSI, HM-MRI, LWI) on a model-
   free grid.

If the partial correlation test is positive, we have "spectrum carries
grading info beyond ADC." If negative, we have "spectrum mechanistically
explains what ADC is measuring." Either way, the paper is publishable —
just with different headlines.
