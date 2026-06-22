# Fig 1(b) — Bayesian (van Trees) CRLB: code + math for review

**For:** Sandy
**From:** Patrick (drafted with Claude assistance)
**Date:** 2026-06-21
**Supersedes:** `notes/CRLB_NOTE_FOR_SANDY.txt` (2026-05-26) — adds the exact code map,
the λ clarification, and a sharper statement of the one validation question.

This is the only unvalidated claim in the manuscript. It gates **Fig 1(b)** (the
3-bar per-component uncertainty panel), the matching **Theory** paragraph
(`theory.tex:55–59`), and the **Discussion** paragraph (`discussion.tex:26`).
Everything else (panels a and c, the figure mechanics) is standard and not in question.

---

## 1. What the panel claims

Per diffusivity bin, three estimates of the standard deviation of the estimated
volume fraction, at the cohort-median SNR = 303:

1. **Unconstrained classical CRLB** — data only, no prior, no constraint.
2. **Bayesian (van Trees) CRLB** — data + a Gaussian prior of precision λ.
3. **Empirical NUTS posterior SD** — data + half-Normal prior + non-negativity (R ≥ 0).

The panel annotates two improvement factors: **prior gain** (1→2) and
**constraint gain** (2→3). The narrative: the raw inverse problem is hopeless for
intermediate D (unconstrained CRLB ≫ 1), the prior buys ~1–2 orders of magnitude,
and positivity + one-sidedness buys a further 2–74×.

---

## 2. The math

### Forward model
Normalized signal decay, K = 8 diffusivity bins, M = 15 b-values:

    s / S0 = U R,    U_{ij} = exp(-b_i D_j),    R_j >= 0,    U is 15 x 8

D = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] µm²/ms
b = [0, 250, ..., 3500] s/mm² (= [0, 0.25, ..., 3.5] ms/µm²)
cond(U) = 2.78e5,  so cond(UᵀU) = 7.7e10  (severely ill-conditioned).

### Likelihood and data Fisher information
Gaussian noise with σ = 1/SNR on the normalized signal:

    s/S0 ~ Normal(U R, σ² I)
    F_data = (1/σ²) UᵀU = SNR² UᵀU

### Bar 1 — classical CRLB
    crlb_unc_j = sqrt( [F_data⁻¹]_{jj} )

### Bar 2 — Bayesian / van Trees CRLB
Gaussian prior R ~ Normal(0, σ_R² I) with σ_R² = 1/λ, i.e. prior precision Λ = λ I.
Posterior information = data information + prior information:

    J_post = F_data + λ I = SNR² UᵀU + λ I
    crlb_bay_j = sqrt( [J_post⁻¹]_{jj} )

### Bar 3 — empirical NUTS posterior SD
Drawn from the sampler (Section 4), normalized to fractions, SD across draws, then
averaged across the 149 ROIs.

### Numbers (SNR = 303, λ = 0.1)
    sigma_R = 1/sqrt(0.1) = 3.162

      D    bar1 uncCRLB   bar2 BayesCRLB   bar3 NUTS_SD   prior(1/2)  constr(2/3)
    0.25       5.797          0.246          0.0145          24x        16.9x
    0.50      44.054          1.276          0.0305          35x        41.9x
    0.75     121.746          2.334          0.0317          52x        73.7x
    1.00     131.048          2.064          0.0410          63x        50.4x
    1.50      78.387          2.195          0.0826          36x        26.6x
    2.00      35.680          1.691          0.1210          21x        14.0x
    3.00       4.932          0.448          0.0777          11x         5.8x
    20.00      0.080          0.028          0.0158           3x         1.7x

---

## 3. The one thing to validate: the Gaussian approximation in bar 2

`J_post = F_data + λ I` is the **exact** posterior information only for a *full
Gaussian* prior (its log-density has constant curvature λ everywhere). The prior
NUTS actually uses is a **half-Normal** (R ≥ 0). Two consequences:

1. **Regularity conditions.** The van Trees bound assumes a smooth prior density
   vanishing at the boundary of the parameter space. The half-Normal has a hard wall
   at R = 0. So `+λI` is not literally the half-Normal's Bayesian CRLB — it is the
   bound for the **matched-scale Gaussian** (same σ_R), used as a stand-in. This is
   "the Gaussian approximation."

2. **It under-counts two effects.** Near R = 0 (where intermediate-bin posteriors
   concentrate): non-negativity truncates the feasible set, and the one-sided prior
   concentrates mass at 0 more sharply than a symmetric Gaussian of equal scale.
   Both shrink the realized SD.

**Therefore bar 2 is a bound for the Gaussian-prior problem; NUTS solves the
constrained half-Normal problem, whose true bound is smaller. NUTS coming in below
bar 2 is expected, not a CRLB violation — the 2→3 gap is exactly the value of
positivity + one-sidedness ("constraint gain").**

### Questions for you
1. **Is the matched-scale Gaussian van Trees framing acceptable** as the middle bar,
   with the figure/caption labeling it explicitly as the Gaussian-prior bound — or do
   you want the **true constrained / one-sided Bayesian CRLB** (Gorman & Hero 1990
   constrained CRLB; or a Bayesian CRLB for truncated priors)? The latter is cleaner
   but materially more work and would change bar 2's values.
2. **Do you agree with the two-mechanism decomposition** — prior gain (unconstrained →
   Bayesian) and constraint gain (Bayesian → NUTS) — as the right way to attribute the
   total gap, rather than a single unconstrained-CRLB-to-NUTS ratio?

---

## 4. Which λ, and why 0.1 (not the MAP 10⁻³)

There are two regularization strengths in the project:

| Estimator | λ | Where |
|-----------|-----|-------|
| MAP point estimate | **10⁻³** | `methods.tex:24`; `recompute.py:40` `RIDGE_STRENGTH = 1e-3` (retuned for recovery on log-normal GTs) |
| NUTS posterior (the `.nc` files) | **0.1** | `configs/prior/ridge.yaml`; `recompute.py:47` `_PRIOR_CFG` (hash-locked to the existing posteriors) |

Bar 2 uses **0.1** because bar 3 is the NUTS posterior, sampled under the half-Normal
prior with precision λ = 0.1. The van Trees bound must carry the *same* prior the
sampler used. (Using 10⁻³ would raise bar 2 ~8–10×, collapse the prior-gain step, and
misattribute nearly all the gain to the constraint — it would be the bound for a prior
NUTS never used.)

**Manuscript note (Patrick to fix):** `theory.tex:92` currently says the half-Normal
"variance 1/λ matches the ridge penalty, linking the MAP and Bayesian formulations,"
which reads as a single shared λ. MAP (10⁻³) and NUTS (0.1) differ. The text will be
disambiguated; results are robust to prior strength (F8). This does not affect the
derivation above.

---

## 5. Exact code map

**Generator:** `scripts/generate_fisher_figure.py`

| Quantity | Line |
|----------|------|
| Design matrix U = exp(-b ⊗ D) | 55 |
| F_data = SNR² UᵀU | 59 |
| F_post = F_data + λI | 61 |
| crlb_unc = sqrt(diag(inv(F_data))) | 63 |
| crlb_bay = sqrt(diag(inv(F_post))) | 64 |
| Correlation matrix C (panel a) | 67–69 |
| Load NUTS SD from identifiability.csv (bar 3) | 71–75 |
| Panel (b) plotting | 116–140 |
| SNR = 303, λ = 0.1 | 51–52 |

**The prior that defines bar 2's validity:** `src/spectra_estimation_dmri/inference/nuts.py`

| Quantity | Line |
|----------|------|
| σ_R = 1/√strength | 128 |
| R ~ HalfNormal(σ_R) | 129–131 |
| σ ~ HalfCauchy (noise inferred, not fixed) | 145 |
| likelihood Normal(U·R, σ) | 151–156 |

**Bar 3 provenance:** `src/spectra_estimation_dmri/biomarkers/recompute.py`

| Step | Line |
|------|------|
| `load_nuts_posteriors`: read each ROI's `.nc`, normalize draws to fractions, per-bin SD | 185–225 (normalize 212–214, SD 220) |
| `component_identifiability`: average SD across 149 ROIs → `mean_posterior_std` | 647–662 |
| write `results/biomarkers/identifiability.csv` | ~1021 |

**Inputs:** `results/inference_bwh_backup/*.nc` (149 gold-standard posteriors);
`results/biomarkers/identifiability.csv` (bar 3 values).

**Paste-runnable reproduction** (no figure side effects):

```python
import numpy as np, pandas as pd
b = np.array([0,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500])/1000.0
D = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
U = np.exp(-np.outer(b, D)); SNR = 303; lam = 0.1
F_data = (SNR**2) * (U.T @ U)
F_post = F_data + lam*np.eye(8)
crlb_unc = np.sqrt(np.diag(np.linalg.inv(F_data)))
crlb_bay = np.sqrt(np.diag(np.linalg.inv(F_post)))
nuts = pd.read_csv("results/biomarkers/identifiability.csv")["mean_posterior_std"].values
```

---

## 6. Secondary caveats (not blocking; for completeness)

1. **Scale.** Bars 1–2 are SDs of the raw amplitudes R; bar 3 is the SD of the
   normalized fraction R_j/ΣR. Because the signal is S0-normalized, ΣR ≈ 1, so they
   are approximately on the same scale, but not identically. The y-label "fraction"
   is literally exact only for bar 3.
2. **Noise.** Bars 1–2 fix σ = 1/303; NUTS infers σ (half-Cauchy), and the cohort SNR
   spans IQR 176–478. The bars are "at the typical SNR," not per-ROI exact.
