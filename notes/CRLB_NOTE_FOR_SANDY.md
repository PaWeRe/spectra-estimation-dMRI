# CRLB diagnostic — for Sandy (2026-05-26)

## TL;DR

The factor-2000× gap between CRLB and NUTS posterior std is explained by **what CRLB
we are comparing against**. The current paper figure plots the *unconstrained classical*
CRLB versus the *fully Bayesian NUTS* posterior std. Those are different estimators of
different problems. When we additionally compute the **Bayesian CRLB** (van Trees lower
bound that incorporates the HalfNormal prior), the picture is:

| D       | unconstrained CRLB | Bayesian CRLB | NUTS empirical std | unc/NUTS  | Bayes/NUTS |
|---------|--------------------|---------------|--------------------|-----------|------------|
| 0.25    | 5.797              | 0.246         | 0.014              | **399×**  | 16.9×      |
| 0.50    | 44.05              | 1.276         | 0.030              | **1445×** | 41.9×      |
| 0.75    | 121.75             | 2.334         | 0.032              | **3843×** | 73.7×      |
| 1.00    | 131.05             | 2.064         | 0.041              | **3197×** | 50.4×      |
| 1.50    | 78.39              | 2.195         | 0.083              | **949×**  | 26.6×      |
| 2.00    | 35.68              | 1.691         | 0.121              | **295×**  | 14.0×      |
| 3.00    | 4.93               | 0.448         | 0.078              | **63×**   | 5.8×       |
| 20.00   | 0.081              | 0.028         | 0.016              | 5×        | 1.7×       |

Computed at SNR = 303 (cohort median), with HalfNormal prior σ_R = 1/√λ at λ = 0.1.

## What changed

The paper currently compares (a) unconstrained CRLB at hard-coded SNR=150 vs (c)
empirical NUTS std. Two issues:

1. **SNR = 150 is too low.** The cohort median is 303 (IQR 176–478). The hard-code understates achievable precision by ~4× in variance (~2× in std).
2. **Apples-to-oranges.** Unconstrained CRLB ignores the prior. NUTS has a HalfNormal prior that is strongly informative for this ill-conditioned design. Comparing them produces the spectacular 2000× ratio not because NUTS is doing magic but because the comparison sidelines the prior contribution to the inverse problem.

## The Bayesian CRLB

Under a Gaussian prior on R with precision Λ (here diagonal, Λ = λI = 0.1·I), the
posterior Fisher information matrix is the sum of the data Fisher matrix and the prior precision:

```
  J_post = (1/σ²) UᵀU + λ I
```

The Bayesian CRLB (van Trees inequality) is `√diag(J_post⁻¹)`. This is the relevant
lower bound for a Bayesian estimator like NUTS.

A subtlety: the actual prior is **half**-Normal (one-sided), not Gaussian. The
Gaussian-CRLB approximation under-counts the information contribution of the
one-sided support (it gives the prior the same Fisher information density at R=0 as
the full Gaussian would). The half-Normal contributes additional information whenever
the posterior mass is near R=0, which is exactly where intermediate-bin posteriors
concentrate. Hence NUTS comes in *below* the Bayesian CRLB at intermediate bins (the
5–74× gap is the half-Normal + non-negativity advantage over a hypothetical full
Gaussian prior).

## What I propose for the figure

Update Figure 8 (currently CRLB vs NUTS) to show **three bars** per diffusivity:

1. Classical unconstrained CRLB (no prior; the theoretical floor without information beyond data)
2. Bayesian CRLB (van Trees, HalfNormal prior treated as Gaussian)
3. Empirical NUTS posterior std

Caption framing: "The unconstrained CRLB diverges for intermediate-D components
because the design matrix is severely ill-conditioned (cond(U)=2.8×10⁵). The HalfNormal
prior contributes ~2 orders of magnitude of regularization (Bayesian CRLB), and the
non-negativity constraint together with the one-sided prior support contributes a
further ~1 order of magnitude (NUTS empirical std)."

## Code

Diagnostic script (paste-runnable):

```python
import numpy as np

b = np.array([0,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500])/1000.0
D = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
U = np.exp(-np.outer(b, D))
SNR = 303
lam = 0.1                          # HalfNormal prior strength
F_data = (SNR**2) * (U.T @ U)
F_post = F_data + lam * np.eye(8)
crlb_unc = np.sqrt(np.diag(np.linalg.inv(F_data)))
crlb_bay = np.sqrt(np.diag(np.linalg.inv(F_post)))
```

cond(U) ≈ 2.8×10⁵, so F_data is rank-deficient in practice — inverting it directly is what
inflates the unconstrained CRLB. Adding λI = 0.1·I regularises the inverse, which is
exactly what the HalfNormal prior is doing inside NUTS.

## Two questions for Sandy

1. Are you comfortable with the van Trees Gaussian-approximation framing, or would you
   prefer to compute the constrained CRLB (Gorman & Hero 1990; or the Bayesian one-sided
   CRLB)? The latter is cleaner but materially more work.
2. The current Eq. 5 has been rewritten in Theory.tex to express MAP as the constrained
   QP `argmin_{R≥0} ||UR - s/S0||² + λ||R||²` solved via NNLS on the augmented system.
   Please review when you have a moment. Diff is in `paper/sections/theory.tex` lines 65–80.

— Patrick (drafted with Claude assistance 2026-05-26)
