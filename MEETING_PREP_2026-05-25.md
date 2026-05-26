# Meeting prep — 2026-05-25, Sandy + Stephan

**Purpose:** Patrick's personal refresher. Each item is a Q he raised plus the cleanest available A, so the meeting can stay at the level of decisions rather than re-derivation.

Anchor numbers and locations live in `PROJECT_STATE.md` §1–3.

---

## TL;DR — walk into the meeting with this in your head

### The reframed bold claim (test it on Stephan first)

> **At a clinically-feasible prostate dMRI acquisition (15 b-values to 3500 s/mm², SNR ~300), model-free Bayesian spectral inversion supports 2 independent compartment fractions. ADC is a near-equivalent scalar projection of that 2-D information: at ROI level, ADC and the optimal-for-this-cohort linear spectral classifier rank patients identically (r ≈ −0.98, n=81/68, both NUTS and tuned MAP). This is "why ADC works" — not because ADC is mysteriously clever, but because the signal genuinely contains only 2 dimensions at this acquisition. Recovering 3+ compartments requires structural priors as in VERDICT / rVERDICT / HM-MRI / RSI; our work characterises the information ceiling without such priors.**

This claim is *stronger and more defensible* than the original manuscript thesis. The "MAP-smearing" and "sensitivity-vector elegance" framings are gone; in their place, three triangulating findings: F2 (intermediates don't add classification signal), F8 (data limit not prior), and the ROI-scalar r ≈ −0.98 (robust across methods).

### The 3 binary decisions for today (priority-ordered)

1. **Narrative spine** — adopt the reframed claim above? If yes, abstract / intro / discussion rewrites unblock.
2. **Demote grading to supplement** — N=29 cannot support a co-pillar Detection-vs-Grading axis story. Report Spearman ρ table in SI, one sentence in Discussion, no ROC plots. Agreed?
3. **Cut Fig 4 and Fig 5; promote simulation to main as F-new-1** — both ex-figures were λ=0.1 artifacts; simulation now needs main-text status to justify "tuned MAP ≈ NUTS for point estimates, NUTS adds only per-bin σ." Agreed?

If items 1–3 land, the rest of the meeting can confirm Fig 7 directional data with Stephan and discuss SI structure.

### Draft figures being generated in parallel (3 agents, ~1h each)

So the meeting has tangible artifacts to react to rather than abstractions:

- **Fig 1 v2** — 8 representative ROIs, tuned MAP (λ=1e-3) vs NUTS, CV-coloured bars → `paper/figures/fig1_v2.png`
- **Fig 3 v2** — ADC vs spectral discriminant scatter, NUTS + tuned MAP × PZ/TZ, bootstrap CI → `paper/figures/fig3_v2.png`
- **F-new-1** — promoted simulation λ-sweep, log-normal GTs highlighted, NUTS reference → `paper/figures/fnew1_simulation.png`

Check these files before walking into the meeting.

### Final figure plan (target 8–9 main figs, MRM cap satisfied)

| Fig | Main message (one sentence) | Status |
|---|---|---|
| **1** | Bayesian spectrum for representative tumour/normal ROIs; outer bins shift, intermediates wide | Update v2 (NUTS + tuned MAP, agent in flight) |
| **2** | 2-feature outer-bin NUTS-LR matches ADC for tumour detection (PZ 0.93 vs 0.95; TZ 0.94 vs 0.98) | Re-plot with NUTS 2-feat after meeting |
| **3** | ADC and spectral discriminant are scalar-equivalent (r ≈ −0.98) — the "Why ADC works" quantitative anchor | Update v2 (agent in flight) |
| ~~4~~ | DROP — vector-correlation died with F4b | Cut |
| ~~5~~ | DROP — MAP/NUTS divergence dissolves at tuned λ (F1) | Cut |
| **6** | Per-ROI classifier predictions carry NUTS-derived uncertainty; misclassified ROIs cluster at decision boundary — the unique NUTS contribution | Keep, enlarge crosses |
| **7** | Per-direction NUTS confirms ROI-level pooling is direction-robust | Confirm Stephan's tarball data covers this |
| ~~8~~ | Move to SI — Fisher/CRLB not load-bearing in new narrative | SI |
| **9** | Pixelwise feasibility demo, MAP tuned, one slice — feasibility only, not a result | Anonymise, recolor |
| **F-new-1** | Tuned MAP ≈ NUTS on prostate-realistic (log-normal) GTs — methodology honesty | Generating (agent in flight) |
| ~~F-new-2~~ | Demote to SI: per-bin Spearman ρ vs GGG table (no plot at N=29) | SI table |

### Conceptual clarifications resolved today (write these down so you don't get cornered)

1. **Per-ROI posterior CV vs across-ROI biological spread vs classification utility are three different things.** A bin can be precisely estimated per ROI (low posterior σ), highly variable across patients (biological signal), and clinically discriminative — that's the ideal signal-bin profile, e.g. D=3.0 in normal PZ. Don't conflate them when Stephan probes.
2. **The two ADC-vs-spectrum correlations.** (a) ROI-level scalar, n=81/68, r ≈ −0.98, λ-robust → Fig 3, lives. (b) Vector-level sensitivity, n=8, r ∈ [−0.76, −0.98] depending on λ → was Fig 4, dies. The abstract used to muddle these.
3. **"Bayesian" is justified.** Proper priors (HalfNormal R_j, HalfCauchy σ), full posterior via NUTS, posterior summaries reported. F1 (tuned MAP ≈ NUTS on smooth GTs) doesn't undermine this — MAP under Gaussian approximation is just the posterior mode.
4. **"Optimal" is too strong. "Best linear classifier we can identify at this cohort size" is honest.** Quote the DeLong test (all p > 0.17) as the equivalence evidence.
5. **VERDICT/RSI positioning.** We don't contradict them — they recover 3+ compartments *via parametric priors*. We measure the *model-free ceiling*. Two complementary results: parametric methods extract more *if* their priors are right; we measure how much the data alone supports. Reviewers from that camp can't object because we don't claim they're wrong.
6. **Tissue-dependent identifiability (Patrick's Fig 1 observation).** Difference between "D=3.0 green for normals, less green for tumors" is a denominator effect on CV (tumors have less D=3.0 mass, so std/mean inflates). Not a real change in information content — F8 shows the data-limit is bin-dependent, not tissue-dependent. The bin_information_sweep tests the downstream question directly: no subset beats {μ_0.25, μ_3.0} in *any* zone.
7. **D=20 (free water).** Kept as a bin in all classifiers and visualisations (no cherry-picking). Drops out of LR-coef decomposition only because D=20 numerically dominates R_j × D_j arithmetic. Mention this in Methods.

---

## Q1. What was the simulation, and is it representative?

**Setup** (`scripts/simulation_study.py`):
- 9 ground-truth spectra: 4 δ-spectra (δ at D = 0.25, 0.50, 0.75, 3.0); 2 bimodal (tumour-like {0.7, 0.3} and normal-like {0.3, 0.7} at the two outer bins); 1 trimodal; 2 log-normals (smooth, μ ≈ 0.5 and 1.5).
- SNR sweep: {100, 200, 400, 800, 1500} for MAP; {200, 400, 800} for NUTS.
- N reps per (GT, SNR): 100 for MAP, 30 for NUTS.
- Both estimators originally run at **λ = 0.1**, the value used in the manuscript.

### What we found at λ = 0.1 only (the original simulation)

`results/simulation/sim_summary.csv`, MAP @ SNR=400 on GT-A (δ at D=0.25):
- MAP recovers 0.66 of mass at D=0.25, dumps 0.25 to D=0.50, 0.07 to D=0.75, 0.02 to D=20.
- NUTS recovers 0.99 of mass at D=0.25, all other bins ≤ 0.003.

This is what motivated the manuscript's "MAP smearing" finding.

### NEW 2026-05-24: λ-sweep added (Patrick's pushback in Q1)

`scripts/map_lambda_sweep.py` → `results/simulation/map_lambda_sweep_summary.csv` + `map_lambda_sweep_{fraction,mse}.png`. 6 GTs × 3 SNRs × 10 λs × 100 reps. **Headline: the λ choice mattered a lot. λ=0.1 was substantially suboptimal across every GT shape we tested.**

| GT (SNR=400) | MAP λ=0.1 (manuscript) | MAP at best-tested λ | NUTS λ=0.1 (reference) |
|---|---|---|---|
| δ @ D=0.25 | frac=0.66, MSE=0.187 | **λ=1e-4: frac=0.83, MSE=0.045** | frac=0.99, MSE=0.0005 |
| δ @ D=3.00 | frac=0.30, MSE=0.633 | **λ=1e-4: frac=0.73, MSE=0.118** | frac=0.96, MSE=0.002 |
| bimodal {0.25:0.7, 3.0:0.3} (tumour-like) | frac=0.61, MSE=0.130 | **λ=1e-4: frac=0.80, MSE=0.046** | frac=0.94, MSE=0.004 |
| bimodal {0.25:0.3, 3.0:0.7} (normal-like) | frac=0.42, MSE=0.319 | **λ=1e-4: frac=0.75, MSE=0.075** | frac=0.93, MSE=0.005 |
| log-normal μ=0.5 | frac=0.89, MSE=0.021 | **λ=1e-3: frac=0.985, MSE=0.003** | frac=1.01, MSE=0.018 |
| log-normal μ=1.5 | frac=0.90, MSE=0.022 | **λ=1e-3: frac=0.98, MSE=0.006** | frac=0.98, MSE=0.014 |

**Three things this changes:**

1. **The MAP-smearing finding is partially an artifact of λ=0.1.** For log-normal GTs (the most prostate-realistic shapes), MAP at tuned λ recovers ~98% of mass — **slightly better MSE than NUTS in our numbers** (MAP MSE 0.003–0.006 vs NUTS MSE 0.014–0.018 for the two log-normals).
2. **For δ-spectra, NUTS still wins by a clear margin.** Tuned MAP closes ~70% of the gap (0.66 → 0.83 at GT-A SNR=400) but doesn't reach NUTS (0.99). Real prostate spectra are unlikely to be δ-functions, so this is less load-bearing for the application.
3. **For bimodal "real-prostate-like" spectra (closer to what we actually see in BWH), NUTS retains a real ~10–15 percentage-point advantage** even at tuned MAP λ. Not negligible.

**Implication for the manuscript:**
- The "ridge smoothing" claim in Discussion (MAP underestimates D=3.0 by ~35% in PZ-normal) is contingent on λ=0.1.
- For the meeting, **the cleanest framing** is: "Tuned MAP and NUTS converge on smooth spectra (the relevant case). On concentrated spectra NUTS still wins. The case for NUTS is no longer 'MAP smears 35%' — it's 'NUTS gives per-bin posterior uncertainty MAP cannot provide, and is somewhat more robust on extreme spectra.'"

### NEW 2026-05-24: same λ-sweep applied to the actual BWH cohort

`scripts/map_lambda_bwh.py` re-fits MAP on all 149 BWH ROIs at λ ∈ {0.0001, 0.001, 0.01, 0.05, 0.1, 0.5}, then compares to the NUTS posterior means already in `features.csv`. **This is the cleanest evidence for the meeting: the manuscript's "MAP underestimates lumen by ~35%" claim mostly disappears when MAP is re-fitted at the tuned λ.**

**PZ-normal D=3.0 median (acinar lumen fraction):**
| λ | MAP median | deviation from NUTS median (0.484) |
|---|---|---|
| 0.5 | 0.192 | −60% |
| **0.1 (manuscript)** | **0.239** | **−51%** |
| 0.05 | 0.269 | −44% |
| 0.01 | 0.330 | −32% |
| 0.001 | 0.402 | −17% |
| **0.0001** | **0.426** | **−12%** |

**PZ-tumour D=0.25 mean (epithelial proxy):**
| λ | MAP mean | NUTS mean = 0.166 |
|---|---|---|
| 0.1 (manuscript) | 0.119 | −28% |
| 0.001 | 0.187 | +13% (overshoots slightly) |
| 0.0001 | 0.181 | +9% |

**TZ-normal D=3.0 mean (acinar lumen):**
| λ | MAP mean | NUTS mean = 0.299 |
|---|---|---|
| 0.1 (manuscript) | 0.206 | −31% |
| 0.001 | 0.296 | −1% |
| 0.0001 | 0.296 | −1% |

So at λ ≈ 1e-3 to 1e-4, MAP and NUTS are within a few percentage points on the well-identified bins across the full cohort. The intermediate bins (D=0.50, 0.75, 1.00) collapse toward zero under tuned MAP — which is **the same behaviour NUTS shows**, just expressed deterministically.

**What this means for the manuscript narrative:**

1. The current Discussion claim that MAP-NUTS divergence is "a ridge smoothing artifact" is correct in mechanism but the *magnitude* is exaggerated 4–5× by the λ=0.1 choice.
2. **The "MAP-LR coefficient profile ≈ inverted ADC sensitivity vector" elegance (r ≈ −0.98) was re-examined at tuned λ — see Q9 below.** Result: at MAP λ ≈ 1e-4 r drops to −0.85, at NUTS r = −0.79. Most of the elegance was the same regulariser-smearing effect that drove F1.
3. **MAP becomes a legitimate primary inference method again, with a tuned λ.** NUTS retains *one* unique contribution: per-bin posterior uncertainty σ_j. That's the irreducible Bayesian gain — not "better point estimates."
4. The Fig 5 "MAP vs NUTS divergence at D=3.0" panel essentially loses its punch under tuned MAP.

**Concrete suggestion for the meeting:** lock the manuscript on tuned MAP (λ ≈ 0.001) as primary point estimator, use NUTS posterior std for the uncertainty story. Drop the "MAP smears 35% of lumen mass" claim. This is closer to where Stephan was originally headed before the rabbit hole.

Files: `results/biomarkers/map_lambda_bwh.csv`, `results/biomarkers/map_lambda_bwh_summary.csv`.

---

## Q2. What did the LR-coefficient decomposition (Exec 4P1) test, and why?

**The hypothesis:** if ADC is "the right scalar projection of the spectrum" for tumour detection, then the LR coefficient vector w trained on the 8 spectral fractions should be approximately proportional to the diffusivity vector D = [0.25, 0.5, ..., 3.0, 20.0]. (Because ADC ≈ ∑ R_j · D_j — a weighted mean — so the "discriminant direction" should look like a constant times D.)

**What we did** (`scripts/lr_coef_decomp.py`):
- Fitted 8-feature LR on NUTS spectral fractions, twice per task: once for tumour-vs-normal, once for GGG ≥ 3.
- Computed cos(w, D_vec) and cos(w_T, w_G).
- Bootstrap CIs over ROIs (1000 resamples).

**Numbers** (7-bin tissue-only, D=20 stripped because the free-water bin dominates everything numerically):
- pooled cos(w, D_vec) detection: +0.14 [−0.04, +0.25]
- pooled cos(w, D_vec) grading:   −0.35 [−0.60, +0.19]
- pooled cos(w_T, w_G) cross-task: +0.34 [−0.17, +0.75]

**Interpretation:**
- "Detection LR ∝ D_vec" — not clearly supported. CI straddles 0. Mildly negative for Path A' headline ("ADC is the right scalar projection").
- "Detection and grading axes orthogonal in LR-coefficient space" — also not supported. cos(w_T, w_G) is modest-positive with very wide CI.
- The LR-vector version of the axis-separation claim is not clean at N=29 with 8 collinear bins. **The univariate Spearman ρ profile (Q4 below) is the clean evidence.**

---

## Q3. The bin-information sweep — what did it test?

**Setup** (`scripts/bin_information_sweep.py`, `results/biomarkers/bin_information_sweep.csv`, 1782 rows):
- Trained LR on every subset of the 8 bins (2-bin, 3-bin, intermediate-only, σ-only, μ+σ combinations, etc.) — 198 feature sets.
- Three classification tasks × three regularisation strengths C ∈ {0.1, 1.0, 10.0}.
- AUC with bootstrap CI for each.

**Reference:** {μ_D=0.25, μ_D=3.00} → PZ AUC 0.933, TZ 0.937, pooled 0.899.

**Decision rule:** any feature set with ΔAUC > 0.01 over reference AND non-overlapping CI → revision candidate.

**Result:** **zero feature sets met the threshold.** Three findings worth quoting:
1. Intermediate-only LR collapses (ΔAUC ≈ −0.13 in every zone). Direct test that intermediate bins carry no independent tumour-vs-normal signal.
2. TZ triple {0.25, 0.50, 3.00} gives Δ +0.014, CI overlaps reference. Don't elevate.
3. {μ_D=0.25, σ_D=0.25} on TZ marginally helps (Δ +0.010, CI overlaps). σ_D=0.25 contains signal in TZ specifically.

**Method-vs-data question Patrick raised:** are the intermediate bins uninformative because (a) the data don't constrain them, or (b) the half-normal prior is dragging them down? Answer: the bin-information sweep tests (a) directly. It shows intermediates *don't add classification signal even when you include them* — which is a fact about the data + features, independent of the prior. The wide posterior CV is a separate (related but distinct) finding about per-bin identifiability.

---

## Q4. The continuous-GGG Spearman sweep — what did it test?

**Setup** (`scripts/ggg_continuous_sweep.py`, `results/biomarkers/ggg_continuous_sweep.csv`):
- 29 tumour ROIs with valid GGG (PZ 21, TZ 8, low=20, high=9).
- For each spectral feature (μ_D=0.25, μ_D=0.50, ..., σ_D=0.25, ..., 16 features) compute Spearman ρ vs continuous GGG (1–5 ordinal).
- Bonferroni α = 0.05 / 16 = 0.0031.

**Pooled (N=29) Bonferroni-significant findings:**
- **μ_D=0.50: ρ = +0.565**, p = 0.0014
- σ_D=0.50: ρ = +0.566, p = 0.0014
- ADC: ρ = −0.546, p = 0.0022

Strong but below Bonferroni: μ_D=0.75 +0.512, σ_D=0.25 +0.499, μ_D=2.0 −0.474, μ_D=0.25 +0.437.

**What it means:** detection lives at outer bins (D=0.25 ↑, D=3.0 ↓ for tumour); grading lives at intermediate (D=0.50 ↑ with higher GGG) and lumen (D=2.0 ↓ with higher GGG) bins. These are *univariate* per-bin correlations, so they don't suffer the bin-collinearity issue that wrecked the LR-vector decomposition.

This is the clean axis-separation evidence — proposed F-new-2 panel.

---

## Q5. The diagnostic triangulation — what is it and how much should I push it?

The headline "spectrum is diagnostically equivalent to ADC at N=29" is supported by four independent tests (`PROJECT_STATE.md` §3 F5):
1. **Partial Spearman ρ(μ_D=0.50, GGG | std-ADC) = +0.42**, p = 0.026 *uncorrected* — does not survive Bonferroni. "After residualising out ADC, μ_D=0.50 has a small residual signal vs GGG that we can't claim is real at N=29."
2. **2-feature LR + paired DeLong test (`scripts/two_feature_lr_vs_adc.py`)**: every {ADC + one spectral feature} model has ΔAUC ≤ 0 vs ADC alone, all DeLong p ≥ 0.17. "No spectral feature adds AUC."
3. **9-variant ADC sweep (`scripts/adc_variants_sweep.py`)**: std-ADC, ext1500, ext2000, full-15-b, high-b-only, midrange, DKI_D, DKI_K, spec_M1. DKI_D narrowly best (+0.003 detection, ρ=−0.555 grading); spec_M1 is much worse (PZ 0.71 vs std-ADC 0.94). "Spectrum-then-collapse is information-lossy as a scalar."
4. **Spectrum first moment**: see (3).

**How much to push:** for the meeting, one sentence — "we have four independent tests that all agree we can't beat std-ADC at N=29 for grading." Don't dive deep unless asked. The deeper value of the triangulation is that it forecloses "what if you'd used DKI as the reference?" reviewer pushback.

---

## Q6. The half-normal prior — is it pulling intermediate bins to zero?

**Setup recap:** `R_j ~ HalfNormal(σ_R = 1/√λ)` with λ = 0.1, so σ_R ≈ 3.16.

**Patrick's concern:** the HalfNormal pulls mass toward 0. If intermediate bins are pinned near 0 by the prior, their high posterior CV (std/mean) is partly a prior artifact (small mean), not just a data-doesn't-constrain story.

**Earlier σ-prior sanity check (2026-05-16):** Patrick refit 5 representative ROIs with σ pinned at Stephan's legacy formula. NUTS σ posterior is being pulled *down* from the HalfCauchy prior median (so data overrides prior on σ). Pinning σ tighter narrows outer-bin posteriors 20–40% but does not unlock the middle bins. Conclusion: σ calibration is not the bottleneck for intermediate-bin identifiability.

### NEW 2026-05-24: R-prior sanity check (Patrick's pushback in Q6)

`scripts/wider_prior_check.py` — re-ran NUTS on 7 representative ROIs across PZ/TZ × tumour/normal at σ_R ∈ {3.16, 10, 30, 100}, i.e. up to a **30× wider R prior** than the manuscript baseline.

**Headline: the prior is NOT doing the shrinkage. The data are.**

Aggregate ratio of (wider-prior R_mean) / (baseline-prior R_mean), averaged across the 7 ROIs:

| bin D | σ_R=10 | σ_R=30 | σ_R=100 |
|---|---|---|---|
| 0.25 | 0.995 | 0.995 | 0.995 |
| 0.50 | 1.028 | 1.024 | 1.016 |
| 0.75 | 0.984 | 0.990 | 1.001 |
| 1.00 | 0.991 | 0.985 | 0.992 |
| 1.50 | 0.983 | 0.979 | 0.971 |
| 2.00 | 1.010 | 1.003 | 1.016 |
| 3.00 | 0.990 | 0.993 | 0.985 |
| 20.00 | 1.018 | 0.999 | 1.016 |

**Every bin moves by ≤3% across the full σ_R range.** Intermediate-bin CVs stay at 0.75–0.88; outer-bin CVs at 0.05–0.27. Example, ROI `new02_pz_tumor`:

| σ_R | D=0.25 | D=0.50 | D=0.75 | D=1.0 | D=1.5 | D=2.0 | D=3.0 | D=20 |
|---|---|---|---|---|---|---|---|---|
| 3.16 (baseline) | 0.251 | 0.043 | 0.046 | 0.065 | 0.114 | 0.180 | 0.251 | 0.051 |
| 10 | 0.251 | 0.043 | 0.048 | 0.066 | 0.114 | 0.172 | 0.257 | 0.050 |
| 30 | 0.249 | 0.046 | 0.047 | 0.064 | 0.111 | 0.176 | 0.257 | 0.050 |
| 100 | 0.251 | 0.043 | 0.046 | 0.065 | 0.113 | 0.179 | 0.252 | 0.051 |

**Implication for the manuscript:** the "intermediate bins not identifiable" Discussion point is robust to prior choice. The wide posterior CV is a genuine consequence of the b-grid + SNR not constraining intermediate D values, not an artifact of the half-normal pulling mass toward zero. F8 closes.

Files: `results/simulation/wider_prior_check.csv`, `results/simulation/wider_prior_check_summary.csv`.

---

## Q7. "Calibrated per-bin uncertainty" — what does that mean and how do we calibrate?

**The claim** in plain English: NUTS gives us a posterior std σ_j for each bin. We want σ_j to mean what frequentists would call a confidence interval — i.e. "if you ran the experiment many times, the 90% interval covers the truth 90% of the time."

**How we check it:** the ground-truth simulation (Q1). For each (GT, SNR, bin) compute the 90% NUTS CI and ask: does the true R_j fall inside?

**Results** (from `project_nuts_coverage_caveat`, all at SNR = 400):
- δ-spectra (GT-A through GT-D): coverage ≈ 0.00 — basically never.
- Bimodal (GT-E, GT-F): coverage ≈ 0.01.
- Trimodal (GT-G): coverage 0.07–0.12.
- Log-normals (GT-H, GT-I): coverage 0.77–0.87 — pretty close to the nominal 0.90.

**So "calibrated" is conditional:** NUTS posteriors are well-calibrated on *smooth* GTs (which is realistic for prostate where the true spectrum is broadly distributed). They are over-confident on concentrated δ-GTs (the bias and posterior std together don't cover the truth).

**Honest framing for the manuscript:**
- Don't say "calibrated 90% credible intervals." Say "calibrated on smooth ground truths; over-confident on δ-spectra."
- The bias and MSE story is solid (NUTS 50–3000× better than MAP on δ/bimodal GTs at SNR=400). The coverage story needs hedging.
- Goes in Discussion as a limitation, not as a reason to abandon NUTS.

---

## Q8. "Compartment-volume mechanism" — is this defined or just vibes?

**Honest answer: it's a phrase pointing at a real mechanism that exists in the literature, but the manuscript hasn't yet defined it precisely.**

The mechanism, in one sentence: prostate tissue has three principal water compartments — restricted intracellular water in epithelium, hindered extracellular water in stroma, free water in glandular lumen — with characteristic D ranges established ex-vivo by Bourne 2018 (D ≈ 0.3–0.5 epithelium, 0.7–0.9 stroma, 2.0–2.2 lumen). Chatterjee 2015 showed compartment *volume fractions* correlate with Gleason more strongly than cellularity (epithelium ρ=+0.898, lumen ρ=−0.912 vs Gleason pattern, n=553 subimages). So changes in tissue grade manifest as **shifts in compartment volume fractions** at largely fixed compartment-specific D values.

Our 8-bin grid is a discrete approximation that brackets those three compartments plus a free-water bin. The "fraction shift at canonical D" framing Stephan endorsed is literally this: our fixed grid forces the model to express tissue change as a *shift in R_j* rather than as a drift in fitted *D_j* (which is what free-floating biexp does).

**Action:** the manuscript needs a single paragraph in Discussion that says exactly this, citing Chatterjee 2015 + Bourne 2018. Until that paragraph is written, "compartment-volume mechanism" is indeed under-specified vibes. Worth flagging this to Stephan tomorrow.

**On "is it the same as 'why ADC works'?":** essentially yes, but with one extra step. "Why ADC works" = "ADC averages over compartment volumes that all shift in the grade-relevant direction" (the 1-D manifold of tissue change). "Compartment-volume mechanism" = the underlying 3-compartment volume picture that ADC is averaging over. The first is the consequence; the second is the cause.

---

## Q9. ADC sensitivity ≈ inverted LR coefficients — the locality question

**The code** (`src/spectra_estimation_dmri/biomarkers/recompute.py:508` `adc_sensitivity_analysis`):
1. Compute `avg_tumor` and `avg_normal` spectra (mean over all ROIs in zone).
2. Fit one LR on all ROIs in zone (81 PZ, 68 TZ) → one 8-element coefficient vector w.
3. Compute ADC sensitivity ∂ADC/∂R_j at two operating points: at `avg_tumor` and at `avg_normal`. Each gives an 8-element vector.
4. Pearson and Spearman correlation between w (8 elements) and each sensitivity vector (8 elements).

**So the correlation is over 8 bins,** evaluated at one operating point at a time. The "n" is *not* 29 (that's the GGG analysis) and *not* 149 (that's the ROI-level scalar correlation in the abstract). It's a comparison of two 8-element vectors, twice per zone.

**Two distinct "ADC vs spectrum" correlations in the current paper** that easily get muddled:
- **(i) ROI-level scalar correlation:** ADC value vs spectral-discriminant score, per ROI. n = 81 PZ / 68 TZ. r ≈ −0.98 with bootstrap CI. This is what the abstract claims.
- **(ii) Vector-level profile correlation:** 8-element sensitivity ∂ADC/∂R_j vs 8-element LR coef w. r = −0.979 (MAP) / −0.788 (NUTS), evaluated at one operating point per zone. No bootstrap. This is Fig 4.

These are different claims! Patrick's confusion was legitimate — the manuscript text in `results.tex` and `discussion.tex` blurs them.

**Why NUTS gives weaker vector correlation:**
- MAP smears mass into intermediate bins (Q1) → LR fitted on MAP features has non-zero coefficients on intermediate bins → coefficient profile is *smoother* across D → looks more like ADC's monotonic-in-D sensitivity → r ≈ −0.98.
- NUTS gives near-zero estimates at intermediate bins → LR coefficients pile onto outer bins → coefficient profile is *peakier* → less like the monotonic sensitivity → r ≈ −0.79.

**Implication for the meeting:** the elegant MAP version is partly an artifact of the smearing. The NUTS version is the truth about the data + classifier, and it's less visually striking. That's the decision Patrick wants on the table — keep the elegant MAP figure with a caveat? Rebuild on NUTS and accept the messier picture? Or drop the panel entirely and lead the mechanism explanation with the univariate Spearman ρ profile from Q4?

### NEW 2026-05-25: rerun across the λ sweep (`scripts/adc_sensitivity_at_tuned_lambda.py`)

For each MAP λ value, refit LR on the MAP-at-that-λ features and re-correlate the LR coef vector with ∂ADC/∂R. Sanity check: MAP@λ=0.1 PZ tumour reproduces the paper's r=−0.979 exactly, so the pipeline is correct.

| Estimator | PZ r_tumor | TZ r_tumor | LR coef profile (D=0.25 → 20.0) |
|---|---|---|---|
| MAP λ=1e-4 (≈ optimum on BWH) | **−0.853** | **−0.801** | strongly peaked: +1.50 +0.83 +0.91 +0.18 −0.22 +0.46 **−1.36** −0.26 |
| MAP λ=1e-3 | −0.755 | −0.861 | strongly peaked |
| MAP λ=1e-2 | −0.966 | −0.947 | smoother |
| MAP λ=5e-2 | −0.980 | −0.973 | smooth monotonic-in-D |
| MAP λ=0.1 (manuscript) | **−0.979** | −0.931 | smooth monotonic-in-D: +0.71 +0.55 +0.40 +0.07 −0.36 −0.44 −0.55 −0.76 |
| MAP λ=0.5 | −0.947 | −0.849 | smooth, but less weight at outer bins |
| **NUTS** | **−0.788** | **−0.883** | concentrated at outer: +1.66 +0.30 +0.26 +0.25 +0.32 +0.23 **−1.03** −0.03 |

**The r=−0.98 elegance is a property of MAP at λ ∈ [1e-2, 1e-1].** Outside that window — including at the BWH-tuned λ ≈ 1e-4, where MAP best recovers the true spectrum — the correlation drops to roughly −0.80, matching NUTS.

**Why this happens (mechanism):** MAP at moderately high λ smears mass into intermediate bins, which makes the LR coef profile across D smoother and *more monotonic in D*. ADC sensitivity ∂ADC/∂R_j is by construction monotonic in D (negative at low D, positive at high D). A smoother coefficient profile correlates more strongly with that monotonic shape. As you tune λ smaller (or move to NUTS), the LR coefficient profile becomes peakier — concentrated at D=0.25 and D=3.0 — which is what the data + classifier actually want. That peaky profile is *less correlated* with the smooth monotonic ADC sensitivity, even though both methods still anti-correlate.

**Three honest framings for the manuscript:**

1. **Conservative (Patrick-leaning).** "The published r ≈ −0.98 was a property of the regularisation choice. At well-tuned MAP and at NUTS, r ≈ −0.80, still consistent with the spectrum being approximately ADC-projection-like but no longer 'near-perfect'." Drop or soften Fig 4.

2. **Medium.** "ADC's per-bin sensitivity is anti-correlated with the LR discriminant direction across a wide range of regularisation strengths (PZ r in [−0.99, −0.76]). The strength of the alignment depends on how concentrated the spectrum estimate is." Keep Fig 4 with NUTS panel + caveat.

3. **Permissive (current manuscript).** Keep MAP λ=0.1 as the reported number, add a sentence acknowledging the regularisation dependence.

Files: `results/biomarkers/adc_sens_vs_lr_tuned_lambda.csv`.

---

## Q10. What is actually new since the 2026-05-16 email?

For the meeting, here's the delta versus what Stephan + Sandy already know:

**Already in 2026-05-16 email (no need to re-explain):**
- Ground-truth simulation: 9 GTs × 5 SNRs × 30–100 reps; MAP loses 0.34–0.76 of δ-mass, NUTS recovers it; MSE 50–3000× better for NUTS.
- 2-feature NUTS-LR on {D=0.25, D=3.00} matches ADC within 0.02–0.04 AUC.
- Random forest, gradient boosting, SVM-RBF all fail to lift AUC over the 2-feature LR by ≥ 0.05.
- SNR-pinning experiment: NUTS σ is being pulled down from prior median by data; pinning σ doesn't unlock middle bins.
- Circularity concern flagged (Stephan dismissed).
- Wang Y 2024, Wang Q 2018, Quigley 2023, Sjölund 2018, Conlin 2021, RSI/RSIrs, VERDICT INNOVATE.

**NEW since 2026-05-16 (worth surfacing tomorrow):**
1. **Exec 4P1** — LR-coefficient decomposition onto D_vec (Q2 above). Result: detection LR is not a clean first-moment projection at N=29; axis-orthogonality from LR weights does not hold cleanly.
2. **Bin-information sweep** — direct test that intermediate-only LR collapses (Q3).
3. **Continuous-GGG Spearman sweep** — μ_D=0.50 ρ=+0.565 Bonferroni-sig, axis separation in univariate ρ profile (Q4).
4. **ADC-variants sweep** — DKI_K ρ=+0.476 vs GGG (p=0.009); std-ADC is fair as primary reference; high-b ADC is *worse* than std-ADC.
5. **Two-feature LR + DeLong** — no augmentation beats std-ADC.
6. **Partial Spearman ρ(μ_D=0.50, GGG | ADC) = +0.42, p=0.026 uncorrected, not Bonferroni-sig.**
7. **Bourne 2018 + Chatterjee 2015 + Sabouri 2017 pulled in** as biological anchors for the compartment-volume framing.

---

## Open process items for the meeting

- Meeting date / time confirmation. Patrick had been planning 2026-05-25 (Monday); Stephan flexible.
- Patrick has a self-imposed walk-away date of **~2026-05-29** if there is no OK paper by then. This is a real constraint, not aspirational. The decisions made tomorrow set whether 2-week or 1-week submission is feasible.
- Sandy has not responded to the 2026-05-16 email. Best to confirm tomorrow whether she'll be at the meeting or whether Patrick needs to push for written feedback separately.

---

## TL;DR for the 60-minute slot

If only one decision per item, in priority order:
1. **MAP demotion: yes or λ-sweep first?** (Q1) → unblocks Fig 1, Fig 4, Methods rewrite.
2. **ADC-sensitivity panel: keep / rebuild on NUTS / drop?** (Q9) → unblocks Fig 4 + Discussion mechanism paragraph.
3. **Half-normal prior sanity check on intermediate bins: yes or no?** (Q6) → bounds how strongly we claim "intermediates not identifiable" in Discussion.
4. **8-bin grid justification — does Stephan's fixed-grid framing land as I have it (Q8)?** → unblocks Intro + Discussion novelty paragraph.
5. **Confirm Fig 7 satisfies the directional ask** (or rework). → unblocks figure regen pass.

If the meeting only gets through items 1–3, that's already enough to write the Results + Methods rewrite this week.
