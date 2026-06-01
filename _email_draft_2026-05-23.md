Subject: Short list of discussion points for tomorrow

Hi all,

A handful of items to walk through tomorrow — none requiring a written reply, just flagging in advance so we use the meeting time well.

**1. Framing.** Stephan, your "why ADC works" angle and the "compartment-volume mechanism" framing I had drafted feel like the same thing relabelled. Where does the second add anything the first doesn't? I'd like to settle on one before I touch prose.

**2. MAP demotion was premature — partial result already in.** I ran the λ sweep I should have run before forwarding the 2026-05-16 finding. Headline: at tuned λ ≈ 1e-3 to 1e-4 (versus the manuscript's λ = 0.1), MAP recovers ~98% of mass on log-normal ground truths (essentially equal to NUTS), and closes most but not all of the gap on bimodal and δ ground truths. On the actual BWH cohort, tuning λ from 0.1 to 1e-4 closes the "MAP underestimates D=3.0 (lumen) by 51%" claim down to 12% in PZ-normal, and to ~1% in TZ-normal. **The "MAP smearing" Discussion paragraph as currently written is mostly an artifact of λ = 0.1.** The irreducible NUTS contribution is per-bin posterior σ_j (not point-estimate accuracy). Want your reads on whether MAP-tuned should become the primary point estimator in the revised manuscript with NUTS retained for the uncertainty story.

**3. ADC sensitivity ≈ inverted LR coefficients — also checked, also regulariser-dependent.** I re-ran the Fig 4 vector correlation across the same λ sweep. Sanity check: MAP λ=0.1 PZ tumour reproduces the published r=−0.979 exactly. But at the BWH-tuned MAP λ=1e-4 (where MAP recovers the spectrum properly), r drops to −0.85; at NUTS r = −0.79. The "near-perfect" −0.98 only holds in a narrow band of moderately-high λ where MAP smears mass into intermediate bins, which makes the LR coefficient profile artificially smooth and monotonic-in-D (i.e. similar to ADC's sensitivity by construction). The directional claim ("ADC anti-aligned with the discriminant direction") survives at all settings; the *magnitude* doesn't. I want your read on whether to soften Fig 4, rebuild on NUTS with the messier r ≈ −0.80, or drop the panel and lead the mechanism explanation with the univariate Spearman ρ profile from the §10f work.

**4. Why 8 bins when only 2 carry tumour-vs-normal info?** Stephan, your fixed-grid-vs-free-floating-biexp framing is the natural answer — the 8-bin grid is the *test*, the 2-bin collapse is the *result*. I want to make sure I am stating it the way you intended, and that it stands up against the existing biexp / discrete-MC prostate literature.

**5. Half-normal prior on R — checked, prior is not the issue.** I re-ran NUTS on 7 representative ROIs at σ_R ∈ {3.16, 10, 30, 100} (i.e. up to a 30× wider R prior than the manuscript baseline). Posterior R_mean at every bin moves by ≤3% across the full σ_R range; intermediate-bin CVs stay at 0.75–0.88. The "intermediate bins not identifiable" finding is robust to prior choice — a genuine data limit at our b-grid and SNR, not a half-normal artifact. Flagging in case anyone wanted to revisit; no action needed.

**6. Fig 7 directional data.** Stephan, your 2026-05-22 note re-asked for ROI-level directional data — I had used your tarball back in May to build the current Fig 7 (one normal + one tumour ROI from patient 9283, per-direction NUTS + MAP). Does that already match what you had in mind, or do you want a different cut?

**7. Sandy** — no response yet on the 2026-05-16 email. Even one line on whether the MAP-smearing / 2-bin collapse / classifier-comparison findings sit OK with you would help me know where to focus.

**New citations I plan to lean on more heavily in the rewrite:** Wang Y. 2024 (Abdom Radiol — mechanism precedent), Sjölund 2018 (NeuroImage — Bayesian-dMRI uncertainty quantification, brain), Conlin 2021 (JMRI — prostate MC without posterior uncertainty), plus the Chatterjee / Bourne / Sabouri lineage you both know better than I do. Happy to bring a one-pager.

Best,
Patrick
