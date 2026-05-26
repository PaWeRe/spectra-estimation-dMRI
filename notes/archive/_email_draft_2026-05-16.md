Hi all,

Last week I re-visited the ROC curves and AUC and specifically why individual diff buckets performed better than the optimized 8-bin spectral classifier.

Reasoning behind this was that our main claim so far was based on "LR coefficients from the MAP estimated spectrum almost perfectly anticorrelated with ADC sensitivity vector" and we used this observation to explain "Why ADC works".

Looking at the coefficients from the 8-bin MAP LR classifier, the biggest "weight" for the classification task was placed on the .5 diff bucket - this bucket happened also to be amongst the most poorly identifiable buckets in the NUTS spectrum estimate (see high coefficient of variation), pretty fishy (as this seemed to indicate that the high correlation of the coefficients might arise from our choice of regularisation and NOT the actual data).

Main finding when recomputing the spectra and trying out different classifiers (to rule out bad spectra and bad classifiers) was that the MAP coefficients are "smeared" by our regulariser (as feared) … so the correlation that we've been seeing was not (only) due to the data. To prove this rather than just suspect it, I ran a ground-truth simulation: 9 known spectra (δ-spectra at each bin, canonical bimodal tumor/normal, trimodal, log-normals) × 5 SNR levels × 30-100 noise reps. MAP loses 0.34-0.76 of the mass from each δ-bin and dumps it into 2-3 neighboring bins; NUTS recovers them with bias ≤0.10. MSE for NUTS is 50-3000× lower than MAP on concentrated ground truths. The MAP-D=0.50 dominance we saw in the BWH data is reproduced exactly when ground truth is at D=0.25 or D=0.75. I also ruled out "maybe LR is just the wrong classifier" by trying random forest, gradient boosting and SVM-RBF (LOOCV, same protocol) - none beats the 2-feature LR by ΔAUC ≥ 0.05 on any task.

Also I found that the 8-bin NUTS classifier can effectively be collapsed to a purely 2D spectrum (with only .25 and 3.0 fractions) without losing any tumor classification performance, making the rest of the bins irrelevant for the tumor differentiation. Concretely: 2-feature NUTS-LR on {D=0.25, D=3.00} gives PZ AUC 0.933, TZ AUC 0.937 vs. ADC raw rank PZ 0.951, TZ 0.979 - spectral matches ADC within ~0.02-0.04, never beats it. This brings us back to the start and the biexponential models apparently being sufficient to differentiate prostate tissue and the reason why ADC is so good (describes the 2D subspace sufficiently).

One thing I also checked - whether the wide NUTS posteriors are an inference artifact rather than a data limit. Compared NUTS-inferred σ against the legacy formula σ = 1/(√(v_count/16)·150) and against MAP-fit residuals on all 149 ROIs, then refit 5 representative ROIs with σ pinned at the formula. NUTS σ is being pulled *down* from the prior median by the data (not up), and pinning σ at the legacy formula narrows outer-bin posteriors modestly but doesn't unlock the middle bins. So σ calibration isn't the bottleneck.

While the clinical-translation framing weakens, there might still be a way to save this work by refocusing on the Bayesian framework itself - the identifiability analysis with per-bin posterior uncertainty (new in prostate dMRI as far as I can tell), the MAP-smearing finding (methodological caveat useful for anyone fitting discrete spectra in the future), and the joint inference of the noise level. Even though we seem to not need the 8 bins for the classification we are still able to reconstruct them (the simulations show we recover the spectrum well even if for the intermediate bins the std is almost as big as the mean) and "show" that 2 bins are actually enough.

I am not sure how much value this adds to the community - it feels a bit like stating the obvious and so would appreciate both of your views on the matter. In particular, are there recent highly-cited prostate-DWI methods papers from the last few years we should be positioning ourselves against?

Finally, there remains the "circularity" concern that I think we might have underestimated before - we classify ROIs that were drawn by humans based on ADC and then comparing ADC with our spectral classifier again (if there were any microstructural things our method sees that ADC misses we at least could not use the current dataset to prove or disprove this very convincingly…). What we would ideally need are full histopathological information on the entire gland and then see if we can predict the map better than ADC can…but this seems to be out of our reach at the moment … the current 29 cases with Gleason scores seem insufficient for deriving any statistically meaningful conclusion unfortunately. The natural next study would be whole-prostate extended-b imaging with prostatectomy / histopathology ground truth - but that's a different project, not something we can retrofit onto this dataset.

Best,
Patrick

---

P.S. — here's the literature that's now landed on my radar from this re-analysis. Would appreciate your take on whether you know these and whether they belong in our discussion:

**Most directly relevant — mechanism / "MC ≈ ADC" precedent:**
- **Wang Y. et al. 2024**, *Abdom Radiol*, doi:10.1007/s00261-024-04684-z — explicitly states that MC parameters correlated with ADC "may explain limited improvements in AUC". This is the closest published version of our mechanistic claim. I have the PDF.
- **Wang Q. et al. 2018**, *JMRI*, PMID 29812977 — meta-analysis pooling IVIM / DKI vs ADC, concluding AUCs are "comparable, not superior".
- **Quigley D.J., Mitchell D.G. 2023**, *Eur J Radiol*, PMC10623580 — review with the "more accurate signal decay does not imply higher sensitivity" framing.

**Closest methodological precedent (Bayesian dMRI):**
- **Sjölund J. et al. 2018**, *NeuroImage*, PMC6419970 — Bayesian inference for dMRI with posterior uncertainty, but in brain, not prostate. Closest published thing I could find to what we're doing.

**Closest prostate multi-compartment work (different methodology, same problem):**
- **Conlin C.C. et al. 2021**, *JMRI*, doi:10.1002/jmri.27393 — BIC model selection in prostate, no posterior uncertainty.

**Counterpoint we have to address (papers claiming MC > ADC):**
- **RSI / RSIrs** (Conlin, Karow, Liss, Zhong et al., recent *J Urology* / *Radiology*) — report csPCa AUCs of 0.73-0.78 vs ADC AUC of only 0.48-0.54. Their ADC implementation differs from ours (likely high-b only, not PI-RADS-compliant b ≤ 1000), but the disagreement needs to be acknowledged.
- **Singh M. et al. 2022**, *Radiology*, doi:10.1148/radiol.212536 — VERDICT INNOVATE, N=303, modest improvement over ADC.

Do either of you know more recent (2023-2025) prostate-DWI methods papers in this space? I'd rather not find out after submission that we missed something obvious.
