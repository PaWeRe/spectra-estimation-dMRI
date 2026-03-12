# Email Draft to Sandy (Stephan CC)

**Subject:** MRM paper — key result + questions

---

Hi Sandy,

Quick update on the MRM paper. Met with Stephan yesterday.

**New result:** The ADC sensitivity vector ∂ADC/∂Rⱼ (how much ADC changes per unit change in each spectral fraction) correlates at r = −0.97 with the learned LR discriminant — evaluated at all four tissue group spectra (PZ/TZ × tumor/normal). This analytically explains why ADC works: it implicitly performs a near-optimal spectral weighting. Key difference: ADC's weighting is nonlinear (tissue-dependent), the spectral discriminant is a fixed linear projection.

**Current numbers** (LOOCV, 56 patients, 149 ROIs):

| | PZ tumor | TZ tumor | GGG 1-2 vs 3-5 |
|---|:---:|:---:|:---:|
| ADC | 0.940 | 0.964 | 0.778 |
| Spectral LR (MAP) | 0.935 | 0.941 | 0.722 |
| Spectral LR (NUTS) | 0.933 | 0.925 | 0.722 |

MAP and NUTS discriminants correlate at r = 0.997. Only D=0.25 is well-identified (CV=0.20); D=0.5–1.0 have CV>0.80. NUTS is ~1000× slower.

NUTS does provide: (1) prediction uncertainty — misclassified ROIs have 2.3× higher uncertainty, (2) per-component identifiability quantification.

**Questions for you:**

1. **Is uncertainty quantification for spectral estimation a contribution the dMRI community cares about?** The inverse Laplace is severely ill-posed — our NUTS posterior directly quantifies which components are resolvable. But if the community doesn't need the full spectrum (just ADC), the problem may not matter regardless of how we solve it.

2. **If uncertainty IS important — is NUTS the right tool?** MAP gives the same classification. Could bootstrap MAP or the Ridge covariance give similar uncertainty cheaper? Or is proper posterior sampling important for credibility?

3. **Any thoughts on alternative inference** (variational, amortized neural) that could give the same uncertainty faster?

Happy to discuss on a call.

Best,
Patrick

P.S. for both of you — a methodological concern: Langkilde et al. note that tumor ROIs were placed based on multiparametric MRI (including DWI/ADC), not whole-mount histopathology. They write: *"there might be a bias in this study toward lesions that are well delineated on ADC maps."* This means our tumor detection AUCs (where ADC scores 0.94) are partially circular — ADC helped define the ground truth. Only the GGG classification (Gleason from pathology) is truly independent. Should we discuss this in the paper, and does it change how we interpret the ADC vs spectral comparison?
