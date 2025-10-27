# ISMRM Abstract: Uncertainty Propagation Section

## Proposed Text for "Constructing Biomarkers" Section

---

### Constructing biomarkers

**Feature Extraction with Uncertainty Propagation**  
From each posterior distribution \( p(R, \sigma | s) \), we extract clinically relevant biomarkers by propagating uncertainty through nonlinear feature functions via Monte Carlo sampling. For each voxel, we draw \( N = 200 \) posterior samples \( \{R^{(i)}, \sigma^{(i)}\}_{i=1}^N \) and compute features \( f(R^{(i)}) \) including individual diffusivity fractions and engineered combinations (e.g., \( f_{\text{combo}}(R) = R_{D=0.25} + \frac{1}{R_{D=2.0}} + \frac{1}{R_{D=3.0}} \)). This yields feature distributions \( \{f(R^{(1)}), \ldots, f(R^{(N)})\} \) characterized by posterior means and standard deviations, naturally quantifying epistemic uncertainty in derived biomarkers.

**Classification with Leave-One-Out Cross-Validation**  
To predict prostate cancer diagnosis and grade, we train L2-regularized logistic regression classifiers on posterior mean features using Leave-One-Out Cross-Validation (LOOCV). Given small clinical cohorts (\( n \approx 20-50 \)), LOOCV provides unbiased performance estimates by training on \( n-1 \) samples and predicting on each held-out patient iteratively. Features are standardized (zero mean, unit variance) within each fold to prevent data leakage. We evaluate Area Under the ROC Curve (AUC) with bootstrap confidence intervals (1000 iterations) for three tasks: tumor vs normal tissue in peripheral and transition zones, and Gleason Grade stratification (GGG < 3 vs ≥ 3). We compare spectrum-based classifiers against Apparent Diffusion Coefficient (ADC) baselines computed from signal decay.

---

## Alternative Shorter Version (if space-limited)

### Constructing biomarkers

From posterior samples \( \{R^{(i)}, \sigma^{(i)}\}_{i=1}^N \), we extract biomarker features \( f(R) \) (e.g., individual diffusivity fractions, engineered combinations) via Monte Carlo propagation, yielding feature distributions with quantified uncertainty. L2-regularized logistic regression classifiers trained with Leave-One-Out Cross-Validation predict cancer diagnosis and Gleason grade, evaluated via AUC with bootstrap CIs. This approach maximizes data efficiency for small cohorts (\( n \sim 20-50 \)) while propagating Bayesian uncertainty into clinical predictions.

---

## Key Points Highlighted

✅ **Uncertainty propagation**: MC sampling through nonlinear feature functions  
✅ **Posterior characterization**: Means + stds capture epistemic uncertainty  
✅ **Small-sample rigor**: LOOCV for unbiased validation  
✅ **Clinical relevance**: Three tasks (tumor detection PZ/TZ, grade stratification)  
✅ **Baseline comparison**: Against ADC (current standard)  
✅ **Statistical rigor**: Bootstrap CIs for AUC  

This directly connects your Bayesian inference framework to clinically interpretable predictions while maintaining methodological rigor.

