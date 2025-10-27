# Diffusivity Discretization Rationale

## Final Discretization (Uniform Multi-Resolution)

**D = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] μm²/ms**

---

## Mathematical Motivation

### Principle: Uniform Spacing Within Clinical Ranges

The diffusivity spectrum is discretized using uniform spacing adapted to clinical relevance:

**Tumor range (D ≤ 1 μm²/ms):**
- Spacing: Δ = 0.25 μm²/ms
- Bins: [0.25, 0.5, 0.75, 1.0]
- **4 values** in interval [0, 1]
- Rationale: Dense sampling for Gleason grade differentiation

**Normal prostate range (1 < D ≤ 2 μm²/ms):**
- Spacing: Δ = 0.5 μm²/ms  
- Bins: [1.5, 2.0]
- **3 values** in interval [1, 2] (including boundaries 1.0, 2.0)
- Rationale: Intermediate resolution for tissue characterization

**High diffusivity range (2 < D ≤ 3 μm²/ms):**
- Spacing: Δ = 1.0 μm²/ms
- Bins: [3.0]
- **2 values** in interval [2, 3] (including boundary 2.0)
- Rationale: Coarse sampling where clinical distinctions diminish

**Free water (D = 20 μm²/ms):**
- Single compartment for CSF and vascular contribution at b=0

---

## Properties

| Property | Value |
|----------|-------|
| Total bins | 8 |
| Clinical bins (D ≤ 3) | 7 |
| Condition number (κ) | **2.78×10⁵** |
| Tumor resolution | 4 bins (0.25, 0.5, 0.75, 1.0) |
| All whole/simple values | ✓ |
| Includes D=20 for b=0 | ✓ |
| Max clinical D | 3.0 ✓ |

---

## For Paper/Abstract

### Concise Version

*Diffusivity bins were placed using uniform spacing adapted to clinical ranges: Δ = 0.25 μm²/ms in the tumor range (D ≤ 1), Δ = 0.5 μm²/ms in the normal prostate range (1 < D ≤ 2), and Δ = 1.0 μm²/ms at high diffusivity (2 < D ≤ 3), plus free water at D = 20 μm²/ms. This yields [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] μm²/ms, providing 4 bins in the tumor range for Gleason differentiation while maintaining favorable numerical conditioning (κ = 2.78×10⁵).*

### Extended Version (Methods)

**Diffusivity Discretization**

We discretized the diffusivity spectrum using adaptive uniform spacing based on clinical relevance. In the tumor range (D ≤ 1 μm²/ms), where Gleason grades show maximal ADC separation, we employed fine spacing (Δ = 0.25 μm²/ms), yielding 4 bins. In the normal prostate range (1 < D ≤ 2 μm²/ms), we used intermediate spacing (Δ = 0.5 μm²/ms), and for high diffusivity values (2 < D ≤ 3 μm²/ms), where clinical distinctions are less pronounced, we applied coarse spacing (Δ = 1.0 μm²/ms). A free water compartment at D = 20 μm²/ms accounts for CSF and vascular contributions at b = 0.

The resulting discretization [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0] μm²/ms can be described by overlapping intervals: 4 values in [0, 1], 3 values in [1, 2] (sharing boundaries 1.0 and 2.0), and 2 values in [2, 3] (sharing boundary 2.0). This scheme balances spectral resolution for cancer detection with numerical stability (condition number κ = 2.78×10⁵), enabling robust Bayesian inference with moderate ridge regularization (λ = 0.5).

---

## Advantages

1. **Simple & Interpretable**: Uniform spacing within each range
2. **Clinically Motivated**: Dense in tumor range, coarse at high D
3. **Whole Values**: Easy to communicate (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20)
4. **Excellent Conditioning**: κ = 2.78×10⁵ (best among tested configurations)
5. **Supervisor Approved**: Matches intuition of 4-3-2 structure

---

## Comparison with Alternatives

| Configuration | κ | Structure | Issues |
|--------------|---|-----------|---------|
| **uniform_8bins** | **2.78e5** | **Uniform in ranges** | **None** ✓ |
| geometric_8bins | 3.46e5 | Geometric in log(S) | Less intuitive |
| optimal_8bins | 8.37e6 | Signal-aware | Complex, poor κ |
| geometric_9bins | 7.88e6 | Full geometric | Too many divergences |

---

## Implementation

**Simulated data:** `dataset.spectrum_pair=uniform_8bins`  
**BWH data:** Configured by default in `configs/dataset/bwh.yaml`

**Test command:**
```bash
uv run python src/spectra_estimation_dmri/main.py \
  dataset=simulated \
  dataset.spectrum_pair=uniform_8bins \
  dataset.snr=1000 \
  inference=nuts \
  prior=ridge \
  prior.strength=0.5 \
  local=true
```

