#!/bin/bash
# Quick SNR validation: Test if NUTS can recover known SNR
# This runs 10 realizations at SNR=300 to check for bias

echo "=========================================="
echo "Quick SNR Validation Test"
echo "=========================================="
echo "Testing: SNR = 300 with 10 realizations"
echo "Expected: Mean inferred SNR ≈ 300 ± 20"
echo "=========================================="
echo ""

# Run inference with 10 realizations
uv run python src/spectra_estimation_dmri/main.py \
  dataset=simulated \
  dataset.spectrum_pair=optimal_7bins \
  dataset.snr=300 \
  dataset.noise_realizations=10 \
  inference=nuts \
  inference.n_iter=5000 \
  inference.target_accept=0.99 \
  prior=ridge \
  prior.strength=0.5

echo ""
echo "=========================================="
echo "Validation Complete!"
echo "=========================================="
echo ""
echo "Check the plot:"
echo "  open results/plots/plot/snr_posterior_nuts_*.pdf"
echo ""
echo "Look for:"
echo "  ✓ Mean SNR close to 300 (within 10%)"
echo "  ✓ Green line (true SNR=300) runs through boxes"
echo "  ✓ Reasonable spread (not too tight, not too wide)"
echo ""
echo "If this looks good, run full validation:"
echo "  bash run_snr_validation_full.sh"
echo "=========================================="

