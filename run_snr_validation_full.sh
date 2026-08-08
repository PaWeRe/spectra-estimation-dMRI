#!/bin/bash
# Full SNR validation: Test recovery across multiple SNR levels
# This takes ~2-4 hours depending on your machine

echo "=========================================="
echo "Full SNR Validation"
echo "=========================================="
echo "Testing SNR levels: 100, 300, 500, 1000"
echo "Realizations per level: 15"
echo "Estimated time: 2-4 hours"
echo "=========================================="
echo ""

# Array of SNR levels to test
SNR_LEVELS=(100 300 500 1000)

for snr in "${SNR_LEVELS[@]}"; do
  echo ""
  echo "=========================================="
  echo "Testing SNR = $snr"
  echo "=========================================="
  
  uv run python src/spectra_estimation_dmri/main.py \
    dataset=simulated \
    dataset.spectrum_pair=optimal_7bins \
    dataset.snr=$snr \
    dataset.noise_realizations=15 \
    inference=nuts \
    inference.n_iter=5000 \
    inference.target_accept=0.99 \
    prior=ridge \
    prior.strength=0.5
  
  echo "Completed SNR = $snr"
  echo ""
done

echo ""
echo "=========================================="
echo "Full Validation Complete!"
echo "=========================================="
echo ""
echo "Now analyze results:"
echo "  uv run python scripts/analyze_snr_inference.py --mode simulated"
echo ""
echo "This will generate:"
echo "  - results/snr_analysis/snr_recovery_summary.pdf"
echo "  - results/snr_analysis/snr_recovery_results.csv"
echo ""
echo "Look for:"
echo "  ✓ Bias < 10% at all SNR levels"
echo "  ✓ 95% CI coverage ≈ 95%"
echo "  ✓ No systematic trend (bias vs SNR)"
echo "=========================================="

