#!/bin/bash
# Quick test of BWH dataset with optimal 8-bin discretization
# Tests on first few ROIs to verify setup

echo "=========================================="
echo "BWH TEST RUN - Optimal 8-bin Discretization"
echo "=========================================="
echo "This is a quick test to verify configuration."
echo "Should complete in ~10-15 minutes."
echo ""

cd /Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI

uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  inference=nuts \
  inference.n_iter=5000 \
  inference.tune=500 \
  inference.n_chains=4 \
  inference.target_accept=0.95 \
  prior=ridge \
  prior.strength=0.5 \
  local=true

echo ""
echo "=========================================="
echo "Test complete! Check results in:"
echo "  - outputs/ (Hydra logs)"
echo "  - results/inference/ (.nc files)"
echo "  - results/plots/plot/ (diagnostic plots)"
echo ""
echo "If successful, run the full analysis with:"
echo "  bash run_bwh_full.sh"
echo "=========================================="


