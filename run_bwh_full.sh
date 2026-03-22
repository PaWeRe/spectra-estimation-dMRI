#!/bin/bash
# Full BWH dataset analysis with optimal 8-bin discretization
# Estimated runtime: 12-13 hours for 151 ROIs
# Use screen or tmux for long-running jobs!

echo "=========================================="
echo "BWH FULL RUN - Optimal 8-bin Discretization"
echo "=========================================="
echo "Dataset: 56 patients, 151 ROIs"
echo "Discretization: 8 bins (signal-aware)"
echo "  [0.25, 0.46, 0.68, 0.89, 1.10, 1.32, 2.0, 3.0]"
echo "Estimated runtime: ~12-13 hours"
echo ""
echo "Configuration:"
echo "  - Inference: NUTS sampler"
echo "  - Iterations: 10,000 (after 1,000 tuning)"
echo "  - Chains: 4"
echo "  - Prior: Ridge (strength=0.5)"
echo "  - Target acceptance: 0.95"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5
echo ""
echo "Starting analysis..."
echo "=========================================="
echo ""

cd /Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI

# Record start time
start_time=$(date +%s)
echo "Start time: $(date)"
echo ""

# Run the analysis
uv run python src/spectra_estimation_dmri/main.py \
  dataset=bwh \
  inference=nuts \
  inference.n_iter=10000 \
  inference.tune=1000 \
  inference.n_chains=4 \
  inference.target_accept=0.95 \
  prior=ridge \
  prior.strength=0.5 \
  local=false \
  2>&1 | tee bwh_run_$(date +%Y%m%d_%H%M%S).log

# Calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETE!"
echo "=========================================="
echo "End time: $(date)"
echo "Total runtime: ${hours}h ${minutes}m"
echo ""
echo "Results saved to:"
echo "  - results/inference/*.nc (posterior samples)"
echo "  - results/plots/plot/ (diagnostic plots)"
echo "  - outputs/ (Hydra run directory)"
echo "  - wandb (if enabled)"
echo ""
echo "Next steps:"
echo "  1. Check convergence: grep 'CONVERGED' bwh_run_*.log"
echo "  2. Run biomarker analysis (if configured)"
echo "  3. Generate summary plots"
echo "=========================================="


