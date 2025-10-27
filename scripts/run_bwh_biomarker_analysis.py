"""
Quick script to run complete pipeline on BWH prostate data:
1. Load real signal decays from signal_decays.json
2. Run MCMC inference (Gibbs or NUTS) to reconstruct spectra
3. Extract biomarkers with uncertainty quantification
4. Train Gleason score classifier
5. Evaluate performance

Usage:
    uv run python scripts/run_bwh_biomarker_analysis.py --sampler gibbs --n_iter 5000
    uv run python scripts/run_bwh_biomarker_analysis.py --sampler nuts --n_iter 2000 --use_mc
"""

import os
import sys
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spectra_estimation_dmri.data.loaders import load_bwh_signal_decays
from spectra_estimation_dmri.models.spectrum_model import DiffusionModel
from spectra_estimation_dmri.inference.gibbs import GibbsSamplerClean
from spectra_estimation_dmri.inference.nuts import NUTSSampler
from spectra_estimation_dmri.data.data_models import DiffusivitySpectrum, DiffusivitySpectraDataset
from spectra_estimation_dmri.biomarkers.simple_gleason_predictor import simple_biomarker_analysis
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description='Run BWH biomarker analysis')
    parser.add_argument('--sampler', type=str, default='gibbs', choices=['gibbs', 'nuts'],
                       help='MCMC sampler to use')
    parser.add_argument('--n_iter', type=int, default=5000,
                       help='Number of MCMC iterations')
    parser.add_argument('--n_chains', type=int, default=4,
                       help='Number of MCMC chains')
    parser.add_argument('--prior', type=str, default='ridge', choices=['ridge', 'uniform'],
                       help='Prior type')
    parser.add_argument('--prior_strength', type=float, default=0.01,
                       help='Prior strength (for ridge/lasso)')
    parser.add_argument('--use_mc', action='store_true',
                       help='Use Monte Carlo predictions (slower but gives uncertainty)')
    parser.add_argument('--classifier', type=str, default='logistic', 
                       choices=['logistic', 'random_forest'],
                       help='Classifier type')
    parser.add_argument('--output_dir', type=str, default='results/bwh_biomarker_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    signal_decays_path = os.path.join(base_dir, 'src/spectra_estimation_dmri/data/bwh/signal_decays.json')
    metadata_path = os.path.join(base_dir, 'src/spectra_estimation_dmri/data/bwh/metadata.csv')
    inference_dir = os.path.join(args.output_dir, 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    
    # Create minimal config
    cfg = OmegaConf.create({
        'inference': {
            'name': args.sampler,
            'n_iter': args.n_iter,
            'n_chains': args.n_chains,
            'burn_in': int(args.n_iter * 0.2),
            'tune': 1000,
            'target_accept': 0.95
        },
        'prior': {
            'type': args.prior,
            'strength': args.prior_strength
        },
        'dataset': {
            'name': 'bwh',
            'snr': 1000  # High SNR for real data
        },
        'classifier': {
            'name': args.classifier
        },
        'biomarker_analysis': {
            'use_uncertainty': True,
            'use_mc_predictions': args.use_mc
        },
        'local': True
    })
    
    print("="*70)
    print("BWH PROSTATE BIOMARKER ANALYSIS")
    print("="*70)
    print(f"Sampler: {args.sampler}")
    print(f"Iterations: {args.n_iter} (chains: {args.n_chains})")
    print(f"Prior: {args.prior} (strength: {args.prior_strength})")
    print(f"Classifier: {args.classifier}")
    print(f"Monte Carlo predictions: {args.use_mc}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/4] Loading BWH signal decays...")
    signal_decay_dataset = load_bwh_signal_decays(signal_decays_path, metadata_path)
    print(f"  Loaded {len(signal_decay_dataset.samples)} ROIs")
    
    # Filter: Only use samples with Gleason scores
    samples_with_labels = [s for s in signal_decay_dataset.samples 
                          if hasattr(s, 'ggg') and s.ggg is not None]
    print(f"  {len(samples_with_labels)} ROIs have Gleason scores")
    
    # Step 2: Run MCMC inference
    print(f"\n[2/4] Running {args.sampler.upper()} inference...")
    print(f"  This will take ~{len(samples_with_labels) * args.n_iter / 1000 / 60:.1f} minutes...")
    
    spectra = []
    for i, signal_decay in enumerate(samples_with_labels[:5]):  # Limit to first 5 for testing
        print(f"  Processing ROI {i+1}/{min(5, len(samples_with_labels))}: {signal_decay.patient_id}_{signal_decay.roi_id}")
        
        # Create model
        model = DiffusionModel(
            signal_decay=signal_decay,
            config=cfg
        )
        
        # Run inference
        if args.sampler == 'gibbs':
            sampler = GibbsSamplerClean(model, signal_decay, cfg)
        else:
            sampler = NUTSSampler(model, signal_decay, cfg)
        
        spectrum = sampler.run(
            return_idata=True,
            show_progress=False,
            save_dir=inference_dir,
            unique_hash=f"{signal_decay.patient_id}_{signal_decay.roi_id}"
        )
        spectra.append(spectrum)
    
    # Create dataset
    spectra_dataset = DiffusivitySpectraDataset(spectra=spectra)
    print(f"  ✓ Reconstructed {len(spectra)} spectra")
    
    # Step 3: Extract biomarkers and train classifier
    print(f"\n[3/4] Extracting biomarkers and training classifier...")
    
    predictor, X, y, metrics = simple_biomarker_analysis(
        spectra_dataset=spectra_dataset,
        metadata_path=metadata_path,
        output_dir=args.output_dir,
        model_type=args.classifier,
        use_uncertainty=True,
        use_mc_predictions=args.use_mc
    )
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"✓ Processed {len(spectra)} ROIs")
    print(f"✓ Extracted {len(X.columns)} features (uncertainty: {cfg.biomarker_analysis.use_uncertainty})")
    print(f"✓ Trained {args.classifier} classifier")
    print(f"\nPerformance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  AUC: {metrics['auc']:.3f}")
    if 'sensitivity' in metrics:
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"\nResults saved to: {args.output_dir}/")
    print("="*70)
    
    # Print top features
    print("\nTop 3 Most Important Features:")
    for i, (feat, imp) in enumerate(sorted(predictor.feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True)[:3], 1):
        print(f"  {i}. {feat}: {imp:.3f}")


if __name__ == '__main__':
    main()

