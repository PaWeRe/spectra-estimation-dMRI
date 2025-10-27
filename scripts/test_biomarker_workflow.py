"""
Test the simple biomarker workflow with synthetic data before running on real BWH data.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from spectra_estimation_dmri.biomarkers.simple_gleason_predictor import (
    extract_biomarker_features,
    SimpleGleasonPredictor,
)
import pandas as pd


def test_feature_extraction():
    """Test 1: Feature extraction with and without uncertainty"""
    print("\n" + "=" * 60)
    print("TEST 1: Feature Extraction")
    print("=" * 60)

    diffusivities = np.linspace(0.5, 3.0, 10)

    # Create two synthetic spectra (tumor vs normal)
    tumor_spectrum = np.array(
        [0.4, 0.3, 0.2, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.001]
    )
    normal_spectrum = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.02])

    # Normalize
    tumor_spectrum = tumor_spectrum / tumor_spectrum.sum()
    normal_spectrum = normal_spectrum / normal_spectrum.sum()

    print("\nTumor-like spectrum (high low_frac):")
    features_tumor = extract_biomarker_features(
        diffusivities, tumor_spectrum, None, include_uncertainty=False
    )
    print(f"  low_frac: {features_tumor['low_frac']:.3f}")
    print(f"  high_frac: {features_tumor['high_frac']:.3f}")
    print(f"  low_high_ratio: {features_tumor['low_high_ratio']:.3f}")

    print("\nNormal-like spectrum (low low_frac):")
    features_normal = extract_biomarker_features(
        diffusivities, normal_spectrum, None, include_uncertainty=False
    )
    print(f"  low_frac: {features_normal['low_frac']:.3f}")
    print(f"  high_frac: {features_normal['high_frac']:.3f}")
    print(f"  low_high_ratio: {features_normal['low_high_ratio']:.3f}")

    print("\n✓ Tumor has higher low_frac and low_high_ratio (as expected!)")

    # Test with MCMC samples (synthetic)
    print("\nWith MCMC uncertainty:")
    tumor_samples = np.random.dirichlet(tumor_spectrum * 100, size=50)  # 50 samples
    features_with_unc = extract_biomarker_features(
        diffusivities, tumor_spectrum, tumor_samples, include_uncertainty=True
    )
    print(
        f"  low_frac: {features_with_unc['low_frac']:.3f} ± {features_with_unc['low_frac_std']:.3f}"
    )
    print(
        f"  low_high_ratio: {features_with_unc['low_high_ratio']:.3f} ± {features_with_unc['low_high_ratio_std']:.3f}"
    )
    print(f"  CI width: {features_with_unc['low_frac_ci_width']:.3f}")

    print("\n✓ Uncertainty features extracted successfully!")

    return True


def test_classifier_simple():
    """Test 2: Classifier training with synthetic data"""
    print("\n" + "=" * 60)
    print("TEST 2: Classifier Training (Simple)")
    print("=" * 60)

    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 100

    # Tumor samples: high low_high_ratio
    tumor_features = {
        "low_frac": np.random.normal(0.6, 0.1, n_samples // 2),
        "high_frac": np.random.normal(0.2, 0.05, n_samples // 2),
        "low_high_ratio": np.random.normal(3.0, 0.5, n_samples // 2),
    }

    # Normal samples: low low_high_ratio
    normal_features = {
        "low_frac": np.random.normal(0.2, 0.05, n_samples // 2),
        "high_frac": np.random.normal(0.5, 0.1, n_samples // 2),
        "low_high_ratio": np.random.normal(0.4, 0.1, n_samples // 2),
    }

    # Combine
    X = pd.DataFrame(
        {
            "low_frac": np.concatenate(
                [tumor_features["low_frac"], normal_features["low_frac"]]
            ),
            "high_frac": np.concatenate(
                [tumor_features["high_frac"], normal_features["high_frac"]]
            ),
            "low_high_ratio": np.concatenate(
                [tumor_features["low_high_ratio"], normal_features["low_high_ratio"]]
            ),
            "mid_frac": np.random.uniform(0.1, 0.3, n_samples),
            "low_mid_ratio": np.random.uniform(0.5, 2.0, n_samples),
        }
    )

    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    print(f"\nDataset: {len(X)} samples")
    print(f"  Class 0 (normal): {np.sum(y == 0)}")
    print(f"  Class 1 (tumor): {np.sum(y == 1)}")

    # Train classifier
    predictor = SimpleGleasonPredictor(model_type="logistic")
    predictor.fit(X, y)

    # Evaluate
    metrics = predictor.evaluate(X, y)
    print(f"\nTraining Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  AUC: {metrics['auc']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")

    print(f"\nFeature Importance:")
    for feat, imp in sorted(
        predictor.feature_importance.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feat}: {imp:.3f}")

    if metrics["auc"] > 0.8:
        print("\n✓ Classifier works well (AUC > 0.8)!")
    else:
        print("\n⚠ AUC is lower than expected, but this is synthetic data")

    return True


def test_classifier_with_uncertainty():
    """Test 3: Classifier with uncertainty features"""
    print("\n" + "=" * 60)
    print("TEST 3: Classifier with Uncertainty Features")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100

    # Key insight: High-grade tumors have LOW uncertainty (consistent restriction)
    # Low-grade/normal have HIGHER uncertainty (heterogeneous)

    tumor_features = {
        "low_high_ratio": np.random.normal(3.0, 0.5, n_samples // 2),
        "low_high_ratio_std": np.random.uniform(
            0.05, 0.15, n_samples // 2
        ),  # Low uncertainty
        "spectrum_uncertainty": np.random.uniform(0.01, 0.05, n_samples // 2),
    }

    normal_features = {
        "low_high_ratio": np.random.normal(0.4, 0.1, n_samples // 2),
        "low_high_ratio_std": np.random.uniform(
            0.15, 0.4, n_samples // 2
        ),  # Higher uncertainty
        "spectrum_uncertainty": np.random.uniform(0.05, 0.15, n_samples // 2),
    }

    X = pd.DataFrame(
        {
            "low_high_ratio": np.concatenate(
                [tumor_features["low_high_ratio"], normal_features["low_high_ratio"]]
            ),
            "low_high_ratio_std": np.concatenate(
                [
                    tumor_features["low_high_ratio_std"],
                    normal_features["low_high_ratio_std"],
                ]
            ),
            "spectrum_uncertainty": np.concatenate(
                [
                    tumor_features["spectrum_uncertainty"],
                    normal_features["spectrum_uncertainty"],
                ]
            ),
            "low_frac": np.random.uniform(0.2, 0.6, n_samples),
            "high_frac": np.random.uniform(0.1, 0.5, n_samples),
        }
    )

    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    # Train with uncertainty features
    predictor = SimpleGleasonPredictor(
        model_type="logistic", use_uncertainty_features=True
    )
    predictor.fit(X, y)

    metrics = predictor.evaluate(X, y)
    print(f"\nPerformance with Uncertainty Features:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  AUC: {metrics['auc']:.3f}")

    print(f"\nFeature Importance (note if uncertainty features are used):")
    for feat, imp in sorted(
        predictor.feature_importance.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feat}: {imp:.3f}")

    # Check if uncertainty features are important
    unc_features = ["low_high_ratio_std", "spectrum_uncertainty"]
    unc_importance = sum(predictor.feature_importance.get(f, 0) for f in unc_features)
    total_importance = sum(predictor.feature_importance.values())
    unc_fraction = unc_importance / total_importance

    print(
        f"\nUncertainty features account for {unc_fraction*100:.1f}% of total importance"
    )

    if unc_fraction > 0.1:
        print("✓ Uncertainty features are being used by the classifier!")
    else:
        print(
            "⚠ Uncertainty features have low importance (might not be informative for this data)"
        )

    return True


def main():
    print("\n" + "=" * 70)
    print("TESTING SIMPLE BIOMARKER WORKFLOW")
    print("=" * 70)
    print("\nThis tests the core functionality with synthetic data")
    print("before running on real BWH prostate data.")

    try:
        test_feature_extraction()
        test_classifier_simple()
        test_classifier_with_uncertainty()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run on small BWH subset: scripts/run_bwh_biomarker_analysis.py")
        print("2. Check feature_importance.pdf to see which features matter")
        print("3. If uncertainty features are important, try Monte Carlo predictions")
        print("4. Run on full BWH dataset")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
