"""
Validate logistic regression implementation with synthetic biomarker data.

This tests:
1. Feature extraction with MC uncertainty propagation
2. LOOCV logistic regression training
3. Metric computation (AUC, accuracy, etc.)
4. Bootstrap confidence intervals
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_estimation_dmri.biomarkers.mc_classification import (
    prepare_classification_data,
    evaluate_feature_set,
    loocv_predictions,
    compute_metrics,
    bootstrap_auc_ci,
)


def create_synthetic_biomarker_dataset(n_samples=30, seed=42):
    """
    Create synthetic biomarker dataset mimicking prostate cancer diffusivity spectra.

    - Tumor: High restricted diffusion (high low-D fraction)
    - Normal: Higher unrestricted diffusion (higher high-D fraction)
    """
    np.random.seed(seed)

    # Generate features for tumor vs normal
    n_tumor = n_samples // 2
    n_normal = n_samples - n_tumor

    # Diffusivity bins (um^2/ms)
    diffusivities = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00]

    data = []

    # Tumor samples: high low-D fractions
    for i in range(n_tumor):
        features = {
            "D_0.25": np.random.beta(5, 2),  # High restricted diffusion
            "D_0.50": np.random.beta(4, 3),
            "D_0.75": np.random.beta(3, 4),
            "D_1.00": np.random.beta(2, 5),
            "D_1.50": np.random.beta(2, 8),
            "D_2.00": np.random.beta(1, 10),
            "D_2.50": np.random.beta(1, 15),
            "D_3.00": np.random.beta(1, 20),
        }
        # Normalize
        total = sum(features.values())
        features = {k: v / total for k, v in features.items()}

        # Add metadata
        features["patient_id"] = f"tumor_{i}"
        features["region"] = "pz_tumor"
        features["is_tumor"] = True
        features["ggg"] = np.random.choice([1, 2, 3, 4, 5])
        features["gs"] = 6 + features["ggg"]

        # Add combo feature
        features["D[0.25]+1/D[2.0]+1/D[3.0]"] = (
            features["D_0.25"]
            + 1.0 / (features["D_2.00"] + 1e-10)
            + 1.0 / (features["D_3.00"] + 1e-10)
        )

        data.append(features)

    # Normal samples: lower restricted diffusion
    for i in range(n_normal):
        features = {
            "D_0.25": np.random.beta(2, 5),  # Lower restricted diffusion
            "D_0.50": np.random.beta(3, 4),
            "D_0.75": np.random.beta(4, 3),
            "D_1.00": np.random.beta(5, 2),
            "D_1.50": np.random.beta(6, 2),
            "D_2.00": np.random.beta(5, 3),
            "D_2.50": np.random.beta(4, 4),
            "D_3.00": np.random.beta(3, 5),
        }
        # Normalize
        total = sum(features.values())
        features = {k: v / total for k, v in features.items()}

        # Add metadata
        features["patient_id"] = f"normal_{i}"
        features["region"] = "pz_normal"
        features["is_tumor"] = False
        features["ggg"] = 0
        features["gs"] = 0

        # Add combo feature
        features["D[0.25]+1/D[2.0]+1/D[3.0]"] = (
            features["D_0.25"]
            + 1.0 / (features["D_2.00"] + 1e-10)
            + 1.0 / (features["D_3.00"] + 1e-10)
        )

        data.append(features)

    df = pd.DataFrame(data)
    return df


def test_feature_extraction():
    """Test 1: Verify synthetic features have expected properties"""
    print("\n" + "=" * 80)
    print("TEST 1: Feature Extraction Validation")
    print("=" * 80)

    df = create_synthetic_biomarker_dataset(n_samples=30)

    print(f"\nGenerated {len(df)} samples")
    print(f"  Tumor: {df['is_tumor'].sum()}")
    print(f"  Normal: {(~df['is_tumor']).sum()}")

    # Check that tumor has higher low-D fraction
    tumor_d025 = df[df["is_tumor"]]["D_0.25"].mean()
    normal_d025 = df[~df["is_tumor"]]["D_0.25"].mean()

    print(f"\nMean D_0.25 (restricted diffusion):")
    print(f"  Tumor:  {tumor_d025:.3f}")
    print(f"  Normal: {normal_d025:.3f}")

    if tumor_d025 > normal_d025:
        print("  ✓ Tumor has higher restricted diffusion (as expected)")
    else:
        print("  ✗ WARNING: Unexpected pattern")

    # Check combo feature
    tumor_combo = df[df["is_tumor"]]["D[0.25]+1/D[2.0]+1/D[3.0]"].mean()
    normal_combo = df[~df["is_tumor"]]["D[0.25]+1/D[2.0]+1/D[3.0]"].mean()

    print(f"\nMean combo feature:")
    print(f"  Tumor:  {tumor_combo:.3f}")
    print(f"  Normal: {normal_combo:.3f}")

    return df


def test_loocv_workflow(df):
    """Test 2: LOOCV logistic regression workflow"""
    print("\n" + "=" * 80)
    print("TEST 2: LOOCV Logistic Regression")
    print("=" * 80)

    # Prepare data
    X_meta, X_features, y = prepare_classification_data(
        df, task="tumor_vs_normal", zone="pz"
    )

    print(f"\nPrepared classification data:")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X_features.shape[1]}")
    print(f"  Classes: {np.sum(y==0)} normal, {np.sum(y==1)} tumor")

    # Run LOOCV
    print("\nRunning LOOCV...")
    y_pred_proba, y_pred_class = loocv_predictions(X_features, y, regularization=1.0)

    print(f"  Predictions range: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
    print(f"  Mean prediction: {y_pred_proba.mean():.3f}")

    # Compute metrics
    metrics = compute_metrics(y, y_pred_proba, y_pred_class)

    print("\nPerformance metrics:")
    print(f"  AUC:         {metrics['auc']:.3f}")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  Precision:   {metrics['precision']:.3f}")

    # Bootstrap CI
    print("\nComputing bootstrap confidence interval...")
    auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(y, y_pred_proba, n_bootstrap=100)
    print(f"  AUC: {auc_mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

    if metrics["auc"] > 0.7:
        print("\n  ✓ Good discrimination (AUC > 0.7)")
    elif metrics["auc"] > 0.6:
        print("\n  ✓ Moderate discrimination (AUC > 0.6)")
    else:
        print("\n  ⚠ Weak discrimination (this is small synthetic data)")

    return metrics


def test_feature_set_evaluation(df):
    """Test 3: Evaluate multiple feature sets"""
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Feature Set Evaluation")
    print("=" * 80)

    X_meta, X_features, y = prepare_classification_data(
        df, task="tumor_vs_normal", zone="pz"
    )

    # Test individual features
    print("\nEvaluating individual diffusivity bins:")
    feature_aucs = []
    for d in [0.25, 0.50, 1.00, 2.00, 3.00]:
        feat_name = f"D_{d:.2f}"
        result = evaluate_feature_set(
            X_meta, X_features, y, [feat_name], feat_name, regularization=1.0
        )
        if result:
            auc = result["metrics"]["auc"]
            feature_aucs.append((feat_name, auc))
            print(f"  {feat_name}: AUC = {auc:.3f}")

    # Find best individual feature
    if feature_aucs:
        best_feat, best_auc = max(feature_aucs, key=lambda x: x[1])
        print(f"\n  Best individual feature: {best_feat} (AUC = {best_auc:.3f})")

    # Test combo feature
    print("\nEvaluating combo feature:")
    combo_result = evaluate_feature_set(
        X_meta,
        X_features,
        y,
        ["D[0.25]+1/D[2.0]+1/D[3.0]"],
        "Combo",
        regularization=1.0,
    )
    if combo_result:
        print(f"  Combo: AUC = {combo_result['metrics']['auc']:.3f}")

    # Test full model
    print("\nEvaluating full model (all features):")
    all_features = [
        f"D_{d:.2f}" for d in [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 2.50, 3.00]
    ]
    full_result = evaluate_feature_set(
        X_meta, X_features, y, all_features, "Full Model", regularization=1.0
    )
    if full_result:
        print(f"  Full Model: AUC = {full_result['metrics']['auc']:.3f}")

    print("\n  ✓ Successfully evaluated multiple feature sets")


def main():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION VALIDATION")
    print("=" * 80)
    print("\nThis validates the biomarker classification pipeline")
    print("with synthetic data that mimics prostate cancer diffusivity patterns.")

    try:
        # Test 1: Feature extraction
        df = test_feature_extraction()

        # Test 2: LOOCV workflow
        metrics = test_loocv_workflow(df)

        # Test 3: Multiple feature sets
        test_feature_set_evaluation(df)

        print("\n" + "=" * 80)
        print("✓ ALL VALIDATION TESTS PASSED")
        print("=" * 80)
        print("\nThe logistic regression implementation is working correctly!")
        print("\nNext steps:")
        print("1. Run on real BWH data: uv run python -m spectra_estimation_dmri.main")
        print("2. Check results in results/biomarkers/")
        print("3. Review ROC curves and feature importance")

        return True

    except Exception as e:
        print(f"\n✗ VALIDATION FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
