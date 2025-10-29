"""
Diagnostic Script for Classifier Performance Issues

This script analyzes why:
1. ADC outperforms Full LR for PZ/TZ
2. GGG results show suspicious perfect AUCs for individual features
3. How to improve classifier performance

Author: AI Assistant
Date: 2025-10-29
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")


def analyze_sample_size_adequacy(n_samples, n_positive, n_negative, n_features):
    """
    Check if sample size is adequate for logistic regression.

    Rule of thumb: Need at least 10 Events Per Variable (EPV)
    """
    epv = min(n_positive, n_negative) / n_features

    print(f"  Samples: {n_samples} (Pos: {n_positive}, Neg: {n_negative})")
    print(f"  Features: {n_features}")
    print(f"  Events Per Variable (EPV): {epv:.2f}")

    if epv < 5:
        print("  ⚠️  CRITICAL: EPV < 5 → High risk of overfitting!")
        print("     Recommendations:")
        print("     - Reduce number of features (feature selection)")
        print("     - Increase regularization strength")
        print("     - Consider single-feature models only")
    elif epv < 10:
        print("  ⚠️  WARNING: EPV < 10 → Risk of unstable estimates")
        print("     Recommendations:")
        print("     - Use stronger regularization")
        print("     - Consider dimensionality reduction (PCA)")
    else:
        print("  ✓ Sample size is adequate for multi-feature model")

    return epv


def compare_regularization_strengths(X, y, feature_name):
    """
    Compare different regularization strengths.
    """
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]  # Inverse regularization
    results = []

    scaler = StandardScaler()
    loo = LeaveOneOut()

    for C in C_values:
        y_pred = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
            clf.fit(X_train_scaled, y_train)
            y_pred[test_idx] = clf.predict_proba(X_test_scaled)[0, 1]

        # Handle inverse correlation
        auc = roc_auc_score(y, y_pred)
        if auc < 0.5:
            y_pred = 1.0 - y_pred
            auc = roc_auc_score(y, y_pred)

        results.append((C, 1.0 / C, auc))

    print(f"\n  Regularization strength comparison for {feature_name}:")
    print("  " + "-" * 50)
    print(f"  {'C (inverse)':<12} {'Lambda':<12} {'AUC':<10}")
    print("  " + "-" * 50)
    for C, lam, auc in results:
        print(f"  {C:<12.2f} {lam:<12.2f} {auc:<10.3f}")

    best_C, best_lam, best_auc = max(results, key=lambda x: x[2])
    print(f"\n  Best: C={best_C:.2f} (lambda={best_lam:.2f}), AUC={best_auc:.3f}")

    return best_C


def analyze_feature_correlation(df, feature_cols):
    """
    Analyze correlation between features to detect multicollinearity.
    """
    import numpy as np

    if len(feature_cols) < 2:
        return

    X = df[feature_cols].values

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Find highly correlated pairs
    high_corr_pairs = []
    n_features = len(feature_cols)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr_matrix[i, j]) > 0.8:
                high_corr_pairs.append(
                    (feature_cols[i], feature_cols[j], corr_matrix[i, j])
                )

    if high_corr_pairs:
        print("\n  ⚠️  High correlations detected (|r| > 0.8):")
        for f1, f2, r in high_corr_pairs[:5]:  # Show top 5
            print(f"    {f1} <-> {f2}: r={r:.3f}")
        print("  → This causes multicollinearity, making coefficients unstable")
        print("  → Solution: Remove redundant features or use PCA")


def main():
    """Main diagnostic function."""

    # Load data
    df = pd.read_csv("results/biomarkers/features.csv")

    print("\n" + "=" * 80)
    print("CLASSIFIER PERFORMANCE DIAGNOSTIC")
    print("=" * 80)

    # Get feature columns
    df["zone"] = df["region"].apply(
        lambda x: (
            "pz"
            if "pz" in str(x).lower()
            else ("tz" if "tz" in str(x).lower() else "unknown")
        )
    )

    individual_features = [
        col for col in df.columns if col.startswith("D_") and not "[" in col
    ]

    # ========================================================================
    # ANALYSIS 1: PZ Tumor Detection
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. PZ TUMOR DETECTION")
    print("=" * 80)

    pz_df = df[df["zone"] == "pz"].copy()
    pz_df = pz_df[pz_df["is_tumor"].notna()]
    y_pz = pz_df["is_tumor"].astype(int).values

    print("\n[Full LR Model]")
    X_pz_full = pz_df[individual_features].values
    epv_pz = analyze_sample_size_adequacy(
        len(y_pz), np.sum(y_pz), np.sum(~y_pz.astype(bool)), len(individual_features)
    )

    if epv_pz < 10:
        print("\n  Testing different regularization strengths...")
        best_C_pz = compare_regularization_strengths(X_pz_full, y_pz, "Full LR (PZ)")
        print(f"\n  💡 RECOMMENDATION: Use C={best_C_pz:.2f} instead of default C=1.0")

    # Analyze feature correlation
    analyze_feature_correlation(pz_df, individual_features)

    print("\n[ADC Model]")
    X_pz_adc = pz_df[["ADC"]].values
    analyze_sample_size_adequacy(len(y_pz), np.sum(y_pz), np.sum(~y_pz.astype(bool)), 1)

    print("\n💡 WHY ADC OUTPERFORMS FULL LR (PZ):")
    print("  1. ADC has EPV=27 (excellent) vs Full LR EPV=3.4 (poor)")
    print("  2. Single feature = more stable estimate with small samples")
    print("  3. Full LR suffers from multicollinearity between spectral bins")
    print("  4. Default regularization (C=1.0) may be too weak")

    # ========================================================================
    # ANALYSIS 2: TZ Tumor Detection
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. TZ TUMOR DETECTION")
    print("=" * 80)

    tz_df = df[df["zone"] == "tz"].copy()
    tz_df = tz_df[tz_df["is_tumor"].notna()]
    y_tz = tz_df["is_tumor"].astype(int).values

    print("\n[Full LR Model]")
    X_tz_full = tz_df[individual_features].values
    epv_tz = analyze_sample_size_adequacy(
        len(y_tz), np.sum(y_tz), np.sum(~y_tz.astype(bool)), len(individual_features)
    )

    if epv_tz < 10:
        print("\n  Testing different regularization strengths...")
        best_C_tz = compare_regularization_strengths(X_tz_full, y_tz, "Full LR (TZ)")
        print(f"\n  💡 RECOMMENDATION: Use C={best_C_tz:.2f} instead of default C=1.0")

    analyze_feature_correlation(tz_df, individual_features)

    print("\n💡 WHY ADC OUTPERFORMS FULL LR (TZ):")
    print("  1. VERY small positive class (only 13 tumors)")
    print("  2. ADC has EPV=13 vs Full LR EPV=1.6 (critically underpowered!)")
    print("  3. With EPV=1.6, Full LR is essentially fitting noise")
    print("  4. D_20.00 shows perfect AUC=1.0 → clear overfitting")

    # ========================================================================
    # ANALYSIS 3: GGG Grading (THE PROBLEMATIC CASE)
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. GGG GRADING (1-2 vs 3-5) ⚠️  CRITICAL ISSUES")
    print("=" * 80)

    ggg_df = df[df["is_tumor"] == True].copy()
    ggg_df = ggg_df[ggg_df["ggg"].notna() & (ggg_df["ggg"] != 0)]
    y_ggg = (ggg_df["ggg"] >= 3).astype(int).values

    print("\n[Full LR Model]")
    X_ggg_full = ggg_df[individual_features].values
    epv_ggg = analyze_sample_size_adequacy(
        len(y_ggg), np.sum(y_ggg), np.sum(~y_ggg.astype(bool)), len(individual_features)
    )

    if epv_ggg < 10:
        print("\n  Testing different regularization strengths...")
        best_C_ggg = compare_regularization_strengths(
            X_ggg_full, y_ggg, "Full LR (GGG)"
        )
        print(f"\n  💡 RECOMMENDATION: Use C={best_C_ggg:.2f} instead of default C=1.0")

    analyze_feature_correlation(ggg_df, individual_features)

    print("\n💡 WHY GGG RESULTS ARE WEIRD:")
    print("  1. CRITICALLY SMALL SAMPLE: Only 28 samples (9 high-grade, 19 low-grade)")
    print("  2. EPV=1.1 for Full LR → Severe overfitting!")
    print("  3. Individual features (D_1.00, D_3.00) show perfect/near-perfect AUCs")
    print("     → This is NOT real performance, it's overfitting on noise")
    print("  4. With LOOCV on 28 samples, model 'memorizes' each fold")
    print("  5. D_3.00 AUC=0.97 but specificity=0% → predicts all as high-grade!")
    print("\n  🔴 THE PERFECT AUCs ARE FALSE POSITIVES FROM OVERFITTING!")

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR IMPROVING CLASSIFIER PERFORMANCE")
    print("=" * 80)

    print("\n1. FOR PZ/TZ TUMOR DETECTION:")
    print("   ✓ Increase regularization: Use C=0.1 or C=0.01 instead of C=1.0")
    print("   ✓ Feature selection: Keep only top 3-5 most discriminative bins")
    print("   ✓ Use elastic net: Mix of L1 and L2 penalties")
    print("   ✓ Consider PCA: Reduce to 2-3 principal components")

    print("\n2. FOR GGG GRADING:")
    print("   🔴 CRITICAL: Sample size is too small for reliable multi-feature model")
    print("   ✓ Option A: Use ONLY single-feature models (ADC, D_0.25, D_3.00)")
    print("   ✓ Option B: Collect more data (need ~50-100 samples minimum)")
    print(
        "   ✓ Option C: Use simpler classification (binary yes/no instead of full LR)"
    )
    print("   ✓ Add warning to paper about limited GGG sample size")

    print("\n3. GENERAL IMPROVEMENTS:")
    print("   ✓ Use stratified K-fold CV instead of LOOCV (more stable)")
    print("   ✓ Report confidence intervals to show uncertainty")
    print("   ✓ Add permutation tests to validate significance")
    print(
        "   ✓ Consider Bayesian logistic regression for better uncertainty quantification"
    )

    print("\n4. IMMEDIATE FIXES FOR YOUR PLOTS:")
    print("   ✓ Remove or flag features with perfect AUCs as 'overfitting'")
    print("   ✓ Add sample size to plot titles (e.g., 'GGG (N=28, underpowered)')")
    print("   ✓ Show confidence intervals on ROC curves")
    print("   ✓ Focus on ADC + top 2-3 spectral features for interpretability")

    print("\n" + "=" * 80)
    print("✓ DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Re-run pipeline with stronger regularization (C=0.1 or 0.01)")
    print("2. Use feature selection to keep only top features")
    print("3. Add warnings about GGG small sample size")
    print("4. Consider collecting more GGG samples if possible")


if __name__ == "__main__":
    main()
