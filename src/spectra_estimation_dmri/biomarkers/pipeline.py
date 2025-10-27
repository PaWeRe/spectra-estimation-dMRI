"""
Main biomarker analysis pipeline.

Orchestrates the complete workflow:
1. Feature extraction (spectra + ADC)
2. Classification with LOOCV
3. Statistical comparison
4. Visualization
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .features import extract_features_from_dataset, get_feature_names
from .adc_baseline import extract_adc_features
from .mc_classification import prepare_classification_data, evaluate_feature_set
from .biomarker_viz import create_summary_report


def run_biomarker_analysis(
    spectra_dataset,
    output_dir: str = "results/biomarkers",
    n_mc_samples: int = 200,
    regularization: float = 1.0,
    adc_b_range: tuple = (0.0, 1.0),  # in ms units
) -> Dict:
    """
    Run complete biomarker analysis pipeline.

    Args:
        spectra_dataset: DiffusivitySpectraDataset with NUTS posterior samples
        output_dir: Directory for outputs
        n_mc_samples: Number of MC samples for uncertainty propagation
        regularization: L2 regularization for logistic regression
        adc_b_range: b-value range for ADC computation (ms units)

    Returns:
        Dictionary with all results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("BIOMARKER ANALYSIS PIPELINE")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Feature Extraction
    # ========================================================================
    print("\n[STEP 1/4] Extracting features from spectra...")

    # Extract spectrum-based features with MC uncertainty
    features_df, uncertainty_df = extract_features_from_dataset(
        spectra_dataset, n_mc_samples=n_mc_samples
    )

    print(f"  Extracted {len(features_df)} samples")
    print(
        f"  Features: {len([c for c in features_df.columns if c.startswith('D_') or 'D[' in c])}"
    )

    # Extract ADC features (baseline)
    adc_df = extract_adc_features(spectra_dataset, b_range=adc_b_range)

    # Merge ADC with spectrum features
    features_df = features_df.merge(
        adc_df[["patient_id", "adc"]], on="patient_id", how="left"
    )
    features_df = features_df.rename(columns={"adc": "ADC"})

    print(
        f"  ADC features computed (b-range: {adc_b_range[0]:.0f}-{adc_b_range[1]:.0f} ms)"
    )

    # Get diffusivity grid
    diffusivities = spectra_dataset.spectra[0].diffusivities

    # Define feature sets to evaluate
    individual_features = [f"D_{d:.2f}" for d in diffusivities]

    # ========================================================================
    # STEP 2: Classification Tasks
    # ========================================================================
    print("\n[STEP 2/4] Running classification tasks...")

    all_results = {}

    # ------------------------------------------------------------------------
    # Task 1: Tumor vs Normal (PZ)
    # ------------------------------------------------------------------------
    print("\n  Task 1: Tumor vs Normal (PZ)")
    X_meta_pz, X_features_pz, y_pz = prepare_classification_data(
        features_df, task="tumor_vs_normal", zone="pz"
    )

    if len(y_pz) > 0:
        print(
            f"    Samples: {len(y_pz)} (Normal: {np.sum(y_pz==0)}, Tumor: {np.sum(y_pz==1)})"
        )

        results_pz = []

        # Individual features
        for feat in individual_features:
            if feat in X_meta_pz.columns:
                result = evaluate_feature_set(
                    X_meta_pz, X_features_pz, y_pz, [feat], feat, regularization
                )
                results_pz.append(result)

        # Combo feature
        combo_col = "D[0.25]+1/D[2.0]+1/D[3.0]"
        if combo_col in X_meta_pz.columns:
            result = evaluate_feature_set(
                X_meta_pz, X_features_pz, y_pz, [combo_col], combo_col, regularization
            )
            results_pz.append(result)

        # Full model (all individual features)
        result = evaluate_feature_set(
            X_meta_pz,
            X_features_pz,
            y_pz,
            individual_features,
            "Full LR",
            regularization,
        )
        results_pz.append(result)

        # ADC baseline
        if "ADC" in X_meta_pz.columns:
            result = evaluate_feature_set(
                X_meta_pz,
                X_features_pz,
                y_pz,
                ["ADC"],
                "ADC (baseline)",
                regularization,
            )
            results_pz.append(result)

        all_results["tumor_vs_normal_pz"] = results_pz
    else:
        print("    [WARNING] No samples available")

    # ------------------------------------------------------------------------
    # Task 2: Tumor vs Normal (TZ)
    # ------------------------------------------------------------------------
    print("\n  Task 2: Tumor vs Normal (TZ)")
    X_meta_tz, X_features_tz, y_tz = prepare_classification_data(
        features_df, task="tumor_vs_normal", zone="tz"
    )

    if len(y_tz) > 0:
        print(
            f"    Samples: {len(y_tz)} (Normal: {np.sum(y_tz==0)}, Tumor: {np.sum(y_tz==1)})"
        )

        results_tz = []

        # Individual features
        for feat in individual_features:
            if feat in X_meta_tz.columns:
                result = evaluate_feature_set(
                    X_meta_tz, X_features_tz, y_tz, [feat], feat, regularization
                )
                results_tz.append(result)

        # Combo feature
        combo_col = "D[0.25]+1/D[2.0]+1/D[3.0]"
        if combo_col in X_meta_tz.columns:
            result = evaluate_feature_set(
                X_meta_tz, X_features_tz, y_tz, [combo_col], combo_col, regularization
            )
            results_tz.append(result)

        # Full model
        result = evaluate_feature_set(
            X_meta_tz,
            X_features_tz,
            y_tz,
            individual_features,
            "Full LR",
            regularization,
        )
        results_tz.append(result)

        # ADC baseline
        if "ADC" in X_meta_tz.columns:
            result = evaluate_feature_set(
                X_meta_tz,
                X_features_tz,
                y_tz,
                ["ADC"],
                "ADC (baseline)",
                regularization,
            )
            results_tz.append(result)

        all_results["tumor_vs_normal_tz"] = results_tz
    else:
        print("    [WARNING] No samples available")

    # ------------------------------------------------------------------------
    # Task 3: Gleason Grade (<7 vs >=7)
    # ------------------------------------------------------------------------
    print("\n  Task 3: Gleason Grade Group (<7 vs >=7)")
    X_meta_ggg, X_features_ggg, y_ggg = prepare_classification_data(
        features_df, task="ggg", zone=None  # Combined zones
    )

    if len(y_ggg) > 0:
        print(
            f"    Samples: {len(y_ggg)} (<7: {np.sum(y_ggg==0)}, >=7: {np.sum(y_ggg==1)})"
        )

        if len(y_ggg) < 10:
            print("    [WARNING] Very small sample size - results may not be reliable")

        results_ggg = []

        # Individual features
        for feat in individual_features:
            if feat in X_meta_ggg.columns:
                result = evaluate_feature_set(
                    X_meta_ggg, X_features_ggg, y_ggg, [feat], feat, regularization
                )
                results_ggg.append(result)

        # Combo feature
        combo_col = "D[0.25]+1/D[2.0]+1/D[3.0]"
        if combo_col in X_meta_ggg.columns:
            result = evaluate_feature_set(
                X_meta_ggg,
                X_features_ggg,
                y_ggg,
                [combo_col],
                combo_col,
                regularization,
            )
            results_ggg.append(result)

        # Full model
        result = evaluate_feature_set(
            X_meta_ggg,
            X_features_ggg,
            y_ggg,
            individual_features,
            "Full LR",
            regularization,
        )
        results_ggg.append(result)

        # ADC baseline
        if "ADC" in X_meta_ggg.columns:
            result = evaluate_feature_set(
                X_meta_ggg,
                X_features_ggg,
                y_ggg,
                ["ADC"],
                "ADC (baseline)",
                regularization,
            )
            results_ggg.append(result)

        all_results["ggg"] = results_ggg
    else:
        print("    [WARNING] No tumor samples with GGG labels available")

    # ========================================================================
    # STEP 3: Save Feature DataFrames
    # ========================================================================
    print("\n[STEP 3/4] Saving feature tables...")

    features_df.to_csv(os.path.join(output_dir, "features.csv"), index=False)
    uncertainty_df.to_csv(
        os.path.join(output_dir, "feature_uncertainty.csv"), index=False
    )

    print(f"  Saved: {output_dir}/features.csv")
    print(f"  Saved: {output_dir}/feature_uncertainty.csv")

    # ========================================================================
    # STEP 4: Visualization and Summary
    # ========================================================================
    print("\n[STEP 4/4] Creating visualizations and summary...")

    create_summary_report(all_results, output_dir)

    print("\n" + "=" * 80)
    print("✓ BIOMARKER ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")

    return {
        "features_df": features_df,
        "uncertainty_df": uncertainty_df,
        "results": all_results,
    }
