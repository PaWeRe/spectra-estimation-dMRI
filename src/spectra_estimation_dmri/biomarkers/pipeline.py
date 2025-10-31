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


def save_predictions_to_csv(
    result: Dict, task_name: str, output_dir: str, X_meta: pd.DataFrame
):
    """
    Save predictions with uncertainty to CSV file.

    Args:
        result: Dictionary from evaluate_feature_set containing predictions
        task_name: Name of the classification task
        output_dir: Directory to save CSV
        X_meta: Metadata DataFrame with ROI IDs
    """
    if "y_pred_proba" not in result:
        return

    # Create DataFrame with predictions
    pred_data = {
        "roi_id": X_meta["roi_id"].values,
        "y_true": result["y_true"],
        "y_pred_proba": result["y_pred_proba"],
        "y_pred_class": result["y_pred_class"],
    }

    # Add uncertainty if available
    if "y_pred_std" in result and result["y_pred_std"] is not None:
        pred_data["y_pred_std"] = result["y_pred_std"]

    df = pd.DataFrame(pred_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, f"predictions_{task_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"    [INFO] Saved predictions to {csv_path}")


def run_biomarker_analysis(
    spectra_dataset,
    output_dir: str = "results/biomarkers",
    n_mc_samples: int = 200,
    regularization: float = 1.0,
    adc_b_range: tuple = (0.0, 1.0),  # in ms units
    use_feature_selection: bool = True,  # Enable feature selection
    n_features_to_select: int = 5,  # Number of features to select
    feature_selection_method: str = "univariate",  # "univariate" or "all"
    # NEW: Task-specific hyperparameters (overrides defaults if provided)
    regularization_pz: Optional[float] = None,  # PZ-specific regularization
    regularization_tz: Optional[float] = None,  # TZ-specific regularization
    regularization_ggg: Optional[float] = None,  # GGG-specific regularization
    n_features_pz: Optional[int] = None,  # PZ-specific n_features
    n_features_tz: Optional[int] = None,  # TZ-specific n_features
    n_features_ggg: Optional[int] = None,  # GGG-specific n_features
    use_optimized_configs: bool = False,  # Use tuned hyperparameters from grid search
    propagate_uncertainty: bool = False,  # Propagate posterior uncertainty through classifier
    random_seed: int = 42,  # Random seed for reproducibility
) -> Dict:
    """
    Run complete biomarker analysis pipeline.

    Args:
        spectra_dataset: DiffusivitySpectraDataset with NUTS posterior samples
        output_dir: Directory for outputs
        n_mc_samples: Number of MC samples for uncertainty propagation
        regularization: Default L2 regularization (C parameter) for all tasks
        adc_b_range: b-value range for ADC computation (ms units)
        use_feature_selection: Enable automatic feature selection (top-k bins)
        n_features_to_select: Default number of features to select
        feature_selection_method: "univariate" (F-test) or "all" (use all bins)
        regularization_pz/tz/ggg: Task-specific regularization (overrides default)
        n_features_pz/tz/ggg: Task-specific n_features (overrides default)
        use_optimized_configs: Use tuned hyperparameters from grid search
            PZ: C=0.5, n_features=4
            TZ: C=0.5, n_features=5
            GGG: C=1.0, n_features=3
        propagate_uncertainty: If True, propagate posterior uncertainty through classifier
            (samples 200 MC samples per prediction for uncertainty quantification)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with all results

    Examples:
        # Use optimized configs from tuning
        run_biomarker_analysis(..., use_optimized_configs=True)

        # Custom per-task configs
        run_biomarker_analysis(..., regularization_pz=0.5, n_features_pz=4,
                                     regularization_tz=0.5, n_features_tz=5)

        # Standard config (all features, weak regularization)
        run_biomarker_analysis(..., regularization=1.0, use_feature_selection=False)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("BIOMARKER ANALYSIS PIPELINE")
    print("=" * 80)

    # Apply optimized configs if requested
    if use_optimized_configs:
        print("\n[CONFIG] Using optimized hyperparameters from grid search")
        # Optimal configs from tune_lr_classifier.py results
        regularization_pz = regularization_pz or 0.5
        n_features_pz = n_features_pz or 4
        regularization_tz = regularization_tz or 0.5
        n_features_tz = n_features_tz or 5
        regularization_ggg = regularization_ggg or 1.0
        n_features_ggg = n_features_ggg or 3
        use_feature_selection = True
        print(f"  PZ: C={regularization_pz}, n_features={n_features_pz}")
        print(f"  TZ: C={regularization_tz}, n_features={n_features_tz}")
        print(f"  GGG: C={regularization_ggg}, n_features={n_features_ggg}")
    else:
        # Use defaults or provided values
        regularization_pz = regularization_pz or regularization
        regularization_tz = regularization_tz or regularization
        regularization_ggg = regularization_ggg or regularization
        n_features_pz = n_features_pz or n_features_to_select
        n_features_tz = n_features_tz or n_features_to_select
        n_features_ggg = n_features_ggg or n_features_to_select
        print(f"\n[CONFIG] Using default/custom hyperparameters")
        print(f"  Default C={regularization}, n_features={n_features_to_select}")
        if regularization_pz != regularization or n_features_pz != n_features_to_select:
            print(f"  PZ override: C={regularization_pz}, n_features={n_features_pz}")
        if regularization_tz != regularization or n_features_tz != n_features_to_select:
            print(f"  TZ override: C={regularization_tz}, n_features={n_features_tz}")
        if (
            regularization_ggg != regularization
            or n_features_ggg != n_features_to_select
        ):
            print(
                f"  GGG override: C={regularization_ggg}, n_features={n_features_ggg}"
            )

    # ========================================================================
    # STEP 0: Deduplicate spectra by ROI
    # ========================================================================
    print("\n[STEP 0/4] Deduplicating spectra...")
    print(f"  Input: {len(spectra_dataset.spectra)} spectra")
    spectra_dataset = spectra_dataset.deduplicate_by_roi(keep="first")
    print(f"  Output: {len(spectra_dataset.spectra)} unique ROIs")

    # ========================================================================
    # STEP 1: Feature Extraction
    # ========================================================================
    print("\n[STEP 1/4] Extracting features from spectra...")

    # Extract spectrum-based features with MC uncertainty
    features_df, uncertainty_df = extract_features_from_dataset(
        spectra_dataset, n_mc_samples=n_mc_samples, random_seed=random_seed
    )

    print(f"  Extracted {len(features_df)} samples")
    print(
        f"  Features: {len([c for c in features_df.columns if c.startswith('D_') or 'D[' in c])}"
    )

    # Extract ADC features (baseline)
    adc_df = extract_adc_features(spectra_dataset, b_range=adc_b_range)

    # Merge ADC with spectrum features using unique ROI identifier
    features_df = features_df.merge(adc_df[["roi_id", "adc"]], on="roi_id", how="left")
    features_df = features_df.rename(columns={"adc": "ADC"})

    print(
        f"  ADC features computed (b-range: {adc_b_range[0]:.0f}-{adc_b_range[1]:.0f} ms)"
    )

    # Get diffusivity grid
    diffusivities = spectra_dataset.spectra[0].diffusivities

    # Define feature sets to evaluate
    individual_features = [f"D_{d:.2f}" for d in diffusivities]

    # Print configuration
    print(f"\n[CONFIGURATION]")
    print(f"  Regularization (C): {regularization}")
    print(f"  Feature selection: {'Enabled' if use_feature_selection else 'Disabled'}")
    if use_feature_selection:
        print(
            f"  Features to select: top-{n_features_to_select} (method: {feature_selection_method})"
        )
    else:
        print(f"  Using all {len(individual_features)} diffusivity bins")
    print(
        f"  Uncertainty propagation: {'Enabled' if propagate_uncertainty else 'Disabled'}"
    )
    if propagate_uncertainty:
        print(f"  MC samples per prediction: {n_mc_samples}")
    print(f"  Random seed: {random_seed}")

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

        # Individual features (no selection for single features)
        for feat in individual_features:
            if feat in X_meta_pz.columns:
                result = evaluate_feature_set(
                    X_meta_pz,
                    X_features_pz,
                    y_pz,
                    [feat],
                    feat,
                    C=regularization,
                    use_feature_selection=False,  # Single feature, no selection needed
                    spectra_dataset=spectra_dataset,
                    n_mc_samples=n_mc_samples,
                    propagate_uncertainty=False,  # Single features don't use classifier
                )
                results_pz.append(result)

        # Combo feature (no selection)
        combo_col = "D[0.25]+1/D[2.0]+1/D[3.0]"
        if combo_col in X_meta_pz.columns:
            result = evaluate_feature_set(
                X_meta_pz,
                X_features_pz,
                y_pz,
                [combo_col],
                combo_col,
                C=regularization,
                use_feature_selection=False,  # Single feature, no selection needed
                spectra_dataset=spectra_dataset,
                n_mc_samples=n_mc_samples,
                propagate_uncertainty=False,  # Single features don't use classifier
            )
            results_pz.append(result)

        # Full model (use task-specific hyperparameters)
        result = evaluate_feature_set(
            X_meta_pz,
            X_features_pz,
            y_pz,
            individual_features,
            "Full LR",
            C=regularization_pz,  # Task-specific
            use_feature_selection=use_feature_selection,
            n_features_to_select=n_features_pz,  # Task-specific
            spectra_dataset=spectra_dataset,
            n_mc_samples=n_mc_samples,
            propagate_uncertainty=propagate_uncertainty,  # Enabled for multi-feature
        )
        results_pz.append(result)

        # Save predictions with uncertainty
        save_predictions_to_csv(result, "tumor_vs_normal_pz", output_dir, X_meta_pz)

        # ADC baseline (no selection)
        if "ADC" in X_meta_pz.columns:
            result = evaluate_feature_set(
                X_meta_pz,
                X_features_pz,
                y_pz,
                ["ADC"],
                "ADC (baseline)",
                C=regularization,
                use_feature_selection=False,
                spectra_dataset=spectra_dataset,
                n_mc_samples=n_mc_samples,
                propagate_uncertainty=False,  # ADC is a single feature
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

        # Individual features (no selection for single features)
        for feat in individual_features:
            if feat in X_meta_tz.columns:
                result = evaluate_feature_set(
                    X_meta_tz,
                    X_features_tz,
                    y_tz,
                    [feat],
                    feat,
                    C=regularization,
                    use_feature_selection=False,
                    spectra_dataset=spectra_dataset,
                    n_mc_samples=n_mc_samples,
                    propagate_uncertainty=False,  # Single features don't use classifier
                )
                results_tz.append(result)

        # Combo feature (no selection)
        combo_col = "D[0.25]+1/D[2.0]+1/D[3.0]"
        if combo_col in X_meta_tz.columns:
            result = evaluate_feature_set(
                X_meta_tz,
                X_features_tz,
                y_tz,
                [combo_col],
                combo_col,
                C=regularization,
                use_feature_selection=False,
                spectra_dataset=spectra_dataset,
                n_mc_samples=n_mc_samples,
                propagate_uncertainty=False,  # Single features don't use classifier
            )
            results_tz.append(result)

        # Full model (use task-specific hyperparameters)
        result = evaluate_feature_set(
            X_meta_tz,
            X_features_tz,
            y_tz,
            individual_features,
            "Full LR",
            C=regularization_tz,  # Task-specific
            use_feature_selection=use_feature_selection,
            n_features_to_select=n_features_tz,  # Task-specific
            spectra_dataset=spectra_dataset,
            n_mc_samples=n_mc_samples,
            propagate_uncertainty=propagate_uncertainty,  # Enabled for multi-feature
        )
        results_tz.append(result)

        # Save predictions with uncertainty
        save_predictions_to_csv(result, "tumor_vs_normal_tz", output_dir, X_meta_tz)

        # ADC baseline (no selection)
        if "ADC" in X_meta_tz.columns:
            result = evaluate_feature_set(
                X_meta_tz,
                X_features_tz,
                y_tz,
                ["ADC"],
                "ADC (baseline)",
                C=regularization,
                use_feature_selection=False,
                spectra_dataset=spectra_dataset,
                n_mc_samples=n_mc_samples,
                propagate_uncertainty=False,  # ADC is a single feature
            )
            results_tz.append(result)

        all_results["tumor_vs_normal_tz"] = results_tz
    else:
        print("    [WARNING] No samples available")

    # ------------------------------------------------------------------------
    # Task 3: Gleason Grade Group (GGG 1-2 vs 3-5)
    # ------------------------------------------------------------------------
    print("\n  Task 3: Gleason Grade Group (GGG 1-2 vs 3-5)")
    X_meta_ggg, X_features_ggg, y_ggg = prepare_classification_data(
        features_df, task="ggg", zone=None  # Combined zones
    )

    if len(y_ggg) > 0:
        print(
            f"    Samples: {len(y_ggg)} (GGG 1-2: {np.sum(y_ggg==0)}, GGG 3-5: {np.sum(y_ggg==1)})"
        )

        if len(y_ggg) < 10:
            print("    [WARNING] Very small sample size - results may not be reliable")

        results_ggg = []

        # Individual features (no selection for single features)
        for feat in individual_features:
            if feat in X_meta_ggg.columns:
                result = evaluate_feature_set(
                    X_meta_ggg,
                    X_features_ggg,
                    y_ggg,
                    [feat],
                    feat,
                    C=regularization,
                    use_feature_selection=False,
                    spectra_dataset=spectra_dataset,
                    n_mc_samples=n_mc_samples,
                    propagate_uncertainty=False,  # Single features don't use classifier
                )
                results_ggg.append(result)

        # Combo feature (no selection)
        combo_col = "D[0.25]+1/D[2.0]+1/D[3.0]"
        if combo_col in X_meta_ggg.columns:
            result = evaluate_feature_set(
                X_meta_ggg,
                X_features_ggg,
                y_ggg,
                [combo_col],
                combo_col,
                C=regularization,
                use_feature_selection=False,
                spectra_dataset=spectra_dataset,
                n_mc_samples=n_mc_samples,
                propagate_uncertainty=False,  # Single features don't use classifier
            )
            results_ggg.append(result)

        # Full model (use task-specific hyperparameters)
        result = evaluate_feature_set(
            X_meta_ggg,
            X_features_ggg,
            y_ggg,
            individual_features,
            "Full LR",
            C=regularization_ggg,  # Task-specific
            use_feature_selection=use_feature_selection,
            n_features_to_select=n_features_ggg,  # Task-specific
            spectra_dataset=spectra_dataset,
            n_mc_samples=n_mc_samples,
            propagate_uncertainty=propagate_uncertainty,  # Enabled for multi-feature
        )
        results_ggg.append(result)

        # Save predictions with uncertainty
        save_predictions_to_csv(result, "ggg", output_dir, X_meta_ggg)

        # ADC baseline (no selection)
        if "ADC" in X_meta_ggg.columns:
            result = evaluate_feature_set(
                X_meta_ggg,
                X_features_ggg,
                y_ggg,
                ["ADC"],
                "ADC (baseline)",
                C=regularization,
                use_feature_selection=False,
                spectra_dataset=spectra_dataset,
                n_mc_samples=n_mc_samples,
                propagate_uncertainty=False,  # ADC is a single feature
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

    # Create ISMRM-compatible exports
    print("\n[BONUS] Creating ISMRM abstract figures...")
    try:
        from spectra_estimation_dmri.visualization.ismrm_exports import (
            create_all_ismrm_exports,
        )
        from spectra_estimation_dmri.visualization.bwh_plotting import (
            group_spectra_by_region,
        )

        # Group spectra for regional plots
        regions = group_spectra_by_region(spectra_dataset)

        # Check if we have SNR data (simulation with NUTS)
        has_snr_data = False
        for spectrum in spectra_dataset.spectra[:5]:  # Check first few
            if spectrum.inference_method == "nuts" and spectrum.inference_data:
                if os.path.exists(spectrum.inference_data):
                    has_snr_data = True
                    break

        # Create ISMRM figures
        ismrm_dir = os.path.join(output_dir, "ismrm")
        create_all_ismrm_exports(
            spectra_dataset, all_results, regions, ismrm_dir, include_snr=has_snr_data
        )
    except Exception as e:
        print(f"[WARNING] Failed to create ISMRM exports: {e}")

    print("\n" + "=" * 80)
    print("✓ BIOMARKER ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")

    return {
        "features_df": features_df,
        "uncertainty_df": uncertainty_df,
        "results": all_results,
    }
