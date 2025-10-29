"""
Data Validation Script for Biomarker Analysis

Checks for:
1. Duplicate ROI IDs (data leakage in LOOCV)
2. ADC computation correctness
3. ROC curve computation validation
4. Data distribution for all three tasks (PZ, TZ, GGG)
5. Methodological errors

Run this BEFORE biomarker analysis to ensure data integrity.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spectra_estimation_dmri.data.data_models import DiffusivitySpectraDataset
from spectra_estimation_dmri.biomarkers.features import extract_features_from_dataset
from spectra_estimation_dmri.biomarkers.adc_baseline import extract_adc_features
from spectra_estimation_dmri.biomarkers.mc_classification import prepare_classification_data
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


def check_duplicate_rois(features_df: pd.DataFrame) -> dict:
    """
    Check for duplicate ROI IDs which would cause data leakage in LOOCV.
    
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*80)
    print("1. CHECKING FOR DUPLICATE ROI IDs")
    print("="*80)
    
    results = {
        "total_samples": len(features_df),
        "unique_rois": features_df["roi_id"].nunique(),
        "duplicates_found": False,
        "duplicate_details": []
    }
    
    # Count occurrences of each ROI
    roi_counts = features_df["roi_id"].value_counts()
    duplicates = roi_counts[roi_counts > 1]
    
    if len(duplicates) > 0:
        results["duplicates_found"] = True
        print(f"⚠️  WARNING: Found {len(duplicates)} duplicate ROI IDs!")
        print(f"\nDuplicate ROI IDs:")
        for roi_id, count in duplicates.items():
            print(f"  {roi_id}: {count} occurrences")
            dup_rows = features_df[features_df["roi_id"] == roi_id]
            results["duplicate_details"].append({
                "roi_id": roi_id,
                "count": count,
                "regions": dup_rows["region"].tolist(),
                "is_tumor": dup_rows["is_tumor"].tolist(),
            })
        print(f"\n❌ CRITICAL: Duplicates will cause data leakage in LOOCV!")
        print(f"   Each ROI should appear exactly once in the dataset.")
    else:
        print(f"✓ No duplicates found")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Unique ROIs: {results['unique_rois']}")
    
    return results


def validate_adc_computation(spectra_dataset, adc_df: pd.DataFrame) -> dict:
    """
    Validate ADC computation by:
    1. Checking against manual calculation
    2. Verifying expected range (0.5-3.0 x10^-3 mm²/s for prostate)
    3. Checking for NaN values
    """
    print("\n" + "="*80)
    print("2. VALIDATING ADC COMPUTATION")
    print("="*80)
    
    results = {
        "n_samples": len(adc_df),
        "n_nan": adc_df["adc"].isna().sum(),
        "adc_range": (adc_df["adc"].min(), adc_df["adc"].max()),
        "adc_mean": adc_df["adc"].mean(),
        "adc_std": adc_df["adc"].std(),
        "out_of_range": 0,
    }
    
    print(f"ADC Statistics:")
    print(f"  Samples: {results['n_samples']}")
    print(f"  NaN values: {results['n_nan']}")
    print(f"  Range: {results['adc_range'][0]:.4f} - {results['adc_range'][1]:.4f} x10^-3 mm²/s")
    print(f"  Mean: {results['adc_mean']:.4f} ± {results['adc_std']:.4f}")
    
    # Check expected physiological range for prostate (0.5 - 3.0)
    expected_min, expected_max = 0.5, 3.0
    out_of_range = adc_df[(adc_df["adc"] < expected_min) | (adc_df["adc"] > expected_max)]
    results["out_of_range"] = len(out_of_range)
    
    if len(out_of_range) > 0:
        print(f"\n⚠️  WARNING: {len(out_of_range)} samples outside expected range ({expected_min}-{expected_max})")
        print(f"   This may indicate computation errors or unusual tissue.")
    else:
        print(f"✓ All ADC values in expected physiological range")
    
    # Manual validation on first sample
    print(f"\nManual Validation (first sample):")
    spectrum = spectra_dataset.spectra[0]
    signal_decay = spectrum.signal_decay
    b_values = np.array(signal_decay.b_values)
    signal_values = np.array(signal_decay.signal_values)
    
    # Manual ADC calculation (b=0-1000 s/mm²)
    mask = (b_values >= 0) & (b_values <= 1000) & (signal_values > 0)
    if np.any(mask):
        log_signal = np.log(signal_values[mask])
        b_valid = b_values[mask]
        slope, intercept = np.polyfit(b_valid, log_signal, 1)
        manual_adc = -slope
        
        roi_id = f"{signal_decay.patient}_{signal_decay.a_region}"
        computed_adc = adc_df[adc_df["roi_id"] == roi_id]["adc"].values[0] if roi_id in adc_df["roi_id"].values else np.nan
        
        print(f"  ROI: {roi_id}")
        print(f"  Manual ADC: {manual_adc:.4f}")
        print(f"  Computed ADC: {computed_adc:.4f}")
        print(f"  Difference: {abs(manual_adc - computed_adc):.6f}")
        
        if abs(manual_adc - computed_adc) < 0.01:
            print(f"  ✓ Match!")
        else:
            print(f"  ⚠️  WARNING: Large difference detected!")
    
    return results


def check_task_distributions(features_df: pd.DataFrame) -> dict:
    """
    Check data distribution for all three tasks.
    """
    print("\n" + "="*80)
    print("3. CHECKING TASK DISTRIBUTIONS")
    print("="*80)
    
    results = {}
    
    # Task 1: Tumor vs Normal (PZ)
    print("\nTask 1: Tumor vs Normal (PZ)")
    X_meta_pz, X_features_pz, y_pz = prepare_classification_data(
        features_df, task="tumor_vs_normal", zone="pz"
    )
    
    if len(y_pz) > 0:
        n_normal = np.sum(y_pz == 0)
        n_tumor = np.sum(y_pz == 1)
        balance_ratio = min(n_normal, n_tumor) / max(n_normal, n_tumor)
        
        results["pz"] = {
            "total": len(y_pz),
            "normal": n_normal,
            "tumor": n_tumor,
            "balance_ratio": balance_ratio,
        }
        
        print(f"  Total: {len(y_pz)}")
        print(f"  Normal: {n_normal} ({100*n_normal/len(y_pz):.1f}%)")
        print(f"  Tumor: {n_tumor} ({100*n_tumor/len(y_pz):.1f}%)")
        print(f"  Balance ratio: {balance_ratio:.2f}")
        
        if balance_ratio < 0.3:
            print(f"  ⚠️  WARNING: Severe class imbalance!")
        elif balance_ratio < 0.5:
            print(f"  ⚠️  Moderate class imbalance")
        else:
            print(f"  ✓ Reasonable class balance")
            
        # Check for patient-level leakage
        patient_ids_pz = X_meta_pz["patient_id"].values
        unique_patients = len(set(patient_ids_pz))
        print(f"  Unique patients: {unique_patients}")
        results["pz"]["unique_patients"] = unique_patients
    
    # Task 2: Tumor vs Normal (TZ)
    print("\nTask 2: Tumor vs Normal (TZ)")
    X_meta_tz, X_features_tz, y_tz = prepare_classification_data(
        features_df, task="tumor_vs_normal", zone="tz"
    )
    
    if len(y_tz) > 0:
        n_normal = np.sum(y_tz == 0)
        n_tumor = np.sum(y_tz == 1)
        balance_ratio = min(n_normal, n_tumor) / max(n_normal, n_tumor) if max(n_normal, n_tumor) > 0 else 0
        
        results["tz"] = {
            "total": len(y_tz),
            "normal": n_normal,
            "tumor": n_tumor,
            "balance_ratio": balance_ratio,
        }
        
        print(f"  Total: {len(y_tz)}")
        print(f"  Normal: {n_normal} ({100*n_normal/len(y_tz):.1f}%)")
        print(f"  Tumor: {n_tumor} ({100*n_tumor/len(y_tz):.1f}%)")
        print(f"  Balance ratio: {balance_ratio:.2f}")
        
        if balance_ratio < 0.3:
            print(f"  ⚠️  WARNING: Severe class imbalance!")
        elif balance_ratio < 0.5:
            print(f"  ⚠️  Moderate class imbalance")
        else:
            print(f"  ✓ Reasonable class balance")
            
        patient_ids_tz = X_meta_tz["patient_id"].values
        unique_patients = len(set(patient_ids_tz))
        print(f"  Unique patients: {unique_patients}")
        results["tz"]["unique_patients"] = unique_patients
    
    # Task 3: GGG (<7 vs >=7)
    print("\nTask 3: Gleason Grade Group (<7 vs >=7)")
    X_meta_ggg, X_features_ggg, y_ggg = prepare_classification_data(
        features_df, task="ggg", zone=None
    )
    
    if len(y_ggg) > 0:
        n_low = np.sum(y_ggg == 0)
        n_high = np.sum(y_ggg == 1)
        balance_ratio = min(n_low, n_high) / max(n_low, n_high) if max(n_low, n_high) > 0 else 0
        
        results["ggg"] = {
            "total": len(y_ggg),
            "ggg_low": n_low,
            "ggg_high": n_high,
            "balance_ratio": balance_ratio,
        }
        
        print(f"  Total: {len(y_ggg)}")
        print(f"  GGG <7 (GGG 1-2): {n_low} ({100*n_low/len(y_ggg):.1f}%)")
        print(f"  GGG >=7 (GGG 3-5): {n_high} ({100*n_high/len(y_ggg):.1f}%)")
        print(f"  Balance ratio: {balance_ratio:.2f}")
        
        if len(y_ggg) < 20:
            print(f"  ⚠️  WARNING: Very small sample size for GGG task!")
        
        if balance_ratio < 0.3:
            print(f"  ⚠️  WARNING: Severe class imbalance!")
        elif balance_ratio < 0.5:
            print(f"  ⚠️  Moderate class imbalance")
        else:
            print(f"  ✓ Reasonable class balance")
            
        patient_ids_ggg = X_meta_ggg["patient_id"].values
        unique_patients = len(set(patient_ids_ggg))
        print(f"  Unique patients: {unique_patients}")
        results["ggg"]["unique_patients"] = unique_patients
        
        # Show GGG distribution
        ggg_values = X_meta_ggg["ggg"].values
        print(f"\n  GGG Distribution:")
        for ggg_val in sorted(set(ggg_values)):
            count = np.sum(ggg_values == ggg_val)
            print(f"    GGG {int(ggg_val)}: {count} samples")
    
    return results


def validate_roc_computation(features_df: pd.DataFrame) -> dict:
    """
    Validate ROC curve computation by checking:
    1. AUC calculation consistency
    2. ROC curve properties (monotonicity)
    3. Expected ADC performance (should be high for tumor detection)
    """
    print("\n" + "="*80)
    print("4. VALIDATING ROC CURVE COMPUTATION")
    print("="*80)
    
    results = {}
    
    # Test on PZ tumor vs normal with ADC
    X_meta_pz, X_features_pz, y_pz = prepare_classification_data(
        features_df, task="tumor_vs_normal", zone="pz"
    )
    
    if len(y_pz) > 0 and "ADC" in X_meta_pz.columns:
        adc_values = X_meta_pz["ADC"].values
        
        # Compute AUC two ways
        auc_sklearn = roc_auc_score(y_pz, -adc_values)  # Negative because lower ADC = tumor
        
        # Manual AUC calculation (Wilcoxon-Mann-Whitney U-statistic)
        tumor_adc = adc_values[y_pz == 1]
        normal_adc = adc_values[y_pz == 0]
        
        # Count concordant pairs
        n_concordant = np.sum(tumor_adc[:, None] < normal_adc[None, :])
        n_total = len(tumor_adc) * len(normal_adc)
        auc_manual = n_concordant / n_total
        
        results["adc_pz"] = {
            "auc_sklearn": auc_sklearn,
            "auc_manual": auc_manual,
            "difference": abs(auc_sklearn - auc_manual),
            "tumor_adc_mean": np.mean(tumor_adc),
            "normal_adc_mean": np.mean(normal_adc),
            "expected_direction": np.mean(tumor_adc) < np.mean(normal_adc),
        }
        
        print(f"\nADC for PZ Tumor vs Normal:")
        print(f"  AUC (sklearn): {auc_sklearn:.4f}")
        print(f"  AUC (manual):  {auc_manual:.4f}")
        print(f"  Difference: {results['adc_pz']['difference']:.6f}")
        
        if results['adc_pz']['difference'] < 0.01:
            print(f"  ✓ AUC calculations match!")
        else:
            print(f"  ⚠️  WARNING: AUC calculations don't match!")
        
        print(f"\n  Tumor ADC: {results['adc_pz']['tumor_adc_mean']:.4f}")
        print(f"  Normal ADC: {results['adc_pz']['normal_adc_mean']:.4f}")
        
        if results['adc_pz']['expected_direction']:
            print(f"  ✓ Expected direction (Tumor < Normal)")
        else:
            print(f"  ⚠️  WARNING: Unexpected direction (Tumor >= Normal)!")
        
        if auc_sklearn > 0.7:
            print(f"  ✓ Good discriminative performance (AUC > 0.7)")
        elif auc_sklearn > 0.5:
            print(f"  ⚠️  Moderate performance (0.5 < AUC < 0.7)")
        else:
            print(f"  ❌ Poor performance (AUC < 0.5) - possible error!")
        
        # Plot ROC curve for visual validation
        fpr, tpr, thresholds = roc_curve(y_pz, -adc_values)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', lw=2, label=f'ADC (AUC={auc_sklearn:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Validation: ADC for PZ Tumor Detection')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = "results/biomarkers/validation_roc_adc_pz.pdf"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"\n  ROC curve saved: {output_path}")
    
    return results


def plot_adc_distributions(features_df: pd.DataFrame):
    """
    Plot ADC distributions for visual inspection.
    """
    print("\n" + "="*80)
    print("5. PLOTTING ADC DISTRIBUTIONS")
    print("="*80)
    
    # Add zone column
    features_df["zone"] = features_df["region"].apply(
        lambda x: "pz" if "pz" in str(x).lower() else ("tz" if "tz" in str(x).lower() else "unknown")
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: PZ - Tumor vs Normal
    ax = axes[0, 0]
    pz_data = features_df[features_df["zone"] == "pz"]
    if len(pz_data) > 0:
        tumor_pz = pz_data[pz_data["is_tumor"] == True]["ADC"].dropna()
        normal_pz = pz_data[pz_data["is_tumor"] == False]["ADC"].dropna()
        
        ax.hist(normal_pz, bins=15, alpha=0.6, label=f'Normal (n={len(normal_pz)})', color='blue')
        ax.hist(tumor_pz, bins=15, alpha=0.6, label=f'Tumor (n={len(tumor_pz)})', color='red')
        ax.set_xlabel('ADC (×10⁻³ mm²/s)')
        ax.set_ylabel('Count')
        ax.set_title('PZ: Tumor vs Normal')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Plot 2: TZ - Tumor vs Normal
    ax = axes[0, 1]
    tz_data = features_df[features_df["zone"] == "tz"]
    if len(tz_data) > 0:
        tumor_tz = tz_data[tz_data["is_tumor"] == True]["ADC"].dropna()
        normal_tz = tz_data[tz_data["is_tumor"] == False]["ADC"].dropna()
        
        ax.hist(normal_tz, bins=15, alpha=0.6, label=f'Normal (n={len(normal_tz)})', color='blue')
        ax.hist(tumor_tz, bins=15, alpha=0.6, label=f'Tumor (n={len(tumor_tz)})', color='red')
        ax.set_xlabel('ADC (×10⁻³ mm²/s)')
        ax.set_ylabel('Count')
        ax.set_title('TZ: Tumor vs Normal')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Plot 3: GGG distribution (tumor only)
    ax = axes[1, 0]
    tumor_data = features_df[features_df["is_tumor"] == True]
    if len(tumor_data) > 0:
        ggg_low = tumor_data[tumor_data["ggg"] < 3]["ADC"].dropna()
        ggg_high = tumor_data[tumor_data["ggg"] >= 3]["ADC"].dropna()
        
        ax.hist(ggg_low, bins=15, alpha=0.6, label=f'GGG <7 (n={len(ggg_low)})', color='orange')
        ax.hist(ggg_high, bins=15, alpha=0.6, label=f'GGG >=7 (n={len(ggg_high)})', color='darkred')
        ax.set_xlabel('ADC (×10⁻³ mm²/s)')
        ax.set_ylabel('Count')
        ax.set_title('Gleason Grade Stratification')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Plot 4: Overall ADC distribution by tissue type
    ax = axes[1, 1]
    if "ADC" in features_df.columns:
        for tissue, color in [("Normal", "blue"), ("Tumor", "red")]:
            is_tumor = (tissue == "Tumor")
            adc_vals = features_df[features_df["is_tumor"] == is_tumor]["ADC"].dropna()
            if len(adc_vals) > 0:
                ax.hist(adc_vals, bins=20, alpha=0.5, label=f'{tissue} (n={len(adc_vals)})', color=color)
        ax.set_xlabel('ADC (×10⁻³ mm²/s)')
        ax.set_ylabel('Count')
        ax.set_title('Overall ADC Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = "results/biomarkers/validation_adc_distributions.pdf"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ADC distributions saved: {output_path}")


def main():
    """
    Run all validation checks.
    """
    print("\n" + "="*80)
    print("BIOMARKER DATA VALIDATION")
    print("="*80)
    print("This script validates data integrity before biomarker analysis.")
    
    # Load dataset
    print("\nLoading dataset...")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    
    config_dir = str(Path(__file__).parent.parent / "configs")
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[
            "dataset=bwh",
            "inference=nuts",
            "prior=ridge",
            "local=true",
        ])
    
    # Load spectra dataset
    from spectra_estimation_dmri.data.loaders import load_bwh_dataset
    spectra_dataset = load_bwh_dataset(
        data_path=cfg.dataset.data_path,
        num_diffusivities=cfg.dataset.num_diffusivities,
        diffusivity_range=cfg.dataset.diffusivity_range,
        cache_dir=f"outputs/{cfg.run_id}",
    )
    
    print(f"Loaded {len(spectra_dataset.spectra)} spectra")
    
    # Extract features
    print("\nExtracting features...")
    features_df, uncertainty_df = extract_features_from_dataset(
        spectra_dataset, n_mc_samples=200
    )
    
    # Extract ADC
    adc_df = extract_adc_features(spectra_dataset, b_range=(0.0, 1.0))
    features_df = features_df.merge(adc_df[["roi_id", "adc"]], on="roi_id", how="left")
    features_df = features_df.rename(columns={"adc": "ADC"})
    
    # Run validation checks
    all_results = {}
    
    # 1. Check duplicates
    all_results["duplicates"] = check_duplicate_rois(features_df)
    
    # 2. Validate ADC
    all_results["adc"] = validate_adc_computation(spectra_dataset, adc_df)
    
    # 3. Check distributions
    all_results["distributions"] = check_task_distributions(features_df)
    
    # 4. Validate ROC
    all_results["roc"] = validate_roc_computation(features_df)
    
    # 5. Plot distributions
    plot_adc_distributions(features_df)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    critical_issues = []
    warnings = []
    
    if all_results["duplicates"]["duplicates_found"]:
        critical_issues.append("Duplicate ROI IDs found - LOOCV will leak data!")
    
    if all_results["adc"]["n_nan"] > 0:
        warnings.append(f"{all_results['adc']['n_nan']} ADC values are NaN")
    
    if all_results["adc"]["out_of_range"] > 0:
        warnings.append(f"{all_results['adc']['out_of_range']} ADC values out of physiological range")
    
    for task in ["pz", "tz", "ggg"]:
        if task in all_results["distributions"]:
            if all_results["distributions"][task]["balance_ratio"] < 0.3:
                warnings.append(f"{task.upper()}: Severe class imbalance")
    
    if "adc_pz" in all_results["roc"]:
        if all_results["roc"]["adc_pz"]["difference"] > 0.01:
            critical_issues.append("ROC AUC calculation mismatch!")
        if not all_results["roc"]["adc_pz"]["expected_direction"]:
            critical_issues.append("ADC shows unexpected direction (Tumor >= Normal)")
        if all_results["roc"]["adc_pz"]["auc_sklearn"] < 0.7:
            warnings.append(f"Low ADC AUC ({all_results['roc']['adc_pz']['auc_sklearn']:.3f}) - expected >0.7")
    
    if len(critical_issues) > 0:
        print("\n❌ CRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"  - {issue}")
    
    if len(warnings) > 0:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if len(critical_issues) == 0 and len(warnings) == 0:
        print("\n✓ ALL CHECKS PASSED!")
    
    print("\nValidation complete. Review results above before running biomarker analysis.")


if __name__ == "__main__":
    main()
