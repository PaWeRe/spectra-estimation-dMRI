"""
Visualization tools for biomarker analysis results.

Generates:
- ROC curves with confidence intervals
- AUC comparison tables
- Prediction uncertainty plots
- Feature importance visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve
from typing import Dict, List, Optional, Tuple
import os


def plot_roc_curves(
    results_list: List[Dict],
    output_path: str,
    title: str = "ROC Curves",
    figsize: Tuple[float, float] = (10, 8),
):
    """
    Plot ROC curves for multiple classifiers.

    Args:
        results_list: List of result dictionaries from evaluate_feature_set()
        output_path: Path to save PDF
        title: Plot title
        figsize: Figure size
    """
    # Filter out None results
    valid_results = [
        r
        for r in results_list
        if r is not None and not np.isnan(r.get("metrics", {}).get("auc", np.nan))
    ]

    if not valid_results:
        print(f"[BIOMARKER VIZ] No valid results to plot, skipping ROC curve")
        return

    plt.figure(figsize=figsize)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))

    for i, result in enumerate(valid_results):
        y_true = result["y_true"]
        y_pred_proba = result["y_pred_proba"]
        feature_name = result["feature_name"]
        metrics = result["metrics"]

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

        # Plot
        auc = metrics["auc"]
        ci_lower = metrics.get("auc_ci_lower", np.nan)
        ci_upper = metrics.get("auc_ci_upper", np.nan)

        if not np.isnan(ci_lower):
            label = f"{feature_name} (AUC={auc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])"
        else:
            label = f"{feature_name} (AUC={auc:.3f})"

        plt.plot(fpr, tpr, color=colors[i], lw=2, label=label)

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[BIOMARKER VIZ] Saved ROC curve: {output_path}")


def create_auc_table(
    results_list: List[Dict],
    baseline_result: Optional[Dict] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create AUC comparison table with statistical tests.

    Args:
        results_list: List of result dictionaries
        baseline_result: Optional baseline (e.g., ADC) for comparison
        output_path: Optional path to save CSV

    Returns:
        DataFrame with AUC values, CIs, and p-values
    """
    from .mc_classification import delong_test

    rows = []

    for result in results_list:
        if result is None:
            continue

        metrics = result["metrics"]

        row = {
            "Feature": result["feature_name"],
            "N_Features": result["n_features"],
            "N_Samples": result["n_samples"],
            "N_Positive": result["n_positive"],
            "N_Negative": result["n_negative"],
            "AUC": metrics["auc"],
            "AUC_CI_Lower": metrics.get("auc_ci_lower", np.nan),
            "AUC_CI_Upper": metrics.get("auc_ci_upper", np.nan),
            "Accuracy": metrics["accuracy"],
            "Sensitivity": metrics.get("sensitivity", np.nan),
            "Specificity": metrics.get("specificity", np.nan),
        }

        # Compare to baseline if provided
        if baseline_result is not None:
            p_value = delong_test(
                result["y_true"],
                result["y_pred_proba"],
                baseline_result["y_pred_proba"],
            )
            row["p_value_vs_baseline"] = p_value

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by AUC (descending) if we have results
    if not df.empty and "AUC" in df.columns:
        df = df.sort_values("AUC", ascending=False)

    if output_path is not None:
        df.to_csv(output_path, index=False, float_format="%.4f")
        print(f"[BIOMARKER VIZ] Saved AUC table: {output_path}")

    return df


def plot_prediction_uncertainty(
    y_true: np.ndarray,
    pred_proba_mean: np.ndarray,
    pred_proba_std: np.ndarray,
    output_path: str,
    title: str = "Predictions with Uncertainty",
    figsize: Tuple[float, float] = (12, 6),
):
    """
    Plot predictions with uncertainty (error bars).

    Args:
        y_true: True labels
        pred_proba_mean: Mean predicted probabilities
        pred_proba_std: Std of predictions (uncertainty)
        output_path: Path to save PDF
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by true label for visualization
    sort_idx = np.argsort(y_true)
    y_sorted = y_true[sort_idx]
    pred_mean_sorted = pred_proba_mean[sort_idx]
    pred_std_sorted = pred_proba_std[sort_idx]

    x_pos = np.arange(len(y_true))

    # Color by true label
    colors = ["blue" if label == 0 else "red" for label in y_sorted]

    # Scatter plot with error bars
    ax.scatter(x_pos, pred_mean_sorted, c=colors, alpha=0.6, s=50)
    ax.errorbar(
        x_pos,
        pred_mean_sorted,
        yerr=pred_std_sorted,
        fmt="none",
        alpha=0.3,
        color="gray",
        capsize=3,
    )

    # Decision threshold
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.5, label="Threshold=0.5")

    ax.set_xlabel("Sample Index (sorted by true label)", fontsize=12)
    ax.set_ylabel("P(Positive Class)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(["Threshold", "Negative (true)", "Positive (true)"], loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[BIOMARKER VIZ] Saved uncertainty plot: {output_path}")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: str,
    title: str = "Feature Importance",
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Plot feature importance from logistic regression coefficients.

    Args:
        feature_names: List of feature names
        importances: Absolute coefficient values
        output_path: Path to save PDF
        title: Plot title
        figsize: Figure size
    """
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_importances, color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("Absolute Coefficient (Importance)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # Highest importance on top

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[BIOMARKER VIZ] Saved feature importance: {output_path}")


def create_summary_report(
    results_dict: Dict[str, List[Dict]],
    output_dir: str,
):
    """
    Create comprehensive summary report with all visualizations.

    Args:
        results_dict: Dictionary with keys like 'tumor_vs_normal_pz', 'ggg', etc.
                     Each value is a list of result dicts
        output_dir: Directory to save all outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("BIOMARKER ANALYSIS SUMMARY")
    print("=" * 60)

    # Map task names to better titles
    title_map = {
        "tumor_vs_normal_pz": "Tumor vs Normal Classification (Peripheral Zone)",
        "tumor_vs_normal_tz": "Tumor vs Normal Classification (Transition Zone)",
        "ggg": "Gleason Grade Group Classification (GGG <7 vs ≥7)",
    }

    for task_name, results_list in results_dict.items():
        print(f"\n[{task_name.upper()}]")

        # Separate baseline (ADC) from other results
        baseline_result = None
        spectrum_results = []

        for result in results_list:
            if result is None:
                continue
            if "ADC" in result["feature_name"]:
                baseline_result = result
            else:
                spectrum_results.append(result)

        # Add baseline back for plotting
        all_results = (
            [baseline_result] + spectrum_results
            if baseline_result
            else spectrum_results
        )

        # ROC curves
        roc_path = os.path.join(output_dir, f"roc_{task_name}.pdf")
        plot_title = title_map.get(task_name, f"ROC Curves - {task_name}")
        plot_roc_curves(all_results, roc_path, title=plot_title)

        # AUC table
        auc_table_path = os.path.join(output_dir, f"auc_table_{task_name}.csv")
        auc_df = create_auc_table(all_results, baseline_result, auc_table_path)

        # Print summary
        if auc_df.empty:
            print(f"  [No valid results - insufficient data or single class]")
        else:
            print(f"\n  AUC Summary:")
            for _, row in auc_df.iterrows():
                auc_str = f"{row['AUC']:.3f} [{row['AUC_CI_Lower']:.3f}-{row['AUC_CI_Upper']:.3f}]"
                print(f"    {row['Feature']:30s}: {auc_str}")
                if "p_value_vs_baseline" in row and not np.isnan(
                    row["p_value_vs_baseline"]
                ):
                    p_val = row["p_value_vs_baseline"]
                    sig = (
                        "***"
                        if p_val < 0.001
                        else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    )
                    print(f"      └─ vs baseline: p={p_val:.4f} {sig}")

    print(f"\n✓ Summary report saved to: {output_dir}/")
