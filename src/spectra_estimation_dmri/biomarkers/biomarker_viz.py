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
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve
from typing import Dict, List, Optional, Tuple
from scipy import stats
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

    # Add sample counts to title if we have results
    if valid_results:
        n_samples = len(valid_results[0]["y_true"])
        n_positive = np.sum(valid_results[0]["y_true"] == 1)
        n_negative = n_samples - n_positive
        title_with_counts = (
            f"{title}\nN={n_samples} ({n_negative} neg, {n_positive} pos)"
        )
    else:
        title_with_counts = title

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title_with_counts, fontsize=14, fontweight="bold")
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
    figsize: Tuple[float, float] = (14, 10),
):
    """
    Plot predictions with uncertainty (error bars) and uncertainty analysis.

    Args:
        y_true: True labels
        pred_proba_mean: Mean predicted probabilities
        pred_proba_std: Std of predictions (MC uncertainty)
        output_path: Path to save PDF
        title: Plot title
        figsize: Figure size
    """
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Sort by prediction probability for better visualization
    sort_idx = np.argsort(pred_proba_mean)
    y_sorted = y_true[sort_idx]
    pred_sorted = pred_proba_mean[sort_idx]
    std_sorted = pred_proba_std[sort_idx]

    n_samples = len(y_true)
    x_pos = np.arange(n_samples)

    # Identify correct vs incorrect predictions
    y_pred_class = (pred_sorted >= 0.5).astype(int)
    correct = y_pred_class == y_sorted

    # ============================================================
    # SUBPLOT 1: Predictions with Uncertainty Error Bars
    # ============================================================
    ax1 = axes[0]

    # Plot each point with error bar, colored by true label, marker by correctness
    for i, (x, y, std, is_correct, true_label) in enumerate(
        zip(x_pos, pred_sorted, std_sorted, correct, y_sorted)
    ):
        color = "#3498db" if true_label == 0 else "#e74c3c"
        marker = "o" if is_correct else "X"
        markersize = 6 if is_correct else 8
        alpha = 0.6 if is_correct else 0.9

        ax1.errorbar(
            x,
            y,
            yerr=std,
            fmt=marker,
            color=color,
            markersize=markersize,
            alpha=alpha,
            capsize=3,
            capthick=1,
            elinewidth=1.5,
        )

    # Decision threshold
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=2, alpha=0.7)

    # Formatting
    ax1.set_xlabel(
        "Sample Index (sorted by prediction)", fontsize=12, fontweight="bold"
    )
    ax1.set_ylabel("P(Positive Class)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"{title}\nPredictions with MC Uncertainty", fontsize=14, fontweight="bold"
    )
    ax1.grid(alpha=0.3, linestyle=":", linewidth=1)
    ax1.set_ylim([-0.05, 1.05])

    # Enhanced legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#3498db",
            markersize=10,
            label="True Negative (correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=10,
            label="True Positive (correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#3498db",
            markersize=10,
            label="True Negative (incorrect)",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=10,
            label="True Positive (incorrect)",
        ),
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=2, label="Threshold=0.5"
        ),
    ]
    ax1.legend(handles=legend_elements, loc="best", fontsize=9, framealpha=0.9)

    # ============================================================
    # SUBPLOT 2: Uncertainty vs Prediction Confidence
    # ============================================================
    ax2 = axes[1]

    # Compute distance from decision boundary (measure of confidence)
    confidence = np.abs(pred_sorted - 0.5)

    # Scatter plot: confidence vs uncertainty
    ax2.scatter(
        confidence[correct],
        std_sorted[correct],
        c="#2ecc71",
        s=80,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
        label="Correct predictions",
    )
    ax2.scatter(
        confidence[~correct],
        std_sorted[~correct],
        c="#e67e22",
        s=120,
        alpha=0.8,
        edgecolors="black",
        linewidths=1.0,
        marker="X",
        label="Incorrect predictions",
    )

    # Compute correlation
    if np.std(confidence) > 0 and np.std(std_sorted) > 0:
        corr = np.corrcoef(confidence, std_sorted)[0, 1]
        ax2.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax2.transAxes,
            fontsize=11,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Statistics box
    mean_unc_correct = np.mean(std_sorted[correct]) if np.any(correct) else 0
    mean_unc_incorrect = np.mean(std_sorted[~correct]) if np.any(~correct) else 0

    stats_text = (
        f"Mean MC uncertainty:\n"
        f"  Correct: {mean_unc_correct:.4f}\n"
        f"  Incorrect: {mean_unc_incorrect:.4f}\n"
        f"  Ratio: {mean_unc_incorrect/mean_unc_correct:.2f}x"
        if mean_unc_correct > 0
        else "N/A"
    )
    ax2.text(
        0.95,
        0.05,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # Formatting
    ax2.set_xlabel(
        "Prediction Confidence (distance from 0.5)", fontsize=12, fontweight="bold"
    )
    ax2.set_ylabel("MC Uncertainty (std)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Does Uncertainty Correlate with Errors?", fontsize=13, fontweight="bold"
    )
    ax2.grid(alpha=0.3, linestyle=":", linewidth=1)
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[BIOMARKER VIZ] Saved uncertainty plot: {output_path}")
    print(
        f"[BIOMARKER VIZ] Accuracy: {np.sum(correct)}/{n_samples} = {np.mean(correct):.3f}"
    )
    print(f"[BIOMARKER VIZ] Mean uncertainty (correct): {mean_unc_correct:.4f}")
    print(f"[BIOMARKER VIZ] Mean uncertainty (incorrect): {mean_unc_incorrect:.4f}")
    if mean_unc_correct > 0:
        print(
            f"[BIOMARKER VIZ] Uncertainty ratio: {mean_unc_incorrect/mean_unc_correct:.2f}x higher for errors"
        )


def save_feature_importance_table(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """
    Save feature importance as CSV table.

    Args:
        feature_names: List of feature names
        importances: Feature importance values (LR coefficients)
        output_path: Path to save CSV

    Returns:
        DataFrame with feature importance
    """
    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": importances,
            "Abs_Coefficient": np.abs(importances),
        }
    )

    # Sort by absolute importance
    df = df.sort_values("Abs_Coefficient", ascending=False)
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"[BIOMARKER VIZ] Saved feature importance table: {output_path}")

    return df


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

    # Add magnitude info to title
    max_coef = np.max(sorted_importances)
    title_with_info = f"{title}\n(max |coef| = {max_coef:.3f})"
    ax.set_title(title_with_info, fontsize=14, fontweight="bold")

    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # Highest importance on top

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[BIOMARKER VIZ] Saved feature importance: {output_path}")


def create_loocv_feature_importance_plot(
    results_dict: Dict[str, List[Dict]],
    output_path: str,
):
    """
    Create bar plot showing LOOCV feature importance (actual features used in CV).

    Shows mean coefficient ± std across LOOCV folds for each task,
    ranked by magnitude. Format matches combined_feature_importance plot.

    Args:
        results_dict: Dictionary with task results
        output_path: Path to save plot
    """
    # Task configurations (matching combined plot order)
    task_configs = [
        ("tumor_vs_normal_pz", "PZ: Tumor Detection"),
        ("tumor_vs_normal_tz", "TZ: Tumor Detection"),
        ("ggg", "GGG(PZ) 1-2 vs 3-5"),
    ]

    # Extract LOOCV feature importance for each task
    task_data = []
    for task_name, title in task_configs:
        if task_name not in results_dict:
            task_data.append(None)
            continue

        # Find Full LR result with LOOCV feature importance
        loocv_importance = None
        n_samples = None
        n_folds = None
        for result in results_dict[task_name]:
            if result is not None and result.get("feature_name") == "Full LR":
                loocv_importance = result.get("loocv_feature_importance")
                n_samples = result.get("n_samples")
                if loocv_importance:
                    # Count how many folds were tracked
                    n_folds = max(
                        stats["n_folds_selected"] for stats in loocv_importance.values()
                    )
                    break

        if loocv_importance:
            task_data.append(
                {
                    "title": title,
                    "importance": loocv_importance,
                    "n_samples": n_samples,
                    "n_folds": n_folds,
                }
            )
        else:
            task_data.append(None)

    # Create 2x2 subplot layout (matching combined plot)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (task, ax) in enumerate(zip(task_data, axes[:3])):
        if task is None:
            ax.axis("off")
            continue

        importance = task["importance"]

        # Sort features by mean absolute coefficient
        sorted_features = sorted(
            importance.items(),
            key=lambda x: abs(x[1]["mean_coefficient"]),
            reverse=True,
        )

        feature_names = [
            f[0].replace("D_", "D ") for f in sorted_features
        ]  # Format like "D 0.25"
        mean_coefs = [f[1]["mean_coefficient"] for f in sorted_features]
        std_coefs = [f[1]["std_coefficient"] for f in sorted_features]
        sel_freq = [f[1]["selection_frequency"] for f in sorted_features]

        # Color based on sign (matching combined plot style)
        colors = []
        for coef in mean_coefs:
            if coef > 0:  # Positive (increases tumor probability)
                colors.append("steelblue")
            else:  # Negative (decreases tumor probability)
                colors.append("indianred")

        # Create bar plot with error bars
        x_pos = np.arange(len(feature_names))
        abs_mean = np.array([abs(c) for c in mean_coefs])

        ax.bar(
            x_pos,
            abs_mean,
            yerr=std_coefs,
            color=colors,
            alpha=0.7,
            capsize=4,
            error_kw={"linewidth": 1.5, "elinewidth": 1.5},
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Absolute LR Coefficient", fontsize=10)

        # Enhanced title with counts
        title_text = f"{task['title']}\n"
        title_text += f"(n={task['n_samples']} samples, {task['n_folds']} LOOCV folds, "
        title_text += f"{len(feature_names)} features selected)"
        ax.set_title(title_text, fontsize=10, fontweight="bold")

        ax.grid(axis="y", alpha=0.3)

        # Add legend in top-left subplot only
        if idx == 0:
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="steelblue", alpha=0.7, label="Positive"),
                Patch(facecolor="indianred", alpha=0.7, label="Negative"),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
            )

    # Hide 4th subplot
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[BIOMARKER VIZ] Saved LOOCV feature importance: {output_path}")


def create_combined_uncertainty_plot(
    results_dict: Dict[str, List[Dict]],
    output_dir: str,
):
    """
    Create combined ISMRM-style uncertainty plot for all three tasks.

    Args:
        results_dict: Dictionary with task results
        output_dir: Directory to save output
    """
    # Task configurations (task_name, title, labels, default_counts)
    task_configs = [
        ("tumor_vs_normal_pz", "PZ: Tumor Detection", ("Normal", "Tumor")),
        ("tumor_vs_normal_tz", "TZ: Tumor Detection", ("Normal", "Tumor")),
        ("ggg", "GGG(PZ) 1-2 vs 3-5", ("GGG 1-2", "GGG 3-5")),
    ]

    # Extract Full LR results with uncertainty for each task
    task_data = []
    for task_name, title, labels in task_configs:
        if task_name not in results_dict:
            task_data.append(None)
            continue

        # Find Full LR result with uncertainty
        full_lr_result = None
        for result in results_dict[task_name]:
            if result is not None and result.get("feature_name") == "Full LR":
                if result.get("y_pred_std") is not None:
                    full_lr_result = result
                break

        if full_lr_result is None:
            task_data.append(None)
        else:
            task_data.append(
                {
                    "title": title,
                    "labels": labels,
                    "y_true": full_lr_result["y_true"],
                    "y_pred_proba": full_lr_result["y_pred_proba"],
                    "y_pred_std": full_lr_result["y_pred_std"],
                    "n_normal": np.sum(full_lr_result["y_true"] == 0),
                    "n_tumor": np.sum(full_lr_result["y_true"] == 1),
                }
            )

    # Check if we have any data
    if all(d is None for d in task_data):
        print("  [INFO] No uncertainty data available for combined plot")
        return

    # Create figure with 2x2 layout (matching ROC combined plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    stats_summary = []

    for i, (data, (task_name, _, _)) in enumerate(zip(task_data, task_configs)):
        ax = axes[i]
        if data is None:
            # No data for this task
            ax.text(
                0.5,
                0.5,
                "Uncertainty data not available\n(run with propagate_uncertainty=true)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
            )
            ax.set_title(
                task_configs[[tc[0] for tc in task_configs].index(task_name)][1],
                fontsize=12,
                fontweight="bold",
            )
            ax.axis("off")
            continue

        # Extract data
        y_true = data["y_true"]
        y_pred_proba = data["y_pred_proba"]
        y_pred_std = data["y_pred_std"]
        title = data["title"]
        labels = data["labels"]
        n_normal = data["n_normal"]
        n_tumor = data["n_tumor"]

        # Sort by prediction probability
        n_samples = len(y_true)
        sort_idx = np.argsort(y_pred_proba)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred_proba[sort_idx]
        y_std_sorted = y_pred_std[sort_idx]

        # Identify correct vs incorrect
        y_pred_class = (y_pred_sorted >= 0.5).astype(int)
        correct = y_pred_class == y_true_sorted

        x_pos = np.arange(n_samples)

        # Plot each point with error bar
        for i, (x, y, std, is_correct, true_label) in enumerate(
            zip(x_pos, y_pred_sorted, y_std_sorted, correct, y_true_sorted)
        ):
            color = "#3498db" if true_label == 0 else "#e74c3c"
            marker = "o" if is_correct else "X"
            markersize = 6 if is_correct else 8
            alpha = 0.6 if is_correct else 0.9

            ax.errorbar(
                x,
                y,
                yerr=std,
                fmt=marker,
                color=color,
                markersize=markersize,
                alpha=alpha,
                capsize=3,
                capthick=1,
                elinewidth=1.5,
            )

        # Decision threshold
        ax.axhline(0.5, color="black", linestyle="--", linewidth=2, alpha=0.7)

        # Formatting
        ax.set_xlabel(
            "Sample Index (sorted by prediction)", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel(f"P({labels[1]})", fontsize=11, fontweight="bold")

        # Title with sample counts (matching ROC plot style)
        title_with_n = f"{title} (n={n_tumor} {labels[1].lower()}, n={n_normal} {labels[0].lower()})"
        ax.set_title(title_with_n, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, linestyle=":", linewidth=1)
        ax.set_ylim([-0.05, 1.05])

        # Compute statistics
        mean_unc_correct = np.mean(y_std_sorted[correct]) if np.any(correct) else 0
        mean_unc_incorrect = np.mean(y_std_sorted[~correct]) if np.any(~correct) else 0

        # Compute ratio
        if mean_unc_correct > 0:
            ratio = mean_unc_incorrect / mean_unc_correct
        else:
            ratio = 0

        # Statistical test (t-test for difference in means)
        p_value = None
        if np.any(correct) and np.any(~correct):
            try:
                # Use independent t-test to compare uncertainty distributions
                t_stat, p_value = stats.ttest_ind(
                    y_std_sorted[~correct],  # incorrect samples
                    y_std_sorted[correct],  # correct samples
                )
            except:
                p_value = None

        # Build statistics text
        if p_value is not None:
            stats_text = (
                f"Uncertainty:\n"
                f"  Correct: {mean_unc_correct:.4f}\n"
                f"  Incorrect: {mean_unc_incorrect:.4f}\n"
                f"  Ratio: {ratio:.2f}x\n"
                f"  p={p_value:.4f}"
            )
        else:
            stats_text = (
                f"Uncertainty:\n"
                f"  Correct: {mean_unc_correct:.4f}\n"
                f"  Incorrect: {mean_unc_incorrect:.4f}\n"
                f"  Ratio: {ratio:.2f}x"
            )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        stats_summary.append(
            {
                "task": task_name,
                "mean_unc_correct": mean_unc_correct,
                "mean_unc_incorrect": mean_unc_incorrect,
                "ratio": ratio,
                "p_value": p_value,
                "n_correct": np.sum(correct),
                "n_samples": n_samples,
            }
        )

    # Hide the 4th subplot (bottom right) to match ROC layout
    axes[3].axis("off")

    # Add common legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#3498db",
            markersize=10,
            label="True Negative (correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=10,
            label="True Positive (correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#3498db",
            markersize=10,
            label="False Positive (incorrect)",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=10,
            label="False Negative (incorrect)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Decision Threshold (0.5)",
        ),
    ]

    # Add legend to the 4th (empty) subplot
    axes[3].legend(
        handles=legend_elements,
        loc="center",
        fontsize=11,
        framealpha=0.9,
    )

    plt.tight_layout()

    # Save plot
    ismrm_dir = os.path.join(output_dir, "ismrm")
    os.makedirs(ismrm_dir, exist_ok=True)

    output_path = os.path.join(ismrm_dir, "combined_uncertainty_ismrm.png")
    pdf_path = os.path.join(ismrm_dir, "combined_uncertainty_ismrm.pdf")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"\n  [INFO] Saved combined uncertainty plot:")
    print(f"    PNG: {output_path}")
    print(f"    PDF: {pdf_path}")

    plt.close()

    # Print summary statistics
    if stats_summary:
        print("\n  UNCERTAINTY ANALYSIS SUMMARY:")
        for stats in stats_summary:
            task = stats["task"].upper().replace("_", " ")
            ratio = stats["ratio"]
            p_val = stats["p_value"]

            if ratio > 0:
                if p_val is not None:
                    sig_str = (
                        "***"
                        if p_val < 0.001
                        else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    )
                    print(
                        f"    {task}: {ratio:.2f}x higher uncertainty for misclassified (p={p_val:.4f} {sig_str})"
                    )
                else:
                    print(
                        f"    {task}: {ratio:.2f}x higher uncertainty for misclassified"
                    )


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

    # Check if any results have uncertainty propagated
    has_uncertainty = False
    for results_list in results_dict.values():
        for result in results_list:
            if result is not None and result.get("y_pred_std") is not None:
                has_uncertainty = True
                break
        if has_uncertainty:
            break

    if has_uncertainty:
        print(
            "  [INFO] Uncertainty propagation detected - will generate uncertainty plots"
        )

    # Map task names to better titles
    title_map = {
        "tumor_vs_normal_pz": "Tumor vs Normal Classification (Peripheral Zone)",
        "tumor_vs_normal_tz": "Tumor vs Normal Classification (Transition Zone)",
        "ggg": "Gleason Grade Group Classification (GGG 1-2 vs 3-5)",
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

        # Plot feature importance for Full LR model
        full_lr_result = next(
            (
                r
                for r in all_results
                if r is not None and r.get("feature_name") == "Full LR"
            ),
            None,
        )
        if full_lr_result is not None and "feature_importance" in full_lr_result:
            # Save as CSV table
            table_path = os.path.join(output_dir, f"feature_importance_{task_name}.csv")
            save_feature_importance_table(
                full_lr_result["feature_names_ordered"],
                full_lr_result["feature_importance"],
                table_path,
            )

            # Plot
            importance_path = os.path.join(
                output_dir, f"feature_importance_{task_name}.pdf"
            )
            plot_feature_importance(
                full_lr_result["feature_names_ordered"],
                full_lr_result["feature_importance"],
                importance_path,
                title=f"Feature Importance - {title_map.get(task_name, task_name)}",
            )

            # Plot prediction uncertainty if available
            if full_lr_result.get("y_pred_std") is not None:
                uncertainty_path = os.path.join(
                    output_dir, f"prediction_uncertainty_{task_name}.pdf"
                )
                plot_prediction_uncertainty(
                    full_lr_result["y_true"],
                    full_lr_result["y_pred_proba"],
                    full_lr_result["y_pred_std"],
                    uncertainty_path,
                    title=f"Prediction Uncertainty - {title_map.get(task_name, task_name)}",
                )

    # Generate combined uncertainty plot if uncertainty data is available
    if has_uncertainty:
        print("\n[COMBINED UNCERTAINTY PLOT]")
        create_combined_uncertainty_plot(results_dict, output_dir)

    print(f"\n✓ Summary report saved to: {output_dir}/")
