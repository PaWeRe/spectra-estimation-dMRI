"""
Visualization module for biomarker analysis results.

Provides comprehensive plotting tools for:
- ROC and PR curves
- Confusion matrices
- Feature importance plots
- Calibration plots
- Decision curve analysis
- Model comparison visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import wandb
from sklearn.metrics import roc_curve, precision_recall_curve, calibration_curve
import warnings

# Set style for medical publication quality plots
plt.style.use("default")
sns.set_palette("colorblind")


class BiomarkerVisualizer:
    """
    Comprehensive visualization tools for biomarker analysis.

    Creates publication-ready plots for medical AI research.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#C73E1D",
            "neutral": "#6C757D",
        }

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        show_confidence_interval: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot ROC curve with confidence intervals.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            model_name: Name for the model in legend
            show_confidence_interval: Whether to show bootstrap CI
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = np.trapz(tpr, fpr)

        # Plot main ROC curve
        ax.plot(
            fpr,
            tpr,
            color=self.colors["primary"],
            linewidth=2.5,
            label=f"{model_name} (AUC = {auc:.3f})",
        )

        # Add confidence interval if requested
        if show_confidence_interval:
            self._add_roc_confidence_interval(ax, y_true, y_prob)

        # Plot diagonal reference line
        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1.5,
            alpha=0.7,
            label="Random Classifier (AUC = 0.500)",
        )

        # Customize plot
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
        ax.set_title(
            "Receiver Operating Characteristic (ROC) Curve",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Add text box with additional metrics
        textstr = (
            f"n = {len(y_true)}\nPositive: {np.sum(y_true)} ({np.mean(y_true):.1%})"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({"biomarker/roc_curve": wandb.Image(fig)})

        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot Precision-Recall curve."""

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = np.trapz(precision, recall)

        # Plot PR curve
        ax.plot(
            recall,
            precision,
            color=self.colors["secondary"],
            linewidth=2.5,
            label=f"{model_name} (AP = {avg_precision:.3f})",
        )

        # Plot baseline (random classifier)
        baseline = np.mean(y_true)
        ax.axhline(
            y=baseline,
            color="k",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Random Classifier (AP = {baseline:.3f})",
        )

        # Customize plot
        ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
        ax.set_ylabel("Precision (PPV)", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if wandb.run is not None:
            wandb.log({"biomarker/pr_curve": wandb.Image(fig)})

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        normalize: bool = False,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot confusion matrix with clinical styling."""

        from sklearn.metrics import confusion_matrix

        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2%"
            cmap = "Blues"
        else:
            fmt = "d"
            cmap = "Blues"

        # Default class names
        if class_names is None:
            class_names = ["Low Grade\n(GGG ≤3)", "High Grade\n(GGG >3)"]

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )

        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

        # Add performance metrics as text
        tn, fp, fn, tp = (
            cm.ravel()
            if not normalize
            else (cm * np.sum(confusion_matrix(y_true, y_pred))).ravel()
        )

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_text = f"Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}"
        ax.text(
            1.05,
            0.5,
            metrics_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="center",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if wandb.run is not None:
            wandb.log({"biomarker/confusion_matrix": wandb.Image(fig)})

        return fig

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature importance with clinical interpretation."""

        if not feature_importance:
            warnings.warn("No feature importance data available")
            return None

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Take top N features
        if len(sorted_features) > top_n:
            sorted_features = sorted_features[:top_n]

        features, importances = zip(*sorted_features)

        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)), dpi=self.dpi)

        bars = ax.barh(
            range(len(features)), importances, color=self.colors["accent"], alpha=0.8
        )

        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([self._format_feature_name(f) for f in features])
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(
                width + max(importances) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.3f}",
                ha="left",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if wandb.run is not None:
            wandb.log({"biomarker/feature_importance": wandb.Image(fig)})

        return fig

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot calibration curve to assess probability calibration."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)

        # Plot 1: Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        ax1.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            color=self.colors["primary"],
            label=model_name,
        )
        ax1.plot(
            [0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Perfect Calibration"
        )

        ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax1.set_ylabel("Fraction of Positives", fontsize=12)
        ax1.set_title("Calibration Plot", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # Plot 2: Histogram of predicted probabilities
        ax2.hist(
            y_prob[y_true == 0],
            bins=20,
            alpha=0.6,
            color=self.colors["neutral"],
            label="Negative Cases",
            density=True,
        )
        ax2.hist(
            y_prob[y_true == 1],
            bins=20,
            alpha=0.6,
            color=self.colors["primary"],
            label="Positive Cases",
            density=True,
        )

        ax2.set_xlabel("Predicted Probability", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title(
            "Distribution of Predicted Probabilities", fontsize=14, fontweight="bold"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if wandb.run is not None:
            wandb.log({"biomarker/calibration": wandb.Image(fig)})

        return fig

    def plot_threshold_analysis(
        self, y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot threshold analysis showing metrics vs threshold."""

        thresholds = np.linspace(0, 1, 101)
        metrics = {
            "sensitivity": [],
            "specificity": [],
            "precision": [],
            "f1_score": [],
        }

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            if np.sum(y_pred) == 0 or np.sum(y_pred) == len(y_pred):
                # Handle edge cases
                metrics["sensitivity"].append(0)
                metrics["specificity"].append(1 if np.sum(y_pred) == 0 else 0)
                metrics["precision"].append(0)
                metrics["f1_score"].append(0)
                continue

            from sklearn.metrics import precision_score, recall_score, f1_score

            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))

            sensitivity = recall_score(y_true, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            metrics["sensitivity"].append(sensitivity)
            metrics["specificity"].append(specificity)
            metrics["precision"].append(precision)
            metrics["f1_score"].append(f1)

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.plot(
            thresholds,
            metrics["sensitivity"],
            label="Sensitivity",
            color=self.colors["primary"],
            linewidth=2,
        )
        ax.plot(
            thresholds,
            metrics["specificity"],
            label="Specificity",
            color=self.colors["secondary"],
            linewidth=2,
        )
        ax.plot(
            thresholds,
            metrics["precision"],
            label="Precision",
            color=self.colors["accent"],
            linewidth=2,
        )
        ax.plot(
            thresholds,
            metrics["f1_score"],
            label="F1 Score",
            color=self.colors["success"],
            linewidth=2,
        )

        ax.set_xlabel("Classification Threshold", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title(
            "Performance Metrics vs Classification Threshold",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if wandb.run is not None:
            wandb.log({"biomarker/threshold_analysis": wandb.Image(fig)})

        return fig

    def plot_model_comparison(
        self, comparison_results: Dict, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of multiple models."""

        metrics = ["auc", "accuracy", "sensitivity", "specificity", "f1_score"]
        models = comparison_results["models_compared"]

        # Prepare data for plotting
        values = []
        for metric in metrics:
            if metric in comparison_results["metrics_comparison"]:
                values.append(
                    comparison_results["metrics_comparison"][metric]["values"]
                )
            else:
                values.append([0] * len(models))

        values = np.array(values).T  # Transpose for easier plotting

        # Create radar chart
        fig, ax = plt.subplots(
            figsize=(10, 10), dpi=self.dpi, subplot_kw=dict(projection="polar")
        )

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle

        colors = [
            self.colors["primary"],
            self.colors["secondary"],
            self.colors["accent"],
        ]

        for i, model in enumerate(models):
            model_values = np.concatenate(
                (values[i], [values[i][0]])
            )  # Complete the circle
            ax.plot(
                angles,
                model_values,
                "o-",
                linewidth=2,
                label=model,
                color=colors[i % len(colors)],
            )
            ax.fill(angles, model_values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title(
            "Model Performance Comparison", fontsize=14, fontweight="bold", pad=20
        )
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if wandb.run is not None:
            wandb.log({"biomarker/model_comparison": wandb.Image(fig)})

        return fig

    def create_biomarker_dashboard(
        self,
        evaluation_results: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        feature_importance: Dict[str, float] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create comprehensive dashboard with all key plots."""

        fig = plt.figure(figsize=(20, 16), dpi=self.dpi)

        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = np.trapz(tpr, fpr)
        ax1.plot(
            fpr,
            tpr,
            color=self.colors["primary"],
            linewidth=2,
            label=f"AUC = {auc:.3f}",
        )
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.7)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curve
        ax2 = fig.add_subplot(gs[0, 1])
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = np.trapz(precision, recall)
        ax2.plot(
            recall,
            precision,
            color=self.colors["secondary"],
            linewidth=2,
            label=f"AP = {ap:.3f}",
        )
        ax2.axhline(y=np.mean(y_true), color="k", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Confusion Matrix
        ax3 = fig.add_subplot(gs[0, 2])
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax3,
            xticklabels=["Low Grade", "High Grade"],
            yticklabels=["Low Grade", "High Grade"],
        )
        ax3.set_title("Confusion Matrix")

        # Feature Importance (if available)
        if feature_importance:
            ax4 = fig.add_subplot(gs[1, :])
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            features, importances = zip(*sorted_features)
            ax4.barh(
                range(len(features)),
                importances,
                color=self.colors["accent"],
                alpha=0.8,
            )
            ax4.set_yticks(range(len(features)))
            ax4.set_yticklabels([self._format_feature_name(f) for f in features])
            ax4.set_xlabel("Importance Score")
            ax4.set_title("Top 10 Feature Importances")
            ax4.grid(True, axis="x", alpha=0.3)

        # Calibration curve
        ax5 = fig.add_subplot(gs[2, 0])
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax5.plot(prob_pred, prob_true, marker="o", color=self.colors["primary"])
        ax5.plot([0, 1], [0, 1], "k--", alpha=0.7)
        ax5.set_xlabel("Mean Predicted Probability")
        ax5.set_ylabel("Fraction of Positives")
        ax5.set_title("Calibration Plot")
        ax5.grid(True, alpha=0.3)

        # Probability distributions
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(
            y_prob[y_true == 0],
            bins=20,
            alpha=0.6,
            color=self.colors["neutral"],
            label="Negative",
            density=True,
        )
        ax6.hist(
            y_prob[y_true == 1],
            bins=20,
            alpha=0.6,
            color=self.colors["primary"],
            label="Positive",
            density=True,
        )
        ax6.set_xlabel("Predicted Probability")
        ax6.set_ylabel("Density")
        ax6.set_title("Probability Distributions")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Metrics summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")
        metrics = evaluation_results["basic_metrics"]

        summary_text = f"""
Performance Summary:
        
AUC: {metrics['auc']:.3f}
Accuracy: {metrics['accuracy']:.3f}
Sensitivity: {metrics['sensitivity']:.3f}
Specificity: {metrics['specificity']:.3f}
Precision: {metrics['precision']:.3f}
F1 Score: {metrics['f1_score']:.3f}

Sample Size: {len(y_true)}
Positive: {np.sum(y_true)} ({np.mean(y_true):.1%})
"""

        ax7.text(
            0.05,
            0.95,
            summary_text,
            transform=ax7.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        fig.suptitle(
            f'Biomarker Analysis Dashboard - {evaluation_results["model_name"]}',
            fontsize=16,
            fontweight="bold",
        )

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if wandb.run is not None:
            wandb.log({"biomarker/dashboard": wandb.Image(fig)})

        return fig

    def _add_roc_confidence_interval(
        self,
        ax,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bootstrap: int = 100,
        alpha: float = 0.05,
    ):
        """Add confidence interval to ROC curve using bootstrap."""

        bootstrap_aucs = []
        bootstrap_tprs = []

        mean_fpr = np.linspace(0, 1, 100)

        # Bootstrap sampling
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]

            if len(np.unique(y_true_boot)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true_boot, y_prob_boot)
            bootstrap_tprs.append(np.interp(mean_fpr, fpr, tpr))
            bootstrap_tprs[-1][0] = 0.0

        if len(bootstrap_tprs) == 0:
            return

        bootstrap_tprs = np.array(bootstrap_tprs)

        # Calculate confidence intervals
        tpr_lower = np.percentile(bootstrap_tprs, 100 * alpha / 2, axis=0)
        tpr_upper = np.percentile(bootstrap_tprs, 100 * (1 - alpha / 2), axis=0)

        ax.fill_between(
            mean_fpr,
            tpr_lower,
            tpr_upper,
            color=self.colors["primary"],
            alpha=0.2,
            label=f"{int((1-alpha)*100)}% CI",
        )

    def _format_feature_name(self, feature_name: str) -> str:
        """Format feature names for better readability."""

        # Replace underscores with spaces and capitalize
        formatted = feature_name.replace("_", " ").title()

        # Handle specific biomarker terms
        replacements = {
            "Adc": "ADC",
            "Psa": "PSA",
            "Diff": "Diffusivity",
            "Ppv": "PPV",
            "Npv": "NPV",
        }

        for old, new in replacements.items():
            formatted = formatted.replace(old, new)

        return formatted
