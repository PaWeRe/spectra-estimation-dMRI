"""
Comprehensive evaluation module for cancer biomarker classification models.

This module provides:
- Multiple evaluation metrics popular in medical ML
- Statistical testing and confidence intervals
- Performance comparison tools
- Clinical decision curve analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    calibration_curve,
    brier_score_loss,
)
from sklearn.utils import resample
import scipy.stats as stats
import wandb

from ..data.base_classes import ClassificationMetrics


class BiomarkerEvaluator:
    """
    Comprehensive evaluator for biomarker classification models.

    Provides clinically-relevant metrics and statistical analysis
    appropriate for medical classification tasks.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.results_cache = {}

    def evaluate_comprehensive(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
    ) -> Dict:
        """
        Perform comprehensive evaluation with all clinically relevant metrics.

        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Prediction probabilities for positive class
            model_name: Name of the model being evaluated

        Returns:
            Dictionary with comprehensive evaluation results
        """

        # Basic classification metrics
        basic_metrics = self._calculate_basic_metrics(y_true, y_pred, y_prob)

        # Clinical metrics
        clinical_metrics = self._calculate_clinical_metrics(y_true, y_pred, y_prob)

        # Statistical analysis
        statistical_analysis = self._calculate_statistical_analysis(
            y_true, y_pred, y_prob
        )

        # Calibration analysis
        calibration_analysis = self._calculate_calibration_metrics(y_true, y_prob)

        # Threshold analysis
        threshold_analysis = self._analyze_thresholds(y_true, y_prob)

        # Combine all results
        comprehensive_results = {
            "model_name": model_name,
            "basic_metrics": basic_metrics,
            "clinical_metrics": clinical_metrics,
            "statistical_analysis": statistical_analysis,
            "calibration": calibration_analysis,
            "threshold_analysis": threshold_analysis,
            "sample_size": len(y_true),
            "class_distribution": {
                "positive": int(np.sum(y_true)),
                "negative": int(len(y_true) - np.sum(y_true)),
                "prevalence": float(np.mean(y_true)),
            },
        }

        # Cache results
        self.results_cache[model_name] = comprehensive_results

        return comprehensive_results

    def _calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        """Calculate standard classification metrics."""

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "sensitivity": recall_score(y_true, y_pred, zero_division=0),  # TPR
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # TNR
            "precision": precision_score(y_true, y_pred, zero_division=0),  # PPV
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        }

        # Additional metrics
        metrics.update(
            {
                "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        )

        return metrics

    def _calculate_clinical_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        """Calculate clinically-relevant metrics."""

        # Kappa metrics (popular in medical literature)
        kappa = cohen_kappa_score(y_true, y_pred)

        try:
            weighted_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        except:
            weighted_kappa = kappa

        # Likelihood ratios (useful for clinical decision making)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        positive_lr = (
            sensitivity / (1 - specificity) if specificity < 1.0 else float("inf")
        )
        negative_lr = (
            (1 - sensitivity) / specificity if specificity > 0.0 else float("inf")
        )

        # Diagnostic odds ratio
        dor = positive_lr / negative_lr if negative_lr != 0 else float("inf")

        # Youden's J statistic (optimal threshold finder)
        youdens_j = sensitivity + specificity - 1

        clinical_metrics = {
            "cohen_kappa": kappa,
            "quadratic_weighted_kappa": weighted_kappa,
            "positive_likelihood_ratio": positive_lr,
            "negative_likelihood_ratio": negative_lr,
            "diagnostic_odds_ratio": dor,
            "youdens_j_statistic": youdens_j,
        }

        return clinical_metrics

    def _calculate_statistical_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        """Calculate statistical significance and confidence intervals."""

        # Bootstrap confidence intervals for AUC
        auc_ci = self._bootstrap_auc_ci(y_true, y_prob)

        # McNemar's test for comparing to random chance
        mcnemar_result = self._mcnemar_test(y_true, y_pred)

        # Binomial test for better than chance performance
        n_correct = np.sum(y_pred == y_true)
        n_total = len(y_true)
        binomial_p = stats.binom_test(n_correct, n_total, p=0.5, alternative="greater")

        statistical_analysis = {
            "auc_confidence_interval": auc_ci,
            "mcnemar_test": mcnemar_result,
            "binomial_test_p_value": binomial_p,
            "better_than_chance": binomial_p < 0.05,
        }

        return statistical_analysis

    def _calculate_calibration_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        """Calculate model calibration metrics."""

        # Brier score (lower is better)
        brier_score = brier_score_loss(y_true, y_prob)

        # Hosmer-Lemeshow test approximation
        try:
            hl_test = self._hosmer_lemeshow_test(y_true, y_prob)
        except:
            hl_test = {"statistic": np.nan, "p_value": np.nan, "well_calibrated": False}

        # Calibration curve data
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            calibration_curve_data = {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
            }
        except:
            calibration_curve_data = {"prob_true": [], "prob_pred": []}

        calibration_metrics = {
            "brier_score": brier_score,
            "hosmer_lemeshow_test": hl_test,
            "calibration_curve": calibration_curve_data,
        }

        return calibration_metrics

    def _analyze_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Analyze performance across different classification thresholds."""

        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

        # Find optimal threshold using Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]

        # Performance at different thresholds
        thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, optimal_threshold]
        threshold_performance = {}

        for threshold in thresholds_to_test:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()

            threshold_performance[f"threshold_{threshold:.3f}"] = {
                "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                "f1_score": f1_score(y_true, y_pred_thresh, zero_division=0),
            }

        threshold_analysis = {
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            },
            "pr_curve": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            },
            "optimal_threshold": float(optimal_threshold),
            "optimal_youden_j": float(j_scores[optimal_idx]),
            "threshold_performance": threshold_performance,
        }

        return threshold_analysis

    def _bootstrap_auc_ci(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence intervals for AUC."""

        if len(np.unique(y_true)) < 2:
            return (0.5, 0.5)

        aucs = []
        for _ in range(n_bootstrap):
            indices = resample(range(len(y_true)), n_samples=len(y_true))
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]

            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                auc = roc_auc_score(y_true_boot, y_prob_boot)
                aucs.append(auc)
            except ValueError:
                continue

        if len(aucs) == 0:
            return (0.5, 0.5)

        alpha = 1 - self.confidence_level
        lower = np.percentile(aucs, 100 * alpha / 2)
        upper = np.percentile(aucs, 100 * (1 - alpha / 2))

        return (float(lower), float(upper))

    def _mcnemar_test(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Perform McNemar's test against random classifier."""

        # Create random predictions
        np.random.seed(42)
        y_random = np.random.binomial(1, np.mean(y_true), len(y_true))

        # Create contingency table
        correct_model = y_pred == y_true
        correct_random = y_random == y_true

        both_correct = np.sum(correct_model & correct_random)
        model_correct_random_wrong = np.sum(correct_model & ~correct_random)
        model_wrong_random_correct = np.sum(~correct_model & correct_random)
        both_wrong = np.sum(~correct_model & ~correct_random)

        # McNemar's test statistic
        b = model_correct_random_wrong
        c = model_wrong_random_correct

        if (b + c) == 0:
            mcnemar_statistic = 0
            p_value = 1.0
        else:
            mcnemar_statistic = ((abs(b - c) - 1) ** 2) / (b + c)
            p_value = 1 - stats.chi2.cdf(mcnemar_statistic, 1)

        return {
            "statistic": float(mcnemar_statistic),
            "p_value": float(p_value),
            "significantly_better": p_value < 0.05,
            "contingency_table": {
                "both_correct": int(both_correct),
                "model_correct_random_wrong": int(model_correct_random_wrong),
                "model_wrong_random_correct": int(model_wrong_random_correct),
                "both_wrong": int(both_wrong),
            },
        }

    def _hosmer_lemeshow_test(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict:
        """Approximation of Hosmer-Lemeshow goodness-of-fit test."""

        # Create bins based on predicted probabilities
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        hl_statistic = 0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            if bin_upper == 1.0:  # Include the upper boundary for the last bin
                in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)

            if np.sum(in_bin) == 0:
                continue

            observed_pos = np.sum(y_true[in_bin])
            observed_neg = np.sum(in_bin) - observed_pos
            expected_pos = np.sum(y_prob[in_bin])
            expected_neg = np.sum(in_bin) - expected_pos

            if expected_pos > 0 and expected_neg > 0:
                hl_statistic += ((observed_pos - expected_pos) ** 2) / expected_pos
                hl_statistic += ((observed_neg - expected_neg) ** 2) / expected_neg

        # Calculate p-value (approximation)
        degrees_freedom = n_bins - 2
        p_value = 1 - stats.chi2.cdf(hl_statistic, degrees_freedom)

        return {
            "statistic": float(hl_statistic),
            "p_value": float(p_value),
            "well_calibrated": p_value > 0.05,  # Null hypothesis: well calibrated
            "degrees_of_freedom": degrees_freedom,
        }

    def compare_models(self, results_list: List[Dict]) -> Dict:
        """
        Compare multiple model evaluation results.

        Args:
            results_list: List of evaluation result dictionaries

        Returns:
            Comparison analysis
        """

        if len(results_list) < 2:
            raise ValueError("Need at least 2 model results for comparison")

        # Extract key metrics for comparison
        models = [r["model_name"] for r in results_list]
        metrics_comparison = {}

        for metric in [
            "auc",
            "accuracy",
            "sensitivity",
            "specificity",
            "f1_score",
            "quadratic_weighted_kappa",
        ]:
            values = []
            for result in results_list:
                if metric in result["basic_metrics"]:
                    values.append(result["basic_metrics"][metric])
                elif metric in result["clinical_metrics"]:
                    values.append(result["clinical_metrics"][metric])
                else:
                    values.append(0.0)

            metrics_comparison[metric] = {
                "values": values,
                "best_model": models[np.argmax(values)],
                "best_value": max(values),
                "rankings": self._rank_values(values, models),
            }

        # Statistical comparison (if possible)
        statistical_comparison = self._statistical_model_comparison(results_list)

        comparison_results = {
            "models_compared": models,
            "metrics_comparison": metrics_comparison,
            "statistical_comparison": statistical_comparison,
            "overall_ranking": self._overall_ranking(metrics_comparison),
        }

        return comparison_results

    def _rank_values(self, values: List[float], models: List[str]) -> Dict[str, int]:
        """Rank models by metric values."""
        model_value_pairs = list(zip(models, values))
        sorted_pairs = sorted(model_value_pairs, key=lambda x: x[1], reverse=True)
        return {model: rank + 1 for rank, (model, value) in enumerate(sorted_pairs)}

    def _overall_ranking(self, metrics_comparison: Dict) -> Dict[str, float]:
        """Calculate overall ranking based on multiple metrics."""

        # Weight different metrics (can be customized)
        metric_weights = {
            "auc": 0.3,
            "accuracy": 0.2,
            "sensitivity": 0.2,
            "specificity": 0.2,
            "f1_score": 0.1,
        }

        model_scores = {}

        for metric, weight in metric_weights.items():
            if metric in metrics_comparison:
                rankings = metrics_comparison[metric]["rankings"]
                for model, rank in rankings.items():
                    if model not in model_scores:
                        model_scores[model] = 0.0
                    # Lower rank is better, so invert the score
                    model_scores[model] += weight * (1.0 / rank)

        # Sort by overall score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        return {model: rank + 1 for rank, (model, score) in enumerate(sorted_models)}

    def _statistical_model_comparison(self, results_list: List[Dict]) -> Dict:
        """Perform statistical comparison between models."""

        # Placeholder for more sophisticated statistical tests
        # In practice, you'd want to use paired tests if you have
        # predictions on the same dataset

        return {
            "note": "Statistical model comparison requires access to original predictions",
            "recommendation": "Use cross-validation or bootstrap for robust comparison",
        }

    def generate_report(self, evaluation_results: Dict) -> str:
        """Generate a comprehensive evaluation report."""

        model_name = evaluation_results["model_name"]
        basic = evaluation_results["basic_metrics"]
        clinical = evaluation_results["clinical_metrics"]
        sample_info = evaluation_results["class_distribution"]

        report = f"""
=== BIOMARKER EVALUATION REPORT ===
Model: {model_name}

DATASET INFORMATION:
  Total samples: {evaluation_results['sample_size']}
  Positive cases: {sample_info['positive']} ({sample_info['prevalence']:.1%})
  Negative cases: {sample_info['negative']} ({1-sample_info['prevalence']:.1%})

PERFORMANCE METRICS:
  AUC: {basic['auc']:.3f}
  Accuracy: {basic['accuracy']:.3f}
  Balanced Accuracy: {basic['balanced_accuracy']:.3f}
  
  Sensitivity (Recall): {basic['sensitivity']:.3f}
  Specificity: {basic['specificity']:.3f}
  Precision (PPV): {basic['precision']:.3f}
  NPV: {basic['negative_predictive_value']:.3f}
  
  F1 Score: {basic['f1_score']:.3f}
  Cohen's Kappa: {clinical['cohen_kappa']:.3f}
  Weighted Kappa: {clinical['quadratic_weighted_kappa']:.3f}

CLINICAL METRICS:
  Positive LR: {clinical['positive_likelihood_ratio']:.2f}
  Negative LR: {clinical['negative_likelihood_ratio']:.2f}
  Diagnostic OR: {clinical['diagnostic_odds_ratio']:.2f}
  Youden's J: {clinical['youdens_j_statistic']:.3f}

CONFUSION MATRIX:
  True Positives: {basic['true_positives']}
  True Negatives: {basic['true_negatives']}
  False Positives: {basic['false_positives']}
  False Negatives: {basic['false_negatives']}

STATISTICAL ANALYSIS:
  AUC 95% CI: [{evaluation_results['statistical_analysis']['auc_confidence_interval'][0]:.3f}, {evaluation_results['statistical_analysis']['auc_confidence_interval'][1]:.3f}]
  Better than chance: {evaluation_results['statistical_analysis']['better_than_chance']}
  
CALIBRATION:
  Brier Score: {evaluation_results['calibration']['brier_score']:.3f} (lower is better)
  Well calibrated: {evaluation_results['calibration']['hosmer_lemeshow_test']['well_calibrated']}

OPTIMAL THRESHOLD: {evaluation_results['threshold_analysis']['optimal_threshold']:.3f}
"""

        return report
