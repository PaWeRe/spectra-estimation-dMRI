"""
Biomarker classification pipeline for Gleason score prediction from diffusivity spectra.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import wandb
from omegaconf import DictConfig

from ..data.data_models import DiffusivitySpectraDataset, SignalDecay
from ..data.base_classes import (
    GleasonScorePredictor,
    BiomarkerFeatures,
    FeatureEngineer,
    ClassificationMetrics,
    MetricsCalculator,
)


class GleasonScoreBiomarker:
    """
    Main biomarker class for Gleason score prediction using diffusivity spectra.

    This class handles:
    - Feature extraction from diffusivity spectra
    - Model training and prediction
    - Performance evaluation
    - Feature importance analysis
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.classifier_config = config.classifier
        self.biomarker_config = config.biomarker_analysis

        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.metrics_calculator = MetricsCalculator()

        # Model and results storage
        self.model = None
        self.feature_names = None
        self.training_results = None
        self.validation_results = None

    def prepare_targets(
        self, spectra_dataset: DiffusivitySpectraDataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare target variables from spectra dataset.

        Returns:
            targets: Binary labels (0: low grade, 1: high grade)
            ggg_scores: Original GGG scores for multi-class classification
        """
        targets = []
        ggg_scores = []
        threshold = self.biomarker_config.gleason_threshold

        for spectrum in spectra_dataset.spectra:
            signal_decay = spectrum.signal_decay

            if hasattr(signal_decay, "ggg") and signal_decay.ggg is not None:
                ggg = signal_decay.ggg
                # Convert GGG to binary classification
                target = 1 if ggg > threshold else 0
                targets.append(target)
                ggg_scores.append(ggg)

        return np.array(targets), np.array(ggg_scores)

    def extract_all_features(
        self, spectra_dataset: DiffusivitySpectraDataset
    ) -> pd.DataFrame:
        """Extract comprehensive features from all spectra in dataset."""
        features_list = []

        for spectrum in spectra_dataset.spectra:
            biomarker_features = self.feature_engineer.engineer_spectrum_features(
                spectrum
            )
            features_dict = biomarker_features.to_dict()
            features_list.append(features_dict)

        features_df = pd.DataFrame(features_list)

        # Handle missing values
        features_df = features_df.fillna(features_df.median())

        # Store feature names for later use
        self.feature_names = list(features_df.columns)

        return features_df

    def train_model(self, spectra_dataset: DiffusivitySpectraDataset) -> Dict:
        """
        Train the biomarker classification model.

        Returns:
            Dictionary with training results and metrics
        """
        print(f"[INFO] Training {self.classifier_config.name} biomarker model...")

        # Extract features and targets
        # TODO: X not used currenlty, so extract_all_features() unnecessary?
        X = self.extract_all_features(spectra_dataset)
        y, ggg_scores = self.prepare_targets(spectra_dataset)

        if len(y) == 0:
            raise ValueError(
                "No valid targets found in dataset. Check GGG/Gleason score availability."
            )

        print(f"[INFO] Dataset size: {len(y)} samples")
        print(f"[INFO] Class distribution: {np.bincount(y)}")
        print(f"[INFO] Features: {len(self.feature_names)}")

        # Initialize model
        model_type = self.classifier_config.type
        hyperparams = self.classifier_config.hyperparameters

        self.model = GleasonScorePredictor(
            model_type=model_type, target_type="binary", **hyperparams
        )

        # Train model
        self.model.fit(spectra_dataset, y)

        # Evaluate training performance
        y_pred = self.model.predict(spectra_dataset)
        y_prob = self.model.predict_proba(spectra_dataset)

        training_metrics = self.metrics_calculator.calculate_metrics(
            y, y_pred, y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        )

        # Store results
        self.training_results = {
            "metrics": training_metrics,
            "feature_importance": self.model.feature_importance,
            "n_samples": len(y),
            "class_distribution": np.bincount(y),
            "features_used": self.feature_names,
        }

        # Log metrics to wandb
        if wandb.run is not None:
            self._log_training_metrics(training_metrics)

        print(f"[INFO] Training completed. AUC: {training_metrics.auc:.3f}")

        return self.training_results

    def cross_validate(self, spectra_dataset: DiffusivitySpectraDataset) -> Dict:
        """
        Perform cross-validation for robust evaluation on small datasets.

        Returns:
            Dictionary with cross-validation results
        """
        print(
            f"[INFO] Performing {self.biomarker_config.cross_validation.n_folds}-fold cross-validation..."
        )

        # Extract targets
        y, _ = self.prepare_targets(spectra_dataset)

        if len(y) == 0:
            raise ValueError("No valid targets found for cross-validation.")

        # Initialize model for CV
        model_type = self.classifier_config.type
        hyperparams = self.classifier_config.hyperparameters

        cv_model = GleasonScorePredictor(
            model_type=model_type, target_type="binary", **hyperparams
        )

        # Perform cross-validation
        cv_results = cv_model.cross_validate(
            spectra_dataset,
            y,
            cv_folds=self.biomarker_config.cross_validation.n_folds,
            random_state=self.config.seed,
        )

        # Calculate summary statistics
        cv_summary = {}
        for metric, values in cv_results.items():
            cv_summary[f"{metric}_mean"] = np.mean(values)
            cv_summary[f"{metric}_std"] = np.std(values)
            cv_summary[f"{metric}_values"] = values

        self.validation_results = cv_summary

        # Log CV metrics to wandb
        if wandb.run is not None:
            self._log_cv_metrics(cv_summary)

        print(
            f"[INFO] Cross-validation completed. Mean AUC: {cv_summary['auc_mean']:.3f} ± {cv_summary['auc_std']:.3f}"
        )

        return cv_summary

    def predict(
        self, spectra_dataset: DiffusivitySpectraDataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.

        Returns:
            predictions: Binary predictions
            probabilities: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        predictions = self.model.predict(spectra_dataset)
        probabilities = self.model.predict_proba(spectra_dataset)

        return predictions, probabilities

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None or self.model.feature_importance is None:
            return {}

        return self.model.feature_importance

    def _log_training_metrics(self, metrics: ClassificationMetrics):
        """Log training metrics to wandb."""
        wandb.log(
            {
                "biomarker/train_auc": metrics.auc,
                "biomarker/train_accuracy": metrics.accuracy,
                "biomarker/train_sensitivity": metrics.sensitivity,
                "biomarker/train_specificity": metrics.specificity,
                "biomarker/train_precision": metrics.precision,
                "biomarker/train_f1": metrics.f1_score,
                "biomarker/train_kappa": metrics.quadratic_weighted_kappa,
            }
        )

    def _log_cv_metrics(self, cv_summary: Dict):
        """Log cross-validation metrics to wandb."""
        wandb.log(
            {
                "biomarker/cv_auc_mean": cv_summary["auc_mean"],
                "biomarker/cv_auc_std": cv_summary["auc_std"],
                "biomarker/cv_accuracy_mean": cv_summary["accuracy_mean"],
                "biomarker/cv_sensitivity_mean": cv_summary["sensitivity_mean"],
                "biomarker/cv_specificity_mean": cv_summary["specificity_mean"],
                "biomarker/cv_kappa_mean": cv_summary["quadratic_weighted_kappa_mean"],
            }
        )


class BiomarkerPipeline:
    """
    Complete pipeline for biomarker analysis including model comparison and optimization.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.biomarker_config = config.biomarker_analysis

        # Results storage
        self.model_results = {}
        self.comparison_results = None

    def run_single_model(
        self, spectra_dataset: DiffusivitySpectraDataset, model_name: str
    ) -> Dict:
        """Run biomarker analysis with a single model."""
        print(f"\n[INFO] Running biomarker analysis with {model_name}...")

        # Update config for this model
        config_copy = self.config.copy()
        config_copy.classifier = self.config.classifier
        if hasattr(self.config, f"classifier_{model_name}"):
            config_copy.classifier = getattr(self.config, f"classifier_{model_name}")

        # Initialize biomarker
        biomarker = GleasonScoreBiomarker(config_copy)

        # Train model
        training_results = biomarker.train_model(spectra_dataset)

        # Cross-validate if enabled
        cv_results = None
        if self.biomarker_config.cross_validation.enabled:
            cv_results = biomarker.cross_validate(spectra_dataset)

        # Store results
        results = {
            "biomarker": biomarker,
            "training": training_results,
            "cross_validation": cv_results,
            "model_name": model_name,
        }

        self.model_results[model_name] = results

        return results

    def run_model_comparison(self, spectra_dataset: DiffusivitySpectraDataset) -> Dict:
        """Run biomarker analysis comparing multiple models."""
        if not self.biomarker_config.model_comparison.enabled:
            return self.run_single_model(spectra_dataset, self.config.classifier.name)

        print(f"\n[INFO] Running biomarker model comparison...")

        models = self.biomarker_config.model_comparison.models
        comparison_results = {}

        for model_name in models:
            results = self.run_single_model(spectra_dataset, model_name)
            comparison_results[model_name] = results

        # Analyze comparison results
        self.comparison_results = self._analyze_model_comparison(comparison_results)

        # Log comparison to wandb
        if wandb.run is not None:
            self._log_model_comparison()

        return comparison_results

    def _analyze_model_comparison(self, results: Dict) -> Dict:
        """Analyze results from model comparison."""
        comparison = {
            "models": list(results.keys()),
            "training_metrics": {},
            "cv_metrics": {},
            "best_model": None,
            "rankings": {},
        }

        primary_metric = self.biomarker_config.evaluation.primary_metric

        # Extract metrics for each model
        training_scores = {}
        cv_scores = {}

        for model_name, model_results in results.items():
            # Training metrics
            if model_results["training"]:
                training_metrics = model_results["training"]["metrics"]
                training_scores[model_name] = getattr(training_metrics, primary_metric)
                comparison["training_metrics"][model_name] = {
                    "auc": training_metrics.auc,
                    "accuracy": training_metrics.accuracy,
                    "sensitivity": training_metrics.sensitivity,
                    "specificity": training_metrics.specificity,
                    "f1_score": training_metrics.f1_score,
                    "kappa": training_metrics.quadratic_weighted_kappa,
                }

            # CV metrics
            if model_results["cross_validation"]:
                cv_summary = model_results["cross_validation"]
                cv_scores[model_name] = cv_summary[f"{primary_metric}_mean"]
                comparison["cv_metrics"][model_name] = {
                    f"{metric}_mean": cv_summary[f"{metric}_mean"]
                    for metric in [
                        "auc",
                        "accuracy",
                        "sensitivity",
                        "specificity",
                        "f1_score",
                        "quadratic_weighted_kappa",
                    ]
                    if f"{metric}_mean" in cv_summary
                }

        # Determine best model based on CV scores (or training if CV not available)
        scores_for_ranking = cv_scores if cv_scores else training_scores

        if scores_for_ranking:
            best_model = max(
                scores_for_ranking.keys(), key=lambda k: scores_for_ranking[k]
            )
            comparison["best_model"] = best_model

            # Create rankings
            sorted_models = sorted(
                scores_for_ranking.items(), key=lambda x: x[1], reverse=True
            )
            comparison["rankings"] = {
                model: rank + 1 for rank, (model, score) in enumerate(sorted_models)
            }

        return comparison

    def _log_model_comparison(self):
        """Log model comparison results to wandb."""
        if self.comparison_results is None:
            return

        # Log best model
        if self.comparison_results["best_model"]:
            wandb.log({"biomarker/best_model": self.comparison_results["best_model"]})

        # Log metrics for each model
        for model_name in self.comparison_results["models"]:
            if model_name in self.comparison_results["cv_metrics"]:
                metrics = self.comparison_results["cv_metrics"][model_name]
                for metric_name, value in metrics.items():
                    wandb.log(
                        {f"biomarker/comparison/{model_name}/{metric_name}": value}
                    )

    def get_best_model(self) -> Optional[GleasonScoreBiomarker]:
        """Get the best performing model from comparison."""
        if self.comparison_results and self.comparison_results["best_model"]:
            best_model_name = self.comparison_results["best_model"]
            return self.model_results[best_model_name]["biomarker"]
        elif len(self.model_results) == 1:
            return list(self.model_results.values())[0]["biomarker"]

        return None

    def get_summary_report(self) -> str:
        """Generate a summary report of biomarker analysis results."""
        if not self.model_results:
            return "No biomarker analysis results available."

        report = "=== BIOMARKER ANALYSIS SUMMARY ===\n\n"

        if self.comparison_results:
            report += (
                f"Models compared: {', '.join(self.comparison_results['models'])}\n"
            )
            report += f"Best model: {self.comparison_results['best_model']}\n\n"

            # Rankings
            report += "Model Rankings:\n"
            for model, rank in self.comparison_results["rankings"].items():
                report += f"  {rank}. {model}\n"
            report += "\n"

            # CV Results
            if self.comparison_results["cv_metrics"]:
                report += "Cross-Validation Results:\n"
                for model, metrics in self.comparison_results["cv_metrics"].items():
                    report += f"  {model}:\n"
                    for metric, value in metrics.items():
                        report += f"    {metric}: {value:.3f}\n"
                report += "\n"

        else:
            model_name = list(self.model_results.keys())[0]
            results = self.model_results[model_name]
            report += f"Single model analysis: {model_name}\n\n"

            if results["training"]:
                metrics = results["training"]["metrics"]
                report += "Training Results:\n"
                report += f"  AUC: {metrics.auc:.3f}\n"
                report += f"  Accuracy: {metrics.accuracy:.3f}\n"
                report += f"  Sensitivity: {metrics.sensitivity:.3f}\n"
                report += f"  Specificity: {metrics.specificity:.3f}\n"
                report += (
                    f"  Weighted Kappa: {metrics.quadratic_weighted_kappa:.3f}\n\n"
                )

            if results["cross_validation"]:
                cv = results["cross_validation"]
                report += "Cross-Validation Results:\n"
                for metric in [
                    "auc",
                    "accuracy",
                    "sensitivity",
                    "specificity",
                    "quadratic_weighted_kappa",
                ]:
                    if f"{metric}_mean" in cv:
                        mean_val = cv[f"{metric}_mean"]
                        std_val = cv[f"{metric}_std"]
                        report += f"  {metric}: {mean_val:.3f} ± {std_val:.3f}\n"

        return report
