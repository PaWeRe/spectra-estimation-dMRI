from typing import Optional, Dict, List, Union, Literal, Tuple
from abc import ABC, abstractmethod
from .data_models import DiffusivitySpectrum, SignalDecay, DiffusivitySpectraDataset
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd


@dataclass
class BiomarkerFeatures:
    """Container for engineered biomarker features derived from diffusivity spectra"""

    # Raw spectrum features
    spectrum_vector: np.ndarray
    diffusivities: np.ndarray

    # Manual feature combinations
    low_diff_fraction: float  # Sum of fractions for D < 1.0 μm²/ms
    mid_diff_fraction: float  # Sum of fractions for 1.0 ≤ D < 2.5 μm²/ms
    high_diff_fraction: float  # Sum of fractions for D ≥ 2.5 μm²/ms

    # Ratio features
    low_high_ratio: float  # low_diff / high_diff
    mid_low_ratio: float  # mid_diff / low_diff

    # Statistical features
    spectrum_entropy: float
    spectrum_kurtosis: float
    spectrum_skewness: float
    spectrum_peak_height: float
    spectrum_peak_location: float

    # ADC comparison
    adc_value: Optional[float] = None
    spectrum_adc_ratio: Optional[float] = None

    # Clinical metadata
    patient_age: Optional[float] = None
    prostate_volume: Optional[float] = None
    psa_level: Optional[float] = None
    region: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML model input"""
        features = {
            "low_diff_fraction": self.low_diff_fraction,
            "mid_diff_fraction": self.mid_diff_fraction,
            "high_diff_fraction": self.high_diff_fraction,
            "low_high_ratio": self.low_high_ratio,
            "mid_low_ratio": self.mid_low_ratio,
            "spectrum_entropy": self.spectrum_entropy,
            "spectrum_kurtosis": self.spectrum_kurtosis,
            "spectrum_skewness": self.spectrum_skewness,
            "spectrum_peak_height": self.spectrum_peak_height,
            "spectrum_peak_location": self.spectrum_peak_location,
        }

        if self.adc_value is not None:
            features["adc_value"] = self.adc_value
        if self.spectrum_adc_ratio is not None:
            features["spectrum_adc_ratio"] = self.spectrum_adc_ratio
        if self.patient_age is not None:
            features["patient_age"] = self.patient_age
        if self.prostate_volume is not None:
            features["prostate_volume"] = self.prostate_volume
        if self.psa_level is not None:
            features["psa_level"] = self.psa_level

        return features


@dataclass
class ClassificationMetrics:
    """Comprehensive metrics for cancer classification evaluation"""

    auc: float
    accuracy: float
    sensitivity: float  # recall, true positive rate
    specificity: float  # true negative rate
    precision: float  # positive predictive value
    f1_score: float
    quadratic_weighted_kappa: float

    # Confusion matrix elements
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Additional medical metrics
    positive_predictive_value: float  # precision
    negative_predictive_value: float

    # Confidence intervals (if available)
    auc_ci_lower: Optional[float] = None
    auc_ci_upper: Optional[float] = None

    def __post_init__(self):
        # Ensure consistency in naming
        self.positive_predictive_value = self.precision
        if self.true_negatives + self.false_negatives > 0:
            self.negative_predictive_value = self.true_negatives / (
                self.true_negatives + self.false_negatives
            )
        else:
            self.negative_predictive_value = 0.0


# TODO: unsure whether this class is necessary, understand derived features better and spectrum analysis (skweness, kurtosis, entropy necessary?)
class FeatureEngineer:
    """Feature engineering for diffusivity spectrum biomarkers"""

    @staticmethod
    def engineer_spectrum_features(spectrum: DiffusivitySpectrum) -> BiomarkerFeatures:
        """Extract comprehensive features from diffusivity spectrum"""
        import scipy.stats as stats

        diffusivities = np.array(spectrum.diffusivities)
        spectrum_vec = np.array(spectrum.spectrum_vector)

        # Ensure spectrum is normalized
        if np.sum(spectrum_vec) > 0:
            spectrum_vec = spectrum_vec / np.sum(spectrum_vec)

        # Manual feature combinations based on clinical knowledge
        low_mask = diffusivities < 1.0
        mid_mask = (diffusivities >= 1.0) & (diffusivities < 2.5)
        high_mask = diffusivities >= 2.5

        low_diff_fraction = np.sum(spectrum_vec[low_mask])
        mid_diff_fraction = np.sum(spectrum_vec[mid_mask])
        high_diff_fraction = np.sum(spectrum_vec[high_mask])

        # Ratio features with small epsilon to avoid division by zero
        eps = 1e-8
        low_high_ratio = low_diff_fraction / (high_diff_fraction + eps)
        mid_low_ratio = mid_diff_fraction / (low_diff_fraction + eps)

        # Statistical features
        # Spectrum entropy
        spectrum_entropy = -np.sum(spectrum_vec * np.log(spectrum_vec + eps))

        # Higher order moments
        spectrum_kurtosis = stats.kurtosis(spectrum_vec)
        spectrum_skewness = stats.skew(spectrum_vec)

        # Peak characteristics
        peak_idx = np.argmax(spectrum_vec)
        spectrum_peak_height = spectrum_vec[peak_idx]
        spectrum_peak_location = diffusivities[peak_idx]

        # Calculate ADC if signal decay available
        adc_value = None
        spectrum_adc_ratio = None
        if hasattr(spectrum.signal_decay, "fit_adc"):
            try:
                adc_value = spectrum.signal_decay.fit_adc(plot=False)
                # Compare spectrum-derived apparent diffusivity with ADC
                weighted_diffusivity = np.sum(diffusivities * spectrum_vec)
                spectrum_adc_ratio = weighted_diffusivity / (adc_value + eps)
            except:
                pass

        # Extract clinical metadata if available
        patient_age = getattr(spectrum.signal_decay, "patient_age", None)
        prostate_volume = getattr(spectrum.signal_decay, "prostate_volume", None)
        psa_level = getattr(spectrum.signal_decay, "psa_level", None)
        region = getattr(spectrum.signal_decay, "a_region", None)

        return BiomarkerFeatures(
            spectrum_vector=spectrum_vec,
            diffusivities=diffusivities,
            low_diff_fraction=low_diff_fraction,
            mid_diff_fraction=mid_diff_fraction,
            high_diff_fraction=high_diff_fraction,
            low_high_ratio=low_high_ratio,
            mid_low_ratio=mid_low_ratio,
            spectrum_entropy=spectrum_entropy,
            spectrum_kurtosis=spectrum_kurtosis,
            spectrum_skewness=spectrum_skewness,
            spectrum_peak_height=spectrum_peak_height,
            spectrum_peak_location=spectrum_peak_location,
            adc_value=adc_value,
            spectrum_adc_ratio=spectrum_adc_ratio,
            patient_age=patient_age,
            prostate_volume=prostate_volume,
            psa_level=psa_level,
            region=region,
        )


class MetricsCalculator:
    """Calculate comprehensive classification metrics for cancer prediction"""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """Calculate all classification metrics"""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # AUC (requires probabilities)
        auc = 0.5  # default
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                pass

        # Quadratic weighted kappa (treating as ordinal)
        try:
            qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        except:
            qwk = cohen_kappa_score(y_true, y_pred)

        return ClassificationMetrics(
            auc=auc,
            accuracy=accuracy,
            sensitivity=recall,
            specificity=specificity,
            precision=precision,
            f1_score=f1,
            quadratic_weighted_kappa=qwk,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            positive_predictive_value=precision,
            negative_predictive_value=tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        )

    @staticmethod
    def bootstrap_confidence_intervals(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence intervals for AUC"""
        from sklearn.utils import resample

        aucs = []
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(len(y_true)), n_samples=len(y_true))
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]

            # Skip if bootstrap sample has only one class
            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                auc = roc_auc_score(y_true_boot, y_prob_boot)
                aucs.append(auc)
            except ValueError:
                continue

        if len(aucs) == 0:
            return 0.5, 0.5

        alpha = 1 - confidence
        lower = np.percentile(aucs, 100 * alpha / 2)
        upper = np.percentile(aucs, 100 * (1 - alpha / 2))

        return lower, upper


class CancerBiomarker(ABC):
    """Abstract base class for cancer biomarker classification"""

    def __init__(self, target_type: Literal["binary", "multiclass"] = "binary"):
        self.target_type = target_type
        self.feature_engineer = FeatureEngineer()
        self.metrics_calculator = MetricsCalculator()
        self.model = None
        self.feature_importance = None
        self.training_metrics = None
        self.validation_metrics = None

    @abstractmethod
    def fit(self, spectra_dataset: DiffusivitySpectraDataset, targets: np.ndarray):
        """Fit the biomarker model"""
        pass

    @abstractmethod
    def predict(self, spectra_dataset: DiffusivitySpectraDataset) -> np.ndarray:
        """Make predictions"""
        pass

    @abstractmethod
    def predict_proba(self, spectra_dataset: DiffusivitySpectraDataset) -> np.ndarray:
        """Get prediction probabilities"""
        pass

    def extract_features(
        self, spectra_dataset: DiffusivitySpectraDataset
    ) -> pd.DataFrame:
        """Extract features from spectra dataset"""
        features_list = []

        for spectrum in spectra_dataset.spectra:
            biomarker_features = self.feature_engineer.engineer_spectrum_features(
                spectrum
            )
            features_dict = biomarker_features.to_dict()
            features_list.append(features_dict)

        return pd.DataFrame(features_list)

    def cross_validate(
        self,
        spectra_dataset: DiffusivitySpectraDataset,
        targets: np.ndarray,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation for small dataset handling"""
        X = self.extract_features(spectra_dataset)

        # Use stratified k-fold for balanced splits
        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )

        cv_metrics = {
            "auc": [],
            "accuracy": [],
            "sensitivity": [],
            "specificity": [],
            "f1_score": [],
            "quadratic_weighted_kappa": [],
        }

        for train_idx, val_idx in skf.split(X, targets):
            # Create train/val splits
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]

            # Create temporary datasets for train/val
            train_spectra = [spectra_dataset.spectra[i] for i in train_idx]
            val_spectra = [spectra_dataset.spectra[i] for i in val_idx]

            train_dataset = DiffusivitySpectraDataset(spectra=train_spectra)
            val_dataset = DiffusivitySpectraDataset(spectra=val_spectra)

            # Fit model on training fold
            self.fit(train_dataset, y_train)

            # Evaluate on validation fold
            y_pred = self.predict(val_dataset)
            y_prob = self.predict_proba(val_dataset)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(
                y_val, y_pred, y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            )

            cv_metrics["auc"].append(metrics.auc)
            cv_metrics["accuracy"].append(metrics.accuracy)
            cv_metrics["sensitivity"].append(metrics.sensitivity)
            cv_metrics["specificity"].append(metrics.specificity)
            cv_metrics["f1_score"].append(metrics.f1_score)
            cv_metrics["quadratic_weighted_kappa"].append(
                metrics.quadratic_weighted_kappa
            )

        return cv_metrics

    def evaluate(
        self,
        spectra_dataset: DiffusivitySpectraDataset,
        targets: np.ndarray,
        bootstrap_ci: bool = True,
    ) -> ClassificationMetrics:
        """Evaluate model performance"""
        y_pred = self.predict(spectra_dataset)
        y_prob = self.predict_proba(spectra_dataset)

        prob_positive = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        metrics = self.metrics_calculator.calculate_metrics(
            targets, y_pred, prob_positive
        )

        # Add confidence intervals if requested
        if bootstrap_ci and len(np.unique(targets)) > 1:
            ci_lower, ci_upper = self.metrics_calculator.bootstrap_confidence_intervals(
                targets, prob_positive
            )
            metrics.auc_ci_lower = ci_lower
            metrics.auc_ci_upper = ci_upper

        return metrics


class GleasonScorePredictor(CancerBiomarker):
    """Concrete implementation for Gleason score prediction"""

    def __init__(
        self,
        model_type: str = "logistic_regression",
        target_type: str = "binary",
        **model_kwargs,
    ):
        super().__init__(target_type)
        self.model_type = model_type
        self.model_kwargs = model_kwargs

    def _create_model(self):
        """Create the specified model"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC

        if self.model_type == "logistic_regression":
            return LogisticRegression(
                random_state=42, max_iter=1000, **self.model_kwargs
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(random_state=42, **self.model_kwargs)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(random_state=42, **self.model_kwargs)
        elif self.model_type == "svm":
            return SVC(probability=True, random_state=42, **self.model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, spectra_dataset: DiffusivitySpectraDataset, targets: np.ndarray):
        """Fit the Gleason score prediction model"""
        X = self.extract_features(spectra_dataset)

        # Handle missing values
        X = X.fillna(X.median())

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X, targets)

        # Store feature importance if available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = dict(
                zip(X.columns, self.model.feature_importances_)
            )
        elif hasattr(self.model, "coef_"):
            self.feature_importance = dict(zip(X.columns, np.abs(self.model.coef_[0])))

        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)
        self.training_metrics = self.metrics_calculator.calculate_metrics(
            targets, y_pred, y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        )

    def predict(self, spectra_dataset: DiffusivitySpectraDataset) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.extract_features(spectra_dataset)
        X = X.fillna(X.median())

        return self.model.predict(X)

    def predict_proba(self, spectra_dataset: DiffusivitySpectraDataset) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self.extract_features(spectra_dataset)
        X = X.fillna(X.median())

        return self.model.predict_proba(X)
