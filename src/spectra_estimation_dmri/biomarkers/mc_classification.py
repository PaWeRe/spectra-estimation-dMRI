"""
Monte Carlo classification with Leave-One-Out Cross-Validation.

Implements:
- MC-based prediction uncertainty propagation
- LOOCV for robust performance estimation
- Statistical comparison with DeLong test
- Support for individual features, combinations, and full models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings


def prepare_classification_data(
    features_df: pd.DataFrame,
    task: str = "tumor_vs_normal",
    zone: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare features and labels for classification.

    Args:
        features_df: DataFrame with features and metadata
        task: 'tumor_vs_normal' or 'ggg'
        zone: Optional zone filter ('pz' or 'tz')

    Returns:
        (X_meta, X_features, y):
            - X_meta: Full dataframe with metadata
            - X_features: Feature columns only (numpy array)
            - y: Binary labels
    """
    df = features_df.copy()

    # Filter by zone if specified
    if zone is not None:
        # Parse zone from region string
        df["zone"] = df["region"].apply(
            lambda x: (
                "pz"
                if "pz" in str(x).lower()
                else ("tz" if "tz" in str(x).lower() else "unknown")
            )
        )
        df = df[df["zone"] == zone]

    # Create labels based on task
    if task == "tumor_vs_normal":
        # Binary: 0=normal, 1=tumor
        df["label"] = df["is_tumor"].astype(int)
        # Remove samples with missing labels
        df = df[df["label"].notna()]

    elif task == "ggg":
        # Binary: 0=GGG<7 (GGG 1-2), 1=GGG>=7 (GGG 3-5)
        # Only include tumor samples
        df = df[df["is_tumor"] == True]

        # Convert GGG to binary (assuming GGG 1-2 vs 3-5 maps to GS <7 vs >=7)
        # GGG 1: GS 6 (3+3), GGG 2: GS 7 (3+4), GGG 3: GS 7 (4+3)
        # So GGG 1-2 is <7, GGG 3-5 is >=7
        df = df[df["ggg"].notna() & (df["ggg"] != 0)]  # Remove unknown/normal
        df["label"] = (df["ggg"] >= 3).astype(int)

    else:
        raise ValueError(f"Unknown task: {task}")

    # Extract feature columns (exclude metadata)
    metadata_cols = ["patient_id", "region", "ggg", "gs", "is_tumor", "label", "zone"]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X_meta = df
    X_features = df[feature_cols].values
    y = df["label"].values

    return X_meta, X_features, y


def loocv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform LOOCV and return predicted probabilities.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        regularization: L2 regularization strength (C parameter)

    Returns:
        (y_pred_proba, y_pred_class):
            - y_pred_proba: Predicted probabilities for positive class
            - y_pred_class: Predicted class labels
    """
    # Standardize features
    scaler = StandardScaler()

    # Initialize arrays for predictions
    n_samples = len(y)
    y_pred_proba = np.zeros(n_samples)
    y_pred_class = np.zeros(n_samples, dtype=int)

    # Leave-One-Out CV
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Standardize
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(
            C=regularization, max_iter=1000, random_state=42, solver="lbfgs"
        )
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred_proba[test_idx] = clf.predict_proba(X_test_scaled)[0, 1]
        y_pred_class[test_idx] = clf.predict(X_test_scaled)[0]

    return y_pred_proba, y_pred_class


def mc_predictions(
    spectra_dataset,
    trained_model: LogisticRegression,
    trained_scaler: StandardScaler,
    feature_names: List[str],
    n_mc_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with MC uncertainty propagation.

    Args:
        spectra_dataset: DiffusivitySpectraDataset
        trained_model: Fitted LogisticRegression model
        trained_scaler: Fitted StandardScaler
        feature_names: List of feature names to extract
        n_mc_samples: Number of MC samples per prediction

    Returns:
        (pred_proba_mean, pred_proba_std):
            - pred_proba_mean: Mean prediction probability
            - pred_proba_std: Std of prediction (uncertainty)
    """
    from .features import extract_mc_features

    pred_proba_mean = []
    pred_proba_std = []

    for spectrum in spectra_dataset.spectra:
        arrays = spectrum.as_numpy()
        diffusivities = arrays["diffusivities"]
        samples = arrays["spectrum_samples"]

        if samples is None:
            # No uncertainty available
            pred_proba_mean.append(0.5)
            pred_proba_std.append(0.0)
            continue

        # Extract features for MC samples
        _, _, feature_samples = extract_mc_features(
            diffusivities, samples, n_mc_samples
        )

        # Predict for each MC sample
        mc_predictions = []
        for fs in feature_samples:
            # Extract relevant features
            X_sample = np.array([fs[fname] for fname in feature_names]).reshape(1, -1)
            X_sample_scaled = trained_scaler.transform(X_sample)
            pred_proba = trained_model.predict_proba(X_sample_scaled)[0, 1]
            mc_predictions.append(pred_proba)

        pred_proba_mean.append(np.mean(mc_predictions))
        pred_proba_std.append(np.std(mc_predictions))

    return np.array(pred_proba_mean), np.array(pred_proba_std)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_class: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred_class: Predicted class labels

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # AUC
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
    else:
        metrics["auc"] = np.nan

    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred_class)

    # Confusion matrix metrics
    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["tp"] = tp
        metrics["tn"] = tn
        metrics["fp"] = fp
        metrics["fn"] = fn

    return metrics


def delong_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> float:
    """
    DeLong test for comparing two AUC values.

    Args:
        y_true: True labels
        y_pred1: Predicted probabilities for model 1
        y_pred2: Predicted probabilities for model 2

    Returns:
        p_value: Two-sided p-value
    """
    # Simple implementation using normal approximation
    # For small samples, this is approximate
    try:
        auc1 = roc_auc_score(y_true, y_pred1)
        auc2 = roc_auc_score(y_true, y_pred2)

        # Use Hanley-McNeil method for variance estimation
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        # Simplified variance (conservative estimate)
        # Full DeLong requires covariance computation
        var1 = auc1 * (1 - auc1) / min(n_pos, n_neg)
        var2 = auc2 * (1 - auc2) / min(n_pos, n_neg)

        # Z-test (assuming independence - conservative)
        z = (auc1 - auc2) / np.sqrt(var1 + var2 + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))

        return p_value
    except:
        return np.nan


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for AUC.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level (e.g., 0.05 for 95% CI)

    Returns:
        (auc_mean, ci_lower, ci_upper)
    """
    n_samples = len(y_true)
    auc_bootstrap = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y_true[idx]
        pred_boot = y_pred_proba[idx]

        # Check if both classes present
        if len(np.unique(y_boot)) > 1:
            auc_boot = roc_auc_score(y_boot, pred_boot)
            auc_bootstrap.append(auc_boot)

    if len(auc_bootstrap) == 0:
        return np.nan, np.nan, np.nan

    auc_mean = np.mean(auc_bootstrap)
    ci_lower = np.percentile(auc_bootstrap, 100 * alpha / 2)
    ci_upper = np.percentile(auc_bootstrap, 100 * (1 - alpha / 2))

    return auc_mean, ci_lower, ci_upper


def evaluate_feature_set(
    X_meta: pd.DataFrame,
    X_features: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    feature_name: str,
    regularization: float = 1.0,
) -> Dict:
    """
    Evaluate a specific feature set with LOOCV.

    Args:
        X_meta: Full dataframe with metadata
        X_features: Feature matrix (all features)
        y: Labels
        feature_cols: List of feature column names in X_meta
        feature_name: Descriptive name for this feature set
        regularization: L2 regularization strength

    Returns:
        Dictionary with results
    """
    # Select features
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    # Get indices of selected features
    all_feature_cols = [
        col
        for col in X_meta.columns
        if col not in ["patient_id", "region", "ggg", "gs", "is_tumor", "label", "zone"]
    ]

    feature_indices = [
        all_feature_cols.index(fc) for fc in feature_cols if fc in all_feature_cols
    ]

    if len(feature_indices) == 0:
        print(f"[WARNING] No valid features found for {feature_name}")
        return None

    X_selected = X_features[:, feature_indices]

    # Check if we have at least 2 classes
    if len(np.unique(y)) < 2:
        print(
            f"[WARNING] {feature_name}: Only one class present, skipping classification"
        )
        return None

    # LOOCV predictions
    y_pred_proba, y_pred_class = loocv_predictions(X_selected, y, regularization)

    # Compute metrics
    metrics = compute_metrics(y, y_pred_proba, y_pred_class)

    # Bootstrap CI on AUC
    if not np.isnan(metrics["auc"]):
        auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(y, y_pred_proba)
        metrics["auc_ci_lower"] = ci_lower
        metrics["auc_ci_upper"] = ci_upper
    else:
        metrics["auc_ci_lower"] = np.nan
        metrics["auc_ci_upper"] = np.nan

    # Store results
    results = {
        "feature_name": feature_name,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_samples": len(y),
        "n_positive": np.sum(y == 1),
        "n_negative": np.sum(y == 0),
        "y_true": y,
        "y_pred_proba": y_pred_proba,
        "y_pred_class": y_pred_class,
        "metrics": metrics,
    }

    return results
