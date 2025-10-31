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
        # Binary: 0=GGG 1-2 (low-grade), 1=GGG 3-5 (high-grade)
        # Only include tumor samples
        df = df[df["is_tumor"] == True]

        # Convert GGG to binary
        # GGG 1: GS 6 or less (3+3)
        # GGG 2: GS 7 (3+4) - lower risk
        # GGG 3: GS 7 (4+3) - higher risk
        # GGG 4: GS 8
        # GGG 5: GS 9-10
        # Split at GGG 3 is clinically significant threshold
        df = df[df["ggg"].notna() & (df["ggg"] != 0)]  # Remove unknown/normal
        df["label"] = (df["ggg"] >= 3).astype(int)

    else:
        raise ValueError(f"Unknown task: {task}")

    # Extract feature columns (exclude metadata)
    metadata_cols = [
        "roi_id",
        "patient_id",
        "region",
        "ggg",
        "gs",
        "is_tumor",
        "label",
        "zone",
    ]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    X_meta = df
    X_features = df[feature_cols].values
    y = df["label"].values

    return X_meta, X_features, y


def loocv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 10.0,
    spectra_dataset=None,
    X_meta: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    n_mc_samples: int = 200,
    propagate_uncertainty: bool = False,
    use_feature_selection: bool = False,
    n_features_to_select: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Perform LOOCV and return predicted probabilities with optional MC uncertainty propagation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        C: L2 regularization parameter (inverse of regularization strength)
        spectra_dataset: Optional dataset for MC uncertainty propagation
        X_meta: Optional metadata dataframe with roi_id for matching
        feature_cols: Optional list of feature column names for MC extraction
        n_mc_samples: Number of MC samples for uncertainty propagation
        propagate_uncertainty: If True, propagate posterior uncertainty through classifier
        use_feature_selection: If True, select top-k features per fold (prevents leakage)
        n_features_to_select: Number of top features to select per fold

    Returns:
        (y_pred_proba, y_pred_class, y_pred_std, loocv_feature_importance):
            - y_pred_proba: Predicted probabilities for positive class
            - y_pred_class: Predicted class labels
            - y_pred_std: Std of predictions (None if not propagate_uncertainty)
            - loocv_feature_importance: Dict of feature stats across folds (None if no tracking)
    """
    # Initialize arrays for predictions
    n_samples = len(y)
    y_pred_proba = np.zeros(n_samples)
    y_pred_class = np.zeros(n_samples, dtype=int)
    y_pred_std = np.zeros(n_samples) if propagate_uncertainty else None

    # Track feature selection and coefficients across folds
    fold_feature_names = []  # List of selected feature names per fold
    fold_coefficients = []  # List of coefficient arrays per fold

    # Leave-One-Out CV
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Feature selection (per fold to prevent leakage)
        selector = None  # Track selector for MC uncertainty propagation
        selected_features_this_fold = list(
            range(X_train.shape[1])
        )  # Default: all features

        if use_feature_selection and X_train.shape[1] > 1:
            from sklearn.feature_selection import SelectKBest, f_classif

            k = min(n_features_to_select, X_train.shape[1])
            selector = SelectKBest(f_classif, k=k)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

            # Track which features were selected
            selected_features_this_fold = selector.get_support(indices=True).tolist()

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_train_scaled, y_train)

        # Store selected features and coefficients for this fold
        if feature_cols is not None:
            fold_feature_names.append(
                [feature_cols[i] for i in selected_features_this_fold]
            )
            fold_coefficients.append(clf.coef_[0])  # Shape: (n_features_selected,)

        # Predict with optional MC uncertainty propagation
        if propagate_uncertainty and spectra_dataset is not None and X_meta is not None:
            # Get the test sample's ROI ID
            test_roi_id = X_meta.iloc[test_idx[0]]["roi_id"]

            # Find corresponding spectrum in dataset
            test_spectrum = None
            for spectrum in spectra_dataset.spectra:
                patient_id = getattr(spectrum.signal_decay, "patient", None)
                region = getattr(spectrum.signal_decay, "a_region", None)
                is_tumor = getattr(spectrum.signal_decay, "is_tumor", False)
                roi_id = f"{patient_id}_{region}_{'tumor' if is_tumor else 'normal'}"

                if roi_id == test_roi_id:
                    test_spectrum = spectrum
                    break

            if test_spectrum is not None:
                arrays = test_spectrum.as_numpy()
                samples = arrays["spectrum_samples"]

                if samples is not None:
                    # Extract features for MC samples and predict with each
                    from .features import extract_mc_features

                    _, _, feature_samples = extract_mc_features(
                        arrays["diffusivities"],
                        samples,
                        n_mc_samples,
                        random_seed=42 + test_idx[0],
                    )

                    mc_predictions = []
                    for fs in feature_samples:
                        # Extract relevant features
                        X_sample = np.array(
                            [fs[fname] for fname in feature_cols]
                        ).reshape(1, -1)

                        # Apply same feature selection as training (if used)
                        if selector is not None:
                            X_sample = selector.transform(X_sample)

                        X_sample_scaled = scaler.transform(X_sample)
                        pred = clf.predict_proba(X_sample_scaled)[0, 1]
                        mc_predictions.append(pred)

                    y_pred_proba[test_idx] = np.mean(mc_predictions)
                    y_pred_std[test_idx] = np.std(mc_predictions)
                    y_pred_class[test_idx] = int(y_pred_proba[test_idx] >= 0.5)
                else:
                    # No posterior samples, fall back to point estimate
                    y_pred_proba[test_idx] = clf.predict_proba(X_test_scaled)[0, 1]
                    y_pred_class[test_idx] = clf.predict(X_test_scaled)[0]
                    y_pred_std[test_idx] = 0.0
            else:
                # Could not find spectrum, fall back to point estimate
                y_pred_proba[test_idx] = clf.predict_proba(X_test_scaled)[0, 1]
                y_pred_class[test_idx] = clf.predict(X_test_scaled)[0]
                y_pred_std[test_idx] = 0.0
        else:
            # Standard prediction without uncertainty propagation
            y_pred_proba[test_idx] = clf.predict_proba(X_test_scaled)[0, 1]
            y_pred_class[test_idx] = clf.predict(X_test_scaled)[0]

    # Aggregate feature importance across folds
    loocv_feature_importance = None
    if len(fold_feature_names) > 0:
        # Count selection frequency and average coefficients
        from collections import defaultdict

        feature_stats = defaultdict(
            lambda: {"count": 0, "coef_sum": 0.0, "coef_list": []}
        )

        for feat_names, coefs in zip(fold_feature_names, fold_coefficients):
            for fname, coef in zip(feat_names, coefs):
                feature_stats[fname]["count"] += 1
                feature_stats[fname]["coef_sum"] += coef
                feature_stats[fname]["coef_list"].append(coef)

        # Create summary: {feature_name: (selection_freq, mean_coef, std_coef)}
        loocv_feature_importance = {
            fname: {
                "selection_frequency": stats["count"] / n_samples,
                "mean_coefficient": stats["coef_sum"] / stats["count"],
                "std_coefficient": np.std(stats["coef_list"]),
                "n_folds_selected": stats["count"],
            }
            for fname, stats in feature_stats.items()
        }

    return y_pred_proba, y_pred_class, y_pred_std, loocv_feature_importance


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
    C: float = 1.0,
    use_feature_selection: bool = False,
    n_features_to_select: int = 5,
    spectra_dataset=None,
    n_mc_samples: int = 200,
    propagate_uncertainty: bool = False,
) -> Dict:
    """
    Evaluate a specific feature set with LOOCV.

    Args:
        X_meta: Full dataframe with metadata
        X_features: Feature matrix (all features)
        y: Labels
        feature_cols: List of feature column names in X_meta
        feature_name: Descriptive name for this feature set
        C: L2 regularization parameter (inverse of regularization strength)
        use_feature_selection: If True, select top-k features per fold (prevents leakage)
        n_features_to_select: Number of top features to select per fold
        spectra_dataset: Optional dataset for MC uncertainty propagation
        n_mc_samples: Number of MC samples for uncertainty propagation
        propagate_uncertainty: If True, propagate posterior uncertainty through classifier

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
        if col
        not in [
            "roi_id",
            "patient_id",
            "region",
            "ggg",
            "gs",
            "is_tumor",
            "label",
            "zone",
        ]
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

    # Note: Feature selection now happens inside LOOCV loop to prevent data leakage
    # We keep track of original feature names for later
    selected_feature_names = feature_cols.copy()

    # LOOCV predictions
    # IMPORTANT: For single features, use raw values (no training) to avoid overfitting
    # For multi-features, train LR classifier with LOOCV
    y_pred_std = None
    if X_selected.shape[1] == 1:
        # Single feature: Use raw values directly (like ADC)
        # No classifier training - just use the feature value for ranking
        y_pred_proba = X_selected[:, 0]

        # Normalize to [0, 1] range for consistency
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (
            y_pred_proba.max() - y_pred_proba.min() + 1e-10
        )

        # Simple threshold at median for class prediction
        y_pred_class = (y_pred_proba >= 0.5).astype(int)

        print(
            f"  [INFO] {feature_name}: Using raw feature values (single feature, no training)"
        )
    else:
        # Multi-feature: Train LR classifier with LOOCV
        y_pred_proba, y_pred_class, y_pred_std, loocv_feature_importance = (
            loocv_predictions(
                X_selected,
                y,
                C=C,
                spectra_dataset=spectra_dataset,
                X_meta=X_meta,
                feature_cols=selected_feature_names,
                n_mc_samples=n_mc_samples,
                propagate_uncertainty=propagate_uncertainty,
                use_feature_selection=use_feature_selection,
                n_features_to_select=n_features_to_select,
            )
        )

        if propagate_uncertainty and y_pred_std is not None:
            print(
                f"  [INFO] {feature_name}: MC uncertainty propagation enabled "
                f"(mean std: {np.mean(y_pred_std):.4f})"
            )

        if use_feature_selection and len(feature_cols) > 1:
            print(
                f"  [INFO] {feature_name}: Feature selection per-fold enabled "
                f"(selecting top-{n_features_to_select} from {len(feature_cols)} features)"
            )

        # Print LOOCV feature importance summary
        if loocv_feature_importance is not None:
            print(f"  [INFO] {feature_name}: LOOCV Feature Selection Frequencies:")
            for fname, stats in sorted(
                loocv_feature_importance.items(),
                key=lambda x: x[1]["selection_frequency"],
                reverse=True,
            ):
                print(
                    f"         {fname}: {stats['selection_frequency']*100:.1f}% "
                    f"(coef={stats['mean_coefficient']:.3f}±{stats['std_coefficient']:.3f})"
                )

    # Compute metrics
    metrics = compute_metrics(y, y_pred_proba, y_pred_class)

    # Handle inverse correlation: if AUC < 0.5, invert predictions
    # This happens when higher feature values indicate lower probability of positive class
    if not np.isnan(metrics["auc"]) and metrics["auc"] < 0.5:
        print(
            f"  [INFO] {feature_name}: AUC={metrics['auc']:.3f} < 0.5, inverting predictions"
        )
        y_pred_proba = 1.0 - y_pred_proba
        y_pred_class = (y_pred_proba >= 0.5).astype(int)
        metrics = compute_metrics(y, y_pred_proba, y_pred_class)

        # CRITICAL WARNING for suspiciously perfect AUCs
        if metrics["auc"] >= 0.99:
            print(
                f"  [WARNING] {feature_name}: Perfect AUC={metrics['auc']:.3f} after inversion!"
            )
            print(
                f"            This suggests overfitting on noise with small sample size (N={len(y)})"
            )
            print(
                f"            Consider: (1) Larger dataset, (2) Stratified CV, (3) This feature may be unreliable"
            )

    # Bootstrap CI on AUC
    if not np.isnan(metrics["auc"]):
        auc_mean, ci_lower, ci_upper = bootstrap_auc_ci(y, y_pred_proba)
        metrics["auc_ci_lower"] = ci_lower
        metrics["auc_ci_upper"] = ci_upper

        # Additional warning if CI is suspiciously narrow at perfect AUC
        if metrics["auc"] >= 0.99 and (ci_upper - ci_lower) < 0.05:
            print(
                f"  [WARNING] Very narrow CI [{ci_lower:.3f}-{ci_upper:.3f}] for perfect AUC - likely overfitting!"
            )
    else:
        metrics["auc_ci_lower"] = np.nan
        metrics["auc_ci_upper"] = np.nan

    # Train final model on all data to extract feature importance (LR coefficients)
    # For single features, coefficient is just correlation
    if X_selected.shape[1] == 1:
        # Single feature: No model needed, importance is just correlation
        feature_importance = np.array([np.corrcoef(X_selected[:, 0], y)[0, 1]])
    else:
        # Multi-feature: Train LR to get coefficients
        # If feature selection was used during LOOCV, apply it here too for consistency
        X_for_importance = X_selected
        final_selected_features = selected_feature_names.copy()

        if use_feature_selection and len(feature_cols) > 1:
            from sklearn.feature_selection import SelectKBest, f_classif

            k = min(n_features_to_select, X_selected.shape[1])
            final_selector = SelectKBest(f_classif, k=k)
            X_for_importance = final_selector.fit_transform(X_selected, y)

            # Get the names of selected features
            selected_indices = final_selector.get_support(indices=True)
            final_selected_features = [feature_cols[i] for i in selected_indices]

        final_model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

        # Standardize for final model
        final_scaler = StandardScaler()
        X_selected_scaled = final_scaler.fit_transform(X_for_importance)
        final_model.fit(X_selected_scaled, y)
        feature_importance = final_model.coef_[0]  # Shape: (n_features_selected,)

        # Update selected features to reflect what was actually used
        selected_feature_names = final_selected_features

    # Store results
    results = {
        "feature_name": feature_name,
        "feature_cols": feature_cols,  # Original feature cols
        "selected_features": selected_feature_names,  # Actually used features (after selection)
        "n_features": len(selected_feature_names),  # Number of actually used features
        "n_samples": len(y),
        "n_positive": np.sum(y == 1),
        "n_negative": np.sum(y == 0),
        "y_true": y,
        "y_pred_proba": y_pred_proba,
        "y_pred_class": y_pred_class,
        "y_pred_std": y_pred_std,  # Prediction uncertainty (if propagated)
        "metrics": metrics,
        "feature_importance": feature_importance,  # LR coefficients (full-dataset model)
        "feature_names_ordered": selected_feature_names,  # Feature names in order (after selection)
        "loocv_feature_importance": (
            loocv_feature_importance if X_selected.shape[1] > 1 else None
        ),  # LOOCV-based importance
    }

    return results
