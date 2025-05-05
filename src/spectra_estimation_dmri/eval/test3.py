import json
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import yaml
import sys

sys.path.append(os.path.join(os.getcwd() + "/src/models/"))
import gibbs


def create_logistic_model(features, labels):
    """
    Create and evaluate a logistic regression model using Leave-One-Out Cross Validation.

    Parameters:
    -----------
    features : array-like of shape (n_samples, n_features)
        Feature matrix containing only the fraction values
    labels : array-like of shape (n_samples,)
        Binary labels (0 for normal/lower grade, 1 for tumor/higher grade)

    Returns:
    --------
    dict containing:
        - auc: float
            Cross-validated AUC score
        - model: LogisticRegression
            The trained logistic regression model
        - coefficients: array
            Model coefficients for each feature
    """
    # Ensure inputs are numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize logistic regression with L2 regularization
    model = LogisticRegression(
        penalty="l2",
        random_state=42,
        class_weight="balanced",
        max_iter=1000,
        fit_intercept=True,
    )

    # Initialize Leave-One-Out Cross Validation
    cv = LeaveOneOut()

    # Store predictions for AUC calculation
    y_pred = np.zeros_like(y, dtype=float)

    # Perform LOOCV
    for train_idx, test_idx in cv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y[train_idx]

        # Fit model on training data
        model.fit(X_train, y_train)

        # Predict probability for test sample
        y_pred[test_idx] = model.predict_proba(X_test)[:, 1]

    # Calculate AUC using all predictions
    cv_auc = roc_auc_score(y, y_pred)

    # Fit the model on the full dataset for final coefficients
    model.fit(X_scaled, y)

    return {
        "auc": cv_auc,
        "model": model,
        "coefficients": model.coef_[0],
        "scaler": scaler,
    }


def extract_features_from_samples(samples, feature_indices):
    """
    Extract features from samples using numeric indices.

    Parameters:
    -----------
    samples : list of arrays
        List of sample arrays from the Gibbs sampler
    feature_indices : list of int
        List of indices for the features to extract

    Returns:
    --------
    list
        List of mean feature values
    """
    feature_values = []
    for idx in feature_indices:
        values = [sample[idx] for sample in samples]
        feature_values.append(np.mean(values))
    return feature_values


def normal_v_tumor_analysis(rois, feature_indices, normal_tumor_data):
    """
    Perform logistic regression analysis for normal vs tumor classification.
    """
    results = []
    zone_data = {"PZ": {"normal": [], "tumor": []}, "TZ": {"normal": [], "tumor": []}}

    # Extract features for each ROI
    for patient_key, patient_data in normal_tumor_data.items():
        for roi_key, roi_data in patient_data.items():
            a_region = roi_data["anatomical_region"]
            zone = "PZ" if "pz" in a_region.lower() else "TZ"
            tissue_type = "normal" if "normal" in a_region.lower() else "tumor"

            roi = next(
                (r for r in rois[a_region] if r["patient_key"] == patient_key), None
            )
            if roi is None:
                continue

            # Extract features using numeric indices
            samples = roi["sample"]
            feature_values = extract_features_from_samples(
                samples, [idx for _, idx in feature_indices]
            )
            zone_data[zone][tissue_type].append(feature_values)

    # Perform logistic regression for each zone
    for zone in ["PZ", "TZ"]:
        normal_data = np.array(zone_data[zone]["normal"])
        tumor_data = np.array(zone_data[zone]["tumor"])

        if len(normal_data) == 0 or len(tumor_data) == 0:
            continue

        # Combine data and create labels
        X = np.vstack([normal_data, tumor_data])
        y = np.concatenate([np.zeros(len(normal_data)), np.ones(len(tumor_data))])

        # Train and evaluate model
        model_results = create_logistic_model(X, y)

        # Store results
        results.append(
            {
                "Analysis": f"{zone} Normal vs Tumor",
                "AUC": model_results["auc"],
                "Sample_Count": f"Normal: {len(normal_data)}, Tumor: {len(tumor_data)}",
                "Coefficients": dict(
                    zip(
                        [feat[0] for feat in feature_indices],
                        model_results["coefficients"],
                    )
                ),
            }
        )

    return results


def ggg_analysis(rois, feature_indices, ggg_data, comparison):
    """
    Perform logistic regression analysis for Gleason Grade Group comparisons.
    """
    results = []
    ggg_features = {}

    # Extract features for each ROI
    for patient_key, patient_data in ggg_data.items():
        for roi_key, roi_data in patient_data.items():
            a_region = roi_data["anatomical_region"]
            if "tumor" not in a_region.lower():
                continue

            roi = next(
                (r for r in rois[a_region] if r["patient_key"] == patient_key), None
            )
            if roi is None:
                continue

            samples = roi["sample"]
            feature_values = extract_features_from_samples(
                samples, [idx for _, idx in feature_indices]
            )

            ggg = int(float(roi["target"]))
            if ggg not in ggg_features:
                ggg_features[ggg] = []
            ggg_features[ggg].append(feature_values)

    # Define GGG groupings
    groupings = {
        "GGG1 vs GGG2": ([1], [2], "GS ≤ 6", "GS 3+4"),
        "GGG1 vs GGG3": ([1], [3], "GS ≤ 6", "GS 4+3"),
        "GGG1 vs GGG4-5": ([1], [4, 5], "GS ≤ 6", "GS ≥ 8"),
        "GGG1-3 vs GGG4-5": ([1, 2, 3], [4, 5], "GS ≤ 7", "GS ≥ 8"),
    }

    group1_gggs, group2_gggs, group1_name, group2_name = groupings[comparison]

    # Combine data for the groups
    group1_data = np.vstack(
        [ggg_features[ggg] for ggg in group1_gggs if ggg in ggg_features]
    )
    group2_data = np.vstack(
        [ggg_features[ggg] for ggg in group2_gggs if ggg in ggg_features]
    )

    if len(group1_data) == 0 or len(group2_data) == 0:
        return results

    # Combine data and create labels
    X = np.vstack([group1_data, group2_data])
    y = np.concatenate([np.zeros(len(group1_data)), np.ones(len(group2_data))])

    # Train and evaluate model
    model_results = create_logistic_model(X, y)

    # Store results
    results.append(
        {
            "Analysis": f"{group1_name} vs {group2_name}",
            "AUC": model_results["auc"],
            "Sample_Count": f"{group1_name}: {len(group1_data)}, {group2_name}: {len(group2_data)}",
            "Coefficients": dict(
                zip(
                    [feat[0] for feat in feature_indices], model_results["coefficients"]
                )
            ),
        }
    )

    return results


def main(configs: dict) -> None:
    # Load data
    with open(configs["INPUT_D2_DICT"], "r") as f:
        ggg_data = json.load(f)
    with open(configs["INPUT_D1_DICT"], "r") as f:
        tumor_normal_data = json.load(f)

    # Define features with their corresponding indices
    features = [
        (".25", 0),
        (".50", 1),
        (".75", 2),
        ("1.", 3),
        ("1.25", 4),
        ("1.50", 5),
        ("2.", 6),
        ("2.50", 7),
        ("3.", 8),
        ("20.", 9),
    ]

    # Fetch pre-computed model outputs
    gibbs_ggg = gibbs.fetch_gibbs_objects(
        gibbs_json_path=configs["GIBBS_D2_JSON"],
        gibbs_hdf5_path=configs["GIBBS_D2_HDF5"],
    )
    gibbs_tn = gibbs.fetch_gibbs_objects(
        gibbs_json_path=configs["GIBBS_D1_JSON"],
        gibbs_hdf5_path=configs["GIBBS_D1_HDF5"],
    )

    # Perform normal vs tumor analysis
    tn_results = normal_v_tumor_analysis(gibbs_tn, features, tumor_normal_data)

    # Perform GGG analyses
    ggg_results = []
    for comparison in [
        "GGG1 vs GGG2",
        "GGG1 vs GGG3",
        "GGG1 vs GGG4-5",
        "GGG1-3 vs GGG4-5",
    ]:
        results = ggg_analysis(gibbs_ggg, features, ggg_data, comparison)
        ggg_results.extend(results)

    # Save results
    all_results = pd.DataFrame(tn_results + ggg_results)
    all_results.to_csv(
        os.path.join(configs["EVAL_DIR_PATH"], "logistic_regression_results.csv"),
        index=False,
    )


if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), "configs.yaml"), "r") as file:
        configs = yaml.safe_load(file)
    main(configs)
