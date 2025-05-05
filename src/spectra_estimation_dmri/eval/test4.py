import json
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import yaml
import sys

sys.path.append(os.path.join(os.getcwd() + "/src/models/"))
import gibbs


def create_improved_logistic_model(features, labels, feature_names=None):
    """
    Create and evaluate a logistic regression model with improvements for small, imbalanced datasets.

    Parameters:
    -----------
    features : array-like of shape (n_samples, n_features)
        Feature matrix
    labels : array-like of shape (n_samples,)
        Binary labels
    feature_names : list, optional
        Names of the features for feature importance reporting

    Returns:
    --------
    dict containing model results and performance metrics
    """
    from sklearn.model_selection import LeaveOneOut
    from sklearn.feature_selection import SelectFromModel
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import numpy as np

    X = np.array(features)
    y = np.array(labels)

    if len(np.unique(y)) < 2:
        raise ValueError("At least two classes required for classification")

    # Use LOOCV for small datasets
    cv = LeaveOneOut()
    scaler = StandardScaler()

    # Initialize predictions array
    y_pred_proba = np.zeros_like(y, dtype=float)

    # Store feature importances across folds
    feature_importances = np.zeros(X.shape[1])

    # Perform LOOCV
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize model with class weights
        model = LogisticRegression(
            penalty="elasticnet",  # Use both L1 and L2 regularization
            solver="saga",  # Solver that supports elasticnet
            l1_ratio=0.5,  # Equal weight to L1 and L2
            class_weight="balanced",
            random_state=42,
            max_iter=2000,
        )

        # Fit model and predict
        model.fit(X_train_scaled, y_train)
        y_pred_proba[test_idx] = model.predict_proba(X_test_scaled)[:, 1]

        # Accumulate feature importances
        feature_importances += np.abs(model.coef_[0])

    # Calculate overall AUC
    auc = roc_auc_score(y, y_pred_proba)

    # Normalize feature importances
    feature_importances /= len(cv.get_n_splits(X))

    # Feature importance dictionary
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    feature_importance_dict = dict(zip(feature_names, feature_importances))

    # Train final model on all data for coefficient interpretation
    X_scaled_full = scaler.fit_transform(X)
    final_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        class_weight="balanced",
        random_state=42,
        max_iter=2000,
    )
    final_model.fit(X_scaled_full, y)

    return {
        "auc": auc,
        "feature_importances": feature_importance_dict,
        "model": final_model,
        "predictions": y_pred_proba,
        "scaler": scaler,
    }


def analyze_feature_ratios(features, labels, feature_names):
    """
    Analyze various feature ratios to identify potentially useful combinations.

    Parameters:
    -----------
    features : array-like
        Original feature matrix
    labels : array-like
        Binary labels
    feature_names : list
        Names of the features

    Returns:
    --------
    dict containing top performing feature ratios
    """
    X = np.array(features)
    y = np.array(labels)

    ratio_results = []

    # Create ratios of all features
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            ratio = X[:, i] / X[:, j]
            auc = roc_auc_score(y, ratio)
            ratio_results.append(
                {"ratio": f"{feature_names[i]}/{feature_names[j]}", "auc": auc}
            )

    # Sort by AUC
    ratio_results.sort(key=lambda x: x["auc"], reverse=True)

    return ratio_results[:5]  # Return top 5 ratios


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
    if not isinstance(samples, (list, np.ndarray)):
        return None  # Return None for invalid samples

    feature_values = []
    for idx in feature_indices:
        try:
            values = [
                sample[idx]
                for sample in samples
                if isinstance(sample, (list, np.ndarray))
            ]
            if values:  # Only calculate mean if we have valid values
                feature_values.append(np.mean(values))
            else:
                return None
        except (IndexError, TypeError):
            return None
    return feature_values


def normal_v_tumor_analysis(rois, feature_indices, normal_tumor_data):
    """
    Perform improved logistic regression analysis for normal vs tumor classification.
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

            samples = roi.get("sample")
            feature_values = extract_features_from_samples(
                samples, [idx for _, idx in feature_indices]
            )

            if feature_values is not None:  # Only add valid feature values
                zone_data[zone][tissue_type].append(feature_values)

    # Perform analysis for each zone
    for zone in ["PZ", "TZ"]:
        if not zone_data[zone]["normal"] or not zone_data[zone]["tumor"]:
            print(f"Insufficient data for {zone} analysis")
            continue

        normal_data = np.array(zone_data[zone]["normal"])
        tumor_data = np.array(zone_data[zone]["tumor"])

        if normal_data.size == 0 or tumor_data.size == 0:
            print(f"Empty arrays for {zone} analysis")
            continue

        # Combine data and create labels
        X = np.vstack([normal_data, tumor_data])
        y = np.concatenate([np.zeros(len(normal_data)), np.ones(len(tumor_data))])
        feature_names = [feat[0] for feat in feature_indices]

        try:
            # Train and evaluate improved model
            model_results = create_improved_logistic_model(X, y, feature_names)

            # Analyze feature ratios
            ratio_results = analyze_feature_ratios(X, y, feature_names)

            # Store results
            results.append(
                {
                    "Analysis": f"{zone} Normal vs Tumor",
                    "AUC": model_results["auc"],
                    "Sample_Count": f"Normal: {len(normal_data)}, Tumor: {len(tumor_data)}",
                    "Feature_Importances": model_results["feature_importances"],
                    "Top_Ratios": ratio_results,
                }
            )
        except Exception as e:
            print(f"Error in {zone} analysis: {str(e)}")
            continue

    return results


def ggg_analysis(rois, feature_indices, ggg_data, comparison):
    """
    Perform improved logistic regression analysis for Gleason Grade Group comparisons.
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

            samples = roi.get("sample")
            feature_values = extract_features_from_samples(
                samples, [idx for _, idx in feature_indices]
            )

            if feature_values is None:  # Skip invalid feature values
                continue

            try:
                ggg = int(float(roi["target"]))
                if ggg not in ggg_features:
                    ggg_features[ggg] = []
                ggg_features[ggg].append(feature_values)
            except (ValueError, KeyError):
                continue

    # Define GGG groupings
    groupings = {
        "GGG1 vs GGG2": ([1], [2], "GS ≤ 6", "GS 3+4"),
        "GGG1 vs GGG3": ([1], [3], "GS ≤ 6", "GS 4+3"),
        "GGG1 vs GGG4-5": ([1], [4, 5], "GS ≤ 6", "GS ≥ 8"),
        "GGG1-3 vs GGG4-5": ([1, 2, 3], [4, 5], "GS ≤ 7", "GS ≥ 8"),
    }

    if comparison not in groupings:
        return results

    group1_gggs, group2_gggs, group1_name, group2_name = groupings[comparison]

    # Combine data for the groups
    try:
        group1_data = np.vstack(
            [
                ggg_features[ggg]
                for ggg in group1_gggs
                if ggg in ggg_features and len(ggg_features[ggg]) > 0
            ]
        )
        group2_data = np.vstack(
            [
                ggg_features[ggg]
                for ggg in group2_gggs
                if ggg in ggg_features and len(ggg_features[ggg]) > 0
            ]
        )
    except (ValueError, TypeError):
        print(f"Error stacking data for {comparison}")
        return results

    if group1_data.size == 0 or group2_data.size == 0:
        print(f"Insufficient data for {comparison}")
        return results

    # Combine data and create labels
    X = np.vstack([group1_data, group2_data])
    y = np.concatenate([np.zeros(len(group1_data)), np.ones(len(group2_data))])
    feature_names = [feat[0] for feat in feature_indices]

    try:
        # Train and evaluate improved model
        model_results = create_improved_logistic_model(X, y, feature_names)

        # Analyze feature ratios
        ratio_results = analyze_feature_ratios(X, y, feature_names)

        # Store results
        results.append(
            {
                "Analysis": f"{group1_name} vs {group2_name}",
                "AUC": model_results["auc"],
                "Sample_Count": f"{group1_name}: {len(group1_data)}, {group2_name}: {len(group2_data)}",
                "Feature_Importances": model_results["feature_importances"],
                "Top_Ratios": ratio_results,
            }
        )
    except Exception as e:
        print(f"Error in {comparison} analysis: {str(e)}")
        return results

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

    # Perform analyses
    tn_results = normal_v_tumor_analysis(gibbs_tn, features, tumor_normal_data)

    ggg_results = []
    for comparison in [
        "GGG1 vs GGG2",
        "GGG1 vs GGG3",
        "GGG1 vs GGG4-5",
        "GGG1-3 vs GGG4-5",
    ]:
        results = ggg_analysis(gibbs_ggg, features, ggg_data, comparison)
        ggg_results.extend(results)

    # Convert results to DataFrame with expanded columns for feature importances and ratios
    def format_results(results_list):
        formatted_results = []
        for result in results_list:
            base_result = {
                "Analysis": result["Analysis"],
                "AUC": result["AUC"],
                "Sample_Count": result["Sample_Count"],
            }

            # Add feature importances
            for feat, importance in result["Feature_Importances"].items():
                base_result[f"Importance_{feat}"] = importance

            # Add top ratios
            for i, ratio_result in enumerate(result["Top_Ratios"]):
                base_result[f"Top_Ratio_{i+1}"] = (
                    f"{ratio_result['ratio']} (AUC: {ratio_result['auc']:.3f})"
                )

            formatted_results.append(base_result)
        return formatted_results

    # Format and save results
    all_results = pd.DataFrame(format_results(tn_results + ggg_results))
    all_results.to_csv(
        os.path.join(
            configs["EVAL_DIR_PATH"], "improved_logistic_regression_results.csv"
        ),
        index=False,
    )

    # Save detailed results in JSON format for further analysis
    detailed_results = {"normal_vs_tumor": tn_results, "ggg_comparisons": ggg_results}
    with open(
        os.path.join(configs["EVAL_DIR_PATH"], "detailed_results.json"), "w"
    ) as f:
        json.dump(detailed_results, f, indent=4)


if __name__ == "__main__":
    with open(os.path.join(os.getcwd(), "configs.yaml"), "r") as file:
        configs = yaml.safe_load(file)
    main(configs)
