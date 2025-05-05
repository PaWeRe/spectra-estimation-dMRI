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
import importlib.resources

sys.path.append(os.path.join(os.getcwd() + "/src/models/"))
import gibbs


def create_feature_combo(features, importances) -> float:
    """
    Create a linear combination feature: .25 / 2.0.

    Args:
    - features: list of feature values where index 0 corresponds to '.25' and index 6 corresponds to '2.0'

    Returns:
    - float: The computed feature (.25 / 2.0)
    """
    # Normalize weights
    weights = importances / np.sum(importances)

    # Select features and apply weights
    feature_025 = features[0] * weights[0]
    feature_050 = features[1] * weights[1]
    feature_075 = features[2] * weights[2]
    feature_1_0 = features[3] * weights[3]
    feature_1_25 = features[4] * weights[4]
    feature_1_50 = features[5] * weights[5]
    feature_2_0 = 1 / (features[6]) * weights[6]
    feature_2_50 = 1 / features[7] * weights[7]
    feature_3_0 = 1 / features[8] * weights[8]

    # Combine features non-linearly
    feature_combo = feature_025 + 1 / feature_2_0 + 1 / feature_2_50

    # Apply log transformation
    # feature_combo = np.log1p(feature_combo)

    return feature_combo


def calculate_adc_multi(
    signal_values, b_values, a_region, b_range="0-1250", plot=False
):
    """
    Calculate ADC using a monoexponential model.

    Parameters:
    signal_values : array-like
        The measured signal intensities
    b_values : array-like
        The corresponding b-values
    a_region : str
        The anatomical region
    b_range : str, optional
        The range of b-values to use. Either '0-1000', '0-1250' or '250-1250'
    plot : bool, optional
        If True, plot the signal decay and fitted line

    Returns:
    adc : float
        The calculated Apparent Diffusion Coefficient
    """
    if b_range == "0-1000":
        mask = b_values <= 1000
    elif b_range == "0-1250":
        mask = b_values <= 1250
    elif b_range == "250-1000":
        mask = (b_values >= 250) & (b_values <= 1000)
    elif b_range == "250-1250":
        mask = (b_values >= 250) & (b_values <= 1250)
    else:
        raise ValueError(
            "Invalid b_range. Use '0-1000', '0-1250', '250-1000' or '250-1250'"
        )

    valid_mask = (signal_values > 0) & mask  # Exclude non-positive signal values

    if not np.any(valid_mask):
        raise ValueError("No valid signal values available for ADC calculation.")

    log_signal = np.log(signal_values[valid_mask])
    valid_b_values = b_values[valid_mask]

    slope, intercept = np.polyfit(valid_b_values, log_signal, 1)
    adc = -slope

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(b_values, signal_values, label="Original data")
        plt.scatter(valid_b_values, np.exp(log_signal), label="Used for fitting")
        fit_b_values = np.linspace(min(valid_b_values), max(valid_b_values), 100)
        fit_signal = np.exp(intercept - adc * fit_b_values)
        plt.plot(fit_b_values, fit_signal, "r-", label="Fitted line")
        plt.xlabel("b-value (s/mm²)")
        plt.ylabel("Signal intensity")
        plt.title(
            f"Signal Decay and ADC Fit (ADC = {adc:.4f} mm²/s)|{a_region}|{b_range}"
        )
        plt.legend()
        plt.yscale("log")
        plt.grid(True)
        plt.show()

    return adc


def gs_stratified_boxplot(rois: dict, save_filename: str, features: list) -> None:
    """
    Create boxplots for multiple features stratified by Gleason Score.

    Args:
    - rois: dict of ROIs
    - save_filename: str, path to save the PDF
    - features: list of tuples, each containing (feature_name, diffusivity_index)
                e.g. [('.25', 0), ('.50', 1), ('3.0', 8)]
    """
    gs_stratification = {
        "GS 6": [],
        "GS 7 (3+4)": [],
        "GS 7 (4+3)": [],
        "GS 8-10": [],
    }

    # Collect data for all features
    for zone_key, zone_list in rois.items():
        if "tumor" in zone_key:
            for element in zone_list:
                try:
                    target = element["target"]
                    samples = [
                        np.transpose(element["sample"])[idx] for _, idx in features
                    ]

                    if target == "1":
                        gs_stratification["GS 6"].append(samples)
                    elif target == "2":
                        gs_stratification["GS 7 (3+4)"].append(samples)
                    elif target == "3":
                        gs_stratification["GS 7 (4+3)"].append(samples)
                    elif target == "4":
                        gs_stratification["GS 8-10"].append(samples)
                    elif target == "5":
                        gs_stratification["GS 8-10"].append(samples)
                except ValueError:
                    p_key = element["patient_key"]
                    print(f"Something wrong with print_roi, patient: {p_key}")

    # Calculate averages and counts
    for gs, sample_list in gs_stratification.items():
        if sample_list:
            avg_samples = np.mean(np.array(sample_list), axis=0)
            gs_stratification[gs] = [avg_samples, len(sample_list)]
        else:
            gs_stratification[gs] = [np.zeros(len(features)), 0]

    # Create PDF with multiple plots
    with PdfPages(save_filename) as pdf:
        n_features = len(features)
        n_cols = min(2, n_features)
        n_rows = (n_features + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows))
        if n_features == 1:
            axes = np.array([axes])

        for idx, (feature_name, _) in enumerate(features):
            ax = axes.flat[idx] if n_features > 1 else axes

            gs_cats = [sample[0][idx] for sample in gs_stratification.values()]
            labels = [f"{a}, n={b[1]}" for a, b in gs_stratification.items()]

            ax.boxplot(gs_cats, labels=labels)
            ax.set_ylabel(f"Relative Fraction at {feature_name} Diffusivity")
            ax.set_title(
                f"Boxplot of Relative Fraction at {feature_name} Diffusivity by GS Stratification"
            )
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def ggg_stat_analysis(rois, features, ggg_data, comparison):
    """
    Calculate statistics for Gleason Grade Group comparisons, combining PZ and TZ.
    """
    results = []
    ggg_data_processed = {}

    for patient_key, patient_data in ggg_data.items():
        for roi_key, roi_data in patient_data.items():
            a_region = roi_data["anatomical_region"]
            if "tumor" not in a_region.lower():
                continue  # Skip non-tumor regions

            b_values = np.array(roi_data["b_values"])
            signal_values = np.array(roi_data["signal_values"])

            # Find corresponding ROI in rois
            roi = next(
                (r for r in rois[a_region] if r["patient_key"] == patient_key), None
            )
            if roi is None:
                continue

            samples = roi["sample"]

            # Calculate features for Gibbs Sampler
            feature_values = [
                np.mean([sample[feat] for sample in samples]) for _, feat in features
            ]

            # Calculate the std for each feature as a measure of uncertainty and importance
            feature_stds = [
                np.std([sample[feat] for sample in samples]) for _, feat in features
            ]
            feature_importances = [1 / std for std in feature_stds]

            # Create the new feature .25/2.0
            feature_combo = create_feature_combo(feature_values, feature_importances)
            feature_values.append(feature_combo)

            # Calculate ADC for all four b-value ranges
            adc_0_1000 = calculate_adc_multi(
                signal_values, b_values, a_region, b_range="0-1000"
            )
            adc_250_1000 = calculate_adc_multi(
                signal_values, b_values, a_region, b_range="250-1000"
            )
            adc_0_1250 = calculate_adc_multi(
                signal_values, b_values, a_region, b_range="0-1250"
            )
            adc_250_1250 = calculate_adc_multi(
                signal_values, b_values, a_region, b_range="250-1250"
            )
            feature_values.extend([adc_0_1000, adc_250_1000, adc_0_1250, adc_250_1250])

            ggg = int(float(roi["target"]))  # Convert '3.0' to 3
            if ggg not in ggg_data_processed:
                ggg_data_processed[ggg] = []
            ggg_data_processed[ggg].append(feature_values)

    # Define GGG groupings based on the comparison
    if comparison == "GGG1 vs GGG2":
        group1, group2 = [1], [2]
        group1_name, group2_name = "GS ≤ 6", "GS 3+4"
    elif comparison == "GGG1 vs GGG3":
        group1, group2 = [1], [3]
        group1_name, group2_name = "GS ≤ 6", "GS 4+3"
    elif comparison == "GGG1 vs GGG4-5":
        group1, group2 = [1], [4, 5]
        group1_name, group2_name = "GS ≤ 6", "GS ≥ 8"
    elif comparison == "GGG1-3 vs GGG4-5":
        group1, group2 = [1, 2, 3], [4, 5]
        group1_name, group2_name = "GS ≤ 7", "GS ≥ 8"
    else:
        raise ValueError("Invalid comparison")

    # Calculate statistics
    for i, (feature_name, _) in enumerate(
        features
        + [
            (".25/2.0", None),  # Add the new feature
            ("ADC (0-1000)", None),
            ("ADC (250-1000)", None),
            ("ADC (0-1250)", None),
            ("ADC (250-1250)", None),
        ]
    ):
        group1_data = np.concatenate([ggg_data_processed.get(g, []) for g in group1])
        group2_data = np.concatenate([ggg_data_processed.get(g, []) for g in group2])

        if len(group1_data) > 0 and len(group2_data) > 0:
            group1_feature = group1_data[:, i]
            group2_feature = group2_data[:, i]

            _, p_value = stats.ttest_ind(group1_feature, group2_feature)

            y_true = np.concatenate(
                [np.zeros(len(group1_feature)), np.ones(len(group2_feature))]
            )
            y_scores = np.concatenate([group1_feature, group2_feature])
            auc = roc_auc_score(y_true, y_scores)

            # Claude Sonnet 3.5 - rule of thumb:
            # If AUC < 0.5: Use 1 - AUC (indicates inverse correlation)
            # If AUC > 0.5: Use AUC directly (indicates positive correlation)
            # If AUC ≈ 0.5: The feature may not be predictive
            if (
                "ADC" in feature_name
                or feature_name == "2."
                or feature_name == "2.50"
                or feature_name == "3."
                or feature_name == "20."
            ):
                auc = 1 - auc

            if feature_name == ".25/2.0":
                model = "Feature combo"
                parameters = ".25/2.0"
            elif "ADC" not in feature_name:
                model = "Gibbs Sampler"
                parameters = feature_name
            else:
                model = "Monoexponential"
                parameters = feature_name.split()[-1][1:-1]

            b_values = (
                "0-3500"
                if "ADC" not in feature_name
                else feature_name.split()[-1][1:-1]
            )
            results.append(
                {
                    "Model": model,
                    "Parameters": parameters,
                    "B-Values": b_values,
                    "Comparison": f"{group1_name} vs {group2_name}",
                    "AUC": auc,
                    "P-Value": p_value,
                    f"{group1_name} Count": len(group1_feature),
                    f"{group2_name} Count": len(group2_feature),
                }
            )

    return pd.DataFrame(results)


def normal_v_tumor_stat_analysis(rois, features, normal_tumor_data):
    results = []
    zone_data = {"PZ": {"normal": [], "tumor": []}, "TZ": {"normal": [], "tumor": []}}

    for patient_key, patient_data in normal_tumor_data.items():
        for roi_key, roi_data in patient_data.items():
            a_region = roi_data["anatomical_region"]
            zone = "PZ" if "pz" in a_region.lower() else "TZ"
            tissue_type = "normal" if "normal" in a_region.lower() else "tumor"

            b_values = np.array(roi_data["b_values"])
            signal_values = np.array(roi_data["signal_values"])

            roi = next(
                (r for r in rois[a_region] if r["patient_key"] == patient_key), None
            )
            if roi is None:
                continue

            samples = roi["sample"]

            feature_values = [
                np.mean([sample[feat] for sample in samples]) for _, feat in features
            ]

            # Calculate the std for each feature as a measure of uncertainty and importance
            feature_stds = [
                np.std([sample[feat] for sample in samples]) for _, feat in features
            ]
            feature_importances = [1 / std for std in feature_stds]

            # Create combo feature .25/2.0
            feature_combo = create_feature_combo(feature_values, feature_importances)
            feature_values.append(feature_combo)

            # Calculate ADC for all three b-value ranges
            adc_0_1000 = calculate_adc_multi(
                signal_values, b_values, a_region, b_range="0-1000"
            )
            adc_250_1000 = calculate_adc_multi(
                signal_values, b_values, a_region, b_range="250-1000"
            )
            adc_0_1250 = calculate_adc_multi(
                signal_values, b_values, a_region, b_range="0-1250"
            )
            adc_250_1250 = calculate_adc_multi(
                signal_values, b_values, a_region, b_range="250-1250"
            )
            feature_values.extend([adc_0_1000, adc_250_1000, adc_0_1250, adc_250_1250])

            zone_data[zone][tissue_type].append(feature_values)

    for zone in zone_data:
        for tissue_type in zone_data[zone]:
            zone_data[zone][tissue_type] = np.array(zone_data[zone][tissue_type])

    for i, (feature_name, _) in enumerate(
        features
        + [
            (".25/2.0", None),  # Add the new feature
            ("ADC (0-1000)", None),
            ("ADC (250-1000)", None),
            ("ADC (0-1250)", None),
            ("ADC (250-1250)", None),
        ]
    ):
        for zone in ["PZ", "TZ"]:
            normal_data = zone_data[zone]["normal"][:, i]
            tumor_data = zone_data[zone]["tumor"][:, i]

            if len(normal_data) > 0 and len(tumor_data) > 0:
                _, p_value = stats.ttest_ind(normal_data, tumor_data)

                y_true = np.concatenate(
                    [np.zeros(len(normal_data)), np.ones(len(tumor_data))]
                )
                y_scores = np.concatenate([normal_data, tumor_data])
                auc = roc_auc_score(y_true, y_scores)

                # Claude Sonnet 3.5 - rule of thumb:
                # If AUC < 0.5: Use 1 - AUC (indicates inverse correlation)
                # If AUC > 0.5: Use AUC directly (indicates positive correlation)
                # If AUC ≈ 0.5: The feature may not be predictive
                if (
                    "ADC" in feature_name
                    or feature_name == "2."
                    or feature_name == "2.50"
                    or feature_name == "3."
                    or feature_name == "20."
                ):
                    auc = 1 - auc

                if feature_name == ".25/2.0":
                    model = "Feature combo"
                    parameters = ".25/2.0"
                elif "ADC" not in feature_name:
                    model = "Gibbs Sampler"
                    parameters = feature_name
                else:
                    model = "Monoexponential"
                    parameters = feature_name.split()[-1][1:-1]

                b_values = (
                    "0-3500"
                    if "ADC" not in feature_name
                    else feature_name.split()[-1][1:-1]
                )
                results.append(
                    {
                        "Model": model,
                        "Parameters": parameters,
                        "B-Values": b_values,
                        f"AUC for {zone}": auc,
                        f"P-Value for {zone}": p_value,
                        f"{zone} Normal Count": len(normal_data),
                        f"{zone} Tumor Count": len(tumor_data),
                    }
                )

    return pd.DataFrame(results)


def weighted_feature_combination(rois, features, ggg_data, comparison):
    print("Features:", features)  # Add this line
    results = []
    ggg_data_processed = {}

    # Define feature_names here
    feature_names = [name for name, _ in features] + [
        "ADC (0-1000)",
        "ADC (250-1000)",
        "ADC (0-1250)",
        "ADC (250-1250)",
    ]

    for patient_key, patient_data in ggg_data.items():
        for roi_key, roi_data in patient_data.items():
            a_region = roi_data["anatomical_region"]
            if "tumor" not in a_region.lower():
                continue  # Skip non-tumor regions

            b_values = np.array(roi_data["b_values"])
            signal_values = np.array(roi_data["signal_values"])

            roi = next(
                (r for r in rois[a_region] if r["patient_key"] == patient_key), None
            )
            if roi is None:
                continue

            samples = roi["sample"]

            # Calculate features and their uncertainties
            feature_values = []
            feature_uncertainties = []
            for i, feat in enumerate(features):
                if isinstance(feat, tuple):
                    _, idx = feat
                else:
                    idx = i
                feature_samples = [sample[idx] for sample in samples]
                feature_values.append(np.mean(feature_samples))
                feature_uncertainties.append(np.std(feature_samples))

            # Calculate ADC for all four b-value ranges
            adc_values = [
                calculate_adc_multi(signal_values, b_values, a_region, b_range)
                for b_range in ["0-1000", "250-1000", "0-1250", "250-1250"]
            ]
            feature_values.extend(adc_values)
            feature_uncertainties.extend(
                [0.1] * 4
            )  # Assuming fixed uncertainty for ADC

            ggg = int(float(roi["target"]))
            if ggg not in ggg_data_processed:
                ggg_data_processed[ggg] = []
            ggg_data_processed[ggg].append((feature_values, feature_uncertainties))

    # Define GGG groupings based on the comparison
    if comparison == "GGG1 vs GGG2":
        group1, group2 = [1], [2]
        group1_name, group2_name = "GS ≤ 6", "GS 3+4"
    elif comparison == "GGG1 vs GGG3":
        group1, group2 = [1], [3]
        group1_name, group2_name = "GS ≤ 6", "GS 4+3"
    elif comparison == "GGG1 vs GGG4-5":
        group1, group2 = [1], [4, 5]
        group1_name, group2_name = "GS ≤ 6", "GS ≥ 8"
    elif comparison == "GGG1-3 vs GGG4-5":
        group1, group2 = [1, 2, 3], [4, 5]
        group1_name, group2_name = "GS ≤ 7", "GS ≥ 8"
    else:
        raise ValueError("Invalid comparison")

    # Prepare data for logistic regression
    X = []
    y = []
    sample_weights = []

    for group, label in [(group1, 0), (group2, 1)]:
        for ggg in group:
            if ggg in ggg_data_processed:
                for features, uncertainties in ggg_data_processed[ggg]:
                    X.append(features)  # Use all features
                    y.append(label)
                    sample_weights.append(1 / np.mean(uncertainties))

    X = np.array(X)
    y = np.array(y)
    sample_weights = np.array(sample_weights)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("sample_weights shape:", sample_weights.shape)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform cross-validated logistic regression
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logistic = LogisticRegression(penalty="l2", solver="lbfgs", random_state=42)

    cv_scores = cross_val_score(
        logistic,
        X_scaled,
        y,
        cv=cv,
        scoring="roc_auc",
        fit_params={"sample_weight": sample_weights},
    )

    # Fit the model on the entire dataset to get feature importances
    logistic.fit(X_scaled, y, sample_weight=sample_weights)

    # Calculate feature importances
    feature_importances = np.abs(logistic.coef_[0])
    print("Feature importances shape:", feature_importances.shape)

    # Prepare results
    result = {
        "Model": "Weighted Linear Combination",
        "Parameters": "All Features",
        "B-Values": "0-3500",
        "Comparison": f"{group1_name} vs {group2_name}",
        "AUC": np.mean(cv_scores),
        "AUC Std": np.std(cv_scores),
        f"{group1_name} Count": sum(y == 0),
        f"{group2_name} Count": sum(y == 1),
    }

    # Add feature importances to the result
    print("Feature names:", feature_names)
    print("Number of feature names:", len(feature_names))

    for i, feature_name in enumerate(feature_names):
        if i < len(feature_importances):
            result[f"{feature_name} Importance"] = feature_importances[i]
        else:
            result[f"{feature_name} Importance"] = np.nan  # or some default value

    return pd.DataFrame([result])


def main(configs: dict) -> None:

    # Load ggg_d2.json
    with open(configs["INPUT_D2_DICT"], "r") as f:
        ggg_data = json.load(f)

    # Load tvn_d1.json
    with open(configs["INPUT_D1_DICT"], "r") as f:
        tumor_normal_data = json.load(f)

    # Define features for statistical analysis
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

    # Fetch pre-computed model outputs for statistical analysis
    gibbs_ggg = gibbs.fetch_gibbs_objects(
        gibbs_json_path=configs["GIBBS_D2_JSON"],
        gibbs_hdf5_path=configs["GIBBS_D2_HDF5"],
    )
    gibbs_tn = gibbs.fetch_gibbs_objects(
        gibbs_json_path=configs["GIBBS_D1_JSON"],
        gibbs_hdf5_path=configs["GIBBS_D1_HDF5"],
    )

    # Calculate MR-based tumor v normal statistics
    stat_analysis_table = normal_v_tumor_stat_analysis(
        gibbs_tn, features, tumor_normal_data
    )
    stat_analysis_table.to_csv(
        os.path.join(configs["EVAL_DIR_PATH"] + "/normal_v_tumor_stat_analysis.csv"),
        index=False,
    )

    # Calculate diffusivity feature gleason score correlation plot
    gs_stratified_boxplot(
        rois=gibbs_ggg,
        save_filename=os.path.join(
            configs["EVAL_DIR_PATH"] + "gs_stratified_boxplot_multi.pdf"
        ),
        features=features,
    )

    # Calculate Gleason Grade Group statistics
    all_results = []
    comparisons = ["GGG1 vs GGG2", "GGG1 vs GGG3", "GGG1 vs GGG4-5", "GGG1-3 vs GGG4-5"]
    for comparison in comparisons:
        ggg_analysis_table = ggg_stat_analysis(
            gibbs_ggg, features, ggg_data, comparison
        )
        all_results.append(ggg_analysis_table)
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(
        os.path.join(configs["EVAL_DIR_PATH"] + "ggg_stat_analysis_combined.csv"),
        index=False,
    )

    # Lin combo
    # Calculate Gleason Grade Group statistics with weighted linear combination
    # all_results = []
    # comparisons = ["GGG1 vs GGG2", "GGG1 vs GGG3", "GGG1 vs GGG4-5", "GGG1-3 vs GGG4-5"]
    # for comparison in comparisons:
    #     weighted_analysis_table = weighted_feature_combination(
    #         gibbs_ggg, features, ggg_data, comparison
    #     )
    #     all_results.append(weighted_analysis_table)

    # combined_results = pd.concat(all_results, ignore_index=True)
    # combined_results.to_csv(
    #     os.path.join(configs["EVAL_DIR_PATH"] + "weighted_ggg_analysis_combined.csv"),
    #     index=False,
    # )


if __name__ == "__main__":
    # load in YAML configuration
    configs = {}
    with importlib.resources.files("spectra_estimation_dmri").joinpath(
        "configs.yaml"
    ).open("r") as file:
        configs.update(yaml.safe_load(file))
    main(configs)
