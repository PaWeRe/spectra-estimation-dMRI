import json
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score
import yaml
import sys

sys.path.append(os.path.join(os.getcwd() + "/spectra-estimation-dMRI/src/models/"))
import gibbs


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

            # Calculate ADC
            adc = calculate_adc_multi(signal_values, b_values, a_region, plot=False)
            feature_values.append(adc)

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
    for i, (feature_name, _) in enumerate(features + [("ADC", None)]):
        group1_data = np.concatenate([ggg_data_processed.get(g, []) for g in group1])
        group2_data = np.concatenate([ggg_data_processed.get(g, []) for g in group2])

        if len(group1_data) > 0 and len(group2_data) > 0:
            group1_feature = group1_data[:, i]
            group2_feature = group2_data[:, i]

            # Calculate p-value
            _, p_value = stats.ttest_ind(group1_feature, group2_feature)

            # Calculate AUC
            y_true = np.concatenate(
                [np.zeros(len(group1_feature)), np.ones(len(group2_feature))]
            )
            y_scores = np.concatenate([group1_feature, group2_feature])

            # Inverse for ADC
            if feature_name == "ADC":
                y_scores = 1 / y_scores
            auc = roc_auc_score(y_true, y_scores)

            model = "Gibbs Sampler" if feature_name != "ADC" else "Monoexponential"
            results.append(
                {
                    "Model": model,
                    "Parameters": feature_name,
                    "B-Values": "0-3500",
                    "Comparison": f"{group1_name} vs {group2_name}",
                    "AUC": auc,
                    "P-Value": p_value,
                    f"{group1_name} Count": len(group1_feature),
                    f"{group2_name} Count": len(group2_feature),
                }
            )

    return pd.DataFrame(results)


def calculate_adc_multi(signal_values, b_values, a_region, plot=False):
    """
    Calculate ADC using a monoexponential model. Mask out b-values above 1000 s/mm².
    Higher b-values may introduce non-Gaussian diffusion effects that can affect the ADC calculation.
    This approach fits the monoexponential model S = S0 * exp(-b * ADC)
    to your data by transforming it to a linear problem: log(S) = log(S0) - b * ADC.

    Parameters:
    signal_values : array-like
        The measured signal intensities
    b_values : array-like
        The corresponding b-values
    plot : bool, optional
        If True, plot the signal decay and fitted line

    Returns:
    adc : float
        The calculated Apparent Diffusion Coefficient
    """
    mask = b_values <= 1000
    valid_mask = (signal_values > 0) & mask  # Exclude non-positive signal values

    if not np.any(valid_mask):
        raise ValueError("No valid signal values available for ADC calculation.")

    log_signal = np.log(signal_values[valid_mask])
    valid_b_values = b_values[valid_mask]

    slope, intercept = np.polyfit(valid_b_values, log_signal, 1)
    adc = -slope

    if plot:
        plt.figure(figsize=(10, 6))

        # Plot original data points
        plt.scatter(b_values, signal_values, label="Original data")
        plt.scatter(valid_b_values, np.exp(log_signal), label="Used for fitting")

        # Plot fitted line
        fit_b_values = np.linspace(0, max(valid_b_values), 100)
        fit_signal = np.exp(intercept - adc * fit_b_values)
        plt.plot(fit_b_values, fit_signal, "r-", label="Fitted line")

        plt.xlabel("b-value (s/mm²)")
        plt.ylabel("Signal intensity")
        plt.title(f"Signal Decay and ADC Fit (ADC = {adc:.4f} mm²/s)|{a_region}")
        plt.legend()
        plt.yscale(
            "log"
        )  # Use log scale for y-axis to better visualize exponential decay
        plt.grid(True)
        plt.show()

    return adc


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

            # Calculate ADC
            adc = calculate_adc_multi(signal_values, b_values, a_region, plot=False)
            feature_values.append(adc)

            zone_data[zone][tissue_type].append(feature_values)

    # Convert lists to numpy arrays
    for zone in zone_data:
        for tissue_type in zone_data[zone]:
            zone_data[zone][tissue_type] = np.array(zone_data[zone][tissue_type])

    # Calculate statistics
    for i, (feature_name, _) in enumerate(features + [("ADC", None)]):
        for zone in ["PZ", "TZ"]:
            normal_data = zone_data[zone]["normal"][:, i]
            tumor_data = zone_data[zone]["tumor"][:, i]

            if len(normal_data) > 0 and len(tumor_data) > 0:
                # Calculate p-value
                _, p_value = stats.ttest_ind(normal_data, tumor_data)

                # Calculate AUC
                y_true = np.concatenate(
                    [np.zeros(len(normal_data)), np.ones(len(tumor_data))]
                )
                y_scores = np.concatenate([normal_data, tumor_data])

                # Inverse for ADC
                if feature_name == "ADC":
                    y_scores = 1 / y_scores
                auc = roc_auc_score(y_true, y_scores)

                model = "Gibbs Sampler" if feature_name != "ADC" else "Monoexponential"
                results.append(
                    {
                        "Model": model,
                        "Parameters": feature_name,
                        "B-Values": "0-3500",
                        f"AUC for {zone}": auc,
                        f"P-Value for {zone}": p_value,
                        f"{zone} Normal Count": len(normal_data),
                        f"{zone} Tumor Count": len(tumor_data),
                    }
                )

    return pd.DataFrame(results)


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


if __name__ == "__main__":
    # load in YAML configuration
    configs = {}
    base_config_path = os.path.join(
        os.getcwd() + "/spectra-estimation-dMRI/configs.yaml"
    )
    with open(base_config_path, "r") as file:
        configs.update(yaml.safe_load(file))
    main(configs)
