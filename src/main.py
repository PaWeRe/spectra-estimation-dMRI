import datetime
import json
import os
import pickle
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import scipy.stats as stats
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import diffusivity_spectra as ds

CURRENT_DIR = os.path.dirname(__file__)
DATA_DIR_PATH = os.path.join(CURRENT_DIR, "..", "data")
OUTPUT_DIR_PATH = os.path.join(CURRENT_DIR, "..", "output")
FILL_STR = "NaN"
AVG_BOXPLOT_CSV = "/avgs/roi_avgs.csv"
AVG_BOXPLOT_PDF = "/avgs/roi_avgs.pdf"
KEY_TO_PDF_NAME = {
    "normal_pz_s2": "/rois/npz.pdf",
    "normal_tz_s3": "/rois/ntz.pdf",
    "tumor_pz_s1": "/rois/tpz.pdf",
    "tumor_tz_s1": "/rois/ttz.pdf",
    "Neglected!": "/rois/neglected.pdf",
}
KEY_TO_CSV_NAME = {
    "normal_pz_s2": "/rois/npz.csv",
    "normal_tz_s3": "/rois/ntz.csv",
    "tumor_pz_s1": "/rois/tpz.csv",
    "tumor_tz_s1": "/rois/ttz.csv",
    "Neglected!": "/rois/neglected.csv",
}


def filter_processed_data(rois: dict, analysis_dataset: dict) -> dict:
    """
    filter processed data to only include ROIs that are present in the analysis dataset
    """
    filtered_rois = {}
    object_list = []
    for roi_name, roi_data in rois.items():
        for object in roi_data:
            # check if patient is in analysis dataset and also if anatomical region the same
            # (in normal case multiple rois that are not used because no matching pair!)
            if object["patient_key"].lower() in analysis_dataset and any(
                roi["anatomical_region"] == object["a_region"].lower()
                for roi in analysis_dataset[object["patient_key"]].values()
            ):
                object_list.append(object)
        filtered_rois[roi_name] = object_list
        object_list = []
    return filtered_rois


def init_plot_matrix(m, n, diffusivities):
    """
    create graph layout on PDF
    """
    f, axarr = plt.subplots(m, n, sharex="col", sharey="row")
    arr_ij = list(np.ndindex(axarr.shape))
    subplots = [axarr[index] for index in arr_ij]
    for s, splot in enumerate(subplots):
        last_row = m * n - s < n + 1
        first_in_row = s % n == 0
        splot.grid(color="0.75")
        if last_row:
            splot.set_xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)")
            str_diffs = ["a"] + list(map(str, diffusivities)) + ["z"]
            splot.set_xticks(np.arange(len(str_diffs)), labels=str_diffs, rotation=90)
        if first_in_row:
            splot.set_ylabel("Relative Fraction")
    return f, axarr, subplots


def print_all(rois: dict, m: int, n: int) -> None:
    """
    Desc
    Args:
        - rois: dict of ROIs
        - m: number of rows in print
        - n: number of columns in print
    Returns:
        None
    """
    # TODO: remove later
    plt.rcParams.update({"font.size": 6})
    # for every roi generate a PDF with all box plots
    for zone_key, zone_list in tqdm(rois.items(), desc="ROIs Zones", position=0):
        if len(zone_list) == 0:
            continue
        with PdfPages(os.path.join(OUTPUT_DIR_PATH + KEY_TO_PDF_NAME[zone_key])) as pdf:
            # create new empty plot page with annotations
            f, axarr, subplots = init_plot_matrix(
                m, n, zone_list[0]["object"].diffusivities
            )
            # populate plots with data of every sample
            n_pages = 1
            for i, sample_dict in tqdm(
                enumerate(zone_list), desc="Samples in Zone", position=1
            ):
                # title=f'p: {sample_dict["patient_key"]}, gs: {sample_dict["gs"]}, snr:{int(sample_dict["snr"])}, dob:{sample_dict["patient_age"]}'
                title = f'{sample_dict["patient_key"]}|{sample_dict["gs"]}|{sample_dict["target"]}|{int(sample_dict["snr"])}|{sample_dict["patient_age"]}'
                sample_dict["object"].plot(
                    ax=subplots[i - (n_pages - 1) * m * n], title=title
                )
                # in case more samples than fit on one page, create another one
                if i == n_pages * m * n - 1:
                    # save and close pdf
                    pdf.savefig()
                    plt.close(f)
                    # create new empty plot page with annotations
                    n_pages += 1
                    f, axarr, subplots = init_plot_matrix(
                        m, n, zone_list[0]["object"].diffusivities
                    )
            # Done!
            # But don't forget to save to pdf after the last page
            pdf.savefig()
            plt.close(f)

        # Calculate basic stastitics to save in spreadsheet
        df = pd.DataFrame()
        for sample_dict in zone_list:
            diff_dict = sample_dict["object"]
            for diff, sample in dict(
                zip(diff_dict.diffusivities, np.transpose(diff_dict.sample))
            ).items():
                min_val = np.min(sample)
                q1 = np.percentile(sample, 25)
                median = np.median(sample)
                mean = np.mean(sample)
                q3 = np.percentile(sample, 75)
                max_val = np.max(sample)
                # Calculate outliers (if any)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
                boxplot_stats = {
                    "Patient": sample_dict["patient_key"],
                    "ROI": sample_dict["a_region"],
                    "SNR": sample_dict["snr"],
                    "Gleason Score": sample_dict["gs"],
                    "Target": sample_dict["target"],
                    "Patient Age": sample_dict["patient_age"],
                    "Diffusivity": diff,
                    "Min": min_val,
                    "Q1": q1,
                    "Median": median,
                    "Mean": mean,
                    "Q3": q3,
                    "Max": max_val,
                }
                df_temp = pd.DataFrame([boxplot_stats])
                df = pd.concat([df, df_temp], ignore_index=True)
        df.to_csv(
            os.path.join(OUTPUT_DIR_PATH + KEY_TO_CSV_NAME[zone_key]), index=False
        )


def print_avg(rois: dict, diffusivities: list, m: int, n: int) -> None:
    """
    Desc
        Average over all ROIs by concatenating per dimension and create pdf with four boxplots
        (one for every ROI)
    Args:
        - rois: dict of ROIS (anatomical regions, ntz, ttz, npz, tpz)
        - diffusivities: list of diffusivity spectra used during gibbs sampling
        - m: rows of figures on one pdf
        - n: columns of figures on one pdf
    Returns:
    None
    """
    avg_dict = {}
    # Create 4 average samples for every ROI (neglected one should automatically be 0.0)
    for zone_key, zone_list in rois.items():
        # initialize new object of d_spectra_sample
        avg_sample_obj = ds.d_spectra_sample(diffusivities)
        # store averaged samples in sample attributed of avg_sample_obj
        n_const = len(zone_list)  # to take average of normalized samples
        if n_const == 0:
            continue
        avg_sample_obj.sample = (
            1 / n_const * np.sum([d["object"].sample for d in zone_list], axis=0)
        )
        # store avg sample under roi key in dict
        avg_dict[zone_key] = avg_sample_obj

    # Create pdf with 4 boxplots
    with PdfPages(os.path.join(OUTPUT_DIR_PATH + AVG_BOXPLOT_PDF)) as pdf:
        f, axarr, subplots = init_plot_matrix(m, n, diffusivities)
        for i, avg_sample_dict in enumerate(avg_dict.items()):
            if avg_sample_dict[0] != "Neglected!":
                title = f"{avg_sample_dict[0]}"
                avg_sample_dict[1].plot(ax=subplots[i - m * n], title=title)
            else:
                print("Neglected sample!")
        pdf.savefig()
        plt.close(f)

    # Calculate basic stastitics to save in spreadsheet
    df = pd.DataFrame()
    for avg_sample_dict in avg_dict.items():
        title = avg_sample_dict[0]
        diff_dict = avg_sample_dict[1]
        for diff, sample in dict(
            zip(diff_dict.diffusivities, np.transpose(diff_dict.sample))
        ).items():
            min_val = np.min(sample)
            q1 = np.percentile(sample, 25)
            median = np.median(sample)
            mean = np.mean(sample)
            q3 = np.percentile(sample, 75)
            max_val = np.max(sample)
            # Calculate outliers (if any)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
            boxplot_stats = {
                "ROI": title,
                "Diffusivity": diff,
                "Min": min_val,
                "Q1": q1,
                "Median": median,
                "Mean": mean,
                "Q3": q3,
                "Max": max_val,
            }
            df_temp = pd.DataFrame([boxplot_stats])
            df = pd.concat([df, df_temp], ignore_index=True)
    df.to_csv(os.path.join(OUTPUT_DIR_PATH + AVG_BOXPLOT_CSV), index=False)


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
                        np.transpose(element["object"].sample)[idx]
                        for _, idx in features
                    ]

                    if target == "1.0":
                        gs_stratification["GS 6"].append(samples)
                    elif target == "2.0":
                        gs_stratification["GS 7 (3+4)"].append(samples)
                    elif target == "3.0":
                        gs_stratification["GS 7 (4+3)"].append(samples)
                    elif target == "4.0":
                        gs_stratification["GS 8-10"].append(samples)
                    elif target == "5.0":
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


def calculate_auc_table(
    rois: dict,
    features: List[Union[Tuple[str, int], Tuple[str, List[Tuple[int, float]]]]],
) -> pd.DataFrame:
    """
    Calculate AUC of ROC curves and p-values for discriminating normal and tumor tissue for each prostatic zone and feature.
    """
    results = []
    zone_data = {"PZ": {"normal": [], "tumor": []}, "TZ": {"normal": [], "tumor": []}}

    for roi_list in rois.values():
        for roi in roi_list:
            zone = "PZ" if "pz" in roi["a_region"].lower() else "TZ"
            tissue_type = "normal" if "normal" in roi["a_region"].lower() else "tumor"

            samples = roi["object"].sample
            feature_values = []
            for _, feat in features:
                if isinstance(feat, int):
                    feature_values.append(np.mean([sample[feat] for sample in samples]))
                else:
                    feature_values.append(
                        np.mean(
                            [
                                sum(coef * sample[idx] for idx, coef in feat)
                                for sample in samples
                            ]
                        )
                    )

            zone_data[zone][tissue_type].append(feature_values)

    # Convert lists to numpy arrays
    for zone in zone_data:
        for tissue_type in zone_data[zone]:
            zone_data[zone][tissue_type] = np.array(zone_data[zone][tissue_type])

    # Count the number of cases in each group
    case_counts = {
        f"{zone}_{tissue_type}_count": len(data)
        for zone in zone_data
        for tissue_type, data in zone_data[zone].items()
    }

    # Add case counts to results
    results.append({"Feature": "Case Counts", **case_counts})

    for i, (feature_name, _) in enumerate(features):
        feature_results = {"Feature": feature_name}

        for zone in ["PZ", "TZ"]:
            normal_data = zone_data[zone]["normal"]
            tumor_data = zone_data[zone]["tumor"]

            if len(normal_data) > 0 and len(tumor_data) > 0:
                # Calculate p-value
                t_stat, p_value = stats.ttest_ind(normal_data[:, i], tumor_data[:, i])

                # Calculate AUC
                y_true = np.concatenate(
                    [np.zeros(len(normal_data)), np.ones(len(tumor_data))]
                )
                y_scores = np.concatenate([normal_data[:, i], tumor_data[:, i]])
                auc = roc_auc_score(y_true, y_scores)

                # Calculate optimal cutoff using Youden's index
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_cutoff = thresholds[optimal_idx]

                feature_results[f"{zone}_AUC"] = auc
                feature_results[f"{zone}_p_value"] = p_value
                feature_results[f"{zone}_cutoff"] = optimal_cutoff
            else:
                feature_results[f"{zone}_AUC"] = np.nan
                feature_results[f"{zone}_p_value"] = np.nan
                feature_results[f"{zone}_cutoff"] = np.nan

        results.append(feature_results)

    return pd.DataFrame(results)


def calculate_gleason_prediction(
    rois: dict,
    features: List[Union[Tuple[str, int], Tuple[str, List[Tuple[int, float]]]]],
) -> pd.DataFrame:
    """
    Calculate probability of parameters to predict Gleason Grade Groups for (PZ) tumors.
    Calculates AUC and p-value for pairwise comparisons between groups and an additional
    comparison of (GS 6 and GS 3+4) vs (GS 4+3 and GS 8-10).
    """
    results = []
    gs_data = {"GS 6": [], "GS 7 (3+4)": [], "GS 7 (4+3)": [], "GS 8-10": []}

    for roi_list in rois.values():
        for roi in roi_list:
            if "tumor" in roi["a_region"]:
                samples = roi["object"].sample
                feature_values = []
                for _, feat in features:
                    if isinstance(feat, int):
                        feature_values.append(
                            np.mean([sample[feat] for sample in samples])
                        )
                    else:
                        feature_values.append(
                            np.mean(
                                [
                                    sum(coef * sample[idx] for idx, coef in feat)
                                    for sample in samples
                                ]
                            )
                        )

                try:
                    if roi["target"] == "1.0":  # GS 6
                        gs_data["GS 6"].append(feature_values)
                    elif roi["target"] == "2.0":  # GS 7 (3+4)
                        gs_data["GS 7 (3+4)"].append(feature_values)
                    elif roi["target"] == "3.0":  # GS 7 (4+3)
                        gs_data["GS 7 (4+3)"].append(feature_values)
                    elif roi["target"] == "4.0":  # GS 8
                        gs_data["GS 8-10"].append(feature_values)
                    elif roi["target"] == "5.0":  # GS 9-10
                        gs_data["GS 8-10"].append(feature_values)
                except ValueError:
                    p_key = roi["patient_key"]
                    print(f"Smt wrong with print_roi, patient: {p_key}")

    for group in gs_data:
        gs_data[group] = np.array(gs_data[group])

    # Count the number of cases in each group
    case_counts = {group: len(data) for group, data in gs_data.items()}

    # Add case counts to results
    results.append(
        {
            "Feature": "Case Counts",
            **{f"{group}_count": count for group, count in case_counts.items()},
        }
    )

    comparisons = [
        ("GS 6_vs_GS 7 (3+4)", "GS 6", "GS 7 (3+4)"),
        ("GS 6_vs_GS 7 (4+3)", "GS 6", "GS 7 (4+3)"),
        ("GS 6_vs_GS 8-10", "GS 6", "GS 8-10"),
        ("GS 7 (3+4)_vs_GS 7 (4+3)", "GS 7 (3+4)", "GS 7 (4+3)"),
        ("GS 7 (3+4)_vs_GS 8-10", "GS 7 (3+4)", "GS 8-10"),
        ("GS 7 (4+3)_vs_GS 8-10", "GS 7 (4+3)", "GS 8-10"),
        (
            "(GS 6, GS 3+4)_vs_(GS 4+3, GS 8-10)",
            ["GS 6", "GS 7 (3+4)"],
            ["GS 7 (4+3)", "GS 8-10"],
        ),
    ]

    for i, (feature_name, _) in enumerate(features):
        feature_results = {"Feature": feature_name}

        for comp_name, group1, group2 in comparisons:
            if isinstance(group1, list):
                group1_data = np.concatenate([gs_data[g] for g in group1])
                group2_data = np.concatenate([gs_data[g] for g in group2])
            else:
                group1_data = gs_data[group1]
                group2_data = gs_data[group2]

            if len(group1_data) > 0 and len(group2_data) > 0:
                # Calculate p-value
                t_stat, p_value = stats.ttest_ind(group1_data[:, i], group2_data[:, i])

                # Calculate AUC
                y_true = np.concatenate(
                    [np.zeros(len(group1_data)), np.ones(len(group2_data))]
                )
                y_scores = np.concatenate([group1_data[:, i], group2_data[:, i]])
                auc = roc_auc_score(y_true, y_scores)

                feature_results[f"{comp_name}_p_value"] = p_value
                feature_results[f"{comp_name}_auc"] = auc
            else:
                feature_results[f"{comp_name}_p_value"] = np.nan
                feature_results[f"{comp_name}_auc"] = np.nan

        results.append(feature_results)

    return pd.DataFrame(results)


def main():

    # Load the original processed_patient_dict.json
    with open(os.path.join(DATA_DIR_PATH, "processed_patient_dict.json"), "r") as f:
        data = json.load(f)

    # Load ggg_aggressiveness_d2.json
    with open(os.path.join(DATA_DIR_PATH, "ggg_aggressiveness_d2.json"), "r") as f:
        ggg_data = json.load(f)

    # Load tumor_normal_d1.json
    with open(os.path.join(DATA_DIR_PATH, "tumor_normal_d1.json"), "r") as f:
        tumor_normal_data = json.load(f)

    recon_diffusivities = np.array(
        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
    )

    c = 150
    l1_lambda = 0.0
    l2_lambda = 0.00001
    iters = 100000

    # Define dict for printing
    roi_print = {
        "normal_pz_s2": [],
        "normal_tz_s3": [],
        "tumor_pz_s1": [],
        "tumor_tz_s1": [],
        "Neglected!": [],
    }

    # Get GS and DOB for all patients (if available)
    df_gsdob = (
        pd.read_csv(
            # DOB were removed for data privacy reasons
            os.path.join(DATA_DIR_PATH + "/patient_gleason_grading.csv"),
            sep=";",
            header=0,
        )
        .astype(str)
        .fillna(FILL_STR)
        .set_index("patient_key")
    )

    # check if already cached
    if not os.path.isfile(os.path.join(DATA_DIR_PATH + "/processed_data.pkl")):
        # Loop through all patients
        for patient_key in tqdm(data, desc="Patients", position=0):
            # Loop through all ROIs for the current patient
            for roi_key in tqdm(data[patient_key], desc="ROIs", position=1):
                try:
                    if roi_key.startswith("roi"):
                        # Extract the signal data and b values for the current ROI
                        # b_values_roi = np.array(data[patient_key][roi_key]['b_values'][1:])/1000
                        # max_sig = np.array(data[patient_key][roi_key]['signal_values'][1])
                        b_values_roi = (
                            np.array(data[patient_key][roi_key]["b_values"]) / 1000
                        )
                        max_sig = np.array(
                            data[patient_key][roi_key]["signal_values"][0]
                        )
                        signal_values_roi = (
                            np.array(data[patient_key][roi_key]["signal_values"])
                            / max_sig
                        )
                        sig_obj = ds.signal_data(signal_values_roi, b_values_roi)
                        v_count = data[patient_key][roi_key]["v_count"]
                        snr = np.sqrt(v_count / 16) * c
                        sigma = 1.0 / snr
                        print(data[patient_key][roi_key])
                        a_region = data[patient_key][roi_key]["anatomical_region"]

                        # Run Gibbs Sampler for the current ROI
                        my_sampler = ds.make_Gibbs_sampler(
                            signal_data=sig_obj,
                            diffusivities=recon_diffusivities,
                            sigma=sigma,
                            inverse_prior_covariance=None,
                            L1_lambda=l1_lambda,
                            L2_lambda=l2_lambda,
                        )
                        my_sample = my_sampler(iters)

                        # Normalize and discard the first 10000
                        my_sample.normalize()
                        my_sample.sample = my_sample.sample[10000:]
                        my_sample.normalize()

                        # calculate age based on date of birth and current date (e.g. 13.01.1949)
                        try:
                            gs = df_gsdob.loc[patient_key, "gs"]
                        except KeyError:
                            print("gs not found for patient_key:", patient_key)
                            gs = FILL_STR

                        try:
                            dob = df_gsdob.loc[patient_key, "dob"]
                            age = (
                                datetime.datetime.now()
                                - datetime.datetime.strptime(dob, "%Y-%m-%d")
                            ).days / 365.25
                        except KeyError:
                            print("dob not found for patient_key:", patient_key)
                            age = FILL_STR

                        try:
                            target = df_gsdob.loc[patient_key, "targets"]
                        except KeyError:
                            print("targets not found for patient_key:", patient_key)
                            target = FILL_STR

                        # Add meta data
                        sample_dict = {
                            "object": my_sample,
                            "a_region": a_region,
                            "snr": snr,
                            "patient_key": patient_key,
                            "gs": gs,
                            "target": target,
                            "patient_age": age,
                        }

                        # Aggregate samples per roi for later plotting
                        roi_print[a_region].append(sample_dict)
                except:
                    print(
                        "Error for patient_key: {}, roi_key: {}".format(
                            patient_key, roi_key
                        )
                    )

            #     count += 1
            #     if count == 3:
            #         break

            # count1 += 1
            # if count1 == 3:
            #     break

        # save ROI for efficiency
        with open(os.path.join(DATA_DIR_PATH + "/processed_data.pkl"), "wb") as f:
            pickle.dump(roi_print, f)
    else:
        with open(os.path.join(DATA_DIR_PATH + "/processed_data.pkl"), "rb") as f:
            roi_print = pickle.load(f)

    # generate donwstream analysis datasets (only temporary tachtic as very inefficient)
    roi_tn = filter_processed_data(roi_print, tumor_normal_data)
    roi_ggg = filter_processed_data(roi_print, ggg_data)

    # # Plot PDFs with all boxplots per roi
    # print_all(
    #     rois=roi_print,
    #     m=3,
    #     n=2,
    # )

    # # Plot extra PDF with avg boxplots per roi
    # print_avg(
    #     rois=roi_print,
    #     diffusivities=recon_diffusivities,
    #     m=2,
    #     n=2,
    # )

    # Calculate diffusivity feature gleason score correlation plot
    gs_stratified_boxplot(
        rois=roi_ggg,
        save_filename=os.path.join(
            OUTPUT_DIR_PATH + "/other/gs_stratified_boxplot_multi.pdf"
        ),
        features=[
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
        ],
    )

    # Calculate AUC table
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
    auc_table = calculate_auc_table(roi_tn, features)
    auc_table.to_csv(
        os.path.join(OUTPUT_DIR_PATH + "/other/auc_table_all.csv"), index=False
    )

    # Calculate Gleason prediction table
    gleason_table = calculate_gleason_prediction(roi_ggg, features)
    gleason_table.to_csv(
        os.path.join(OUTPUT_DIR_PATH + "/other/gleason_prediction_table_ggg.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
