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

            samples = roi["object"].sample

            # Calculate features for Gibbs Sampler
            feature_values = [
                np.mean([sample[feat] for sample in samples]) for _, feat in features
            ]

            # Calculate ADC
            adc = calculate_adc_multi(signal_values, b_values)
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


def calculate_adc_multi(signal_values, b_values):
    """
    Calculate ADC using monoexponential model. Mask out b-values above 1000s/mm2.
    Higher b-values may introduce non-Gaussian diffusion effects that can affect the ADC calculation.
    TODO: very low AUC for unknown reason?
    """
    mask = b_values <= 1000
    log_signal = np.log(signal_values[mask])
    slope, intercept = np.polyfit(b_values[mask], log_signal, 1)
    return -slope


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

            samples = roi["object"].sample

            # Calculate features for Gibbs Sampler
            feature_values = [
                np.mean([sample[feat] for sample in samples]) for _, feat in features
            ]

            # Calculate ADC
            adc = calculate_adc_multi(signal_values, b_values)
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

    # GGG comparisons
    comparisons = ["GGG1 vs GGG2", "GGG1 vs GGG3", "GGG1 vs GGG4-5", "GGG1-3 vs GGG4-5"]
    all_results = []  # to show all comparisons in one csv

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

    # Check if already cached
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

    # Generate donwstream analysis datasets (only temporary tachtic as very inefficient)
    roi_tn = filter_processed_data(roi_print, tumor_normal_data)
    roi_ggg = filter_processed_data(roi_print, ggg_data)

    # Plot PDFs with all boxplots per roi
    # print_all(
    #     rois=roi_print,
    #     m=3,
    #     n=2,
    # )

    # Plot extra PDF with avg boxplots per roi
    # print_avg(
    #     rois=roi_print,
    #     diffusivities=recon_diffusivities,
    #     m=2,
    #     n=2,
    # )

    # Calculate diffusivity feature gleason score correlation plot
    # gs_stratified_boxplot(
    #     rois=roi_ggg,
    #     save_filename=os.path.join(
    #         OUTPUT_DIR_PATH + "/other/gs_stratified_boxplot_multi.pdf"
    #     ),
    #     features=features
    # )

    # Calculate MR-based tumor v normal statistics
    stat_analysis_table = normal_v_tumor_stat_analysis(
        roi_tn, features, tumor_normal_data
    )
    stat_analysis_table.to_csv(
        os.path.join(OUTPUT_DIR_PATH + "/other/normal_v_tumor_stat_analysis.csv"),
        index=False,
    )

    # Calculate Gleason Grade Group statistics
    for comparison in comparisons:
        ggg_analysis_table = ggg_stat_analysis(roi_ggg, features, ggg_data, comparison)
        all_results.append(ggg_analysis_table)

    # Combine all results into a single DataFrame
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save the combined results to a single CSV file
    combined_results.to_csv(
        os.path.join(OUTPUT_DIR_PATH, "other/ggg_stat_analysis_combined.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
