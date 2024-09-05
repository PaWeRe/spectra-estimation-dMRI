import json
import os
import pandas as pd
from typing import Dict, List, Tuple

import yaml


def create_tumor_normal_pairs(
    json_file: str, csv_file: str
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, Dict[str, Dict]]]:
    with open(json_file, "r") as f:
        data = json.load(f)

    df = pd.read_csv(csv_file, sep=";")

    paired_data = {}
    discarded_data = {}

    for patient_id, patient_data in data.items():
        patient_row = df[df["patient_id"] == patient_id.lower()]

        if patient_row.empty:
            discarded_data[patient_id] = patient_data
            continue

        pz_pair = (
            patient_row["normal_pz_s2"].iloc[0] == 1
            and patient_row["tumor_pz_s1"].iloc[0] == 1
        )
        tz_pair = (
            patient_row["normal_tz_s3"].iloc[0] == 1
            and patient_row["tumor_tz_s1"].iloc[0] == 1
        )

        if pz_pair or tz_pair:
            paired_data[patient_id] = {}

            if pz_pair:
                normal_roi = next(
                    (
                        roi
                        for roi in patient_data.values()
                        if roi.get("anatomical_region") == "normal_pz_s2"
                    ),
                    None,
                )
                tumor_roi = next(
                    (
                        roi
                        for roi in patient_data.values()
                        if roi.get("anatomical_region") == "tumor_pz_s1"
                    ),
                    None,
                )

                if normal_roi and tumor_roi:
                    for roi_key, roi_data in patient_data.items():
                        if roi_data == normal_roi or roi_data == tumor_roi:
                            paired_data[patient_id][roi_key] = roi_data

            if tz_pair:
                normal_roi = next(
                    (
                        roi
                        for roi in patient_data.values()
                        if roi.get("anatomical_region") == "normal_tz_s3"
                    ),
                    None,
                )
                tumor_roi = next(
                    (
                        roi
                        for roi in patient_data.values()
                        if roi.get("anatomical_region") == "tumor_tz_s1"
                    ),
                    None,
                )

                if normal_roi and tumor_roi:
                    for roi_key, roi_data in patient_data.items():
                        if roi_data == normal_roi or roi_data == tumor_roi:
                            paired_data[patient_id][roi_key] = roi_data
        else:
            discarded_data[patient_id] = patient_data

    return paired_data, discarded_data


def create_tumor_dataset(
    json_file: str, gleason_csv: str
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, Dict[str, Dict]]]:
    with open(json_file, "r") as f:
        data = json.load(f)

    # Load Gleason Grade Group data
    df_gleason = pd.read_csv(gleason_csv, sep=";")

    # Convert targets to int and filter
    df_gleason["targets"] = (
        pd.to_numeric(df_gleason["targets"], errors="coerce").fillna(0).astype(int)
    )
    df_gleason = df_gleason[df_gleason["targets"].between(1, 5)]

    valid_patients = set(df_gleason["patient_key"].str.lower())

    tumor_data = {}
    discarded_data = {}

    for patient_id, patient_data in data.items():
        tumor_rois = {
            roi_key: roi_data
            for roi_key, roi_data in patient_data.items()
            if roi_data.get("anatomical_region") in ["tumor_pz_s1", "tumor_tz_s1"]
        }

        if tumor_rois and patient_id.lower() in valid_patients:
            tumor_data[patient_id] = tumor_rois
        else:
            discarded_data[patient_id] = patient_data

    return tumor_data, discarded_data


def save_dataset(data: Dict[str, Dict[str, Dict]], filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def main(configs: dict) -> None:
    json_file = os.path.join(
        configs["JSON_INPUT_DIR_PATH"] + "processed_patient_dict.json"
    )
    csv_file = os.path.join(configs["METADATA_DIR"] + "prostate_rois.csv")
    gleason_file = os.path.join(configs["METADATA_DIR"] + "patient_gleason_grading.csv")
    d1_file = os.path.join(configs["JSON_INPUT_DIR_PATH"] + "tumor_normal_d1.json")
    d1_file_discarded = os.path.join(
        configs["JSON_INPUT_DIR_PATH"] + "tumor_normal_d1_discarded.json"
    )
    d2_file = os.path.join(
        configs["JSON_INPUT_DIR_PATH"] + "ggg_aggressiveness_d2.json"
    )
    d2_file_discarded = os.path.join(
        configs["JSON_INPUT_DIR_PATH"] + "ggg_aggressiveness_d2_discarded.json"
    )

    # Create and save the tumor-normal pairs dataset
    paired_data, paired_discarded = create_tumor_normal_pairs(json_file, csv_file)
    save_dataset(paired_data, d1_file)
    save_dataset(paired_discarded, d1_file_discarded)

    # Create and save the tumor dataset
    tumor_data, tumor_discarded = create_tumor_dataset(json_file, gleason_file)
    save_dataset(tumor_data, d2_file)
    save_dataset(tumor_discarded, d2_file_discarded)

    print(f"Number of patients in tumor-normal pairs dataset: {len(paired_data)}")
    print(
        f"Number of patients discarded from tumor-normal pairs dataset: {len(paired_discarded)}"
    )
    print(f"Number of patients in tumor dataset: {len(tumor_data)}")
    print(f"Number of patients discarded from tumor dataset: {len(tumor_discarded)}")


if __name__ == "__main__":
    # load in YAML configuration
    configs = {}
    base_config_path = os.path.join(
        os.getcwd() + "/spectra-estimation-dMRI/configs.yaml"
    )
    with open(base_config_path, "r") as file:
        configs.update(yaml.safe_load(file))
    main(configs)
