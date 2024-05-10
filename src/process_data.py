import os
import nrrd
import glob
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
ROIS_PATH = os.getenv("ROIS_PATH")
IMG_DATA_PATH = os.getenv("IMG_DATA_PATH")
LABEL_DESCR = os.getenv("LABEL_DESCR")
ALT_LABEL_DESCR = os.getenv("ALT_LABEL_DESCR")
DATA_HEADER_DESCR = os.getenv("DATA_HEADER_DESCR")
ALT_DATA_HEADER_DESCR = os.getenv("ALT_DATA_HEADER_DESCR")


def sort_by_number(name):
    """
    extract the numeric part of the directory name and convert to integer
    """
    return int(name[3:])


data_dict = {}
df_a_regions = pd.read_csv(ROIS_PATH, delimiter=";")
dir_names = [
    name
    for name in os.listdir(IMG_DATA_PATH)
    if os.path.isdir(os.path.join(IMG_DATA_PATH, name))
]
dir_names = sorted(dir_names, key=sort_by_number)

for patient_folder in dir_names:
    patient_path = os.path.join(IMG_DATA_PATH, patient_folder)
    if not os.path.isdir(patient_path):
        continue
    label_path = os.path.join(patient_path, LABEL_DESCR)
    data_path = os.path.join(patient_path, DATA_HEADER_DESCR)
    if os.path.exists(label_path):
        label_data, label_header = nrrd.read(label_path)
    else:
        label_path = glob.glob(os.path.join(patient_path, ALT_LABEL_DESCR))[0]
        label_data, label_header = nrrd.read(label_path)
    if os.path.exists(data_path):
        data, data_header = nrrd.read(data_path)
    else:
        data_path = glob.glob(os.path.join(patient_path, ALT_DATA_HEADER_DESCR))[0]
        data, data_header = nrrd.read(data_path)

    b_values = np.array(data_header["MultiVolume.FrameLabels"].split(","), dtype=float)

    for roi_num in range(1, 5):
        roi_name = f"roi{roi_num}"
        roi_data = {}
        # Get index for label map
        if roi_num in label_data:
            roi_idx = np.where(label_data == roi_num)
            # Loop over each b-value volume in the data array
            signal_values = []
            v_count = len(roi_idx[0])
            for b in range(data.shape[0]):
                # Select the matching pixels from the data array
                if pd.notna(roi_idx):
                    seg_pixels = data[b][roi_idx]
                    avg_signal = np.mean(seg_pixels)
                    signal_values.append(avg_signal)
            # Store signal values, b-values, and pixel count in ROI data
            roi_data["signal_values"] = signal_values
            roi_data["b_values"] = b_values.tolist()
            roi_data["v_count"] = v_count
            # Anatomical region assignment to ROI
            patient_folder = patient_folder.lower()
            if roi_num == 1:
                if (
                    df_a_regions[df_a_regions["patient_id"] == patient_folder][
                        "tumor_pz_s1"
                    ].any()
                    == 1
                ):
                    roi_data["anatomical_region"] = "tumor_pz_s1"
                elif (
                    df_a_regions[df_a_regions["patient_id"] == patient_folder][
                        "tumor_tz_s1"
                    ].any()
                    == 1
                ):
                    roi_data["anatomical_region"] = "tumor_tz_s1"
                else:
                    roi_data["anatomical_region"] = "Neglected!"
            elif roi_num == 2:
                if (
                    df_a_regions[df_a_regions["patient_id"] == patient_folder][
                        "normal_pz_s2"
                    ].any()
                    == 1
                ):
                    roi_data["anatomical_region"] = "normal_pz_s2"
                else:
                    roi_data["anatomical_region"] = "Neglected!"
            elif roi_num == 3:
                if (
                    df_a_regions[df_a_regions["patient_id"] == patient_folder][
                        "normal_tz_s3"
                    ].any()
                    == 1
                ):
                    roi_data["anatomical_region"] = "normal_tz_s3"
                else:
                    roi_data["anatomical_region"] = "Neglected!"
            # Store ROI data in patient data
            if patient_folder.lower() not in data_dict:
                # .lower() to harmonize NewXY and newXY
                data_dict[patient_folder.lower()] = {}
            data_dict[patient_folder.lower()][roi_name] = roi_data
            print(data_dict)

with open("processed-data.json", "w") as outfile:
    json.dump(data_dict, outfile)
