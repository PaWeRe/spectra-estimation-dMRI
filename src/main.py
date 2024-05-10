import json
import os
import pickle
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
KEY_TO_PDF_NAME = {
    "normal_pz_s2": "/npz.pdf",
    "normal_tz_s3": "/ntz.pdf",
    "tumor_pz_s1": "/tpz.pdf",
    "tumor_tz_s1": "/ttz.pdf",
    "Neglected!": "/neglected.pdf",
}


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
    with PdfPages(os.path.join(OUTPUT_DIR_PATH + "/roi_avg.pdf")) as pdf:
        f, axarr, subplots = init_plot_matrix(m, n, diffusivities)
        for i, avg_sample_dict in enumerate(avg_dict.items()):
            if avg_sample_dict[0] != "Neglected!":
                title = f"{avg_sample_dict[0]}"
                avg_sample_dict[1].plot(ax=subplots[i - m * n], title=title)
            else:
                print("Neglected sample!")
        pdf.savefig()
        plt.close(f)


def gs_log_classifier(rois: dict, save_filename: str, save_filename_db: str) -> None:
    """
    Desc
        3) How well am I able to distinguish between (0,1) with x1 (nROI), and x2 (tROI), using a logisticregression classifier?

    Args:
        - TBD
        -
    Return:
        None
    """
    ggg = []
    # preprocessing
    for zone_key, zone_list in rois.items():
        for element in zone_list:
            try:
                if (
                    element["target"] == "0.0"
                    or element["target"] == "1.0"
                    or element["target"] == "2.0"
                    or element["target"] == "3.0"
                    or element["target"] == "4.0"
                ):
                    if "tumor" in element["a_region"]:
                        ggg.append(element)
                    else:
                        a_r = element["a_region"]
                        print(f"Not tumor ROI! {a_r}")
                else:
                    print("No ggg!")
            except ValueError:
                p_key = element["patient_key"]
                print(f"Smt wrong with print_roi, patient: {p_key}")

    # list comprehensions for getting features for correlation matrix and classifier
    # .25
    xm_025 = np.array(
        [np.mean([d025["object"].sample[i][0] for i in range(90000)]) for d025 in ggg]
    )
    # .5
    xm_05 = np.array(
        [np.mean([d025["object"].sample[i][1] for i in range(90000)]) for d025 in ggg]
    )
    # .75
    xm_075 = np.array(
        [np.mean([d025["object"].sample[i][2] for i in range(90000)]) for d025 in ggg]
    )
    # 3.0
    xm_3 = np.array(
        [np.mean([d025["object"].sample[i][8] for i in range(90000)]) for d025 in ggg]
    )
    # 2.5
    xm_25 = np.array(
        [np.mean([d025["object"].sample[i][7] for i in range(90000)]) for d025 in ggg]
    )
    # 0.25/3.0
    xm_0253 = xm_025 / xm_3
    # 0.25/(3.0+2.5)
    xm_025325 = xm_025 / (xm_3 + xm_25)
    # (0.25+0.5+0.75)/(2.5+3.0)
    xm_025575325 = (xm_025 + xm_05 + xm_075) / (xm_25 + xm_3)
    # y
    y = np.array([ycs for ycs in [tcs_element["target"] for tcs_element in ggg]])
    y = y.astype(float).astype(int)
    y_binary = np.where(y <= 1, 0, 1)

    # 3)
    # # use .25 diff fraction as feature to predict
    # #X = xm_025.reshape(-1,1)
    # X = xm_025575325.reshape(-1,1)

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

    # # fit a logistic regression model
    # model = LogisticRegression()
    # model.fit(X_train, y_train)

    # # Predict probabilities of the positive class for the test set
    # y_pred_proba = model.predict_proba(X_test)[:, 1]

    # # Calculate the AUC-ROC score
    # auc = roc_auc_score(y_test, y_pred_proba)
    # print("AUC-ROC Score:", auc)

    # # Compute the false positive rate, true positive rate, and thresholds for the ROC curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # # Plot the ROC curve
    # plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.3f)' % auc)
    # plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'ROC Curve | xm_025575325 | test size 0.3 (11)')
    # plt.legend()
    # if save_filename is not None:
    #     plt.savefig(save_filename)
    # plt.show()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Define your features and target variable
    X = xm_025575325.reshape(-1, 1)
    y = y_binary

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Fit a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict probabilities of the positive class for the test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate the AUC-ROC score
    auc = roc_auc_score(y_test, y_pred_proba)
    print("AUC-ROC Score:", auc)

    # Compute the false positive rate, true positive rate, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.3f)" % auc)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve | xm_025575325 | test size 0.4 (14)")
    plt.legend()
    if save_filename is not None:
        plt.savefig(save_filename)
    plt.show()

    # Plot the decision boundary
    plt.scatter(X_test, y_test, color="black", label="Actual")
    plt.scatter(
        X_test, model.predict(X_test), color="red", marker="x", label="Predicted"
    )
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Logistic Regression Decision Boundary")
    if save_filename_db is not None:
        plt.savefig(save_filename_db)
    plt.legend()
    plt.show()


def pr_visualization(rois: dict, save_filename: str) -> None:
    """
    Desc
        2) Visualize fitted line for x1 (nROI) and all 0 cases {0,1}, and fitted line for x2 (tROI) and all 1 cases {2,3,4} (visualize anatomical regions!)

    Args:
        - TBD
        -
    Return:
        None
    """
    ggg = []
    # preprocessing
    for zone_key, zone_list in rois.items():
        for element in zone_list:
            try:
                if (
                    element["target"] == "0.0"
                    or element["target"] == "1.0"
                    or element["target"] == "2.0"
                    or element["target"] == "3.0"
                    or element["target"] == "4.0"
                ):
                    if "tumor" in element["a_region"]:
                        ggg.append(element)
                    else:
                        a_r = element["a_region"]
                        print(f"Not tumor ROI! {a_r}")
                else:
                    print("No ggg!")
            except ValueError:
                p_key = element["patient_key"]
                print(f"Smt wrong with print_roi, patient: {p_key}")

    # list comprehensions for getting features for correlation matrix and classifier
    # .25
    xm_025 = np.array(
        [np.mean([d025["object"].sample[i][0] for i in range(90000)]) for d025 in ggg]
    )
    # .5
    xm_05 = np.array(
        [np.mean([d025["object"].sample[i][1] for i in range(90000)]) for d025 in ggg]
    )
    # .75
    xm_075 = np.array(
        [np.mean([d025["object"].sample[i][2] for i in range(90000)]) for d025 in ggg]
    )
    # 3.0
    xm_3 = np.array(
        [np.mean([d025["object"].sample[i][8] for i in range(90000)]) for d025 in ggg]
    )
    # 2.5
    xm_25 = np.array(
        [np.mean([d025["object"].sample[i][7] for i in range(90000)]) for d025 in ggg]
    )
    # 0.25/3.0
    xm_0253 = xm_025 / xm_3
    # 0.25/(3.0+2.5)
    xm_025325 = xm_025 / (xm_3 + xm_25)
    # (0.25+0.5+0.75)/(2.5+3.0)
    xm_025575325 = (xm_025 + xm_05 + xm_075) / (xm_25 + xm_3)
    # y
    y = np.array([ycs for ycs in [tcs_element["target"] for tcs_element in ggg]])
    y = y.astype(float).astype(int)
    y_binary = np.where(y <= 1, 0, 1)

    # 2)
    slope, intercept, r, p, stderr = stats.linregress(y, xm_025575325)
    line = f"Regression line: xm_025575325={intercept:.2f}+{slope:.2f}x, r={r:.2f}"
    fig, ax = plt.subplots()
    ax.plot(
        y,
        xm_025575325,
        linewidth=0,
        marker="s",
        label="Mean diffusivity rel fraction ratio",
    )
    ax.plot(y, intercept + slope * y, label=line)
    ax.set_ylabel("(0.25+0.5+0.75)/(2.5+3.0)")
    ax.set_xlabel("Gleason Grade Groups")
    ax.set_title("Pearsons r for GGGs and mean diffusivity relative fraction ratio")
    ax.legend(facecolor="white")
    plt.savefig(save_filename)


def gs_corr_matrix(rois: dict, save_filename: str) -> None:
    """
    Desc
        1) How do x1 (nROI), x2 (tROI) correlate with (0,1) based on correlation matrix and pearson's r?

    Args:
        - rois: dict of ROIS (anatomical regions, ntz, ttz, npz, tpz)
        - diff_compartment: Specifying which diffusivity fraction we want to use for calculating the correlation
    Return:
        None
    """
    # tcs = []
    # tncs = []
    # ncs = []
    # nncs = []
    # # preprocessing
    # # getting all tROI for the ncs and cs cases (for retrieving correct ncs case afterwards)
    # for zone_key,zone_list in rois.items():
    #     for element in zone_list:
    #         try:
    #             if (element['target'] == "2.0" or element['target'] == "3.0" or element['target'] == "4.0"):
    #                     if 'tumor' in element['a_region']:
    #                         tcs.append(element)
    #                     else:
    #                         print('Tumor case!')
    #             elif (element['target'] == "0.0" or element['target'] == "1.0"):
    #                     if 'tumor' in element['a_region']:
    #                         tncs.append(element)
    #                     else:
    #                         print('Normal case!')
    #         except(ValueError):
    #             p_key = element['patient_key']
    #             print(f'Smt wrong with print_roi, patient: {p_key}')
    # # getting the matching nROIs for ncs and cs (there can be up to two nROIs!)
    # for zone_key,zone_list in rois.items():
    #     for element in zone_list:
    #         try:
    #             if (element['target'] == "0.0" or element['target'] == "1.0"):
    #                 for ncs_element in tncs:
    #                     if ("normal" in element["a_region"] and "pz" in ncs_element["a_region"] and "pz" in element["a_region"]) and (ncs_element["patient_key"] == element["patient_key"]):
    #                         nncs.append(element)
    #                     elif ("normal" in element["a_region"] and "tz" in ncs_element["a_region"] and "tz" in element["a_region"]) and (ncs_element["patient_key"] == element["patient_key"]):
    #                         nncs.append(element)
    #                     else:
    #                         print("Wrong normal case to tumor case!")
    #             elif (element['target'] == "2.0" or element['target'] == "3.0" or element['target'] == "4.0"):
    #                 for cs_element in tcs:
    #                     if ("normal" in element["a_region"] and "pz" in cs_element["a_region"] and "pz" in element["a_region"]) and (cs_element["patient_key"] == element["patient_key"]):
    #                         ncs.append(element)
    #                     elif ("normal" in element["a_region"] and "tz" in cs_element["a_region"] and "tz" in element["a_region"]) and (cs_element["patient_key"] == element["patient_key"]):
    #                         ncs.append(element)
    #                     else:
    #                         print("Wrong normal case to tumor case!")
    #         except(ValueError):
    #                     p_key = element['patient_key']
    #                     print(f'Smt wrong with print_roi, patient: {p_key}')
    # # make sure that tncs&nncs and tcs&ncs are of equal length, if not delete all items that occur only in one list
    # if len(ncs) != len(tcs):
    #     ncs = [ncs_element for ncs_element in ncs if ncs_element['patient_key'] in [tcs_element['patient_key'] for tcs_element in tcs]]
    #     tcs = [tcs_element for tcs_element in tcs if tcs_element['patient_key'] in [ncs_element['patient_key'] for ncs_element in ncs]]
    # if len(nncs) != len(tncs):
    #     nncs = [nncs_element for nncs_element in nncs if nncs_element['patient_key'] in [tncs_element['patient_key'] for tncs_element in tncs]]
    #     tcs = [tncs_element for tncs_element in tncs if tncs_element['patient_key'] in [nncs_element['patient_key'] for nncs_element in nncs]]

    # # list comprehensions for getting features for correlation matrix and classifier
    # # .25
    # xm_025tcs = np.array([np.mean([d025['object'].sample[i][0] for i in range(90000)]) for d025 in tcs])
    # xm_025ncs = np.array([np.mean([d025['object'].sample[i][0] for i in range(90000)]) for d025 in ncs])
    # xm_025tncs = np.array([np.mean([d025['object'].sample[i][0] for i in range(90000)]) for d025 in tncs])
    # xm_025nncs = np.array([np.mean([d025['object'].sample[i][0] for i in range(90000)]) for d025 in nncs])
    # # 3.0
    # xm_3tcs = np.array([np.mean([d025['object'].sample[i][8] for i in range(90000)]) for d025 in tcs])
    # xm_3ncs = np.array([np.mean([d025['object'].sample[i][8] for i in range(90000)]) for d025 in ncs])
    # xm_3tncs = np.array([np.mean([d025['object'].sample[i][8] for i in range(90000)]) for d025 in tncs])
    # xm_3nncs = np.array([np.mean([d025['object'].sample[i][8] for i in range(90000)]) for d025 in tncs])
    # # 2.5
    # xm_25tcs = np.array([np.mean([d025['object'].sample[i][7] for i in range(90000)]) for d025 in tcs])
    # xm_25ncs = np.array([np.mean([d025['object'].sample[i][7] for i in range(90000)]) for d025 in ncs])
    # xm_25tncs = np.array([np.mean([d025['object'].sample[i][7] for i in range(90000)]) for d025 in tncs])
    # xm_25nncs = np.array([np.mean([d025['object'].sample[i][7] for i in range(90000)]) for d025 in nncs])
    # # 0.25/3.0
    # xm_0253ncs = xm_025ncs / xm_3ncs
    # xm_0253tcs = xm_025tcs / xm_3tcs
    # xm_0253nncs = xm_025nncs / xm_3nncs
    # xm_0253tncs = xm_025tncs / xm_3tncs
    # # 0.25/(3.0+2.5)
    # xm_025325ncs = xm_025ncs / (xm_3ncs + xm_25ncs)
    # xm_025325tcs = xm_025tcs / (xm_3tcs + xm_25tcs)
    # xm_025325nncs = xm_025nncs / (xm_3nncs + xm_25nncs)
    # xm_025325tncs = xm_025tncs / (xm_3tncs + xm_25tncs)
    # # y_cs and y_ncs (both tcs+tncs and ncs+nncs can be used)
    # y_cs = np.array([ycs for ycs in [tcs_element['target'] for tcs_element in tcs]])
    # y_ncs = np.array([yncs for yncs in [tncs_element['target'] for tncs_element in tncs]])
    # # binarize y_cs and y_ncs, so that we have one y with values 0-nonsignificant, 1-significant
    # y_cs = y_cs.astype(float).astype(int)
    # y_ncs = y_ncs.astype(float).astype(int)
    # y = np.concatenate((y_cs, y_ncs))
    # y_binary = np.where(y <= 1, 0, 1)
    # #1)
    # cs_matrix = np.column_stack((
    #     xm_025tcs,
    #     xm_025ncs,
    #     xm_0253tcs,
    #     xm_0253ncs,
    #     xm_025325tcs,
    #     xm_025325ncs,
    #     y_cs,
    # ))
    # ncs_matrix = np.column_stack((
    #     xm_025tncs,
    #     xm_025nncs,
    #     xm_0253tncs,
    #     xm_0253nncs,
    #     xm_025325tncs,
    #     xm_025325nncs,
    #     y_ncs,
    # ))
    # # Plot correlation matrix of all features for cs_matrix (using Pearson's r)
    # corr_matrix_cs = np.corrcoef(cs_matrix, rowvar=False).round(decimals=2)
    # corr_matrix_ncs = np.corrcoef(ncs_matrix, rowvar=False).round(decimals=2)

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # # Plot for cs_matrix
    # cs_im = axs[0].imshow(corr_matrix_cs, vmin=-1, vmax=1)
    # axs[0].set_title('cs_matrix')
    # axs[0].set_xticks(range(7))
    # axs[0].set_xticklabels(('xm_025tcs', 'xm_025ncs', 'xm_0253tcs', 'xm_0253ncs', 'xm_025325tcs', 'xm_025325ncs', 'y_cs'), rotation=45)
    # axs[0].set_yticks(range(7))
    # axs[0].set_yticklabels(('xm_025tcs', 'xm_025ncs', 'xm_0253tcs', 'xm_0253ncs', 'xm_025325tcs', 'xm_025325ncs', 'y_cs'))
    # for i in range(7):
    #     for j in range(7):
    #         axs[0].text(j, i, corr_matrix_cs[i, j], ha='center', va='center', color='r')

    # # Plot for ncs_matrix
    # ncs_im = axs[1].imshow(corr_matrix_ncs, vmin=-1, vmax=1)
    # axs[1].set_title('ncs_matrix')
    # axs[1].set_xticks(range(7))
    # axs[1].set_xticklabels(('xm_025tncs', 'xm_025nncs', 'xm_0253tncs', 'xm_0253nncs', 'xm_025325tncs', 'xm_025325nncs', 'y_ncs'), rotation=45)
    # axs[1].set_yticks(range(7))
    # axs[1].set_yticklabels(('xm_025tncs', 'xm_025nncs', 'xm_0253tncs', 'xm_0253nncs', 'xm_025325tncs', 'xm_025325nncs', 'y_ncs'))
    # for i in range(7):
    #     for j in range(7):
    #         axs[1].text(j, i, corr_matrix_ncs[i, j], ha='center', va='center', color='r')

    # # Create a common colorbar
    # cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # cbar = fig.colorbar(cs_im, cax=cax)
    # cbar.set_label('Pearson correlation')

    # # Adjust spacing between subplots
    # plt.subplots_adjust(wspace=0.4, left=0.1, right=0.84)

    # # Save the plots to a PDF file
    # with PdfPages(save_filename) as pdf:
    #     pdf.savefig(fig)

    # plt.show()

    ggg = []
    # preprocessing
    for zone_key, zone_list in rois.items():
        for element in zone_list:
            try:
                if (
                    element["target"] == "0.0"
                    or element["target"] == "1.0"
                    or element["target"] == "2.0"
                    or element["target"] == "3.0"
                    or element["target"] == "4.0"
                ):
                    if "tumor" in element["a_region"]:
                        ggg.append(element)
                    else:
                        a_r = element["a_region"]
                        print(f"Not tumor ROI! {a_r}")
                else:
                    print("No ggg!")
            except ValueError:
                p_key = element["patient_key"]
                print(f"Smt wrong with print_roi, patient: {p_key}")

    # list comprehensions for getting features for correlation matrix and classifier
    # .25
    xm_025 = np.array(
        [np.mean([d025["object"].sample[i][0] for i in range(90000)]) for d025 in ggg]
    )
    # .5
    xm_05 = np.array(
        [np.mean([d025["object"].sample[i][1] for i in range(90000)]) for d025 in ggg]
    )
    # .75
    xm_075 = np.array(
        [np.mean([d025["object"].sample[i][2] for i in range(90000)]) for d025 in ggg]
    )
    # 3.0
    xm_3 = np.array(
        [np.mean([d025["object"].sample[i][8] for i in range(90000)]) for d025 in ggg]
    )
    # 2.5
    xm_25 = np.array(
        [np.mean([d025["object"].sample[i][7] for i in range(90000)]) for d025 in ggg]
    )
    # 0.25/3.0
    xm_0253 = xm_025 / xm_3
    # 0.25/(3.0+2.5)
    xm_025325 = xm_025 / (xm_3 + xm_25)
    # (0.25+0.5+0.75)/(2.5+3.0)
    xm_025575325 = (xm_025 + xm_05 + xm_075) / (xm_25 + xm_3)
    # y
    y = np.array([ycs for ycs in [tcs_element["target"] for tcs_element in ggg]])
    y = y.astype(float).astype(int)
    y_binary = np.where(y <= 1, 0, 1)

    # 1)
    # Create the DataFrame
    df_features = pd.DataFrame(
        [xm_025, xm_0253, xm_025325, xm_025575325, y]
    ).T  # Transpose the data to have arrays as columns

    # Assign column names
    df_features.columns = ["xm_025", "xm_0253", "xm_025325", "xm_025575325", "y"]

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="white")
    corr = df_features.corr()
    heatmap = sns.heatmap(corr, annot=True, cmap="Blues", fmt=".3g")

    # Save the plots to a PDF file
    plt.savefig(save_filename)
    plt.show()


def main():

    with open(
        os.path.join(DATA_DIR_PATH + "/processed_patient_dict.json"),
        "r",
    ) as f:
        data = json.load(f)

    count = 0
    count1 = 0

    # Diffusivities
    # Exp0
    # recon_diffusivities = np.arange(0.25, 3.1, 3.0 / 12)
    # Ex1
    # recon_diffusivities = np.arange(0.25, 3.1, 3.0 / 12)
    # recon_diffusivities = np.append(recon_diffusivities, 20.0)
    # recon_diffusivities = np.append(recon_diffusivities, 100.0)
    # Ex2
    # recon_diffusivities = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00])
    # Ex3
    # recon_diffusivities = np.array([0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00])
    # Ex4
    # recon_diffusivities = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 10.00])
    # Ex5
    # recon_diffusivities = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00])
    # Ex6
    # recon_diffusivities = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 100.00])
    # Ex7
    # recon_diffusivities = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 10.00])
    # Ex8
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
    if not os.path.isfile(os.path.join(OUTPUT_DIR_PATH + "/processed_data.pkl")):
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

                        # calculate age based on date of birth and current date (e.g. 13.01.1949
                        try:
                            gs = df_gsdob.loc[patient_key, "gs"]
                            dob = df_gsdob.loc[patient_key, "dob"]
                            target = df_gsdob.loc[patient_key, "targets"]
                        except:
                            print("Patient not found in gsdob.csv")
                            gs = "NaN"
                            dob = "NaN"
                            target = "NaN"
                        # if dob == FILL_STR or not dob or dob == 'nan':
                        #     age = dob
                        # else:
                        #     age = (datetime.datetime.now() - datetime.datetime.strptime(dob, '%Y-%m-%d')).days / 365.25

                        # Add meta data
                        sample_dict = {
                            "object": my_sample,
                            "a_region": a_region,
                            "snr": snr,
                            "patient_key": patient_key,
                            "gs": gs,
                            "target": target,
                            "patient_age": dob,
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
        with open(os.path.join(OUTPUT_DIR_PATH + "/processed_data.pkl"), "wb") as f:
            pickle.dump(roi_print, f)
    else:
        with open(os.path.join(OUTPUT_DIR_PATH + "/processed_data.pkl"), "rb") as f:
            roi_print = pickle.load(f)

    # Plot PDFs with all boxplots per roi
    print_all(
        rois=roi_print,
        m=3,
        n=2,
    )

    # Plot extra PDF with avg boxplots per roi
    print_avg(
        rois=roi_print,
        diffusivities=recon_diffusivities,
        m=2,
        n=2,
    )

    gs_corr_matrix(
        rois=roi_print,
        save_filename=os.path.join(OUTPUT_DIR_PATH + "/correlation_matrix.pdf"),
    )

    gs_log_classifier(
        rois=roi_print,
        save_filename=os.path.join(OUTPUT_DIR_PATH + "/log_classifier_all.pdf"),
        save_filename_db=os.path.join(OUTPUT_DIR_PATH + "/log_classifier_db.pdf"),
    )

    pr_visualization(
        rois=roi_print,
        save_filename=os.path.join(OUTPUT_DIR_PATH + "/pr_visualization_025.pdf"),
    )


if __name__ == "__main__":
    main()

# TODO:
# - Run Gibbs sampler experiments with following configurations
# Exp0 (gibbs_exp0.pkl):
# Do NOT include signal for b0 and use max signal to signal_value[1]
# Diffusivity range [0.25,   0.5 ,   0.75,   1.  ,   1.25,   1.5 ,   1.75,   2.  , 2.25,   2.5 ,   2.75,   3.]
# Exp1 (gibbs_exp1.pkl):
# Include signal for b0 and adjust max signal to signal_value[0]
# Diffusivity range [0.25,   0.5 ,   0.75,   1.  ,   1.25,   1.5 ,   1.75,   2.  , 2.25,   2.5 ,   2.75,   3.  ,  20.  , 100.  ]
# Exp2 gibbs_exp2.pkl:
# Include signal for b0 and adjust max signal to signal_value[0]
# Diffusivity range [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
# Exp3 gibbs_exp3.pkl:
##Include signal for b0 and adjust max signal to signal_value[0]
# Diffusivity range [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
# Exp4 gibbs_exp4.pkl:
# Include signal for b0 and adjust max signal to signal_value[0]
# Diffusivity range [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 10.00]
# Exp5 gibbs_exp5.pkl:
# Include signal for b0 and adjust max signal to signal_value[0]
# Diffusivity range [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
# Exp6 gibbs_exp6.pkl:
# Include signal for b0 and adjust max signal to signal_value[0]
# Diffusivity range [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 100.00]
# Exp7 gibbs_exp7.pkl:
# Include signal for b0 and adjust max signal to signal_value[0]
# Diffusivity range [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 10.00]
# Only on subset of 36 subjects with gleason scoring
# Pearson's correlation coefficient between mean .25 diffusivity fraction and GGG for every subject
# Pearson's correlation coefficient for all significant vs. non-significant GGG
# Correlation heatmap (matrix) for all diffusivities ...

# Run and compare
# 10, 20, 100 with 0.0 compartment
# Generate average and standard deviation
# Average: Concatenate all
# Show average in visualizations (in addition and also without mean)
# Calculate correlation between GS and .25 diffusivity tissue compartment!!
# Needs to be one score
# Ask Stephan for ADC calculate
