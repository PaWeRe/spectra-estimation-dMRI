import datetime
import json
import os
import pickle
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cvxopt import matrix, solvers
import pandas as pd
from tqdm import tqdm
import yaml
import h5py

sys.path.append(os.path.join(os.getcwd() + "/src/utils/"))
import generate_analysis_datasets as gad


class signal_data:
    def __init__(self, signal_values, b_values):
        self.signal_values = signal_values
        self.b_values = b_values
        # self.v_count = v_count
        # self.patient_id = patient_id

    def plot(self, save_filename=None, title=None):
        plt.plot(self.b_values, self.signal_values, linestyle="None", marker=".")
        plt.xlabel("B Value")
        plt.ylabel("Signal")
        if title is not None:
            plt.title(title)
        # plt.show()
        if save_filename is not None:
            plt.savefig(save_filename)
        else:
            plt.show()

    # Plot 3rd dimension (snr/p_count as well)
    def plot3d(self, title=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.b_values, self.signal_values, self.v_count, marker=".")
        ax.set_xlabel("B Value")
        ax.set_ylabel("Signal")
        ax.set_zlabel("V Count")
        if title is not None:
            ax.set_title(title)
        plt.show()


# package up fractions and diffusivities
class d_spectrum:
    def __init__(self, fractions, diffusivities):
        self.fractions = fractions
        self.diffusivities = diffusivities

    def plot(self, title=None):
        # plt.plot(diffusivities, fractions, 'o')
        plt.vlines(self.diffusivities, np.zeros(len(self.fractions)), self.fractions)
        plt.xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)")
        plt.ylabel("Relative Fraction")
        plt.xlim(left=0)
        if title is not None:
            plt.title(title)
        plt.show()


# a collection of diffusivity spectra, probably
# result of Gibbs' sampling
class d_spectra_sample:
    def __init__(self, diffusivities):
        self.diffusivities = diffusivities
        self.sample = []  # a list of samples

    def plot(
        self, save_filename=None, title=None, start=0, end=-1, skip=False, ax=None
    ):
        if not ax:
            fig, ax = plt.subplots()
            ax.set_xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)")
            ax.set_ylabel("Relative Fraction")
            tick_labels = []
            for number in self.diffusivities:
                tick_labels.append(str(number))
            if skip:
                for i in range(0, len(self.diffusivities)):
                    if i % 2 == 1:
                        tick_labels[i] = ""
            print(tick_labels)
            ax.set_xticklabels(tick_labels, rotation=45)
        if title is not None:
            ax.set_title(title, fontdict={"fontsize": 12})
        sample_arr = np.asarray(self.sample)[start:end]
        ax.boxplot(
            sample_arr,
            showfliers=False,
            manage_ticks=False,
            showmeans=True,
            meanline=True,
        )
        # store = ax.boxplot(sample_arr, showfliers=True, manage_ticks=False, showmeans=True, meanline=True, whis=10)
        # sum_means = np.sum([store['means'][i]._y[0] for i in range(10)])
        # print(sum_means)
        if save_filename is not None:
            plt.savefig(save_filename)

    def diagnostic_plot(self):
        plt.plot(np.asarray(self.sample))
        plt.show()

    def normalize(self):
        for i in range(0, len(self.sample)):
            sum = np.sum(self.sample[i])
            self.sample[i] = self.sample[i] / sum


# A multi compartment signal generator
#
# predict a single or multiple signals corresponding to the specified b value(s),
# fractions are the mixing coefficients of the corresponding diffusivities
# they are supposed to sum to one
def predict_signal(s_0, b_value, diffusivities, fractions):
    signal = 0
    for i in range(len(diffusivities)):
        signal += s_0 * fractions[i] * np.exp(-b_value * diffusivities[i])
    return signal


# make a vector of signals that correspond to a vector of b_values
# with optional noise added
# (with noise, sigs can be negative)
def generate_signals(s_0, b_values, d_spectrum, sigma=0.0):
    predictions = predict_signal(
        s_0, b_values, d_spectrum.diffusivities, d_spectrum.fractions
    )
    if sigma > 0.0:
        predictions += npr.normal(0, sigma, len(predictions))
    return signal_data(predictions, b_values)


###################################################################
# Calculate the parameters of the multivariate normal that characterizes
# the posterior distribution on R, the 'concentrations'
# with optional prior inverse covariance
# and optional L2 regularizer, which overides inverse_prior_covariance
# calculating the mean entails solving a linear equation in Sigma_inverse
# which may not be full rank.  Prints rank, if deficient can be fixed
# by using L1_lambda > 0.0


def calculate_MVN_posterior_params(
    signal_data,
    recon_diffusivities,
    sigma,
    inverse_prior_covariance=None,
    L1_lambda=0.0,
    L2_lambda=0.0,
):
    M_count = signal_data.signal_values.shape[0]
    N = recon_diffusivities.shape[0]
    u_vec_tuple = ()
    for i in range(M_count):
        u_vec_tuple += (np.exp((-signal_data.b_values[i] * recon_diffusivities)),)
    U_vecs = np.vstack(u_vec_tuple)
    U_outer_prods = np.zeros((N, N))
    for i in range(M_count):
        U_outer_prods += np.outer(U_vecs[i], U_vecs[i])
    Sigma_inverse = (1.0 / (sigma * sigma)) * U_outer_prods
    if L2_lambda > 0.0:
        inverse_prior_covariance = L2_lambda * np.eye(N)
    if inverse_prior_covariance is not None:
        Sigma_inverse += inverse_prior_covariance
    print(
        "Size: {}, Rank: {}".format(
            Sigma_inverse.shape[0], np.linalg.matrix_rank(Sigma_inverse)
        )
    )
    Sigma = np.linalg.inv(Sigma_inverse)
    weighted_U_vecs = np.zeros(N)
    for i in range(M_count):
        weighted_U_vecs += signal_data.signal_values[i] * U_vecs[i]
    One_vec = np.ones(N)
    Sigma_inverse_M = (1.0 / (sigma * sigma) * weighted_U_vecs) - L1_lambda * One_vec
    # avoid inverse
    M = np.linalg.solve(Sigma_inverse, Sigma_inverse_M)
    return (M, Sigma_inverse, Sigma_inverse_M)


# find mode of truncated MVN by convex optimization
# def calculate_trunc_MVN_mode(M, Sigma, Sigma_inverse, Sigma_inverse_M):
def calculate_trunc_MVN_mode(M, Sigma_inverse, Sigma_inverse_M):
    N = Sigma_inverse.shape[0]
    P = matrix(Sigma_inverse)
    Q = matrix(-Sigma_inverse_M)
    G = matrix(-np.identity(N), tc="d")
    H = matrix(np.zeros(N))
    sol = solvers.qp(P, Q, G, H)
    mode = sol["x"]
    return mode
    #  M, Sigma_inverse, and mode are established


# given signals etc, calculate the MVN parameters, then solve for the mode
def trunc_MVN_mode(
    signal_data,
    recon_diffusivities,
    sigma,
    inverse_prior_covariance=None,
    L1_lambda=0.0,
    L2_lambda=0.0,
):
    M, Sigma_inverse, Sigma_inverse_M = calculate_MVN_posterior_params(
        signal_data,
        recon_diffusivities,
        sigma,
        inverse_prior_covariance,
        L1_lambda=L1_lambda,
        L2_lambda=L2_lambda,
    )
    mode = calculate_trunc_MVN_mode(M, Sigma_inverse, Sigma_inverse_M)
    return d_spectrum(np.asarray(mode), recon_diffusivities)


#############################################################################
## this one had problems, so coded custom one (next)
## sample from specified univariate normal dist truncated to non zero values
# def sample_normal_non_neg(mu, sigma):
#     z = mu / sigma
#     # if z < -10.0:
#     if z < -5.0:
#         value = 0.0
#     else:
#         X = stats.truncnorm(
#             - mu / sigma, np.inf, loc=mu, scale=sigma)
#         value = X.rvs(1)[0]
#     # print('mu: {}, sigma: {}, value: {}'.format(mu, sigma, value))
#     return value
#############################################################################


# sample from a univariate normal distribution
# that is truncated to be non-negative
def sample_normal_non_neg_new(mu, sigma):
    if mu >= 0:
        # standard rejection sampler
        while True:
            u = npr.normal(mu, sigma)
            if u > 0.0:
                return u
    else:
        # mu is negative, use 'robert sampler',
        # a sampler for a standard normal truncated on the left
        # at sigma_minus, which should be non negative
        # follows method in:
        #     'Simulation of Truncated Normal Variables'
        #     Cristian P. Robert, ArXiv 2009
        sigma_minus = -mu / sigma
        alpha_star = (sigma_minus + np.sqrt(sigma_minus**2 + 4)) / 2.0
        while True:
            x = npr.exponential(1.0 / alpha_star)  # arugment is 1 / lambda
            z = sigma_minus + x
            eta = np.exp(-((z - alpha_star) ** 2) / 2.0)
            u = npr.uniform()
            if u <= eta:
                # z is result of robert sampler, proposition 2.3 in paper
                # transform result back for more general non standard normal
                return mu + sigma * z


def make_Gibbs_sampler(
    signal_data,
    diffusivities,
    sigma,
    inverse_prior_covariance=None,
    L1_lambda=0.0,
    L2_lambda=0.0,
):
    signals = signal_data.signal_values
    b_values = signal_data.b_values
    M, Sigma_inverse, weighted_U_vecs = calculate_MVN_posterior_params(
        signal_data,
        diffusivities,
        sigma,
        inverse_prior_covariance,
        L1_lambda=L1_lambda,
        L2_lambda=L2_lambda,
    )
    # initialize R to mode of the truncated MVN
    R = np.array(calculate_trunc_MVN_mode(M, Sigma_inverse, weighted_U_vecs)).T[0]
    N = Sigma_inverse.shape[0]
    ###################################################
    print("\n\nSetting Up Gibbs Sampler")
    sigma_i = np.empty(N, dtype=object)
    Sigma_inverse_quotient = np.empty(N, dtype=object)
    M_slash_i = np.empty(N, dtype=object)
    for i in range(N):
        Sigma_inverse_ii = Sigma_inverse[i][i]
        sigma_i[i] = np.sqrt(1.0 / Sigma_inverse_ii)
        Sigma_inverse_i_slash_i = np.delete(Sigma_inverse[i], i, axis=0)
        Sigma_inverse_quotient[i] = Sigma_inverse_i_slash_i / Sigma_inverse_ii
        M_slash_i[i] = np.delete(M, i, 0)

    def Gibbs_sampler(iterations, the_sample=None):
        if the_sample is None:
            the_sample = d_spectra_sample(diffusivities)
        count = 0
        for j in range(iterations):
            if (count % 100) == 0:
                print(".", end="")
            count += 1
            for i in range(N):
                # for i in range(1):
                # next 3 are rowvecs...
                R_slash_i = np.delete(R, i, 0)
                dot_prod = np.dot(Sigma_inverse_quotient[i], (M_slash_i[i] - R_slash_i))
                # Patrick wonders if dot_prod needs index by i
                mu_i = M[i] + dot_prod
                # value = sample_normal_non_neg(mu_i, sigma_i[i])
                value = sample_normal_non_neg_new(mu_i, sigma_i[i])
                R[i] = value
            # sample_collection.append(np.copy(R))
            the_sample.sample.append(np.copy(R))
        return the_sample

    return Gibbs_sampler


###########################################################################
# Gaussian process stuff
# Kernel for GP prior
def nep_kernel(i, j, k_sigma, scale):
    return np.square(k_sigma) * np.exp(-np.square((i - j) / scale) / 2.0)


# covariance for GP prior
def make_prior_cov(k_sigma, scale, dims):
    def inner_nep_kernel(i, j):
        return nep_kernel(i, j, k_sigma, scale)

    return np.fromfunction(inner_nep_kernel, (dims, dims))


############################################################################
def predict_signals_from_diffusivity_sample(d_specta_sample, b_values):
    diffusivities = d_spectra_sample.diffusivities
    for i in range(len(d_spectra_sample.sample)):
        fractions = d_spectra_sample.sample[i]
        sigs = generate_signals(1.0, b_values, diffusivities, fractions)
        print(sigs)


def init_plot_matrix(m, n, diffusivities):
    """
    Create graph layout on PDF with adjustments for better fitting
    """
    # Increase figure size
    fig, axarr = plt.subplots(m, n, sharex="col", sharey="row", figsize=(10, 10))

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.2, wspace=0.1)

    arr_ij = list(np.ndindex(axarr.shape))
    subplots = [axarr[index] for index in arr_ij]

    for s, splot in enumerate(subplots):
        last_row = m * n - s < n + 1
        first_in_row = s % n == 0
        splot.grid(color="0.75")
        if last_row:
            splot.set_xlabel(r"Diffusivity Value ($\mu$m$^2$/ ms.)", fontsize=10)
            str_diffs = [""] + list(map(str, diffusivities)) + [""]
            splot.set_xticks(
                np.arange(len(str_diffs)), labels=str_diffs, rotation=90, fontsize=6
            )
        if first_in_row:
            splot.set_ylabel("Relative Fraction", fontsize=10)

        # Set tick parameters for both axes
        splot.tick_params(axis="both", which="major", labelsize=10)

    # plt.tight_layout()
    return fig, axarr, subplots


def compute_gibbs_object(
    input_dict_path,
    output_hdf5_path,
    output_json_path,
    metadata_path,
    output_print_path,
    recon_diffusivities,
    iters=100000,
    c=150,
    l1_lambda=0.0,
    l2_lambda=0.00001,
    inverse_prior_covariance=None,
) -> None:
    """
    For given patient dict containing signal value decay per ROI, execute Gibbs Sampler and
    store sample objects with metadata in HDF5 file and JSON file respectively.
    """

    # Define output dict (structured by ROI)
    output_dict = {
        "normal_pz_s2": [],
        "normal_tz_s3": [],
        "tumor_pz_s1": [],
        "tumor_tz_s1": [],
        "Neglected!": [],
    }

    # Get Metadata table for patients (e.g. gleason score)
    df_gsdob = (
        pd.read_csv(metadata_path, sep=";", header=0)
        .astype(str)
        .fillna("NaN")
        .set_index("patient_key")
    )

    # Load in input dict (with only signal decay)
    if not os.path.isfile(input_dict_path):
        raise ValueError(f"The provided path {input_dict_path} does not exist.")
    with open(input_dict_path, "r") as f:
        input_dict = json.load(f)

    # Create HDF5 file
    counter = 0
    with h5py.File(output_hdf5_path, "w") as hf:
        # Compute Gibbs Sampler Object and add Metadata for every ROI
        for patient_key in tqdm(input_dict, desc="Patients", position=0):
            for roi_key in tqdm(input_dict[patient_key], desc="ROIs", position=1):
                if counter > 1:
                    break
                if roi_key.startswith("roi"):
                    b_values_roi = (
                        np.array(input_dict[patient_key][roi_key]["b_values"]) / 1000
                    )
                    max_sig = np.array(
                        input_dict[patient_key][roi_key]["signal_values"][0]
                    )
                    signal_values_roi = (
                        np.array(input_dict[patient_key][roi_key]["signal_values"])
                        / max_sig
                    )
                    sig_obj = signal_data(signal_values_roi, b_values_roi)
                    v_count = input_dict[patient_key][roi_key]["v_count"]
                    snr = np.sqrt(v_count / 16) * c
                    sigma = 1.0 / snr
                    a_region = input_dict[patient_key][roi_key]["anatomical_region"]

                    # Run Gibbs Sampler for the current ROI
                    my_sampler = make_Gibbs_sampler(
                        signal_data=sig_obj,
                        diffusivities=recon_diffusivities,
                        sigma=sigma,
                        inverse_prior_covariance=inverse_prior_covariance,
                        L1_lambda=l1_lambda,
                        L2_lambda=l2_lambda,
                    )
                    my_sample = my_sampler(iters)

                    # Normalize and discard the first 10000 (burn-in)
                    my_sample.normalize()
                    my_sample.sample = my_sample.sample[10000:]

                    # Add metadata to computed sample
                    try:
                        gs = df_gsdob.loc[patient_key][0]
                    except:
                        gs = "NaN"
                    try:
                        target = df_gsdob.loc[patient_key][1]
                    except:
                        target = "NaN"
                    # dob = df_gsdob.loc[patient_key][2]
                    # try:
                    #     age = (
                    #         datetime.datetime.now()
                    #         - datetime.datetime.strptime(dob, "%Y-%m-%d")
                    #     ).days / 365.25
                    # except:
                    #     age = "NaN"
                    age = "NaN"

                    # Create gibbs sampler object with metadata
                    gibbs_sample_dict = {
                        "a_region": a_region,
                        "snr": snr,
                        "patient_key": patient_key,
                        "gs": gs,
                        "target": target,
                        "patient_age": age,
                        "diffusivities": recon_diffusivities.tolist(),
                    }

                    # Store large array in HDF5
                    dataset_name = f"{patient_key}_{roi_key}"
                    hf.create_dataset(
                        dataset_name, data=my_sample.sample, compression="gzip"
                    )

                    # Add reference to HDF5 dataset in metadata
                    gibbs_sample_dict["hdf5_dataset"] = dataset_name

                    # Aggregate samples per roi for later plotting
                    output_dict[a_region].append(gibbs_sample_dict)
                    counter += 1

    # Save metadata as JSON
    gad.save_dataset(data=output_dict, filename=output_json_path)

    print_all(
        output_path=output_print_path,
        rois=output_dict,
        hdf5_path=output_hdf5_path,
        m=3,
        n=2,
    )

    print_avg(
        output_path=output_print_path,
        rois=output_dict,
        hdf5_path=output_hdf5_path,
        diffusivities=recon_diffusivities,
        m=2,
        n=2,
    )


def print_all(output_path: str, rois: dict, hdf5_path: str, m: int, n: int) -> None:
    KEY_TO_PDF_NAME = {
        "normal_pz_s2": "/npz.pdf",
        "normal_tz_s3": "/ntz.pdf",
        "tumor_pz_s1": "/tpz.pdf",
        "tumor_tz_s1": "/ttz.pdf",
        "Neglected!": "/neglected.pdf",
    }
    KEY_TO_CSV_NAME = {
        "normal_pz_s2": "/npz.csv",
        "normal_tz_s3": "/ntz.csv",
        "tumor_pz_s1": "/tpz.csv",
        "tumor_tz_s1": "/ttz.csv",
        "Neglected!": "/neglected.csv",
    }

    plt.rcParams.update({"font.size": 6})
    with h5py.File(hdf5_path, "r") as hf:
        for zone_key, zone_list in tqdm(rois.items(), desc="ROIs Zones", position=0):
            if len(zone_list) == 0:
                continue
            with PdfPages(os.path.join(output_path + KEY_TO_PDF_NAME[zone_key])) as pdf:
                f, axarr, subplots = init_plot_matrix(
                    m, n, zone_list[0]["diffusivities"]
                )
                n_pages = 1
                for i, sample_dict in tqdm(
                    enumerate(zone_list), desc="Samples in Zone", position=1
                ):
                    dataset = hf[sample_dict["hdf5_dataset"]]
                    sample = d_spectra_sample(sample_dict["diffusivities"])
                    sample.sample = dataset[()]

                    title = f'{sample_dict["patient_key"]}|{sample_dict["gs"]}|{sample_dict["target"]}|{int(sample_dict["snr"])}|{sample_dict["patient_age"]}'
                    sample.plot(ax=subplots[i - (n_pages - 1) * m * n], title=title)
                    if i == n_pages * m * n - 1:
                        pdf.savefig()
                        plt.close(f)
                        n_pages += 1
                        f, axarr, subplots = init_plot_matrix(
                            m, n, zone_list[0]["diffusivities"]
                        )
                pdf.savefig()
                plt.close(f)

            df = pd.DataFrame()
            for sample_dict in zone_list:
                dataset = hf[sample_dict["hdf5_dataset"]]
                for diff, sample in zip(
                    sample_dict["diffusivities"], np.transpose(dataset[()])
                ):
                    boxplot_stats = {
                        "Patient": sample_dict["patient_key"],
                        "ROI": sample_dict["a_region"],
                        "SNR": sample_dict["snr"],
                        "Gleason Score": sample_dict["gs"],
                        "Target": sample_dict["target"],
                        "Patient Age": sample_dict["patient_age"],
                        "Diffusivity": diff,
                        "Min": np.min(sample),
                        "Q1": np.percentile(sample, 25),
                        "Median": np.median(sample),
                        "Mean": np.mean(sample),
                        "Q3": np.percentile(sample, 75),
                        "Max": np.max(sample),
                    }
                    df = pd.concat(
                        [df, pd.DataFrame([boxplot_stats])], ignore_index=True
                    )
            df.to_csv(
                os.path.join(output_path + KEY_TO_CSV_NAME[zone_key]), index=False
            )


def print_avg(
    output_path: str, rois: dict, hdf5_path: str, diffusivities: list, m: int, n: int
) -> None:
    avg_dict = {}
    with h5py.File(hdf5_path, "r") as hf:
        for zone_key, zone_list in rois.items():
            if len(zone_list) == 0:
                continue
            avg_sample_obj = d_spectra_sample(diffusivities)
            avg_sample_obj.sample = np.mean(
                [hf[d["hdf5_dataset"]][()] for d in zone_list], axis=0
            )
            avg_dict[zone_key] = avg_sample_obj

    with PdfPages(os.path.join(output_path + "roi_avgs.pdf")) as pdf:
        f, axarr, subplots = init_plot_matrix(m, n, diffusivities)
        for i, (key, avg_sample) in enumerate(avg_dict.items()):
            if key != "Neglected!":
                avg_sample.plot(ax=subplots[i - m * n], title=key)
        pdf.savefig()
        plt.close(f)

    df = pd.DataFrame()
    for title, avg_sample in avg_dict.items():
        for diff, sample in zip(diffusivities, np.transpose(avg_sample.sample)):
            boxplot_stats = {
                "ROI": title,
                "Diffusivity": diff,
                "Min": np.min(sample),
                "Q1": np.percentile(sample, 25),
                "Median": np.median(sample),
                "Mean": np.mean(sample),
                "Q3": np.percentile(sample, 75),
                "Max": np.max(sample),
            }
            df = pd.concat([df, pd.DataFrame([boxplot_stats])], ignore_index=True)
    df.to_csv(os.path.join(output_path + "roi_avgs.csv"), index=False)


def fetch_gibbs_objects(gibbs_json_path: str, gibbs_hdf5_path: str) -> dict:
    """
    Fetching pre-computed gibbs objects from JSON and HDF5 files.
    """
    if not os.path.isfile(gibbs_json_path) or not os.path.isfile(gibbs_hdf5_path):
        raise FileNotFoundError(f"One or both of the provided paths do not exist.")

    with open(gibbs_json_path, "r") as f:
        gibbs_metadata = json.load(f)

    with h5py.File(gibbs_hdf5_path, "r") as hf:
        for roi_type in gibbs_metadata:
            for sample in gibbs_metadata[roi_type]:
                sample["sample"] = hf[sample["hdf5_dataset"]][()]

    return gibbs_metadata


def main(configs: dict) -> None:

    # Execute Sampler with configuration params
    recon_diffusivities = np.array(
        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
    )
    compute_gibbs_object(
        input_dict_path=configs["INPUT_DICT_PATH"],
        output_hdf5_path=configs["OUTPUT_HDF5_PATH"],
        output_json_path=configs["OUTPUT_JSON_PATH"],
        metadata_path=configs["METADATA_PATH"],
        output_print_path=configs["OUTPUT_PRINT_ROI_PATH"],
        recon_diffusivities=recon_diffusivities,
    )

    # Print boxplots
    output_dir = {}
    with open(configs["OUTPUT_JSON_PATH"], "r") as f:
        output_dir = json.load(f)
    print_avg(
        output_path=configs["OUTPUT_PRINT_ROI_PATH"],
        rois=output_dir,
        hdf5_path=configs["OUTPUT_HDF5_PATH"],
        diffusivities=recon_diffusivities,
        m=2,
        n=2,
    )
    print_all(
        output_path=configs["OUTPUT_PRINT_ROI_PATH"],
        rois=output_dir,
        hdf5_path=configs["OUTPUT_HDF5_PATH"],
        m=2,
        n=2,
    )


if __name__ == "__main__":
    # load in YAML configuration
    configs = {}
    base_config_path = os.path.join(os.getcwd() + "/configs.yaml")
    with open(base_config_path, "r") as file:
        configs.update(yaml.safe_load(file))
    main(configs)
