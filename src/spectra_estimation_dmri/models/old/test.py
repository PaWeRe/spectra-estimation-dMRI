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

from spectra_estimation_dmri.utils import plotting


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


import torch
import torch.nn as nn
import torch.optim as optim


class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.u = nn.Parameter(torch.randn(dim, dtype=torch.float32))

    def forward(self, z):
        wzb = torch.sum(self.w * z, dim=1) + self.b
        return z + self.u * torch.tanh(wzb.unsqueeze(1))

    def log_det_jacobian(self, z):
        wzb = torch.sum(self.w * z, dim=1) + self.b
        return torch.log(
            torch.abs(1 + torch.sum(self.u * self.w) * (1 - torch.tanh(wzb) ** 2))
        )


class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_flows):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=torch.float32)
        for flow in self.flows:
            log_det += flow.log_det_jacobian(z)
            z = flow(z)
        return z, log_det


class MFVB_TruncatedMVN_NormalizingFlow:
    def __init__(
        self,
        signal_data,
        recon_diffusivities,
        sigma,
        n_flows=5,
        L1_lambda=0.0,
        L2_lambda=0.0,
    ):
        self.signal_data = signal_data
        self.recon_diffusivities = torch.tensor(
            recon_diffusivities, dtype=torch.float32
        )
        self.sigma = torch.tensor(sigma, dtype=torch.float32)
        self.L1_lambda = torch.tensor(L1_lambda, dtype=torch.float32)
        self.L2_lambda = torch.tensor(L2_lambda, dtype=torch.float32)
        self.m = len(recon_diffusivities)
        self.n = len(signal_data.b_values)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute U matrix
        self.U = torch.exp(
            -torch.outer(
                torch.tensor(signal_data.b_values, dtype=torch.float32),
                self.recon_diffusivities,
            )
        ).to(self.device)

        # Calculate MVN posterior parameters
        self.M, self.Sigma_inv, self.Sigma_inv_M = self.calculate_MVN_posterior_params()

        self.flow = NormalizingFlow(self.m, n_flows).to(self.device)
        self.optimizer = optim.Adam(self.flow.parameters(), lr=0.01)

    def calculate_MVN_posterior_params(self):
        U_outer_prods = torch.matmul(self.U.T, self.U)
        Sigma_inv = (
            1.0 / (self.sigma * self.sigma)
        ) * U_outer_prods + self.L2_lambda * torch.eye(
            self.m, dtype=torch.float32, device=self.device
        )
        weighted_U_vecs = torch.matmul(
            torch.tensor(
                self.signal_data.signal_values, dtype=torch.float32, device=self.device
            ),
            self.U,
        )
        Sigma_inv_M = (
            1.0 / (self.sigma * self.sigma) * weighted_U_vecs
        ) - self.L1_lambda * torch.ones(self.m, dtype=torch.float32, device=self.device)
        M = torch.linalg.solve(Sigma_inv, Sigma_inv_M)
        return M, Sigma_inv, Sigma_inv_M

    def objective(self, z):
        x, log_det = self.flow(z)

        # Apply exponential to ensure non-negativity
        R = torch.exp(x)

        # Compute log-likelihood
        E_R = R
        E_R2 = R * R

        log_likelihood = -0.5 * torch.sum(
            torch.diag(self.Sigma_inv) * E_R2
        ) + torch.sum(self.Sigma_inv_M * E_R)

        # Add log-det-jacobian for the exponential transformation
        log_det += torch.sum(x, dim=1)

        return -(log_likelihood + log_det).mean()

    def update_parameters(self, n_iterations=1000):
        for _ in range(n_iterations):
            self.optimizer.zero_grad()
            z = torch.randn(
                100, self.m, device=self.device, dtype=torch.float32
            )  # Sample from base distribution
            loss = self.objective(z)
            loss.backward()
            self.optimizer.step()

    def sample(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.m, device=self.device, dtype=torch.float32)
            x, _ = self.flow(z)
            R = torch.exp(x)
        return R.cpu().numpy()


def compute_mfvb_object(
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
    n_flows=5,
):
    # Load input data
    with open(input_dict_path, "r") as f:
        input_dict = json.load(f)

    # Create output dict
    output_dict = {
        "normal_pz_s2": [],
        "normal_tz_s3": [],
        "tumor_pz_s1": [],
        "tumor_tz_s1": [],
        "Neglected!": [],
    }

    counter = 0
    # Create HDF5 file
    with h5py.File(output_hdf5_path, "w") as hf:
        for patient_key in tqdm(input_dict, desc="Patients"):
            for roi_key in tqdm(input_dict[patient_key], desc="ROIs"):
                if counter == 5:
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

                    # Run MFVB for the current ROI
                    mfvb = MFVB_TruncatedMVN_NormalizingFlow(
                        sig_obj,
                        recon_diffusivities,
                        sigma,
                        n_flows=n_flows,
                        L1_lambda=l1_lambda,
                        L2_lambda=l2_lambda,
                    )
                    mfvb.update_parameters(
                        n_iterations=1000
                    )  # You may need to adjust this
                    samples = mfvb.sample(iters)

                    # Normalize samples
                    samples /= samples.sum(axis=1, keepdims=True)

                    # Discard the first 10000 (burn-in)
                    samples = samples[10000:]

                    # Create metadata dict
                    mfvb_sample_dict = {
                        "a_region": a_region,
                        "snr": snr,
                        "patient_key": patient_key,
                        "gs": "NaN",
                        "target": "NaN",
                        "patient_age": "NaN",
                        "diffusivities": recon_diffusivities.tolist(),
                    }

                    # Store large array in HDF5
                    dataset_name = f"{patient_key}_{roi_key}"
                    hf.create_dataset(dataset_name, data=samples, compression="gzip")

                    # Add reference to HDF5 dataset in metadata
                    mfvb_sample_dict["hdf5_dataset"] = dataset_name

                    # Aggregate samples per roi for later plotting
                    output_dict[a_region].append(mfvb_sample_dict)
                    counter += 1

    # Save metadata as JSON
    with open(output_json_path, "w") as f:
        json.dump(output_dict, f)

    print_all(output_print_path, output_dict, output_hdf5_path, 3, 2)
    print_avg(
        output_print_path, output_dict, output_hdf5_path, recon_diffusivities, 2, 2
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


if __name__ == "__main__":
    import yaml

    configs = {}
    base_config_path = os.path.join(os.getcwd() + "/configs.yaml")
    with open(base_config_path, "r") as file:
        configs.update(yaml.safe_load(file))

    recon_diffusivities = np.array(
        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
    )
    compute_mfvb_object(
        input_dict_path=configs["INPUT_DICT_PATH"],
        output_hdf5_path=configs["OUTPUT_HDF5_PATH"],
        output_json_path=configs["OUTPUT_JSON_PATH"],
        metadata_path=configs["METADATA_PATH"],
        output_print_path=configs["OUTPUT_PRINT_ROI_PATH"],
        recon_diffusivities=recon_diffusivities,
        n_flows=5,  # You can adjust this value
    )
