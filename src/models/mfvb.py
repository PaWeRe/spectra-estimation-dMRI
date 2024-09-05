import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class MFVBDiffusionModel:
    def __init__(self, signal_data, b_values, diffusivities, sigma, initial_fractions):
        self.signal_data = torch.tensor(signal_data, dtype=torch.float32)
        self.b_values = torch.tensor(b_values, dtype=torch.float32)
        self.diffusivities = torch.tensor(diffusivities, dtype=torch.float32)
        self.sigma = sigma
        self.m = len(diffusivities)
        self.n = len(b_values)

        # Initialize variational parameters based on initial fractions
        self.alpha = torch.nn.Parameter(
            torch.tensor([f * 1000 + 1 for f in initial_fractions], dtype=torch.float32)
        )
        self.beta = torch.nn.Parameter(torch.ones(self.m, dtype=torch.float32) * 1000)

        # Compute U matrix
        self.U = torch.exp(-torch.outer(self.b_values, self.diffusivities))

        # Compute Sigma_inv and M
        self.Sigma_inv = (1 / (self.sigma**2)) * torch.mm(self.U.T, self.U)
        self.M = (1 / (self.sigma**2)) * torch.mm(
            self.U.T, self.signal_data.unsqueeze(1)
        )

    def elbo(self):
        E_R = self.alpha / self.beta
        E_R_squared = (self.alpha * (self.alpha + 1)) / (self.beta**2)

        V = torch.diag(E_R_squared - E_R**2)
        trace_term = 0.5 * torch.trace(
            torch.mm(self.Sigma_inv, V + torch.outer(E_R, E_R))
        )

        q = torch.mv(self.Sigma_inv, self.M.squeeze())
        q_term = torch.dot(q, E_R)

        entropy = torch.sum(
            self.alpha
            - torch.log(self.beta)
            + torch.lgamma(self.alpha)
            + (1 - self.alpha) * torch.digamma(self.alpha)
        )

        return -trace_term + q_term + entropy

    def optimize(self, num_iterations=1000, lr=0.01):
        optimizer = torch.optim.Adam([self.alpha, self.beta], lr=lr)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            loss = -self.elbo()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.alpha, self.beta], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                self.alpha.clamp_(min=1e-6)
                self.beta.clamp_(min=1e-6)

    def get_posterior_mean(self):
        fractions = (self.alpha / self.beta).detach().numpy()
        return fractions / np.sum(fractions)  # Normalize to sum to 1


def run_mfvb_and_plot(
    data, diffusivities, initial_fractions, num_iterations=1000, lr=0.01
):
    results = []

    for sample in tqdm(data):
        sigma = 1.0 / np.sqrt(sample["v_count"] / 16 * 150)
        model = MFVBDiffusionModel(
            sample["signal_values"],
            sample["b_values"],
            diffusivities,
            sigma,
            initial_fractions[sample["anatomical_region"]],
        )
        model.optimize(num_iterations=num_iterations, lr=lr)
        results.append(
            {
                "fractions": model.get_posterior_mean(),
                "anatomical_region": sample["anatomical_region"],
            }
        )

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.ravel()

    for i, region in enumerate(
        ["normal_pz_s2", "normal_tz_s3", "tumor_pz_s1", "tumor_tz_s1"]
    ):
        region_data = [
            r["fractions"] for r in results if r["anatomical_region"] == region
        ]
        if region_data:
            axs[i].boxplot(np.array(region_data).T)
            axs[i].set_title(f"MFVB: {region}")
            axs[i].set_xlabel(r"Diffusivity Value ($\mu m^2$/ ms.)")
            axs[i].set_ylabel("Relative Fraction")

            tick_locations = range(1, len(diffusivities) + 1)
            axs[i].set_xticks(tick_locations)
            axs[i].set_xticklabels([f"{d:.2f}" for d in diffusivities])
            axs[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("mfvb_boxplots.pdf")
    plt.close()


if __name__ == "__main__":
    initial_fractions = {
        "normal_pz_s2": [0.1, 0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.1, 0.04, 0.01],
        "normal_tz_s3": [0.12, 0.06, 0.06, 0.11, 0.16, 0.19, 0.18, 0.09, 0.02, 0.01],
        "tumor_pz_s1": [0.2, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.05, 0.04, 0.01],
        "tumor_tz_s1": [0.22, 0.12, 0.11, 0.14, 0.14, 0.09, 0.09, 0.04, 0.04, 0.01],
    }

    # Define diffusivities
    diffusivities = np.array(
        [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 20.00]
    )

    # Sample data
    sample_data = [
        {
            "signal_values": np.array(
                [
                    1.0,
                    0.5723,
                    0.3635,
                    0.2407,
                    0.1748,
                    0.1296,
                    0.1017,
                    0.0861,
                    0.0744,
                    0.0667,
                    0.0593,
                    0.0547,
                    0.0518,
                    0.0449,
                    0.0442,
                ]
            ),
            "b_values": np.array(
                [
                    0.0,
                    250.0,
                    500.0,
                    750.0,
                    1000.0,
                    1250.0,
                    1500.0,
                    1750.0,
                    2000.0,
                    2250.0,
                    2500.0,
                    2750.0,
                    3000.0,
                    3250.0,
                    3500.0,
                ]
            ),
            "v_count": 214,
            "anatomical_region": "normal_pz_s2",
        },
        {
            "signal_values": np.array(
                [
                    1.0,
                    0.5756,
                    0.3863,
                    0.2593,
                    0.1907,
                    0.1421,
                    0.1136,
                    0.0956,
                    0.0828,
                    0.0727,
                    0.0642,
                    0.0564,
                    0.0494,
                    0.0463,
                    0.0414,
                ]
            ),
            "b_values": np.array(
                [
                    0.0,
                    250.0,
                    500.0,
                    750.0,
                    1000.0,
                    1250.0,
                    1500.0,
                    1750.0,
                    2000.0,
                    2250.0,
                    2500.0,
                    2750.0,
                    3000.0,
                    3250.0,
                    3500.0,
                ]
            ),
            "v_count": 214,
            "anatomical_region": "normal_tz_s3",
        },
        {
            "signal_values": np.array(
                [
                    1.0,
                    0.7264,
                    0.5680,
                    0.4513,
                    0.3836,
                    0.3343,
                    0.2945,
                    0.2493,
                    0.2390,
                    0.2180,
                    0.1942,
                    0.1941,
                    0.1665,
                    0.1613,
                    0.1463,
                ]
            ),
            "b_values": np.array(
                [
                    0.0,
                    250.0,
                    500.0,
                    750.0,
                    1000.0,
                    1250.0,
                    1500.0,
                    1750.0,
                    2000.0,
                    2250.0,
                    2500.0,
                    2750.0,
                    3000.0,
                    3250.0,
                    3500.0,
                ]
            ),
            "v_count": 134,
            "anatomical_region": "tumor_pz_s1",
        },
        {
            "signal_values": np.array(
                [
                    1.0,
                    0.7187,
                    0.5686,
                    0.4513,
                    0.3837,
                    0.3343,
                    0.2945,
                    0.2493,
                    0.2390,
                    0.2181,
                    0.1943,
                    0.1942,
                    0.1666,
                    0.1613,
                    0.1464,
                ]
            ),
            "b_values": np.array(
                [
                    0.0,
                    250.0,
                    500.0,
                    750.0,
                    1000.0,
                    1250.0,
                    1500.0,
                    1750.0,
                    2000.0,
                    2250.0,
                    2500.0,
                    2750.0,
                    3000.0,
                    3250.0,
                    3500.0,
                ]
            ),
            "v_count": 134,
            "anatomical_region": "tumor_tz_s1",
        },
    ]

    # Run MFVB and plot results
    run_mfvb_and_plot(
        sample_data, diffusivities, initial_fractions, num_iterations=5000, lr=0.001
    )

    # Print estimated fractions for each ROI
    for sample in sample_data:
        print(f"\nEstimated fractions for {sample['anatomical_region']}:")
        sigma = 1.0 / np.sqrt(sample["v_count"] / 16 * 150)
        model = MFVBDiffusionModel(
            sample["signal_values"],
            sample["b_values"],
            diffusivities,
            sigma,
            initial_fractions[sample["anatomical_region"]],
        )
        model.optimize(num_iterations=5000, lr=0.001)
        print(model.get_posterior_mean())
