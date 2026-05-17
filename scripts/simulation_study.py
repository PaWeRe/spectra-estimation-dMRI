"""
Ground-truth simulation study.

For each ground-truth spectrum GT-A..GT-I, sweep SNR and noise reps; fit both
MAP (ridge λ=0.1, closed-form) and NUTS (reduced settings); collect per-bin
bias, MSE, recovery probability, NUTS 90% CI coverage.

Outputs:
  results/simulation/sim_results.csv     # raw per-rep predictions
  results/simulation/sim_summary.csv     # per (GT, SNR, estimator, bin)
  results/simulation/bias_heatmap.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from spectra_estimation_dmri.biomarkers.recompute import (
    B_VALUES_MS,
    DIFFUSIVITIES,
    build_design_matrix,
    compute_map_spectrum,
)


OUTPUT_DIR = "results/simulation"
SEED = 20260516


# ---------------- Ground truth spectra ----------------------------------

D = DIFFUSIVITIES  # [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]


def delta(idx: int) -> np.ndarray:
    r = np.zeros(8); r[idx] = 1.0; return r


def lognormal_on_grid(mu: float, sigma: float) -> np.ndarray:
    """Discrete log-normal at the 8 bins, normalized to sum=1."""
    logD = np.log(D)
    p = np.exp(-0.5 * ((logD - np.log(mu)) / sigma) ** 2)
    p /= p.sum()
    return p


GROUND_TRUTHS = {
    "GT-A_d0.25":    delta(0),
    "GT-B_d0.50":    delta(1),
    "GT-C_d0.75":    delta(2),
    "GT-D_d3.00":    delta(6),
    "GT-E_bi-tumor": np.array([0.7, 0, 0, 0, 0, 0, 0.3, 0]),
    "GT-F_bi-norm":  np.array([0.3, 0, 0, 0, 0, 0, 0.7, 0]),
    "GT-G_tri":      np.array([0.4, 0, 0, 0, 0.2, 0, 0.4, 0]),
    "GT-H_lognorm0.5": lognormal_on_grid(0.5, 0.6),
    "GT-I_lognorm1.5": lognormal_on_grid(1.5, 0.5),
}


SNRS = [100, 200, 400, 800, 1500]
N_REPS_MAP = 100
# NUTS is slow; run fewer reps at fewer SNRs
SNRS_NUTS = [200, 400, 800]
N_REPS_NUTS = 30


def normalize(R: np.ndarray) -> np.ndarray:
    """Normalize spectrum to sum to 1."""
    s = R.sum()
    return R / s if s > 0 else R


def simulate_signal(R_true: np.ndarray, U: np.ndarray, snr: float,
                    rng: np.random.Generator) -> np.ndarray:
    """y_norm = U @ R_true + N(0, 1/snr).  All work in normalized signal."""
    mu = U @ R_true
    sigma = 1.0 / snr
    return mu + rng.normal(0.0, sigma, size=mu.shape)


def fit_map(y_norm: np.ndarray, U: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """Ridge NNLS MAP (closed form, then clip)."""
    n_d = U.shape[1]
    A = U.T @ U + lam * np.eye(n_d)
    R = np.linalg.solve(A, U.T @ y_norm)
    R = np.maximum(R, 0.0)
    return normalize(R)


def fit_nuts(y_norm: np.ndarray, U: np.ndarray, lam: float = 0.1,
             draws: int = 1000, tune: int = 500, chains: int = 2,
             target_accept: float = 0.9, seed: int = 0) -> dict:
    """Return dict with R_mean (8,), R_std (8,), R_q05, R_q95, sigma_mean."""
    import pymc as pm
    sigma_R = 1.0 / np.sqrt(lam)
    n_d = U.shape[1]
    with pm.Model():
        R = pm.HalfNormal("R", sigma=sigma_R, shape=n_d)
        sigma = pm.HalfCauchy("sigma", beta=0.01)
        mu = pm.math.dot(U, R)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_norm)
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
            progressbar=False, random_seed=seed, return_inferencedata=True,
        )
    R_samples = idata.posterior["R"].values.reshape(-1, n_d)  # (n, 8)
    # Normalize each posterior sample to sum=1 before computing summaries
    row_sums = R_samples.sum(axis=1, keepdims=True)
    R_norm = R_samples / np.maximum(row_sums, 1e-10)
    sigma_post = idata.posterior["sigma"].values.flatten()
    return {
        "R_mean":  R_norm.mean(axis=0),
        "R_std":   R_norm.std(axis=0),
        "R_q05":   np.quantile(R_norm, 0.05, axis=0),
        "R_q95":   np.quantile(R_norm, 0.95, axis=0),
        "sigma_mean": float(sigma_post.mean()),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    U = build_design_matrix()
    rng = np.random.default_rng(SEED)

    # ---------------- MAP across all (GT, SNR, rep) ----------------------
    print("=== MAP ridge sweep ===")
    t0 = time.time()
    map_rows = []
    for gt_name, R_true in GROUND_TRUTHS.items():
        R_true_norm = normalize(R_true.astype(float))
        for snr in SNRS:
            for rep in range(N_REPS_MAP):
                y = simulate_signal(R_true_norm, U, snr, rng)
                R_hat = fit_map(y, U, lam=0.1)
                for i in range(8):
                    map_rows.append({
                        "estimator": "MAP",
                        "gt": gt_name,
                        "snr": snr,
                        "rep": rep,
                        "bin": i,
                        "D": D[i],
                        "R_true": R_true_norm[i],
                        "R_hat":  R_hat[i],
                    })
    print(f"  MAP done: {len(map_rows)} rows in {time.time()-t0:.1f}s")

    # ---------------- NUTS reduced sweep ---------------------------------
    print("\n=== NUTS reduced sweep (this is slow) ===")
    nuts_rows = []
    rng_nuts = np.random.default_rng(SEED + 1)
    t0 = time.time()
    n_total = len(GROUND_TRUTHS) * len(SNRS_NUTS) * N_REPS_NUTS
    counter = 0
    for gt_name, R_true in GROUND_TRUTHS.items():
        R_true_norm = normalize(R_true.astype(float))
        for snr in SNRS_NUTS:
            for rep in range(N_REPS_NUTS):
                counter += 1
                y = simulate_signal(R_true_norm, U, snr, rng_nuts)
                res = fit_nuts(y, U, lam=0.1, draws=800, tune=400,
                               chains=2, target_accept=0.9, seed=SEED + counter)
                for i in range(8):
                    nuts_rows.append({
                        "estimator": "NUTS",
                        "gt": gt_name,
                        "snr": snr,
                        "rep": rep,
                        "bin": i,
                        "D": D[i],
                        "R_true":  R_true_norm[i],
                        "R_hat":   res["R_mean"][i],
                        "R_std":   res["R_std"][i],
                        "R_q05":   res["R_q05"][i],
                        "R_q95":   res["R_q95"][i],
                        "sigma_post": res["sigma_mean"],
                    })
                if counter % 10 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / counter * (n_total - counter)
                    print(f"  [{counter}/{n_total}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    print(f"  NUTS done in {time.time()-t0:.0f}s")

    # ---------------- Save raw + summary ---------------------------------
    df_map  = pd.DataFrame(map_rows)
    df_nuts = pd.DataFrame(nuts_rows)
    df_all  = pd.concat([df_map, df_nuts], ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_DIR, "sim_results.csv"), index=False)

    # Summary: per (estimator, gt, snr, bin)
    rows = []
    for (est, gt, snr, b), grp in df_all.groupby(["estimator", "gt", "snr", "bin"]):
        R_true_val = grp["R_true"].iloc[0]
        R_hat = grp["R_hat"].values
        bias  = R_hat.mean() - R_true_val
        mse   = ((R_hat - R_true_val) ** 2).mean()
        recovery = (R_hat > 0.05).mean()  # bin "recovered" if mass > 5%
        row = {
            "estimator": est, "gt": gt, "snr": snr, "bin": b,
            "D": D[b], "R_true": R_true_val,
            "R_hat_mean": R_hat.mean(),
            "R_hat_std":  R_hat.std(),
            "bias": bias, "mse": mse, "recovery_prob": recovery,
        }
        if est == "NUTS":
            q05 = grp["R_q05"].values; q95 = grp["R_q95"].values
            cover = ((q05 <= R_true_val) & (R_true_val <= q95)).mean()
            row["coverage_90"] = cover
        rows.append(row)
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(os.path.join(OUTPUT_DIR, "sim_summary.csv"), index=False)
    print(f"Saved sim_results.csv ({len(df_all)} rows) and sim_summary.csv ({len(df_sum)} rows)")

    # ---------------- Bias heatmap (MAP vs NUTS at SNR ~ realistic) ------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax_row, snr_target in zip(axes, [200, 400]):
        for ax, est in zip(ax_row, ["MAP", "NUTS"]):
            sub = df_sum[(df_sum["estimator"] == est) & (df_sum["snr"] == snr_target)]
            if sub.empty: continue
            piv = sub.pivot(index="gt", columns="D", values="bias")
            piv = piv.reindex(list(GROUND_TRUTHS.keys()))
            im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r",
                           vmin=-0.4, vmax=0.4)
            ax.set_xticks(range(8)); ax.set_xticklabels([f"{d:.2g}" for d in D], fontsize=8)
            ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8)
            ax.set_title(f"{est}  SNR={snr_target}  (R_hat - R_true)")
            ax.set_xlabel("D (μm²/ms)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Per-bin bias by ground truth and SNR (closer to 0 is better)", y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bias_heatmap.png"), dpi=140, bbox_inches="tight")
    print("Saved bias_heatmap.png")


if __name__ == "__main__":
    main()
