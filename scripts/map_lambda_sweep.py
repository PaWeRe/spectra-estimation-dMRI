"""
MAP λ sweep — answer to the meeting-prep Q1 ("MAP smearing at λ=0.1 may be
fixable by tuning λ").

Forward-simulates a representative subset of ground-truth spectra at a
range of SNRs, then fits MAP (closed-form ridge NNLS, then non-negativity
projection, then normalise to sum=1) across a λ grid spanning four orders
of magnitude. NUTS @ λ=0.1 from the previous simulation_study.py run is
the reference floor for "how well an ill-posed inverse Laplace can be
inverted at this SNR / b-grid."

Question this script settles:
  Does ANY λ recover δ-spectra (GT-A, GT-D) as cleanly as NUTS, or is
  the smearing inherent to ridge-NNLS regardless of λ?

Outputs:
  results/simulation/map_lambda_sweep.csv
  results/simulation/map_lambda_sweep_summary.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectra_estimation_dmri.biomarkers.recompute import (
    B_VALUES_MS, DIFFUSIVITIES, build_design_matrix,
)


OUTPUT_DIR = "results/simulation"
SEED = 20260524

D = DIFFUSIVITIES  # [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]


def delta(idx: int) -> np.ndarray:
    r = np.zeros(8); r[idx] = 1.0
    return r


def lognormal_on_grid(mu: float, sigma: float) -> np.ndarray:
    logD = np.log(D)
    p = np.exp(-0.5 * ((logD - np.log(mu)) / sigma) ** 2)
    p /= p.sum()
    return p


GROUND_TRUTHS = {
    "GT-A_d0.25":     delta(0),
    "GT-D_d3.00":     delta(6),
    "GT-E_bi-tumor":  np.array([0.7, 0, 0, 0, 0, 0, 0.3, 0]),
    "GT-F_bi-norm":   np.array([0.3, 0, 0, 0, 0, 0, 0.7, 0]),
    "GT-H_lognorm0.5": lognormal_on_grid(0.5, 0.6),
    "GT-I_lognorm1.5": lognormal_on_grid(1.5, 0.5),
}

SNRS = [200, 400, 800]
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]
N_REPS = 100


def normalize(R: np.ndarray) -> np.ndarray:
    s = R.sum()
    return R / s if s > 0 else R


def simulate_signal(R_true: np.ndarray, U: np.ndarray, snr: float,
                    rng: np.random.Generator) -> np.ndarray:
    mu = U @ R_true
    sigma = 1.0 / snr
    return mu + rng.normal(0.0, sigma, size=mu.shape)


def fit_map(y_norm: np.ndarray, U: np.ndarray, lam: float) -> np.ndarray:
    n_d = U.shape[1]
    A = U.T @ U + lam * np.eye(n_d)
    R = np.linalg.solve(A, U.T @ y_norm)
    R = np.maximum(R, 0.0)
    return normalize(R)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    U = build_design_matrix()
    rng = np.random.default_rng(SEED)

    print(f"=== MAP λ sweep: {len(GROUND_TRUTHS)} GTs × {len(SNRS)} SNRs × "
          f"{len(LAMBDAS)} λs × {N_REPS} reps = "
          f"{len(GROUND_TRUTHS)*len(SNRS)*len(LAMBDAS)*N_REPS} MAP fits ===")
    t0 = time.time()
    rows = []
    for gt_name, R_true in GROUND_TRUTHS.items():
        R_true_norm = normalize(R_true.astype(float))
        for snr in SNRS:
            for lam in LAMBDAS:
                R_hats = np.zeros((N_REPS, 8))
                for rep in range(N_REPS):
                    y = simulate_signal(R_true_norm, U, snr, rng)
                    R_hats[rep] = fit_map(y, U, lam=lam)
                R_hat_mean = R_hats.mean(axis=0)
                R_hat_std = R_hats.std(axis=0)
                for i in range(8):
                    rows.append({
                        "gt": gt_name,
                        "snr": snr,
                        "lambda": lam,
                        "bin": i,
                        "D": D[i],
                        "R_true": R_true_norm[i],
                        "R_hat_mean": R_hat_mean[i],
                        "R_hat_std": R_hat_std[i],
                        "bias": R_hat_mean[i] - R_true_norm[i],
                        "mse": ((R_hats[:, i] - R_true_norm[i]) ** 2).mean(),
                    })
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "map_lambda_sweep.csv")
    df.to_csv(out, index=False)
    print(f"  Wrote {out} ({len(df)} rows)")

    # Summary table: per (gt, snr, lambda), aggregate "how much mass at the
    # true peak bin?" plus a total-MSE metric across all 8 bins.
    print("\n=== Per-GT recovery summary (mass at true peak vs total MSE) ===")
    summary_rows = []
    for gt_name, R_true in GROUND_TRUTHS.items():
        R_true_norm = normalize(R_true.astype(float))
        peak_bins = np.where(R_true_norm > 0.1)[0].tolist()
        peak_label = ",".join(f"D={D[i]}" for i in peak_bins)
        for snr in SNRS:
            for lam in LAMBDAS:
                sub = df[(df["gt"] == gt_name) & (df["snr"] == snr)
                         & (df["lambda"] == lam)]
                mass_at_peaks = sub.iloc[peak_bins]["R_hat_mean"].sum()
                total_mse = sub["mse"].sum()
                summary_rows.append({
                    "gt": gt_name,
                    "peak_bins": peak_label,
                    "snr": snr,
                    "lambda": lam,
                    "mass_at_true_peaks": mass_at_peaks,
                    "true_mass_at_peaks": R_true_norm[peak_bins].sum(),
                    "fraction_recovered": mass_at_peaks / R_true_norm[peak_bins].sum(),
                    "total_mse_8bins": total_mse,
                })

    summary_df = pd.DataFrame(summary_rows)
    out2 = os.path.join(OUTPUT_DIR, "map_lambda_sweep_summary.csv")
    summary_df.to_csv(out2, index=False)
    print(f"  Wrote {out2} ({len(summary_df)} rows)")

    # Print human-readable table for the meeting
    print("\n=== Best λ per (GT, SNR) by total 8-bin MSE ===")
    for gt_name in GROUND_TRUTHS:
        for snr in SNRS:
            sub = summary_df[(summary_df["gt"] == gt_name)
                             & (summary_df["snr"] == snr)]
            best = sub.loc[sub["total_mse_8bins"].idxmin()]
            print(f"  {gt_name:18s} SNR={snr:4d} | best λ={best['lambda']:>6.3f} "
                  f"| MSE={best['total_mse_8bins']:.4f} "
                  f"| frac@peak={best['fraction_recovered']:.3f}")

    print("\n=== MAP @ λ=0.1 (baseline used in manuscript) vs best-tuned λ ===")
    for gt_name in GROUND_TRUTHS:
        for snr in SNRS:
            baseline = summary_df[(summary_df["gt"] == gt_name)
                                  & (summary_df["snr"] == snr)
                                  & (summary_df["lambda"] == 0.1)].iloc[0]
            sub = summary_df[(summary_df["gt"] == gt_name)
                             & (summary_df["snr"] == snr)]
            best = sub.loc[sub["total_mse_8bins"].idxmin()]
            print(f"  {gt_name:18s} SNR={snr:4d} | λ=0.1: "
                  f"frac={baseline['fraction_recovered']:.3f}, "
                  f"MSE={baseline['total_mse_8bins']:.4f} | "
                  f"best (λ={best['lambda']:.3f}): "
                  f"frac={best['fraction_recovered']:.3f}, "
                  f"MSE={best['total_mse_8bins']:.4f}")


if __name__ == "__main__":
    main()
