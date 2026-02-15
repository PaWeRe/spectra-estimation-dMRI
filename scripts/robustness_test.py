"""
Robustness Test: Solver Performance Across Spectral Shapes and SNR.

Tests NUTS sampler robustness by:
1. Constructing diverse synthetic spectra:
   - Typical tumor spectrum (high restricted diffusion)
   - Typical normal tissue spectrum (high free diffusion)
   - "Inverse" spectrum (opposite of typical tumor pattern)
   - Flat/uniform spectrum
   - Bimodal spectrum (two peaks)
   - Single-peak spectrum
2. Running NUTS at multiple SNR levels (50, 100, 200, 500, 1000)
3. Reporting: RMSE, bias, R-hat, ESS, credible interval coverage

Usage:
    uv run python scripts/robustness_test.py [--fast]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "robustness_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BWH diffusivity grid and b-values
DIFF_VALUES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
B_VALUES = np.array([0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75,
                      2., 2.25, 2.5, 2.75, 3., 3.25, 3.5])

# Ridge regularization
RIDGE_STRENGTH = 0.5

# NUTS parameters (reduced for speed; increase for paper-quality results)
NUTS_CONFIG = {
    "fast": {"n_iter": 500, "tune": 200, "n_chains": 2, "target_accept": 0.90},
    "paper": {"n_iter": 2000, "tune": 500, "n_chains": 4, "target_accept": 0.95},
}

# SNR levels to test
SNR_LEVELS = [50, 100, 200, 500, 1000]

# Number of noise realizations per (spectrum, SNR) pair
N_REALIZATIONS = 5


def create_design_matrix(b_values, diffusivities):
    return np.exp(-np.outer(b_values, diffusivities))


def define_test_spectra(diff_values):
    """
    Define diverse synthetic spectra for robustness testing.

    Returns dict of spectrum_name -> spectrum_vector (normalized).
    """
    n = len(diff_values)

    spectra = {}

    # 1. Typical tumor (from ttz_default config, mapped to 8 bins)
    tumor = np.array([0.267, 0.037, 0.038, 0.058, 0.103, 0.116, 0.226, 0.116])
    spectra["Tumor (typical)"] = tumor / tumor.sum()

    # 2. Normal PZ tissue (from npz_default, mapped to 8 bins)
    normal = np.array([0.059, 0.026, 0.024, 0.027, 0.036, 0.047, 0.328, 0.053])
    spectra["Normal PZ"] = normal / normal.sum()

    # 3. Inverse of tumor: high at D>1, low at D<1
    inverse_tumor = tumor[::-1]
    spectra["Inverse tumor"] = inverse_tumor / inverse_tumor.sum()

    # 4. Flat/uniform
    flat = np.ones(n) / n
    spectra["Uniform"] = flat

    # 5. Bimodal: peaks at D=0.25 and D=3.0
    bimodal = np.zeros(n)
    bimodal[0] = 0.4  # D=0.25
    bimodal[6] = 0.4  # D=3.0
    bimodal[1:6] = 0.04  # Small fill
    bimodal[7] = 0.0  # No free water
    spectra["Bimodal"] = bimodal / bimodal.sum()

    # 6. Single peak at D=1.0 (intermediate)
    single_peak = np.zeros(n)
    idx_10 = np.argmin(np.abs(diff_values - 1.0))
    single_peak[idx_10] = 0.7
    single_peak[max(0, idx_10-1)] = 0.1
    single_peak[min(n-1, idx_10+1)] = 0.1
    remaining = 1.0 - single_peak.sum()
    single_peak[0] = remaining / 2
    single_peak[-1] = remaining / 2
    spectra["Single peak (D=1.0)"] = single_peak / single_peak.sum()

    return spectra


def simulate_signal(spectrum, b_values, diff_values, snr, rng=None):
    """Generate noisy signal decay from a known spectrum."""
    if rng is None:
        rng = np.random.default_rng()

    U = create_design_matrix(b_values, diff_values)
    signal_clean = U @ spectrum
    sigma = 1.0 / snr  # Noise level on normalized signal
    noise = rng.normal(0, sigma, size=signal_clean.shape)
    signal_noisy = signal_clean + noise
    return signal_noisy, signal_clean, sigma


def run_nuts_single(signal, b_values, diff_values, ridge_strength, nuts_config, seed=42):
    """
    Run NUTS on a single signal decay.

    Returns dict with samples, diagnostics, and timing.
    """
    import pymc as pm
    import arviz as az

    U = create_design_matrix(b_values, diff_values)
    n_dim = U.shape[1]

    sigma_R = 1.0 / np.sqrt(ridge_strength) if ridge_strength > 0 else 10.0

    t0 = time.time()
    with pm.Model() as model:
        R = pm.HalfNormal("R", sigma=sigma_R, shape=n_dim)
        sigma = pm.HalfCauchy("sigma", beta=0.01)
        mu = pm.math.dot(U, R)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=signal)

        idata = pm.sample(
            draws=nuts_config["n_iter"],
            tune=nuts_config["tune"],
            chains=nuts_config["n_chains"],
            target_accept=nuts_config["target_accept"],
            return_inferencedata=True,
            progressbar=False,
            random_seed=seed,
        )
    elapsed = time.time() - t0

    # Extract samples
    samples = idata.posterior["R"].values  # (chains, draws, n_dim)
    n_chains, n_draws, n_dim = samples.shape
    samples_flat = samples.reshape(-1, n_dim)

    # Diagnostics
    summary = az.summary(idata, var_names=["R", "sigma"])
    r_hat_values = summary["r_hat"].values
    ess_bulk_values = summary["ess_bulk"].values
    ess_tail_values = summary["ess_tail"].values

    return {
        "samples": samples_flat,
        "mean": samples_flat.mean(axis=0),
        "std": samples_flat.std(axis=0),
        "q025": np.percentile(samples_flat, 2.5, axis=0),
        "q975": np.percentile(samples_flat, 97.5, axis=0),
        "r_hat_max": r_hat_values[:-1].max(),  # Exclude sigma
        "r_hat_all": r_hat_values[:-1],
        "ess_bulk_min": ess_bulk_values[:-1].min(),
        "ess_tail_min": ess_tail_values[:-1].min(),
        "elapsed_sec": elapsed,
    }


def map_estimate_batch(U, signals, ridge_strength=0.5):
    """Vectorized MAP."""
    n_dim = U.shape[1]
    UU = U.T @ U + ridge_strength * np.eye(n_dim)
    UU_inv_Ut = np.linalg.solve(UU, U.T)
    spectra = (UU_inv_Ut @ signals.T).T
    return np.maximum(spectra, 0)


def compute_metrics(true_spectrum, estimated_mean, estimated_std, q025, q975):
    """Compute recovery metrics."""
    # RMSE (on normalized spectra)
    true_norm = true_spectrum / true_spectrum.sum()
    est_norm = estimated_mean / (estimated_mean.sum() + 1e-10)

    rmse = np.sqrt(np.mean((true_norm - est_norm) ** 2))
    mae = np.mean(np.abs(true_norm - est_norm))
    bias = np.mean(est_norm - true_norm)

    # Credible interval coverage: fraction of true values within [q025, q975]
    within = np.sum((true_spectrum >= q025) & (true_spectrum <= q975))
    coverage = within / len(true_spectrum)

    # Mean CI width
    ci_width = np.mean(q975 - q025)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "coverage_95": coverage,
        "ci_width": ci_width,
    }


def run_full_experiment(spectra_dict, snr_levels, diff_values, b_values,
                        ridge_strength, nuts_config, n_realizations, use_nuts=True):
    """
    Run the full robustness experiment.

    Returns DataFrame with all results.
    """
    U = create_design_matrix(b_values, diff_values)
    all_results = []

    total_runs = len(spectra_dict) * len(snr_levels) * n_realizations
    run_count = 0

    for spec_name, true_spectrum in spectra_dict.items():
        for snr in snr_levels:
            for realization in range(n_realizations):
                run_count += 1
                rng = np.random.default_rng(42 + realization + snr)
                signal, _, sigma = simulate_signal(
                    true_spectrum, b_values, diff_values, snr, rng
                )

                row = {
                    "spectrum_name": spec_name,
                    "snr": snr,
                    "realization": realization,
                    "true_spectrum": true_spectrum.tolist(),
                }

                # MAP estimate (always fast)
                map_est = map_estimate_batch(U, signal[None, :], ridge_strength)[0]
                map_metrics = compute_metrics(
                    true_spectrum, map_est, np.zeros_like(map_est),
                    map_est, map_est  # No CI for MAP
                )
                row["map_rmse"] = map_metrics["rmse"]
                row["map_bias"] = map_metrics["bias"]

                # NUTS (slower)
                if use_nuts:
                    seed = 42 + realization * 100 + snr
                    print(f"  [{run_count}/{total_runs}] {spec_name}, SNR={snr}, "
                          f"r={realization+1}/{n_realizations}...", end="", flush=True)
                    try:
                        nuts_result = run_nuts_single(
                            signal, b_values, diff_values, ridge_strength,
                            nuts_config, seed=seed
                        )
                        nuts_metrics = compute_metrics(
                            true_spectrum,
                            nuts_result["mean"],
                            nuts_result["std"],
                            nuts_result["q025"],
                            nuts_result["q975"],
                        )
                        row["nuts_rmse"] = nuts_metrics["rmse"]
                        row["nuts_bias"] = nuts_metrics["bias"]
                        row["nuts_coverage"] = nuts_metrics["coverage_95"]
                        row["nuts_ci_width"] = nuts_metrics["ci_width"]
                        row["r_hat_max"] = nuts_result["r_hat_max"]
                        row["ess_bulk_min"] = nuts_result["ess_bulk_min"]
                        row["ess_tail_min"] = nuts_result["ess_tail_min"]
                        row["elapsed_sec"] = nuts_result["elapsed_sec"]
                        row["converged"] = nuts_result["r_hat_max"] < 1.05
                        row["nuts_mean"] = nuts_result["mean"].tolist()
                        row["nuts_std"] = nuts_result["std"].tolist()
                        print(f" R̂={nuts_result['r_hat_max']:.3f}, "
                              f"RMSE={nuts_metrics['rmse']:.4f}, "
                              f"{nuts_result['elapsed_sec']:.1f}s")
                    except Exception as e:
                        print(f" FAILED: {e}")
                        row["nuts_rmse"] = np.nan
                        row["converged"] = False
                        row["elapsed_sec"] = np.nan

                all_results.append(row)

    return pd.DataFrame(all_results)


def plot_robustness_heatmap(results_df, metric="nuts_rmse", output_path=None):
    """
    Heatmap of a metric across spectra x SNR.
    """
    # Average over realizations
    pivot = results_df.groupby(["spectrum_name", "snr"])[metric].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Spectrum Type")

    # Add values to cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > np.nanmedian(pivot.values) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label=metric.replace("_", " ").title())

    title_map = {
        "nuts_rmse": "NUTS RMSE (normalized spectra)",
        "r_hat_max": "Max R-hat (convergence)",
        "nuts_coverage": "95% CI Coverage",
        "elapsed_sec": "Runtime (seconds)",
    }
    ax.set_title(title_map.get(metric, metric), fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def plot_spectrum_recovery(results_df, diff_values, output_path=None):
    """
    Visual comparison of true vs recovered spectra for each shape at best/worst SNR.
    """
    spec_names = results_df["spectrum_name"].unique()
    n = len(spec_names)
    snr_levels = sorted(results_df["snr"].unique())

    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i, spec_name in enumerate(spec_names):
        subset = results_df[results_df["spectrum_name"] == spec_name]

        for j, snr in enumerate([snr_levels[0], snr_levels[-1]]):  # Worst and best
            ax = axes[i, j]
            snr_subset = subset[subset["snr"] == snr]

            if snr_subset.empty or "nuts_mean" not in snr_subset.columns:
                ax.set_visible(False)
                continue

            # Get true spectrum
            true = np.array(snr_subset.iloc[0]["true_spectrum"])
            true_norm = true / true.sum()

            # Get NUTS means across realizations
            nuts_means = []
            nuts_stds = []
            for _, row in snr_subset.iterrows():
                if isinstance(row.get("nuts_mean"), list):
                    mean = np.array(row["nuts_mean"])
                    nuts_means.append(mean / (mean.sum() + 1e-10))
                if isinstance(row.get("nuts_std"), list):
                    nuts_stds.append(np.array(row["nuts_std"]))

            x = np.arange(len(diff_values))
            bar_width = 0.35

            # True spectrum
            ax.bar(x - bar_width/2, true_norm, bar_width,
                   color="#3498db", alpha=0.8, label="True")

            # NUTS mean (averaged over realizations)
            if nuts_means:
                avg_mean = np.mean(nuts_means, axis=0)
                avg_std = np.std(nuts_means, axis=0)  # Across realizations
                ax.bar(x + bar_width/2, avg_mean, bar_width,
                       color="#e74c3c", alpha=0.8, label="NUTS")
                ax.errorbar(x + bar_width/2, avg_mean, yerr=avg_std,
                            fmt="none", color="black", capsize=3, linewidth=1)

            # Convergence indicator
            converged = snr_subset["converged"].all() if "converged" in snr_subset.columns else True
            r_hat = snr_subset["r_hat_max"].mean() if "r_hat_max" in snr_subset.columns else 0
            status = "CONVERGED" if converged else f"R̂={r_hat:.2f}"
            color = "green" if converged else "red"
            ax.text(0.02, 0.95, status, transform=ax.transAxes,
                    fontsize=8, va="top", color=color, fontweight="bold")

            snr_label = f"SNR={snr}" + (" (lowest)" if j == 0 else " (highest)")
            ax.set_title(f"{spec_name} | {snr_label}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d:.2f}" for d in diff_values], fontsize=7, rotation=45)
            ax.set_ylabel("Fraction")
            ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Spectrum Recovery: True vs NUTS Estimate\n"
        "Left = lowest SNR, Right = highest SNR",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


def plot_convergence_summary(results_df, output_path=None):
    """
    Summary plot of convergence diagnostics across all experiments.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: R-hat vs SNR
    ax = axes[0]
    for spec_name in results_df["spectrum_name"].unique():
        subset = results_df[results_df["spectrum_name"] == spec_name]
        avg = subset.groupby("snr")["r_hat_max"].mean()
        ax.plot(avg.index, avg.values, "o-", label=spec_name, markersize=5)
    ax.axhline(y=1.05, color="red", linestyle="--", alpha=0.5, label="R̂ < 1.05 threshold")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Max R̂")
    ax.set_title("Convergence (R̂) vs SNR")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # Panel 2: RMSE vs SNR
    ax = axes[1]
    for spec_name in results_df["spectrum_name"].unique():
        subset = results_df[results_df["spectrum_name"] == spec_name]
        avg = subset.groupby("snr")["nuts_rmse"].mean()
        ax.plot(avg.index, avg.values, "o-", label=spec_name, markersize=5)
    ax.set_xlabel("SNR")
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title("Recovery Error vs SNR")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Panel 3: Runtime vs SNR
    ax = axes[2]
    for spec_name in results_df["spectrum_name"].unique():
        subset = results_df[results_df["spectrum_name"] == spec_name]
        avg = subset.groupby("snr")["elapsed_sec"].mean()
        ax.plot(avg.index, avg.values, "o-", label=spec_name, markersize=5)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("NUTS Runtime vs SNR")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    fig.suptitle(
        "NUTS Robustness: Convergence, Accuracy, and Efficiency",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness test for NUTS sampler")
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced NUTS parameters for quick testing")
    parser.add_argument("--map-only", action="store_true",
                        help="Only run MAP estimation (skip NUTS)")
    parser.add_argument("--snr", nargs="+", type=int, default=None,
                        help="Subset of SNR levels to test")
    parser.add_argument("--realizations", type=int, default=None,
                        help="Number of noise realizations")
    args = parser.parse_args()

    mode = "fast" if args.fast else "paper"
    nuts_config = NUTS_CONFIG[mode]
    snr_levels = args.snr if args.snr else SNR_LEVELS
    n_realizations = args.realizations if args.realizations else N_REALIZATIONS
    use_nuts = not args.map_only

    print("=" * 70)
    print("ROBUSTNESS TEST: NUTS Solver Performance")
    print("=" * 70)
    print(f"  Mode: {mode}")
    print(f"  NUTS config: {nuts_config}")
    print(f"  SNR levels: {snr_levels}")
    print(f"  Realizations: {n_realizations}")
    print(f"  Use NUTS: {use_nuts}")

    # Define test spectra
    print("\n[Step 1] Defining test spectra...")
    test_spectra = define_test_spectra(DIFF_VALUES)
    for name, spec in test_spectra.items():
        print(f"  {name}: max={spec.max():.3f} at D={DIFF_VALUES[spec.argmax()]:.2f}")

    # Run experiment
    print(f"\n[Step 2] Running experiment ({len(test_spectra)} spectra x "
          f"{len(snr_levels)} SNRs x {n_realizations} realizations "
          f"= {len(test_spectra)*len(snr_levels)*n_realizations} runs)...")

    results_df = run_full_experiment(
        test_spectra, snr_levels, DIFF_VALUES, B_VALUES,
        RIDGE_STRENGTH, nuts_config, n_realizations, use_nuts=use_nuts
    )

    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, "robustness_results.csv"), index=False)
    print(f"\n  Saved results to: {os.path.join(OUTPUT_DIR, 'robustness_results.csv')}")

    # Generate figures
    print("\n[Step 3] Creating visualizations...")

    if use_nuts:
        plot_robustness_heatmap(
            results_df, metric="nuts_rmse",
            output_path=os.path.join(OUTPUT_DIR, "heatmap_rmse.png"),
        )
        plot_robustness_heatmap(
            results_df, metric="r_hat_max",
            output_path=os.path.join(OUTPUT_DIR, "heatmap_rhat.png"),
        )
        plot_robustness_heatmap(
            results_df, metric="nuts_coverage",
            output_path=os.path.join(OUTPUT_DIR, "heatmap_coverage.png"),
        )
        plot_spectrum_recovery(
            results_df, DIFF_VALUES,
            output_path=os.path.join(OUTPUT_DIR, "spectrum_recovery.png"),
        )
        plot_convergence_summary(
            results_df,
            output_path=os.path.join(OUTPUT_DIR, "convergence_summary.png"),
        )

    # Print summary table
    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST SUMMARY")
    print("=" * 70)

    if use_nuts:
        summary = results_df.groupby(["spectrum_name", "snr"]).agg(
            rmse_mean=("nuts_rmse", "mean"),
            rmse_std=("nuts_rmse", "std"),
            rhat_mean=("r_hat_max", "mean"),
            converged_pct=("converged", "mean"),
            runtime_mean=("elapsed_sec", "mean"),
        ).reset_index()

        print(f"\n{'Spectrum':<22} {'SNR':>5} {'RMSE':>10} {'R̂_max':>8} {'Conv%':>6} {'Time(s)':>8}")
        print("-" * 62)
        for _, row in summary.iterrows():
            print(f"{row['spectrum_name']:<22} {row['snr']:>5} "
                  f"{row['rmse_mean']:>8.4f}±{row['rmse_std']:.4f} "
                  f"{row['rhat_mean']:>8.3f} "
                  f"{row['converged_pct']*100:>5.0f}% "
                  f"{row['runtime_mean']:>7.1f}")

    print(f"\nAll outputs in: {OUTPUT_DIR}")
