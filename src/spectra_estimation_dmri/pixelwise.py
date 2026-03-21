"""
Pixel-wise diffusivity spectrum estimation.

Pure functions operating on numpy arrays — no config objects, no dataclass wrappers.
Consistent normalization: all fitting is done on S/S0 ∈ [0, 1].

Usage:
    dwi = load_prostate_dwi()
    mask = load_mask(...)
    signals, coords = dwi.pixel_signal_array(mask)

    adc = compute_adc(signals, B_VALUES_S_MM2)
    spectra = compute_map_spectra(signals, U, ridge_strength=0.1)
    nuts_results = run_nuts_all(signals, U, ...)

    adc_map = assemble_map(adc, coords, (64, 64))
"""

import numpy as np
from pathlib import Path
from typing import Optional

try:
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------
#TODO: make sure to compute the design matrix in the same way as the one used in the BWH dataset, to ensure that the results are comparable.
def build_design_matrix(b_values: np.ndarray, diffusivities: np.ndarray) -> np.ndarray:
    """U[i,j] = exp(-b_i * d_j).  Shape: (n_b, n_d)."""
    return np.exp(-np.outer(b_values, diffusivities))


# ---------------------------------------------------------------------------
# ADC — vectorized across all pixels
# ---------------------------------------------------------------------------

def compute_adc(
    signals: np.ndarray,
    b_values_s_mm2: np.ndarray,
    b_max_s_mm2: float = 1250.0,
) -> np.ndarray:
    """Monoexponential ADC fit per pixel via least squares.

    Model: log(S) = log(S0) - b * ADC

    Args:
        signals: (n_pixels, n_b) raw signal intensities.
        b_values_s_mm2: (n_b,) b-values in s/mm².
        b_max_s_mm2: upper b-value cutoff for fitting.

    Returns:
        adc: (n_pixels,) ADC in mm²/s.
    """
    mask = b_values_s_mm2 <= b_max_s_mm2
    b_fit = b_values_s_mm2[mask]
    S_fit = np.maximum(signals[:, mask], 1e-10)
    log_S = np.log(S_fit)

    # Design matrix for log-linear fit: [1, -b] @ [log(S0), ADC]^T = log(S)
    A = np.column_stack([np.ones_like(b_fit), -b_fit])
    # Solve for all pixels: A @ params = log_S^T  →  params shape (2, n_pixels)
    params, _, _, _ = np.linalg.lstsq(A, log_S.T, rcond=None)
    return params[1]  # ADC for each pixel


# ---------------------------------------------------------------------------
# MAP (Ridge NNLS) — closed-form, vectorized across all pixels
# ---------------------------------------------------------------------------

def compute_map_spectra(
    signals: np.ndarray,
    U: np.ndarray,
    ridge_strength: float = 0.1,
) -> np.ndarray:
    """Closed-form Ridge regression with non-negativity projection.

    R = clip((U'U + λI)^{-1} U' S_norm, min=0)

    Normalizes each pixel by its S(b=0) before fitting, so output
    spectra are in fractional units consistent with NUTS.

    Args:
        signals: (n_pixels, n_b) raw signal intensities.
        U: (n_b, n_d) design matrix from build_design_matrix().
        ridge_strength: L2 regularization λ.

    Returns:
        spectra: (n_pixels, n_d) non-negative spectrum per pixel.
    """
    S0 = np.maximum(signals[:, 0:1], 1e-10)
    signals_norm = signals / S0

    n_d = U.shape[1]
    # (U'U + λI)^{-1} U'  →  projection matrix (n_d, n_b)
    A = U.T @ U + ridge_strength * np.eye(n_d)
    projection = np.linalg.solve(A, U.T)

    # Apply to all pixels: (n_d, n_b) @ (n_b, n_pixels) → (n_d, n_pixels)
    spectra = (projection @ signals_norm.T).T
    return np.maximum(spectra, 0.0)


# ---------------------------------------------------------------------------
# NUTS — per pixel
# ---------------------------------------------------------------------------

def run_nuts_pixel(
    signal: np.ndarray,
    U: np.ndarray,
    ridge_strength: float = 0.1,
    n_draws: int = 2000,
    n_tune: int = 200,
    n_chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
) -> dict:
    """Run NUTS for a single pixel.

    Same model as the existing NUTSSampler:
        R ~ HalfNormal(σ_R = 1/√λ)
        σ ~ HalfCauchy(β = 0.01)
        S_obs ~ Normal(U @ R, σ)

    Normalizes signal by S(b=0) before fitting.

    Args:
        signal: (n_b,) raw signal for one pixel.
        U: (n_b, n_d) design matrix.
        ridge_strength: prior strength λ → σ_R = 1/√λ.
        n_draws, n_tune, n_chains, target_accept: NUTS parameters.
        random_seed: for reproducibility.

    Returns:
        dict with keys:
            spectrum_mean: (n_d,) posterior mean of R
            spectrum_std:  (n_d,) posterior std of R
            sigma_mean:    float, posterior mean of noise σ
            sigma_std:     float, posterior std of noise σ
            snr:           float, 1 / sigma_mean
            r_hat_max:     float, worst-case convergence diagnostic
    """
    if not PYMC_AVAILABLE:
        raise ImportError("PyMC is required for NUTS. Install with: uv add pymc")

    n_d = U.shape[1]
    S0 = signal[0] if signal[0] > 0 else 1.0
    signal_norm = signal / S0

    sigma_R = 1.0 / np.sqrt(ridge_strength) if ridge_strength > 0 else 10.0

    with pm.Model():
        R = pm.HalfNormal("R", sigma=sigma_R, shape=n_d)
        sigma = pm.HalfCauchy("sigma", beta=0.01)
        mu = pm.math.dot(U, R)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=signal_norm)

        idata = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            progressbar=False,
            random_seed=random_seed,
        )

    R_samples = idata.posterior["R"].values
    n_ch, n_dr, _ = R_samples.shape
    R_flat = R_samples.reshape(n_ch * n_dr, n_d)

    sigma_samples = idata.posterior["sigma"].values.flatten()

    import arviz as az

    summary = az.summary(idata, var_names=["R", "sigma"])
    r_hat_max = float(summary["r_hat"].max())

    return {
        "spectrum_mean": R_flat.mean(axis=0),
        "spectrum_std": R_flat.std(axis=0),
        "sigma_mean": float(sigma_samples.mean()),
        "sigma_std": float(sigma_samples.std()),
        "snr": float(1.0 / sigma_samples.mean()),
        "r_hat_max": r_hat_max,
    }


def run_nuts_all(
    signals: np.ndarray,
    U: np.ndarray,
    ridge_strength: float = 0.1,
    n_draws: int = 2000,
    n_tune: int = 200,
    n_chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
    checkpoint_path: Optional[str] = None,
) -> dict:
    """Run NUTS on all pixels with progress reporting and checkpointing.

    Args:
        signals: (n_pixels, n_b) raw signal intensities.
        U: (n_b, n_d) design matrix.
        checkpoint_path: if set, saves intermediate results as .npz every 10 pixels.
        (other args same as run_nuts_pixel)

    Returns:
        dict with keys (each is a numpy array over pixels):
            spectrum_mean: (n_pixels, n_d)
            spectrum_std:  (n_pixels, n_d)
            sigma_mean:    (n_pixels,)
            sigma_std:     (n_pixels,)
            snr:           (n_pixels,)
            r_hat_max:     (n_pixels,)
    """
    import time

    n_pixels, n_b = signals.shape
    n_d = U.shape[1]

    spectrum_mean = np.zeros((n_pixels, n_d))
    spectrum_std = np.zeros((n_pixels, n_d))
    sigma_mean = np.zeros(n_pixels)
    sigma_std = np.zeros(n_pixels)
    snr = np.zeros(n_pixels)
    r_hat_max = np.zeros(n_pixels)

    # Resume from checkpoint if it exists
    start_idx = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = np.load(checkpoint_path)
        start_idx = int(ckpt["completed"])
        spectrum_mean[:start_idx] = ckpt["spectrum_mean"][:start_idx]
        spectrum_std[:start_idx] = ckpt["spectrum_std"][:start_idx]
        sigma_mean[:start_idx] = ckpt["sigma_mean"][:start_idx]
        sigma_std[:start_idx] = ckpt["sigma_std"][:start_idx]
        snr[:start_idx] = ckpt["snr"][:start_idx]
        r_hat_max[:start_idx] = ckpt["r_hat_max"][:start_idx]
        print(f"Resumed from checkpoint: {start_idx}/{n_pixels} pixels done")

    t_start = time.time()
    for i in range(start_idx, n_pixels):
        t_pixel = time.time()
        result = run_nuts_pixel(
            signals[i], U, ridge_strength,
            n_draws, n_tune, n_chains, target_accept, random_seed,
        )
        spectrum_mean[i] = result["spectrum_mean"]
        spectrum_std[i] = result["spectrum_std"]
        sigma_mean[i] = result["sigma_mean"]
        sigma_std[i] = result["sigma_std"]
        snr[i] = result["snr"]
        r_hat_max[i] = result["r_hat_max"]

        elapsed = time.time() - t_pixel
        total_elapsed = time.time() - t_start
        done = i - start_idx + 1
        avg = total_elapsed / done
        remaining = avg * (n_pixels - i - 1)
        converged = "ok" if result["r_hat_max"] < 1.05 else "WARN"
        print(
            f"[{i+1}/{n_pixels}] {elapsed:.1f}s | "
            f"avg {avg:.1f}s/px | ETA {remaining/60:.0f}min | "
            f"R-hat {result['r_hat_max']:.3f} ({converged})"
        )

        if checkpoint_path and (i + 1) % 10 == 0:
            _save_checkpoint(
                checkpoint_path, i + 1,
                spectrum_mean, spectrum_std,
                sigma_mean, sigma_std, snr, r_hat_max,
            )

    # Final save
    if checkpoint_path:
        _save_checkpoint(
            checkpoint_path, n_pixels,
            spectrum_mean, spectrum_std,
            sigma_mean, sigma_std, snr, r_hat_max,
        )

    return {
        "spectrum_mean": spectrum_mean,
        "spectrum_std": spectrum_std,
        "sigma_mean": sigma_mean,
        "sigma_std": sigma_std,
        "snr": snr,
        "r_hat_max": r_hat_max,
    }


def _save_checkpoint(path, completed, spectrum_mean, spectrum_std,
                     sigma_mean, sigma_std, snr, r_hat_max):
    np.savez(
        path,
        completed=completed,
        spectrum_mean=spectrum_mean,
        spectrum_std=spectrum_std,
        sigma_mean=sigma_mean,
        sigma_std=sigma_std,
        snr=snr,
        r_hat_max=r_hat_max,
    )
    print(f"  Checkpoint saved: {completed} pixels → {path}")


# ---------------------------------------------------------------------------
# Tumor probability via logistic regression
# ---------------------------------------------------------------------------

def train_tumor_lr(
    signal_decay_dataset,
    U: np.ndarray,
    diffusivities: np.ndarray,
    ridge_strength: float = 0.1,
    zone: str = "pz",
    C: float = 1.0,
) -> dict:
    """Train a logistic regression on ROI-level spectra for tumor detection.

    Computes MAP spectra from signal decays, then trains LR on the
    normalized spectra. Trains on ALL ROIs in the specified zone.

    Args:
        signal_decay_dataset: SignalDecayDataset from load_bwh_signal_decays().
        U: (n_b, n_d) design matrix.
        diffusivities: (n_d,) diffusivity bin values.
        ridge_strength: for MAP spectrum computation.
        zone: 'pz' or 'tz' — anatomical zone to train on.
        C: inverse regularization strength.

    Returns:
        dict with 'scaler', 'model', 'diffusivities', 'auc_loocv', etc.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score

    features_list = []
    labels = []

    for sd in signal_decay_dataset.samples:
        region = getattr(sd, "a_region", "")
        if zone not in str(region).lower():
            continue

        sig = np.array(sd.signal_values).reshape(1, -1)
        spec = compute_map_spectra(sig, U, ridge_strength)[0]
        spec_norm = spec / (spec.sum() + 1e-10)
        features_list.append(spec_norm)
        labels.append(1 if sd.is_tumor else 0)

    X = np.array(features_list)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(X_scaled, y)

    # Quick LOOCV AUC for reference
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X_scaled):
        m = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
        m.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = m.predict_proba(X_scaled[test_idx])[:, 1]
    auc = roc_auc_score(y, y_pred) if len(np.unique(y)) > 1 else np.nan

    print(f"Trained tumor LR ({zone}): {len(y)} ROIs, "
          f"{y.sum()} tumor, AUC(LOOCV)={auc:.3f}")
    print(f"  Coefficients: {dict(zip([f'D={d}' for d in diffusivities], model.coef_[0].round(3)))}")

    return {
        "scaler": scaler,
        "model": model,
        "diffusivities": diffusivities,
        "auc_loocv": auc,
        "coef": model.coef_[0],
        "intercept": model.intercept_[0],
        "n_samples": len(y),
        "n_tumor": int(y.sum()),
    }


def compute_tumor_probability(
    spectra: np.ndarray,
    lr_result: dict,
) -> np.ndarray:
    """Apply trained LR to pixel spectra to get P(tumor).

    Args:
        spectra: (n_pixels, n_d) spectra (from MAP or NUTS mean).
            Will be normalized to sum-to-1 internally.
        lr_result: output from train_tumor_lr().

    Returns:
        prob: (n_pixels,) tumor probability per pixel.
    """
    spec_norm = spectra / (spectra.sum(axis=1, keepdims=True) + 1e-10)
    X_scaled = lr_result["scaler"].transform(spec_norm)
    return lr_result["model"].predict_proba(X_scaled)[:, 1]


def compute_tumor_probability_with_uncertainty(
    nuts_results: dict,
    lr_result: dict,
    n_mc: int = 200,
    random_seed: int = 42,
) -> tuple:
    """Propagate NUTS posterior uncertainty through the LR classifier.

    For each pixel, draw MC samples from a Gaussian approximation
    of the posterior (mean ± std), run each through the LR, and
    compute mean and std of P(tumor).

    Args:
        nuts_results: output from run_nuts_all().
        lr_result: output from train_tumor_lr().
        n_mc: number of MC samples per pixel.
        random_seed: for reproducibility.

    Returns:
        (prob_mean, prob_std): each (n_pixels,).
    """
    rng = np.random.RandomState(random_seed)
    mean = nuts_results["spectrum_mean"]
    std = nuts_results["spectrum_std"]
    n_pixels, n_d = mean.shape

    prob_mean = np.zeros(n_pixels)
    prob_std = np.zeros(n_pixels)

    for i in range(n_pixels):
        samples = np.maximum(
            rng.normal(mean[i], std[i], size=(n_mc, n_d)), 0.0
        )
        probs = compute_tumor_probability(samples, lr_result)
        prob_mean[i] = probs.mean()
        prob_std[i] = probs.std()

    return prob_mean, prob_std


# ---------------------------------------------------------------------------
# Map assembly
# ---------------------------------------------------------------------------

def assemble_map(
    pixel_values: np.ndarray,
    coords: np.ndarray,
    image_shape: tuple,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Scatter pixel-level values back into a 2D spatial map.

    Args:
        pixel_values: (n_pixels,) or (n_pixels, n_channels).
        coords: (n_pixels, 2) row/col coordinates.
        image_shape: (rows, cols) of the output map.
        fill_value: value for pixels outside the mask.

    Returns:
        If pixel_values is 1D: (rows, cols) array.
        If pixel_values is 2D: (rows, cols, n_channels) array.
    """
    if pixel_values.ndim == 1:
        out = np.full(image_shape, fill_value)
        out[coords[:, 0], coords[:, 1]] = pixel_values
    else:
        n_channels = pixel_values.shape[1]
        out = np.full((*image_shape, n_channels), fill_value)
        out[coords[:, 0], coords[:, 1], :] = pixel_values
    return out
