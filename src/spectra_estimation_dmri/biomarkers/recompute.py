"""
Clean recomputation of ALL paper numbers from source data.

Produces:
- MAP features (Ridge NNLS from raw signal)
- NUTS features (posterior means from 149 .nc files)
- ADC (monoexponential fit, b <= 1000 s/mm²)
- LOOCV classification AUCs at multiple C values
- MAP vs NUTS comparison statistics
- Definitive features.csv and auc_table.csv

Usage:
    uv run python -m spectra_estimation_dmri.biomarkers.recompute
"""

import json
import hashlib
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants (must match configs exactly)
# ---------------------------------------------------------------------------

B_VALUES_MS = np.array([0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75,
                         2., 2.25, 2.5, 2.75, 3., 3.25, 3.5])
B_VALUES_S_MM2 = B_VALUES_MS * 1000  # [0, 250, ..., 3500]
DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
D_COLS = [f"D_{d:.2f}" for d in DIFFUSIVITIES]
# MAP ridge regularizer, tuned via simulation sweep (scripts/map_lambda_sweep.py).
# Original manuscript value was 0.1; lambda=1e-3 better matches NUTS on prostate-realistic
# log-normal ground truths (see F1 / F-new-1).
RIDGE_STRENGTH = 1e-3

# Config dicts for spectra_id hash (must match what generated the .nc files).
# Do NOT change strength here — it is the lookup key for the existing NUTS .nc files
# computed at the original prior strength. NUTS is robust to wider priors (F8).
_LIKELIHOOD_CFG = {"type": "gaussian"}
_PRIOR_CFG = {"type": "ridge", "strength": 0.1, "nonnegative": True}
_INFERENCE_CFG = {
    "name": "nuts", "n_iter": 2000, "tune": 200, "n_chains": 4,
    "target_accept": 0.95, "sampler_snr": None, "init": "map",
}

# Default paths (relative to project root)
SIGNAL_JSON = "src/spectra_estimation_dmri/data/bwh/signal_decays.json"
METADATA_CSV = "src/spectra_estimation_dmri/data/bwh/metadata.csv"
NC_DIR = "results/inference_bwh_backup"
OUTPUT_DIR = "results/biomarkers"


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def build_design_matrix() -> np.ndarray:
    """U[i,j] = exp(-b_i * d_j). Shape: (15, 8)."""
    return np.exp(-np.outer(B_VALUES_MS, DIFFUSIVITIES))


def compute_map_spectrum(signal: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Constrained ridge MAP estimate: argmin_{R >= 0} ||U R - s/S0||^2 + lambda ||R||^2.

    Solved as NNLS on the augmented system [U; sqrt(lambda) I] R = [s/S0; 0].
    Projecting an unconstrained Gaussian MAP onto the non-negative orthant is NOT
    the same as the constrained MAP whenever the unconstrained optimum is infeasible
    (Sandy 2026-05-25 counter-example).
    """
    from scipy.optimize import nnls
    S0 = signal[0] if signal[0] > 0 else 1.0
    s_norm = signal / S0
    n_d = U.shape[1]
    U_aug = np.vstack([U, np.sqrt(RIDGE_STRENGTH) * np.eye(n_d)])
    s_aug = np.concatenate([s_norm, np.zeros(n_d)])
    spectrum, _ = nnls(U_aug, s_aug)
    return spectrum


def compute_adc(signal: np.ndarray, b_max_s_mm2: float = 1000.0) -> float:
    """Monoexponential ADC from raw signal.

    Args:
        signal: (15,) raw signal intensities (same order as B_VALUES_S_MM2).
        b_max_s_mm2: upper b-value cutoff in s/mm².

    Returns:
        ADC in mm²/s (= μm²/ms × 1e-3).
    """
    mask = B_VALUES_S_MM2 <= b_max_s_mm2
    S_fit = signal[mask]
    valid = S_fit > 0
    if valid.sum() < 2:
        return np.nan
    log_S = np.log(S_fit[valid])
    b_fit = B_VALUES_S_MM2[mask][valid]
    slope, _ = np.polyfit(b_fit, log_S, 1)
    return -slope


def compute_spectra_id(signal_values, b_values, snr) -> str:
    """Reproduce the MD5 hash used to name .nc files."""
    def to_serializable(obj):
        return obj.tolist() if hasattr(obj, "tolist") else obj

    hash_dict = {
        "signal_values": to_serializable(signal_values),
        "b_values": to_serializable(b_values),
        "snr": snr,
        "likelihood": _LIKELIHOOD_CFG,
        "prior": _PRIOR_CFG,
        "inference": _INFERENCE_CFG,
        "spectrum_pair": None,
    }
    json_repr = json.dumps(hash_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(json_repr.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(signal_json: str = SIGNAL_JSON,
                 metadata_csv: str = METADATA_CSV) -> list[dict]:
    """Load raw signal data + metadata, return list of ROI dicts.

    Each dict has: patient, region, is_tumor, ggg, gs, signal, b_values, snr, roi_id.
    """
    with open(signal_json) as f:
        signal_data = json.load(f)

    metadata = {}
    with open(metadata_csv, newline="") as f:
        import csv
        for row in csv.DictReader(f):
            metadata[row["patient_id"]] = row

    rois = []
    for patient_id, patient_rois in signal_data.items():
        meta = metadata.get(patient_id, {})
        gs = meta.get("gs") if meta else None
        ggg = None
        try:
            targets = meta.get("targets", "")
            ggg = int(targets) if targets.isdigit() else None
        except Exception:
            pass

        for roi_name, roi in patient_rois.items():
            anat = roi["anatomical_region"]
            is_tumor = "tumor" in anat
            if "pz" in anat:
                region = "pz"
            elif "tz" in anat:
                region = "tz"
            else:
                continue  # skip "Neglected!" regions

            voxel_count = roi["v_count"]
            snr = float(np.sqrt(voxel_count / 16) * 150)
            roi_id = f"{patient_id}_{region}_{'tumor' if is_tumor else 'normal'}"

            rois.append({
                "patient": patient_id,
                "region": region,
                "is_tumor": is_tumor,
                "ggg": ggg,
                "gs": gs,
                "signal": np.array(roi["signal_values"]),
                "b_values": roi["b_values"],
                "snr": snr,
                "voxel_count": voxel_count,
                "roi_id": roi_id,
            })

    return rois


def load_nuts_posteriors(rois: list[dict], nc_dir: str = NC_DIR) -> dict:
    """Load NUTS posterior means and stds from .nc files.

    Returns dict: roi_id -> {"mean": (8,), "std": (8,), "sigma_mean": float}.
    """
    import arviz as az

    var_names = [f"diff_{d:.2f}" for d in DIFFUSIVITIES]
    results = {}

    for roi in rois:
        spectra_id = compute_spectra_id(roi["signal"], roi["b_values"], roi["snr"])
        nc_path = os.path.join(nc_dir, f"{spectra_id}.nc")

        if not os.path.exists(nc_path):
            print(f"  [WARN] Missing .nc for {roi['roi_id']}: {nc_path}")
            continue

        idata = az.from_netcdf(nc_path)

        # Extract posterior samples for each diffusivity component
        samples_list = []
        for vn in var_names:
            var_samples = idata.posterior[vn].values  # (chains, draws)
            samples_list.append(var_samples.flatten())
        samples = np.column_stack(samples_list)  # (n_total, 8)

        # Normalize each sample to sum to 1 (fractional spectrum)
        row_sums = samples.sum(axis=1, keepdims=True)
        samples_norm = samples / np.maximum(row_sums, 1e-10)

        sigma_samples = idata.posterior["sigma"].values.flatten()

        results[roi["roi_id"]] = {
            "mean": samples_norm.mean(axis=0),
            "std": samples_norm.std(axis=0),
            "sigma_mean": float(sigma_samples.mean()),
            "sigma_std": float(sigma_samples.std()),
        }

    return results


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def build_features_df(rois: list[dict], nuts_posteriors: dict,
                      U: np.ndarray) -> pd.DataFrame:
    """Build comprehensive features DataFrame with MAP, NUTS, and ADC.

    Columns:
        roi_id, patient, region, is_tumor, ggg, gs, zone,
        map_D_0.25 .. map_D_20.00 (MAP fractions),
        nuts_D_0.25 .. nuts_D_20.00 (NUTS posterior mean fractions),
        nuts_std_D_0.25 .. nuts_std_D_20.00 (NUTS posterior stds),
        adc (monoexponential, b<=1000 s/mm²)
    """
    rows = []
    for roi in rois:
        signal = roi["signal"]

        # MAP spectrum
        map_spec = compute_map_spectrum(signal, U)
        map_norm = map_spec / (map_spec.sum() + 1e-10)

        # NUTS spectrum
        nuts_data = nuts_posteriors.get(roi["roi_id"])
        if nuts_data is None:
            nuts_mean = np.full(8, np.nan)
            nuts_std = np.full(8, np.nan)
        else:
            nuts_mean = nuts_data["mean"]
            nuts_std = nuts_data["std"]

        # ADC
        adc = compute_adc(signal, b_max_s_mm2=1000.0)

        # Zone
        zone = "pz" if roi["region"] == "pz" else "tz"

        row = {
            "roi_id": roi["roi_id"],
            "patient": roi["patient"],
            "region": roi["region"],
            "is_tumor": roi["is_tumor"],
            "ggg": roi["ggg"],
            "gs": roi["gs"],
            "zone": zone,
            "adc": adc,
        }

        for i, d in enumerate(DIFFUSIVITIES):
            col = f"D_{d:.2f}"
            row[f"map_{col}"] = map_norm[i]
            row[f"nuts_{col}"] = nuts_mean[i]
            row[f"nuts_std_{col}"] = nuts_std[i]

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def loocv_auc(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> tuple:
    """LOOCV logistic regression AUC.

    Returns (auc, y_pred_proba, coefs_full_model).
    """
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_train, y[train_idx])
        y_pred[test_idx] = clf.predict_proba(X_test)[0, 1]

    auc = roc_auc_score(y, y_pred) if len(np.unique(y)) > 1 else np.nan

    # Full-data model for coefficients
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
    clf_full.fit(X_full, y)

    return auc, y_pred, clf_full.coef_[0], clf_full.intercept_[0], scaler_full


def raw_rank_auc(feature: np.ndarray, y: np.ndarray) -> float:
    """AUC using raw feature values as scores (no classifier training)."""
    auc = roc_auc_score(y, feature)
    return max(auc, 1 - auc)  # handle inverse correlation


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the AUC of a fixed score vector.

    Resamples (label, score) pairs. This is the project-wide interval method
    (matches Fig 2 / Fig 3, which also bootstrap); the older Hanley--McNeil
    analytic SE is retired. Paired AUC comparisons use DeLong's test (see
    scripts/two_feature_lr_vs_adc.py).
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) > 1:
            aucs.append(roc_auc_score(y[idx], score[idx]))
    if not aucs:
        return np.nan, np.nan
    return (float(np.percentile(aucs, 100 * alpha / 2)),
            float(np.percentile(aucs, 100 * (1 - alpha / 2))))


def run_classification(df: pd.DataFrame, C_values: list[float] = [0.1, 1.0, 10.0],
                       ) -> pd.DataFrame:
    """Run all classification tasks and return AUC table.

    Tasks: PZ tumor detection, TZ tumor detection, GGG grading.
    Methods: ADC (raw rank), ADC (LR), MAP Full LR, NUTS Full LR,
             MAP 2-feat {D=0.25,3.0}, NUTS 2-feat {D=0.25,3.0},
             MAP 6-inner, NUTS 6-inner.

    The 2-feature rows use only the two outer compartments (restricted
    D=0.25 + free-water D=3.0); they test the "collapse" claim (Fig 2) that
    detection needs only these two bins. The 6-inner rows are the MIRROR
    ablation: the six intermediate/dump fractions with the two outer bins
    REMOVED -- they show the intermediate bins remain individually informative
    (AUC ~0.78-0.82) yet redundant once the outer bins are present, i.e.
    "redundancy, not uselessness". AUCs carry percentile bootstrap 95% CIs
    (auc_lo, auc_hi).
    """
    map_cols = [f"map_{c}" for c in D_COLS]
    nuts_cols = [f"nuts_{c}" for c in D_COLS]
    # Outer-bin "collapse" pair: D=0.25 (restricted) and D=3.00 (free water).
    outer_idx = [0, 6]
    map_outer = [map_cols[i] for i in outer_idx]
    nuts_outer = [nuts_cols[i] for i in outer_idx]
    # Mirror ablation: the six inner fractions (everything EXCEPT the two outer
    # compartments) -> D in {0.50, 0.75, 1.00, 1.50, 2.00, 20.00}.
    inner_idx = [1, 2, 3, 4, 5, 7]
    map_inner = [map_cols[i] for i in inner_idx]
    nuts_inner = [nuts_cols[i] for i in inner_idx]
    # "Redundancy, not uselessness" probe: the four MOST poorly-identified bins
    # (within-ROI posterior CV > 0.7) -> D in {0.50, 0.75, 1.00, 1.50}. Used
    # alone they still detect tumor at AUC ~0.81-0.82 (cited in Results; not a
    # Table 1 row). Confirms the inner bins are individually informative.
    poorid_idx = [1, 2, 3, 4]
    map_poorid = [map_cols[i] for i in poorid_idx]
    nuts_poorid = [nuts_cols[i] for i in poorid_idx]

    results = []

    tasks = {
        "PZ": {"filter": df["zone"] == "pz", "label_col": "is_tumor"},
        "TZ": {"filter": df["zone"] == "tz", "label_col": "is_tumor"},
    }

    # GGG task: tumor ROIs with known GGG, binary split at GGG >= 3
    ggg_mask = (df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)
    tasks["GGG"] = {"filter": ggg_mask, "label_col": "_ggg_binary"}

    for task_name, task_cfg in tasks.items():
        task_df = df[task_cfg["filter"]].copy()

        if task_name == "GGG":
            task_df["_ggg_binary"] = (task_df["ggg"] >= 3).astype(int)

        y = task_df[task_cfg["label_col"]].astype(int).values
        n_pos = y.sum()
        n_neg = len(y) - n_pos

        if len(np.unique(y)) < 2:
            print(f"  [SKIP] {task_name}: only one class (n={len(y)})")
            continue

        print(f"\n  {task_name}: n={len(y)} (pos={n_pos}, neg={n_neg})")

        # ADC raw rank AUC (directed score so the bootstrap CI is on the
        # tumor-positive orientation: lower ADC -> tumor).
        adc_vals = task_df["adc"].values
        adc_score = adc_vals if roc_auc_score(y, adc_vals) >= 0.5 else -adc_vals
        adc_raw_auc = roc_auc_score(y, adc_score)
        lo, hi = bootstrap_auc_ci(y, adc_score)
        results.append({"task": task_name, "method": "ADC (raw rank)",
                        "C": "-", "auc": adc_raw_auc,
                        "auc_lo": lo, "auc_hi": hi, "n": len(y)})
        print(f"    ADC raw rank:    {adc_raw_auc:.4f} [{lo:.3f}, {hi:.3f}]")

        # Per-bin single-feature raw-rank AUC (matches the individual-fraction
        # ROC curves in Fig 2; oriented so AUC >= 0.5, bootstrap CI on the
        # oriented score as for ADC raw rank above).
        for est, cols in [("MAP", map_cols), ("NUTS", nuts_cols)]:
            for c, col in zip(D_COLS, cols):
                feat = task_df[col].values
                score = feat if roc_auc_score(y, feat) >= 0.5 else -feat
                auc_bin = roc_auc_score(y, score)
                lo_b, hi_b = bootstrap_auc_ci(y, score)
                results.append({"task": task_name,
                                "method": f"{est} {c.replace('D_', 'D=')}",
                                "C": "-", "auc": auc_bin,
                                "auc_lo": lo_b, "auc_hi": hi_b, "n": len(y)})

        for C in C_values:
            # ADC via LR
            auc_adc, pred_adc, _, _, _ = loocv_auc(adc_vals.reshape(-1, 1), y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_adc)
            results.append({"task": task_name, "method": "ADC (LR)",
                            "C": C, "auc": auc_adc,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})

            # MAP Full LR (8 features)
            X_map = task_df[map_cols].values
            auc_map, pred_map, coefs_map, _, _ = loocv_auc(X_map, y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_map)
            results.append({"task": task_name, "method": "MAP Full LR",
                            "C": C, "auc": auc_map,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})

            # NUTS Full LR (8 features)
            X_nuts = task_df[nuts_cols].values
            auc_nuts, pred_nuts, coefs_nuts, _, _ = loocv_auc(X_nuts, y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_nuts)
            results.append({"task": task_name, "method": "NUTS Full LR",
                            "C": C, "auc": auc_nuts,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})

            # MAP 2-feature {D=0.25, 3.0} (the "collapse" classifier)
            auc_map2, pred_map2, _, _, _ = loocv_auc(task_df[map_outer].values, y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_map2)
            results.append({"task": task_name, "method": "MAP 2-feat {0.25,3.0}",
                            "C": C, "auc": auc_map2,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})

            # NUTS 2-feature {D=0.25, 3.0}
            auc_nuts2, pred_nuts2, _, _, _ = loocv_auc(task_df[nuts_outer].values, y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_nuts2)
            results.append({"task": task_name, "method": "NUTS 2-feat {0.25,3.0}",
                            "C": C, "auc": auc_nuts2,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})

            # MAP 6-inner (mirror ablation: drop the two outer compartments)
            auc_map6, pred_map6, _, _, _ = loocv_auc(task_df[map_inner].values, y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_map6)
            results.append({"task": task_name, "method": "MAP 6-inner",
                            "C": C, "auc": auc_map6,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})

            # NUTS 6-inner (mirror ablation)
            auc_nuts6, pred_nuts6, _, _, _ = loocv_auc(task_df[nuts_inner].values, y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_nuts6)
            results.append({"task": task_name, "method": "NUTS 6-inner",
                            "C": C, "auc": auc_nuts6,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})

            # 4 most poorly-identified bins (CV>0.7) used alone -- redundancy probe
            auc_mapP, pred_mapP, _, _, _ = loocv_auc(task_df[map_poorid].values, y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_mapP)
            results.append({"task": task_name, "method": "MAP 4-poorly-id (CV>0.7)",
                            "C": C, "auc": auc_mapP,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})
            auc_nutsP, pred_nutsP, _, _, _ = loocv_auc(task_df[nuts_poorid].values, y, C=C)
            lo, hi = bootstrap_auc_ci(y, pred_nutsP)
            results.append({"task": task_name, "method": "NUTS 4-poorly-id (CV>0.7)",
                            "C": C, "auc": auc_nutsP,
                            "auc_lo": lo, "auc_hi": hi, "n": len(y)})

            print(f"    C={C:5.1f}  ADC LR={auc_adc:.4f}  "
                  f"MAP={auc_map:.4f} (2f {auc_map2:.4f} / 6in {auc_map6:.4f})  "
                  f"NUTS={auc_nuts:.4f} (2f {auc_nuts2:.4f} / 6in {auc_nuts6:.4f})")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# MAP vs NUTS comparison
# ---------------------------------------------------------------------------

def compare_map_nuts(df: pd.DataFrame) -> dict:
    """Compare MAP and NUTS features: per-component correlation, discriminant correlation."""
    map_cols = [f"map_{c}" for c in D_COLS]
    nuts_cols = [f"nuts_{c}" for c in D_COLS]

    comparison = {}

    # Per-component correlation across all 149 ROIs
    for mc, nc, d in zip(map_cols, nuts_cols, DIFFUSIVITIES):
        r, p = stats.pearsonr(df[mc].values, df[nc].values)
        comparison[f"r_D_{d:.2f}"] = r

    # Overall MAP vs NUTS spectrum correlation (flatten all)
    map_all = df[map_cols].values.flatten()
    nuts_all = df[nuts_cols].values.flatten()
    r_overall, _ = stats.pearsonr(map_all, nuts_all)
    comparison["r_overall"] = r_overall

    # Discriminant score comparison for PZ
    for zone in ["pz", "tz"]:
        zone_df = df[df["zone"] == zone]
        y = zone_df["is_tumor"].astype(int).values
        if len(np.unique(y)) < 2:
            continue

        X_map = zone_df[map_cols].values
        X_nuts = zone_df[nuts_cols].values

        _, _, _, map_int, map_scaler = loocv_auc(X_map, y, C=1.0)
        _, _, _, nuts_int, nuts_scaler = loocv_auc(X_nuts, y, C=1.0)

        # Get full-model discriminant scores
        scaler_m = StandardScaler()
        X_map_s = scaler_m.fit_transform(X_map)
        clf_m = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf_m.fit(X_map_s, y)
        disc_map = X_map_s @ clf_m.coef_[0] + clf_m.intercept_[0]

        scaler_n = StandardScaler()
        X_nuts_s = scaler_n.fit_transform(X_nuts)
        clf_n = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        clf_n.fit(X_nuts_s, y)
        disc_nuts = X_nuts_s @ clf_n.coef_[0] + clf_n.intercept_[0]

        r_disc, _ = stats.pearsonr(disc_map, disc_nuts)
        comparison[f"r_discriminant_{zone}"] = r_disc

    return comparison


# ---------------------------------------------------------------------------
# ADC vs discriminant correlation
# ---------------------------------------------------------------------------

def adc_discriminant_correlation(df: pd.DataFrame,
                                 C_values: list[float] = [1.0, 10.0]) -> pd.DataFrame:
    """Compute ADC vs spectral discriminant score correlation per zone."""
    results = []

    for feature_prefix in ["map", "nuts"]:
        feat_cols = [f"{feature_prefix}_{c}" for c in D_COLS]

        for zone in ["pz", "tz"]:
            zone_df = df[df["zone"] == zone]
            y = zone_df["is_tumor"].astype(int).values
            adc_vals = zone_df["adc"].values
            X = zone_df[feat_cols].values

            for C in C_values:
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)
                clf = LogisticRegression(C=C, max_iter=1000, random_state=42,
                                         solver="lbfgs")
                clf.fit(X_s, y)
                disc_scores = X_s @ clf.coef_[0] + clf.intercept_[0]

                r, p = stats.pearsonr(adc_vals, disc_scores)
                results.append({
                    "zone": zone.upper(),
                    "features": feature_prefix.upper(),
                    "C": C,
                    "r_adc_disc": r,
                    "p_value": p,
                    "n": len(y),
                })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# ADC sensitivity analysis
# ---------------------------------------------------------------------------

def compute_adc_from_spectrum(R: np.ndarray, b_max: float = 1.0) -> float:
    """ADC from spectral fractions via synthetic signal.

    Units: b in ms/um², D in um²/ms, ADC in um²/ms.
    """
    signal = np.zeros(len(B_VALUES_MS))
    for Rj, Dj in zip(R, DIFFUSIVITIES):
        signal += Rj * np.exp(-B_VALUES_MS * Dj)
    b_mask = B_VALUES_MS <= b_max
    log_signal = np.log(np.maximum(signal[b_mask], 1e-20))
    slope, _ = np.polyfit(B_VALUES_MS[b_mask], log_signal, 1)
    return -slope


def compute_sensitivity(spectrum: np.ndarray, b_max: float = 1.0,
                        epsilon: float = 0.0001) -> np.ndarray:
    """Numerical dADC/dR_j for each component."""
    adc_base = compute_adc_from_spectrum(spectrum, b_max)
    sensitivity = np.zeros(len(spectrum))
    for j in range(len(spectrum)):
        R_perturbed = spectrum.copy()
        R_perturbed[j] += epsilon
        sensitivity[j] = (compute_adc_from_spectrum(R_perturbed, b_max) - adc_base) / epsilon
    return sensitivity


def adc_sensitivity_analysis(df: pd.DataFrame,
                             C_values: list[float] = [1.0]) -> pd.DataFrame:
    """Compare ADC sensitivity vector with LR coefficient vector.

    For each (zone, feature_type, C, operating_point), computes:
    - dADC/dR_j (sensitivity at average tumor/normal spectrum)
    - LR coefs (standardized)
    - Pearson and Spearman correlation between them

    This is the VECTOR-LEVEL analysis (8 elements), distinct from the
    ROI-LEVEL ADC-discriminant correlation (N=81/68 elements).
    """
    results = []

    for prefix, label in [("map", "MAP"), ("nuts", "NUTS")]:
        feat_cols = [f"{prefix}_{c}" for c in D_COLS]

        for zone in ["pz", "tz"]:
            zone_df = df[df["zone"] == zone]
            y = zone_df["is_tumor"].astype(int).values
            X = zone_df[feat_cols].values
            if len(np.unique(y)) < 2:
                continue

            avg_tumor = X[y == 1].mean(axis=0)
            avg_normal = X[y == 0].mean(axis=0)

            for C in C_values:
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)
                clf = LogisticRegression(C=C, max_iter=1000, random_state=42,
                                         solver="lbfgs")
                clf.fit(X_s, y)
                coefs = clf.coef_[0]

                for name, spectrum in [("tumor", avg_tumor), ("normal", avg_normal)]:
                    sens = compute_sensitivity(spectrum, b_max=1.0)
                    r_pearson, p_pearson = stats.pearsonr(sens, coefs)
                    r_spearman, p_spearman = stats.spearmanr(sens, coefs)

                    results.append({
                        "zone": zone.upper(),
                        "features": label,
                        "C": C,
                        "operating_point": name,
                        "r_pearson": r_pearson,
                        "p_pearson": p_pearson,
                        "r_spearman": r_spearman,
                        "p_spearman": p_spearman,
                        "sensitivity": sens.tolist(),
                        "lr_coefs": coefs.tolist(),
                    })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Per-component identifiability
# ---------------------------------------------------------------------------

def component_identifiability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-component CV from NUTS posterior std."""
    rows = []
    for d in DIFFUSIVITIES:
        col = f"D_{d:.2f}"
        mean_frac = df[f"nuts_{col}"].mean()
        mean_std = df[f"nuts_std_{col}"].mean()
        mean_cv = (df[f"nuts_std_{col}"] / (df[f"nuts_{col}"] + 1e-10)).mean()
        rows.append({
            "component": col,
            "D_um2_ms": d,
            "mean_fraction": mean_frac,
            "mean_posterior_std": mean_std,
            "mean_CV": mean_cv,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Posterior-uncertainty propagation (the "Prediction confidence" claims)
# ---------------------------------------------------------------------------
# Canonical source for the manuscript's resolved-negative uncertainty result
# (results.tex "Prediction confidence"; discussion.tex; PROJECT_STATE S4/S11/
# S12/S13). The question: does propagating the NUTS posterior through the trained
# detector add diagnostic value beyond the point estimate? Answer: NO. These
# functions reproduce the exact numbers cited in the text so they are
# reproducible from the single source of truth rather than only inside the Fig 6
# plotting script (scripts/fig6_uncertainty_classifier.py).

def load_nuts_draws(rois: list[dict], nc_dir: str = NC_DIR) -> dict:
    """Load FULL NUTS posterior draws (not just mean/std).

    Mirrors load_nuts_posteriors, but keeps every draw so the posterior can be
    propagated through the classifier. Returns dict:
        roi_id -> (n_draws, 8) array, each draw normalized to a fractional
        spectrum (sums to 1), columns ordered by DIFFUSIVITIES.
    """
    import arviz as az

    var_names = [f"diff_{d:.2f}" for d in DIFFUSIVITIES]
    draws = {}
    for roi in rois:
        spectra_id = compute_spectra_id(roi["signal"], roi["b_values"], roi["snr"])
        nc_path = os.path.join(nc_dir, f"{spectra_id}.nc")
        if not os.path.exists(nc_path):
            continue
        idata = az.from_netcdf(nc_path)
        cols = [idata.posterior[vn].values.flatten() for vn in var_names]
        S = np.column_stack(cols)
        S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-10)
        draws[roi["roi_id"]] = S
    return draws


def _propagate_zone(df_zone: pd.DataFrame, draws: dict, C: float = 1.0,
                    ci: tuple = (5.0, 95.0)) -> pd.DataFrame:
    """Within-zone LOOCV detection LR (8 NUTS posterior-mean fractions),
    propagating each held-out ROI's full posterior draws through the fixed
    classifier to obtain the predictive distribution of P(tumor).

    Identical pipeline to scripts/fig6_uncertainty_classifier.py (C=1, 8 NUTS
    features, StandardScaler, seed 42, 90% CI) so the canonical numbers match
    Figure 6. Returns one row per ROI with:
        p_point  -- P(tumor) at the posterior-mean features (the Fig-2 label)
        ci_width -- width of the 90% credible interval on P(tumor) (prob space)
        z_std    -- std of the propagated draws in LOGIT space (removes the
                    sigmoid P(1-P) geometry; the "genuine" predictive spread)
        dist     -- |p_point - 0.5|, distance to the decision boundary
        correct  -- whether the point-estimate label is correct
    """
    nuts_cols = [f"nuts_{c}" for c in D_COLS]
    df_zone = df_zone[df_zone["roi_id"].isin(draws)].reset_index(drop=True)
    y = df_zone["is_tumor"].astype(int).values
    Xmean = df_zone[nuts_cols].values
    roi_ids = df_zone["roi_id"].values
    n = len(y)

    p_point = np.zeros(n)
    p_lo = np.zeros(n)
    p_hi = np.zeros(n)
    z_std = np.zeros(n)

    for tr, te in LeaveOneOut().split(Xmean):
        i = te[0]
        sc = StandardScaler().fit(Xmean[tr])
        clf = LogisticRegression(C=C, max_iter=2000, random_state=42,
                                 solver="lbfgs").fit(sc.transform(Xmean[tr]), y[tr])
        p_point[i] = clf.predict_proba(sc.transform(Xmean[i:i + 1]))[0, 1]
        probs = clf.predict_proba(sc.transform(draws[roi_ids[i]]))[:, 1]
        p_lo[i], p_hi[i] = np.percentile(probs, ci)
        z = np.log(np.clip(probs, 1e-6, 1 - 1e-6)
                   / np.clip(1 - probs, 1e-6, 1 - 1e-6))
        z_std[i] = z.std()

    return pd.DataFrame({
        "roi_id": roi_ids,
        "zone": df_zone["zone"].values,
        "y": y,
        "p_point": p_point,
        "ci_width": p_hi - p_lo,
        "z_std": z_std,
        "dist": np.abs(p_point - 0.5),
        "correct": (p_point >= 0.5).astype(int) == y,
    })


def _oriented_auc(label: np.ndarray, score: np.ndarray) -> float:
    """AUC oriented so it is >= 0.5 (the score's discriminative magnitude)."""
    a = roc_auc_score(label, score)
    return max(a, 1 - a)


def _paired_auc_delta_ci(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                         n_boot: int = 2000, seed: int = 42) -> tuple:
    """Bootstrap 95% CI for AUC(pred_b) - AUC(pred_a) on paired held-out
    predictions (resample ROIs, recompute both AUCs on the same resample)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], pred_b[idx])
                      - roc_auc_score(y[idx], pred_a[idx]))
    return (float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)))


def uncertainty_propagation(df: pd.DataFrame, draws: dict, C: float = 1.0,
                            n_boot: int = 2000, seed: int = 42) -> tuple:
    """CANONICAL test: does the NUTS posterior add diagnostic value to the
    tumor-vs-normal decision beyond the point estimate? (Resolved: NO.)

    Reproduces every number the manuscript "Prediction confidence" subsection
    and the Discussion uncertainty paragraph rely on:

      (1) CI-width ratio (misclassified / correct) in PROBABILITY space vs
          LOGIT space, with Mann-Whitney significance. The 2.4x probability
          effect collapses to ~1.3x (n.s.) in logit space -> it is sigmoid
          geometry, not the spectral posterior.
      (2) Controlled logistic  error ~ z(logit_width) + z(distance) : does the
          propagated interval width predict misclassification once distance to
          the boundary is accounted for? Reports the width coefficient with a
          bootstrap p / 95% CI, plus the univariate error-AUC of each predictor.
      (3) Uncertainty-as-features: per-zone detection LOOCV AUC using posterior
          MEANS only vs MEANS+STDS (the fair test, since the LR is trained on
          means and cannot otherwise see uncertainty), plus STDS-only and the
          whole-spectrum posterior spread as a univariate flag.

    Returns (summary_df, per_roi_df).
    """
    nuts_cols = [f"nuts_{c}" for c in D_COLS]
    std_cols = [f"nuts_std_{c}" for c in D_COLS]

    # ---- propagate per zone (within-zone LOOCV), then pool --------------------
    per_zone = []
    for z in ("pz", "tz"):
        dz = df[df["zone"] == z]
        per_zone.append(_propagate_zone(dz, draws, C=C))
    per_roi = pd.concat(per_zone, ignore_index=True)
    err = (~per_roi["correct"]).astype(int).values   # 1 = misclassified

    rows = []
    n_miss = int(err.sum())
    rows.append({"metric": "n_roi", "value": len(per_roi)})
    rows.append({"metric": "n_misclassified", "value": n_miss})

    # ---- (1) CI-width ratio: probability vs logit space -----------------------
    cw_c = per_roi.loc[per_roi["correct"], "ci_width"]
    cw_m = per_roi.loc[~per_roi["correct"], "ci_width"]
    zs_c = per_roi.loc[per_roi["correct"], "z_std"]
    zs_m = per_roi.loc[~per_roi["correct"], "z_std"]
    mw_prob = stats.mannwhitneyu(cw_m, cw_c, alternative="two-sided")
    mw_logit = stats.mannwhitneyu(zs_m, zs_c, alternative="two-sided")
    rows += [
        {"metric": "ci_width_correct_mean_prob", "value": cw_c.mean()},
        {"metric": "ci_width_miss_mean_prob", "value": cw_m.mean()},
        {"metric": "ci_width_ratio_prob", "value": cw_m.mean() / cw_c.mean()},
        {"metric": "ci_width_ratio_prob_mw_p", "value": mw_prob.pvalue},
        {"metric": "spread_correct_mean_logit", "value": zs_c.mean()},
        {"metric": "spread_miss_mean_logit", "value": zs_m.mean()},
        {"metric": "spread_ratio_logit", "value": zs_m.mean() / zs_c.mean()},
        {"metric": "spread_ratio_logit_mw_p", "value": mw_logit.pvalue},
    ]

    # ---- (2) controlled logistic: error ~ z(logit_width) + z(distance) --------
    zw = stats.zscore(per_roi["z_std"].values)
    zd = stats.zscore(per_roi["dist"].values)
    Xc = np.column_stack([zw, zd])
    full = LogisticRegression(penalty=None, max_iter=5000,
                              solver="lbfgs").fit(Xc, err)
    coef_w, coef_d = full.coef_[0]
    rng = np.random.RandomState(seed)
    boot_w = []
    nC = len(err)
    for _ in range(n_boot):
        idx = rng.choice(nC, nC, replace=True)
        if len(np.unique(err[idx])) < 2:
            continue
        try:
            b = LogisticRegression(penalty=None, max_iter=5000,
                                   solver="lbfgs").fit(Xc[idx], err[idx])
            boot_w.append(b.coef_[0][0])
        except Exception:
            continue
    boot_w = np.array(boot_w)
    p_w = 2.0 * min((boot_w <= 0).mean(), (boot_w >= 0).mean())
    rows += [
        {"metric": "ctrl_logit_width_coef", "value": coef_w},
        {"metric": "ctrl_logit_width_coef_lo", "value": float(np.percentile(boot_w, 2.5))},
        {"metric": "ctrl_logit_width_coef_hi", "value": float(np.percentile(boot_w, 97.5))},
        {"metric": "ctrl_logit_width_coef_boot_p", "value": float(min(p_w, 1.0))},
        {"metric": "ctrl_distance_coef", "value": coef_d},
        {"metric": "err_auc_logit_width", "value": _oriented_auc(err, per_roi["z_std"].values)},
        {"metric": "err_auc_distance", "value": _oriented_auc(err, -per_roi["dist"].values)},
    ]

    # ---- (3) uncertainty-as-features: does posterior std lift detection AUC? ---
    for z in ("pz", "tz"):
        dz = df[df["zone"] == z].reset_index(drop=True)
        dz = dz[dz["roi_id"].isin(draws)].reset_index(drop=True)
        yz = dz["is_tumor"].astype(int).values
        auc_mean, pred_mean, _, _, _ = loocv_auc(dz[nuts_cols].values, yz, C=C)
        auc_both, pred_both, _, _, _ = loocv_auc(
            dz[nuts_cols + std_cols].values, yz, C=C)
        auc_std, _, _, _, _ = loocv_auc(dz[std_cols].values, yz, C=C)
        dlo, dhi = _paired_auc_delta_ci(yz, pred_mean, pred_both,
                                        n_boot=n_boot, seed=seed)
        # whole-spectrum posterior spread (sum of per-bin std) as a flag
        spread = dz[std_cols].sum(axis=1).values
        rows += [
            {"metric": f"{z}_auc_means_only", "value": auc_mean},
            {"metric": f"{z}_auc_means_plus_stds", "value": auc_both},
            {"metric": f"{z}_delta_auc_addstds", "value": auc_both - auc_mean},
            {"metric": f"{z}_delta_auc_addstds_lo", "value": dlo},
            {"metric": f"{z}_delta_auc_addstds_hi", "value": dhi},
            {"metric": f"{z}_auc_stds_only", "value": auc_std},
        ]

    # whole-spectrum spread vs error (pooled) -- S13
    spread_all = df.set_index("roi_id").loc[per_roi["roi_id"], std_cols].sum(axis=1).values
    rows.append({"metric": "err_auc_whole_spectrum_spread",
                 "value": _oriented_auc(err, spread_all)})

    summary = pd.DataFrame(rows)
    return summary, per_roi


# ---------------------------------------------------------------------------
# Main recomputation
# ---------------------------------------------------------------------------

def recompute_all(signal_json: str = SIGNAL_JSON,
                  metadata_csv: str = METADATA_CSV,
                  nc_dir: str = NC_DIR,
                  output_dir: str = OUTPUT_DIR,
                  C_values: list[float] = [0.1, 1.0, 10.0]) -> dict:
    """Run complete recomputation from source data.

    Returns dict with all DataFrames and comparison results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("CLEAN RECOMPUTATION — ALL PAPER NUMBERS")
    print("=" * 80)

    # Step 1: Load data
    print("\n[1/6] Loading raw data...")
    rois = load_dataset(signal_json, metadata_csv)
    print(f"  Loaded {len(rois)} ROIs from {len(set(r['patient'] for r in rois))} patients")
    n_tumor = sum(1 for r in rois if r["is_tumor"])
    n_normal = len(rois) - n_tumor
    print(f"  Tumor: {n_tumor}, Normal: {n_normal}")

    # Verify new39 GGG
    new39 = [r for r in rois if r["patient"] == "new39" and r["is_tumor"]]
    if new39:
        print(f"  new39 tumor GGG = {new39[0]['ggg']} (expected: 1)")

    # Step 2: Compute MAP features
    print("\n[2/6] Computing MAP spectra (Ridge NNLS, lambda={})...".format(RIDGE_STRENGTH))
    U = build_design_matrix()
    print(f"  Design matrix: {U.shape}, cond={np.linalg.cond(U):.0f}")
    # MAP is computed inside build_features_df

    # Step 3: Load NUTS posteriors
    print("\n[3/6] Loading NUTS posteriors from .nc files...")
    nuts_posteriors = load_nuts_posteriors(rois, nc_dir)
    print(f"  Loaded {len(nuts_posteriors)}/{len(rois)} NUTS posteriors")

    # Step 4: Build features DataFrame
    print("\n[4/6] Assembling features...")
    df = build_features_df(rois, nuts_posteriors, U)
    print(f"  Features shape: {df.shape}")
    print(f"  Zones: PZ={len(df[df['zone']=='pz'])}, TZ={len(df[df['zone']=='tz'])}")

    # GGG check
    ggg_valid = df[(df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)]
    n_low = (ggg_valid["ggg"] <= 2).sum()
    n_high = (ggg_valid["ggg"] >= 3).sum()
    print(f"  GGG: n={len(ggg_valid)} (low-grade={n_low}, high-grade={n_high})")

    # Step 5: Classification
    print("\n[5/6] Running LOOCV classification...")
    auc_df = run_classification(df, C_values=C_values)

    # Step 6: Comparisons
    print("\n[6/6] Computing comparisons...")

    # MAP vs NUTS
    print("\n  MAP vs NUTS comparison:")
    map_nuts = compare_map_nuts(df)
    for k, v in sorted(map_nuts.items()):
        print(f"    {k}: {v:.4f}")

    # ADC vs discriminant
    print("\n  ADC vs discriminant correlation:")
    adc_disc_df = adc_discriminant_correlation(df)
    for _, row in adc_disc_df.iterrows():
        print(f"    {row['zone']} {row['features']} C={row['C']}: "
              f"r={row['r_adc_disc']:.4f} (p={row['p_value']:.2e})")

    # ADC sensitivity
    print("\n  ADC sensitivity (vector-level: dADC/dR vs LR coefs):")
    sens_df = adc_sensitivity_analysis(df, C_values=[1.0])
    for _, row in sens_df.iterrows():
        print(f"    {row['zone']} {row['features']:4s} {row['operating_point']:6s}: "
              f"r_pearson={row['r_pearson']:.4f} (p={row['p_pearson']:.4f})  "
              f"r_spearman={row['r_spearman']:.4f}")

    # Identifiability
    print("\n  Per-component identifiability:")
    ident_df = component_identifiability(df)
    print(f"    {'Component':10s} {'Mean Frac':>10s} {'Mean Std':>10s} {'CV':>8s}")
    for _, row in ident_df.iterrows():
        print(f"    {row['component']:10s} {row['mean_fraction']:10.4f} "
              f"{row['mean_posterior_std']:10.4f} {row['mean_CV']:8.2f}")

    # Posterior-uncertainty propagation ("Prediction confidence" claims).
    # Needs the full posterior draws (not just mean/std), loaded separately.
    print("\n  Posterior-uncertainty propagation (does it add diagnostic value?):")
    unc_df = None
    unc_per_roi = None
    try:
        draws = load_nuts_draws(rois, nc_dir)
        print(f"    Loaded posterior draws for {len(draws)}/{len(rois)} ROIs")
        unc_df, unc_per_roi = uncertainty_propagation(df, draws, C=1.0)
        _u = dict(zip(unc_df["metric"], unc_df["value"]))
        print(f"    CI-width ratio (miss/correct): prob {_u['ci_width_ratio_prob']:.2f}x "
              f"(MW p={_u['ci_width_ratio_prob_mw_p']:.3f}) | "
              f"logit {_u['spread_ratio_logit']:.2f}x "
              f"(MW p={_u['spread_ratio_logit_mw_p']:.3f})")
        print(f"    Controlled (error ~ logit-width + distance): width coef "
              f"{_u['ctrl_logit_width_coef']:+.2f} (boot p={_u['ctrl_logit_width_coef_boot_p']:.2f}, "
              f"CI [{_u['ctrl_logit_width_coef_lo']:+.2f},{_u['ctrl_logit_width_coef_hi']:+.2f}]); "
              f"err-AUC width {_u['err_auc_logit_width']:.2f} vs distance {_u['err_auc_distance']:.2f}")
        print(f"    Uncertainty-as-features dAUC: PZ {_u['pz_delta_auc_addstds']:+.3f} "
              f"[{_u['pz_delta_auc_addstds_lo']:+.3f},{_u['pz_delta_auc_addstds_hi']:+.3f}], "
              f"TZ {_u['tz_delta_auc_addstds']:+.3f} "
              f"[{_u['tz_delta_auc_addstds_lo']:+.3f},{_u['tz_delta_auc_addstds_hi']:+.3f}]")
        print("    -> posterior adds NO independent error signal beyond the point estimate.")
    except Exception as e:
        print(f"    [SKIP] uncertainty propagation failed: {e}")

    # Save outputs
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    features_path = os.path.join(output_dir, "features.csv")
    auc_path = os.path.join(output_dir, "auc_table.csv")
    adc_disc_path = os.path.join(output_dir, "adc_discriminant.csv")
    ident_path = os.path.join(output_dir, "identifiability.csv")
    map_nuts_path = os.path.join(output_dir, "map_nuts_comparison.csv")
    sens_path = os.path.join(output_dir, "adc_sensitivity.csv")
    unc_path = os.path.join(output_dir, "uncertainty_propagation.csv")
    unc_roi_path = os.path.join(output_dir, "uncertainty_propagation_per_roi.csv")

    df.to_csv(features_path, index=False)
    auc_df.to_csv(auc_path, index=False)
    adc_disc_df.to_csv(adc_disc_path, index=False)
    ident_df.to_csv(ident_path, index=False)
    pd.DataFrame([map_nuts]).to_csv(map_nuts_path, index=False)
    sens_df.to_csv(sens_path, index=False)
    if unc_df is not None:
        unc_df.to_csv(unc_path, index=False)
        unc_per_roi.to_csv(unc_roi_path, index=False)

    print(f"  {features_path}")
    print(f"  {auc_path}")
    print(f"  {adc_disc_path}")
    print(f"  {ident_path}")
    print(f"  {map_nuts_path}")
    print(f"  {sens_path}")
    if unc_df is not None:
        print(f"  {unc_path}")
        print(f"  {unc_roi_path}")

    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL AUC TABLE")
    print("=" * 80)
    pivot = auc_df.pivot_table(index=["method", "C"], columns="task",
                                values="auc", aggfunc="first")
    print(pivot.to_string(float_format="{:.4f}".format))

    print("\n" + "=" * 80)
    print("RECOMPUTATION COMPLETE")
    print("=" * 80)

    return {
        "features_df": df,
        "auc_df": auc_df,
        "adc_disc_df": adc_disc_df,
        "ident_df": ident_df,
        "map_nuts_comparison": map_nuts,
        "sensitivity_df": sens_df,
        "uncertainty_df": unc_df,
        "uncertainty_per_roi_df": unc_per_roi,
    }


if __name__ == "__main__":
    recompute_all()
