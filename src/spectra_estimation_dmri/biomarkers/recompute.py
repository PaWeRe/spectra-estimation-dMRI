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
RIDGE_STRENGTH = 0.1

# Config dicts for spectra_id hash (must match what generated the .nc files)
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
    """Ridge NNLS MAP estimate for a single ROI.

    Normalizes signal by S(b=0), solves (U'U + λI)^{-1} U' s, clips to >=0.
    Returns normalized spectrum (fractions summing to ~1).
    """
    S0 = signal[0] if signal[0] > 0 else 1.0
    s_norm = signal / S0
    n_d = U.shape[1]
    A = U.T @ U + RIDGE_STRENGTH * np.eye(n_d)
    projection = np.linalg.solve(A, U.T)
    spectrum = projection @ s_norm
    return np.maximum(spectrum, 0.0)


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


def run_classification(df: pd.DataFrame, C_values: list[float] = [0.1, 1.0, 10.0],
                       ) -> pd.DataFrame:
    """Run all classification tasks and return AUC table.

    Tasks: PZ tumor detection, TZ tumor detection, GGG grading.
    Methods: ADC (raw rank), ADC (LR), MAP Full LR, NUTS Full LR.
    """
    map_cols = [f"map_{c}" for c in D_COLS]
    nuts_cols = [f"nuts_{c}" for c in D_COLS]

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

        # ADC raw rank AUC
        adc_vals = task_df["adc"].values
        adc_raw_auc = raw_rank_auc(adc_vals, y)
        results.append({"task": task_name, "method": "ADC (raw rank)",
                        "C": "-", "auc": adc_raw_auc, "n": len(y)})
        print(f"    ADC raw rank:    {adc_raw_auc:.4f}")

        for C in C_values:
            # ADC via LR
            auc_adc, _, _, _, _ = loocv_auc(adc_vals.reshape(-1, 1), y, C=C)
            results.append({"task": task_name, "method": "ADC (LR)",
                            "C": C, "auc": auc_adc, "n": len(y)})

            # MAP Full LR (8 features)
            X_map = task_df[map_cols].values
            auc_map, _, coefs_map, _, _ = loocv_auc(X_map, y, C=C)
            results.append({"task": task_name, "method": "MAP Full LR",
                            "C": C, "auc": auc_map, "n": len(y)})

            # NUTS Full LR (8 features)
            X_nuts = task_df[nuts_cols].values
            auc_nuts, _, coefs_nuts, _, _ = loocv_auc(X_nuts, y, C=C)
            results.append({"task": task_name, "method": "NUTS Full LR",
                            "C": C, "auc": auc_nuts, "n": len(y)})

            print(f"    C={C:5.1f}  ADC LR={auc_adc:.4f}  "
                  f"MAP={auc_map:.4f}  NUTS={auc_nuts:.4f}")

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

    df.to_csv(features_path, index=False)
    auc_df.to_csv(auc_path, index=False)
    adc_disc_df.to_csv(adc_disc_path, index=False)
    ident_df.to_csv(ident_path, index=False)
    pd.DataFrame([map_nuts]).to_csv(map_nuts_path, index=False)
    sens_df.to_csv(sens_path, index=False)

    print(f"  {features_path}")
    print(f"  {auc_path}")
    print(f"  {adc_disc_path}")
    print(f"  {ident_path}")
    print(f"  {map_nuts_path}")
    print(f"  {sens_path}")

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
    }


if __name__ == "__main__":
    recompute_all()
