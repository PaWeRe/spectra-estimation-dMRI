"""
Verification script for key paper findings.

Recomputes all reported numbers from features.csv to confirm correctness.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# Units: b in ms/μm², D in μm²/ms (as in the config)
B_VALUES = np.array([0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.,
                      2.25, 2.5, 2.75, 3., 3.25, 3.5])
DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
D_COLS = [f"D_{d:.2f}" for d in DIFFUSIVITIES]


def load_and_prepare(features_path: str):
    df = pd.read_csv(features_path)
    df["zone"] = df["region"].apply(
        lambda x: "pz" if "pz" in str(x).lower() else ("tz" if "tz" in str(x).lower() else "unknown")
    )
    print(f"Loaded {len(df)} ROIs, {df['patient_id'].nunique()} patients")
    print(f"Tumor: {df['is_tumor'].sum()}, Normal: {(~df['is_tumor']).sum()}")
    return df


def compute_loocv_auc(X, y, C=1.0):
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_train, y[train_idx])
        y_pred[test_idx] = clf.predict_proba(X_test)[0, 1]
    return roc_auc_score(y, y_pred), y_pred


def get_lr_coefficients(X, y, C=1.0):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(X_scaled, y)
    return clf.coef_[0], clf.intercept_[0], scaler


def compute_discriminant_scores(X, coefs, intercept, scaler):
    X_scaled = scaler.transform(X)
    return X_scaled @ coefs + intercept


def compute_adc_from_spectrum(R, b_max=1.0):
    """ADC from spectral fractions. Units: b in ms/μm², D in μm²/ms, ADC in μm²/ms."""
    signal = np.zeros(len(B_VALUES))
    for j, (Rj, Dj) in enumerate(zip(R, DIFFUSIVITIES)):
        signal += Rj * np.exp(-B_VALUES * Dj)
    b_mask = B_VALUES <= b_max
    log_signal = np.log(np.maximum(signal[b_mask], 1e-20))
    slope, _ = np.polyfit(B_VALUES[b_mask], log_signal, 1)
    return -slope


def compute_sensitivity(spectrum, b_max=1.0, epsilon=0.0001):
    adc_base = compute_adc_from_spectrum(spectrum, b_max)
    sensitivity = np.zeros(len(spectrum))
    for j in range(len(spectrum)):
        R_perturbed = spectrum.copy()
        R_perturbed[j] += epsilon
        adc_perturbed = compute_adc_from_spectrum(R_perturbed, b_max)
        sensitivity[j] = (adc_perturbed - adc_base) / epsilon
    return sensitivity, adc_base


# ===========================================================================
print("=" * 80)
print("VERIFICATION OF KEY PAPER FINDINGS")
print("=" * 80)

df = load_and_prepare("results/biomarkers/features.csv")

# ===========================================================================
print("\n" + "=" * 80)
print("1. CLASSIFICATION AUCs")
print("=" * 80)

tasks = {
    "PZ": (df[df["zone"] == "pz"], "is_tumor"),
    "TZ": (df[df["zone"] == "tz"], "is_tumor"),
}
ggg_df = df[(df["is_tumor"] == True) & (df["ggg"].notna()) & (df["ggg"] != 0)].copy()
ggg_df["label"] = (ggg_df["ggg"] >= 3).astype(int)
tasks["GGG"] = (ggg_df, "label")

for task_name, (task_df, label_col) in tasks.items():
    y = task_df[label_col].astype(int).values
    print(f"\n  {task_name}: N={len(y)}, pos={y.sum()}, neg={len(y)-y.sum()}")

    # ADC raw AUC
    adc_vals = task_df["ADC"].values
    raw_auc = roc_auc_score(y, adc_vals)
    if raw_auc < 0.5:
        raw_auc = 1 - raw_auc
    print(f"    ADC raw feature AUC:  {raw_auc:.4f}")

    # ADC with LR
    for C in [1.0, 10.0]:
        auc, _ = compute_loocv_auc(adc_vals.reshape(-1, 1), y, C=C)
        print(f"    ADC LR LOOCV (C={C:4.1f}): {auc:.4f}")

    # Full LR (NUTS features)
    X_full = task_df[D_COLS].values
    for C in [1.0, 10.0]:
        auc, _ = compute_loocv_auc(X_full, y, C=C)
        print(f"    Full LR 8-feat (C={C:4.1f}): {auc:.4f}")

print("\n  REPORTED VALUES (from FINDINGS.md):")
print("    PZ: ADC=0.940, MAP C=10=0.935, NUTS C=10=0.933")
print("    TZ: ADC=0.964, MAP C=10=0.941, NUTS C=10=0.925")
print("    GGG: ADC=0.778, MAP C=10=0.722, NUTS C=10=0.722")

# ===========================================================================
print("\n" + "=" * 80)
print("2. ADC vs DISCRIMINANT SCORE CORRELATION (across ROIs)")
print("=" * 80)

for zone, C_val in [("pz", 10.0), ("tz", 10.0)]:
    zone_df = df[df["zone"] == zone]
    y = zone_df["is_tumor"].astype(int).values
    X = zone_df[D_COLS].values
    adc_vals = zone_df["ADC"].values

    coefs, intercept, scaler = get_lr_coefficients(X, y, C=C_val)
    disc_scores = compute_discriminant_scores(X, coefs, intercept, scaler)

    r, p = stats.pearsonr(adc_vals, disc_scores)
    print(f"\n  {zone.upper()} (C={C_val}): ADC vs discriminant score r = {r:.4f} (p = {p:.2e})")
    print(f"    LR coefficients: {np.round(coefs, 4)}")

    # Also check at various C
    for C in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        coefs_c, intercept_c, scaler_c = get_lr_coefficients(X, y, C=C)
        disc_c = compute_discriminant_scores(X, coefs_c, intercept_c, scaler_c)
        r_c, _ = stats.pearsonr(adc_vals, disc_c)
        print(f"    C={C:5.1f}: r = {r_c:.4f}, coefs = {np.round(coefs_c, 3)}")

# ===========================================================================
print("\n" + "=" * 80)
print("3. ADC SENSITIVITY VECTOR vs LR FEATURE VECTOR")
print("=" * 80)

for zone in ["pz", "tz"]:
    zone_df = df[df["zone"] == zone]
    y = zone_df["is_tumor"].astype(int).values
    X = zone_df[D_COLS].values

    tumor_mask = y == 1
    normal_mask = y == 0
    avg_tumor = X[tumor_mask].mean(axis=0)
    avg_normal = X[normal_mask].mean(axis=0)

    print(f"\n  --- {zone.upper()} ---")
    print(f"  Avg tumor:  {np.round(avg_tumor, 4)}")
    print(f"  Avg normal: {np.round(avg_normal, 4)}")

    for C in [1.0, 10.0]:
        coefs, _, _ = get_lr_coefficients(X, y, C=C)
        print(f"\n  LR coefs (C={C}): {np.round(coefs, 4)}")

        for name, spectrum in [("tumor", avg_tumor), ("normal", avg_normal)]:
            sens, adc_val = compute_sensitivity(spectrum, b_max=1.0)
            r_val, p_val = stats.pearsonr(sens, coefs)
            print(f"    Sensitivity at {name} (ADC={adc_val:.4f}):")
            print(f"      dADC/dR = {np.round(sens, 4)}")
            print(f"      Corr with LR: r = {r_val:.4f} (p = {p_val:.3f})")

# ===========================================================================
print("\n" + "=" * 80)
print("4. PER-COMPONENT IDENTIFIABILITY (CV)")
print("=" * 80)

unc_df = pd.read_csv("results/biomarkers/feature_uncertainty.csv")
print(f"\n  {'Component':10s} | {'Mean Fraction':13s} | {'Mean Std':10s} | {'Mean CV':8s}")
print(f"  {'-'*50}")
for col in D_COLS:
    mean_feat = df[col].mean()
    mean_std = unc_df[col].mean()
    mean_cv = (unc_df[col] / (df[col] + 1e-10)).mean()
    print(f"  {col:10s} | {mean_feat:13.4f} | {mean_std:10.4f} | {mean_cv:8.4f}")

print("\n  REPORTED (FINDINGS.md): D_0.25 CV=0.20, D_0.50-1.00 CV>0.80, D_3.0 CV=0.32")

# ===========================================================================
print("\n" + "=" * 80)
print("5. GGG SAMPLE SIZE CHECK")
print("=" * 80)

ggg_all = df[(df["is_tumor"] == True) & (df["ggg"].notna())].copy()
print(f"\n  All tumor ROIs with GGG: {len(ggg_all)}")
print(f"  GGG distribution:")
for g in sorted(ggg_all["ggg"].unique()):
    n = (ggg_all["ggg"] == g).sum()
    print(f"    GGG {g:.0f}: n={n}")
ggg_nonzero = ggg_all[ggg_all["ggg"] != 0]
print(f"\n  GGG != 0: n={len(ggg_nonzero)}")
print(f"  Low-grade (1-2): {(ggg_nonzero['ggg'] <= 2).sum()}")
print(f"  High-grade (3-5): {(ggg_nonzero['ggg'] >= 3).sum()}")

# ===========================================================================
print("\n" + "=" * 80)
print("6. EXTRACTING MAP FEATURES FROM .nc FILES")
print("=" * 80)

import os
import arviz as az

nc_dir = "results/inference_bwh_backup"
nc_files = [f for f in os.listdir(nc_dir) if f.endswith(".nc")]
print(f"  Found {len(nc_files)} .nc files")

# Load one to check structure
test_nc = os.path.join(nc_dir, nc_files[0])
idata = az.from_netcdf(test_nc)
print(f"\n  Test file: {nc_files[0]}")
print(f"  Groups: {list(idata.groups())}")
print(f"  Posterior variables: {list(idata.posterior.data_vars)}")

# Check for MAP init (spectrum_init) in the data
# MAP features would need to be extracted from the model
# For now, let's check if we can build MAP features

# Load the signal_decays.json to get raw data
import json
json_path = "src/spectra_estimation_dmri/data/bwh/signal_decays.json"
with open(json_path) as f:
    signal_data = json.load(f)
print(f"\n  Signal decays JSON: {len(signal_data)} entries")
print(f"  First entry keys: {list(signal_data[0].keys()) if isinstance(signal_data, list) else list(signal_data.keys())}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE — SEE DISCREPANCIES ABOVE")
print("=" * 80)
