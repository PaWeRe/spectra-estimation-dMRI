"""
Re-gridding robustness check (read-only on gold-standard signals).

Question (final-draft due diligence): had we followed our OWN bin-placement rule
(dD ~ D^1.5, finer at low D), would the spectrum look different, classify better,
or attack the main claim that ADC ~ the optimal spectral discriminant?

Method: re-fit TUNED MAP (lambda=1e-3) on all 149 ROIs under three grids, then run
the EXACT paper detection pipeline (StandardScaler + LogisticRegression C=1, LOOCV)
per zone, plus the ADC<->discriminant Spearman. MAP is a valid first pass (F1: tuned
MAP ~ NUTS for classification). Nothing is written or committed; gold-standard .nc
files are untouched.

FINDING (2026-06-22): detection AUC and the ADC<->discriminant correlation are
essentially unchanged across grids, and the tumor-normal contrast stays at the two
outer compartments (D=0.25 up, D=3.0 down) regardless of binning. The main claim is
robust to discretization -- this confirms theory.tex's "regardless of bin placement"
assertion empirically. Detection only (well-powered); says nothing about grading (n=29).

    grid                       AUC_full pz/tz   2-outer pz/tz    rho_ADCdisc pz/tz
    baseline_current           0.917 / 0.952    0.925 / 0.965    -0.980 / -0.929
    constant_precision_rule    0.903 / 0.952    0.936 / 0.969    -0.975 / -0.974
    finer_low_D                0.906 / 0.950    0.937 / 0.964    -0.976 / -0.973

Run: uv run python scripts/regrid_robustness.py
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from spectra_estimation_dmri.biomarkers.recompute import (
    load_dataset, compute_adc, B_VALUES_MS, RIDGE_STRENGTH,
)

GRIDS = {
    "baseline_current": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0],
    "constant_precision_rule": [0.25, 0.32, 0.43, 0.60, 0.90, 1.51, 3.0, 20.0],
    "finer_low_D": [0.25, 0.3, 0.4, 0.5, 0.6, 0.9, 1.5, 3.0, 20.0],
}


def map_fractions(signal, D):
    """Tuned-MAP spectrum on a custom grid, normalized to fractions."""
    U = np.exp(-np.outer(B_VALUES_MS, np.array(D)))
    S0 = signal[0] if signal[0] > 0 else 1.0
    s_norm = signal / S0
    n_d = U.shape[1]
    U_aug = np.vstack([U, np.sqrt(RIDGE_STRENGTH) * np.eye(n_d)])
    s_aug = np.concatenate([s_norm, np.zeros(n_d)])
    spec, _ = nnls(U_aug, s_aug)
    return spec / (spec.sum() + 1e-10)


def loocv_auc(X, y, C=1.0):
    loo = LeaveOneOut()
    yp = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        yp[te] = clf.predict_proba(Xte)[0, 1]
    return roc_auc_score(y, yp)


def disc_adc_corr(X, y, adc):
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    clf.fit(Xs, y)
    disc = Xs @ clf.coef_[0] + clf.intercept_[0]
    return stats.spearmanr(disc, adc).correlation


def main():
    rois = load_dataset()
    adc = np.array([compute_adc(r["signal"], 1000.0) for r in rois])
    zone = np.array(["pz" if r["region"] == "pz" else "tz" for r in rois])
    ytum = np.array([int(r["is_tumor"]) for r in rois])
    print(f"Loaded {len(rois)} ROIs | PZ {np.sum(zone=='pz')} TZ {np.sum(zone=='tz')} | "
          f"tumor {ytum.sum()} normal {(1-ytum).sum()}\n")

    rows = []
    for name, D in GRIDS.items():
        D = np.array(D)
        frac = np.array([map_fractions(r["signal"], D) for r in rois])
        out = {"grid": name, "nbins": len(D)}
        for z in ["pz", "tz"]:
            m = zone == z
            Xz, yz, az = frac[m], ytum[m], adc[m]
            out[f"AUC_full_{z}"] = loocv_auc(Xz, yz, 1.0)
            i025 = int(np.argmin(np.abs(D - 0.25)))
            i30 = int(np.argmin(np.abs(D - 3.0)))
            out[f"AUC_2outer_{z}"] = loocv_auc(Xz[:, [i025, i30]], yz, 1.0)
            out[f"rho_ADCdisc_{z}"] = disc_adc_corr(Xz, yz, az)
        rows.append(out)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== Detection AUC + ADC<->discriminant Spearman per grid ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Cohort-mean tumor-vs-normal fractions (PZ): where does contrast live? ===")
    for name, D in GRIDS.items():
        D = np.array(D)
        frac = np.array([map_fractions(r["signal"], D) for r in rois])
        m = zone == "pz"
        ft = frac[m][ytum[m] == 1].mean(0)
        fn = frac[m][ytum[m] == 0].mean(0)
        print(f"\n[{name}]  D = {np.round(D, 2)}")
        print(f"  tumor-normal: {np.round(ft - fn, 3)}")


if __name__ == "__main__":
    main()
