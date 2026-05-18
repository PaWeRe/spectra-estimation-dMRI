"""
Two-feature LR vs ADC: does adding μ_D=0.50 (the grading-axis residual
from partial-Spearman) lift AUC for GGG≥3 over ADC alone?

Companion to partial_corr_ggg.py. Partial Spearman ρ(μ_D=0.50, GGG | ADC)
= +0.420, uncorrected p=0.026, doesn't survive Bonferroni at N=29. This
script asks the same question in clinical-utility frame: AUC for GGG≥3
with {ADC} vs {ADC + μ_D=0.50}, paired DeLong test.

Endpoint: GGG ≥ 3 (binary), on the 29 tumor ROIs with valid GGG.
Classifier: standardized LogisticRegression (C=1.0), LOOCV prediction.
Inference:
- AUC bootstrap percentile 95% CI (B=2000, seed=42).
- Paired DeLong test on LOOCV predicted probabilities vs ADC baseline
  (proper U-statistic covariance — not the conservative independence
  approximation used elsewhere in this repo).

Reference for paired DeLong: DeLong, DeLong & Clarke-Pearson, Biometrics
44(3):837-845, 1988. Fast implementation via midranks
(Sun & Xu 2014, IEEE Signal Process Lett 21(11):1389-1393).

Outputs:
- results/biomarkers/two_feature_lr_vs_adc.csv
- console summary

Usage:
    uv run python scripts/two_feature_lr_vs_adc.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"
OUTPUT_CSV = REPO_ROOT / "results" / "biomarkers" / "two_feature_lr_vs_adc.csv"

N_BOOTSTRAP = 2000
ALPHA = 0.05
LR_C = 1.0
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Paired DeLong (Sun & Xu 2014, midrank-based, exact for ties)
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank for tied values (DeLong needs this)."""
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # midrank, 1-indexed
        i = j
    T_unordered = np.empty(n, dtype=float)
    T_unordered[order] = T
    return T_unordered


def _delong_components(preds: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute V10 (positive structural components), V01 (negative
    structural components), and AUC vector for K models.

    preds: shape (K, n) predicted scores.
    y: shape (n,) binary labels.
    """
    pos = y == 1
    neg = y == 0
    m = int(pos.sum())
    n_neg = int(neg.sum())
    K = preds.shape[0]

    aucs = np.zeros(K)
    V10 = np.zeros((K, m))
    V01 = np.zeros((K, n_neg))
    for k in range(K):
        s = preds[k]
        sp = s[pos]
        sn = s[neg]
        # Midranks among combined
        s_all = np.concatenate([sp, sn])
        TR = _compute_midrank(s_all)
        TX = _compute_midrank(sp)
        TY = _compute_midrank(sn)
        TR_pos = TR[:m]
        TR_neg = TR[m:]
        # V10 (per-positive component)
        V10[k] = (TR_pos - TX) / n_neg
        # V01 (per-negative component)
        V01[k] = 1.0 - (TR_neg - TY) / m
        aucs[k] = V10[k].mean()
    return V10, V01, aucs


def paired_delong(preds_a: np.ndarray, preds_b: np.ndarray, y: np.ndarray) -> dict:
    """Paired DeLong test for AUC_a - AUC_b.

    Returns dict with auc_a, auc_b, delta, var, z, p (two-sided).
    """
    preds = np.vstack([preds_a, preds_b])
    V10, V01, aucs = _delong_components(preds, y)
    m = V10.shape[1]
    n_neg = V01.shape[1]
    S10 = np.cov(V10, ddof=1)  # (2, 2)
    S01 = np.cov(V01, ddof=1)  # (2, 2)
    S = S10 / m + S01 / n_neg
    delta = aucs[0] - aucs[1]
    var_delta = float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1])
    if var_delta <= 0:
        return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]), delta=float(delta),
                    var=var_delta, z=float("nan"), p=float("nan"))
    z = delta / np.sqrt(var_delta)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return dict(auc_a=float(aucs[0]), auc_b=float(aucs[1]),
                delta=float(delta), var=var_delta, z=float(z), p=float(p))


# ---------------------------------------------------------------------------
# LOOCV LR + bootstrap CI
# ---------------------------------------------------------------------------

def loocv_lr_probs(X: np.ndarray, y: np.ndarray, C: float = LR_C) -> np.ndarray:
    n = len(y)
    yp = np.zeros(n)
    for tr, te in LeaveOneOut().split(X):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, random_state=42).fit(
            scaler.transform(X[tr]), y[tr]
        )
        yp[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    return yp


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return (
        float(roc_auc_score(y, p)),
        float(np.percentile(aucs, 100 * ALPHA / 2)),
        float(np.percentile(aucs, 100 * (1 - ALPHA / 2))),
    )


# ---------------------------------------------------------------------------
# Zone preparation
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    sub = df[(df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)].copy()
    if zone != "pooled":
        sub = sub[sub["zone"] == zone]
    return sub


# ---------------------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------------------

FEATURE_SETS = [
    ("ADC", ["adc"]),
    ("ADC + μ_D=0.50", ["adc", "nuts_D_0.50"]),
    ("ADC + μ_D=0.50 + μ_D=2.00", ["adc", "nuts_D_0.50", "nuts_D_2.00"]),
    ("ADC + μ_D=0.50 + σ_D=0.50", ["adc", "nuts_D_0.50", "nuts_std_D_0.50"]),
    ("ADC + μ_D=0.50 + μ_D=0.25 + μ_D=3.00",
        ["adc", "nuts_D_0.50", "nuts_D_0.25", "nuts_D_3.00"]),
    ("μ_D=0.50 alone", ["nuts_D_0.50"]),
    ("8-bin μ NUTS", [f"nuts_D_{d:.2f}" for d in
                      [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]]),
]


def run() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features.csv: {len(df)} ROIs")

    rows: list[dict] = []
    for zone in ("pooled", "pz", "tz"):
        sub = prepare(df, zone)
        n = len(sub)
        if n < 8:
            print(f"  {zone}: N={n} too small, skipping")
            continue
        y = (sub["ggg"].values >= 3).astype(int)
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        print(f"\n[{zone}] N={n}  GGG≥3 = {n_pos}/{n}  GGG<3 = {n_neg}/{n}")

        # Run all models, store LOOCV probs
        loocv_probs = {}
        for label, cols in FEATURE_SETS:
            X = sub[cols].values.astype(float)
            if X.shape[1] == 0:
                continue
            p = loocv_lr_probs(X, y)
            auc, lo, hi = bootstrap_auc(y, p)
            loocv_probs[label] = p
            rows.append(dict(
                zone=zone, n=n, n_pos=n_pos, n_neg=n_neg,
                model=label, k_features=X.shape[1],
                features=",".join(cols),
                auc=auc, auc_ci_lo=lo, auc_ci_hi=hi,
                delong_z=np.nan, delong_p=np.nan, delta_auc=np.nan,
            ))

        # Paired DeLong vs ADC baseline
        if "ADC" in loocv_probs:
            base = loocv_probs["ADC"]
            for label, p in loocv_probs.items():
                if label == "ADC":
                    continue
                res = paired_delong(p, base, y)
                # Update the row for this (zone, model)
                for r in rows:
                    if r["zone"] == zone and r["model"] == label:
                        r["delta_auc"] = res["delta"]
                        r["delong_z"] = res["z"]
                        r["delong_p"] = res["p"]
                        break

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV} ({len(out)} rows)")
    return out


def summarize(out: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(f"  LOOCV LR  •  endpoint: GGG ≥ 3  •  paired DeLong vs ADC baseline  •  α={ALPHA}")
    print("=" * 100)
    for zone in ("pooled", "pz", "tz"):
        sub = out[out["zone"] == zone]
        if sub.empty:
            continue
        n = int(sub["n"].iloc[0])
        n_pos = int(sub["n_pos"].iloc[0])
        print(f"\n[{zone}] N={n}  pos(GGG≥3)={n_pos}")
        print(f"  {'model':40s}  {'k':>2s}  {'AUC':>5s}  {'95% CI':>16s}  {'ΔAUC':>6s}  {'DeLong p':>9s}")
        for _, r in sub.iterrows():
            ci = f"[{r['auc_ci_lo']:.3f},{r['auc_ci_hi']:.3f}]"
            d = "  —   " if not np.isfinite(r["delta_auc"]) else f"{r['delta_auc']:+.3f}"
            pp = "  —    " if not np.isfinite(r["delong_p"]) else f"{r['delong_p']:.4f}"
            flag = ""
            if np.isfinite(r["delong_p"]) and r["delong_p"] < ALPHA:
                flag = "*"
            print(f"  {r['model']:40s}  {int(r['k_features']):>2d}  {r['auc']:.3f}  {ci:>16s}  {d:>6s}  {pp:>9s}  {flag}")


if __name__ == "__main__":
    out = run()
    summarize(out)
