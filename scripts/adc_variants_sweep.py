"""
ADC-variants sweep: a fair upper bound for the ADC reference.

Motivation. Partial-Spearman (partial_corr_ggg.py) and two-feature LR +
DeLong (two_feature_lr_vs_adc.py) both pointed to the "spectrum
mechanistically richer but diagnostically equivalent" framing using
ADC(b≤1000). But our ADC was Session-6-optimized for tumor-vs-normal;
the grading reference might be stronger at a different b-range. If
even a grading-optimized ADC doesn't change the picture, Path A' is
locked.

A second question Patrick raised 2026-05-17: are different b-ranges
optimal for different tasks? I.e., does a low-b ADC dominate detection
while a high-b ADC dominates grading? If yes, that's a novel finding
about ADC itself, independent of the spectrum.

ADC variants computed per ROI (b in s/mm²):
  std       :  0 ≤ b ≤ 1000   (current paper baseline, 5 points)
  ext1500   :  0 ≤ b ≤ 1500   (7 points)
  ext2000   :  0 ≤ b ≤ 2000   (9 points)
  full      :  0 ≤ b ≤ 3500   (all 15 points)
  high      : 1000 ≤ b ≤ 3500 (high-b only, 11 points; drops perfusion-
                                contaminated low-b region. PI-RADS-style.)
  midrange  :  250 ≤ b ≤ 1500 (no b=0, intermediate; "fitted ADC")
  spec_M1   : ∫ R(D) · D dD  (first moment of NUTS posterior mean
                                spectrum — model-based ADC analog. Tests
                                Chatterjee 2015 mechanism: standard ADC
                                ≈ projection of spectrum onto D-axis.)
  DKI_D     : Kurtosis-fit ADC, S = S0 · exp(-bD + (bD)² · K/6),
              fit on 0 ≤ b ≤ 2500 (DKI's stable range).
  DKI_K     : The kurtosis parameter K from the same fit.

For each variant we evaluate three tasks:
  Detection (149 ROIs, per zone): LOOCV LR AUC, bootstrap 95% CI.
  Grading-continuous (29 tumor ROIs with valid GGG, pooled): Spearman
    ρ vs continuous GGG.
  Grading-binary (29 ROIs, GGG≥3, pooled): LOOCV LR AUC + paired
    DeLong vs std-ADC, bootstrap 95% CI.

And the partial-Spearman question is repeated with each variant as the
covariate:
  ρ(μ_D=0.50, GGG | ADC_variant)

Outputs:
- results/biomarkers/adc_variants.csv               (per-ROI ADC values)
- results/biomarkers/adc_variants_summary.csv       (one row per
                                                     variant × task)
- console summary

Usage:
    uv run python scripts/adc_variants_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_JSON = REPO_ROOT / "src" / "spectra_estimation_dmri" / "data" / "bwh" / "signal_decays.json"
METADATA_CSV = REPO_ROOT / "src" / "spectra_estimation_dmri" / "data" / "bwh" / "metadata.csv"
FEATURES_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"
OUT_ROI = REPO_ROOT / "results" / "biomarkers" / "adc_variants.csv"
OUT_SUM = REPO_ROOT / "results" / "biomarkers" / "adc_variants_summary.csv"

# b in s/mm². ADC stored in units of (s/mm²)^-1 (i.e. mm²/s); multiplied
# by 1000 if you want ×10⁻³ mm²/s. We keep raw units throughout.
ADC_RANGES = {
    "std":       (0,    1000),
    "ext1500":   (0,    1500),
    "ext2000":   (0,    2000),
    "full":      (0,    3500),
    "high":      (1000, 3500),
    "midrange":  (250,  1500),
}
DKI_BMAX = 2500
DIFF_BINS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
MU_COLS = [f"nuts_D_{d:.2f}" for d in DIFF_BINS]

N_BOOT = 2000
ALPHA = 0.05
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# ADC fitters
# ---------------------------------------------------------------------------

def fit_monoexp(bvals: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    """Log-linear monoexponential. Returns (adc, r2)."""
    mask = sig > 0
    if mask.sum() < 2:
        return float("nan"), float("nan")
    b = bvals[mask]
    y = np.log(sig[mask])
    slope, intercept = np.polyfit(b, y, 1)
    yhat = slope * b + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-slope), r2


def fit_dki(bvals: np.ndarray, sig: np.ndarray) -> tuple[float, float, float]:
    """Kurtosis fit S = S0 · exp(-bD + (bD)²·K/6). Returns (D, K, r2).

    Uses log-linear-fit ADC as starting point, K=0.8 (typical prostate).
    """
    mask = (sig > 0) & (bvals <= DKI_BMAX)
    if mask.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    b = bvals[mask].astype(float)
    s = sig[mask].astype(float)
    s0_init = s.max()
    d_init, _ = fit_monoexp(b, s)
    if not np.isfinite(d_init) or d_init <= 0:
        d_init = 1e-3
    p0 = (s0_init, d_init, 0.8)

    def model(b, S0, D, K):
        return S0 * np.exp(-b * D + (b * D) ** 2 * K / 6.0)

    try:
        popt, _ = curve_fit(
            model, b, s, p0=p0,
            bounds=([1e-6, 1e-6, 0.0], [np.inf, 1e-2, 3.0]),
            maxfev=5000,
        )
        S0, D, K = popt
        yhat = model(b, S0, D, K)
        ss_res = float(np.sum((s - yhat) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(D), float(K), r2
    except Exception:
        return float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Per-ROI computation
# ---------------------------------------------------------------------------

def build_adc_table(features: pd.DataFrame) -> pd.DataFrame:
    with open(SIGNAL_JSON, "r") as f:
        signal_data = json.load(f)

    rows = []
    for patient_id, rois in signal_data.items():
        for roi_name, roi in rois.items():
            anatomical_region = roi["anatomical_region"]
            is_tumor = "tumor" in anatomical_region
            if "tz" in anatomical_region:
                zone = "tz"
            elif "pz" in anatomical_region:
                zone = "pz"
            else:
                continue
            roi_id = f"{patient_id}_{zone}_{'tumor' if is_tumor else 'normal'}"
            bvals = np.asarray(roi["b_values"], dtype=float)
            sig = np.asarray(roi["signal_values"], dtype=float)

            entry: dict = {"roi_id": roi_id, "zone": zone, "is_tumor": is_tumor}
            # Monoexp variants
            for name, (b_lo, b_hi) in ADC_RANGES.items():
                m = (bvals >= b_lo) & (bvals <= b_hi)
                adc, r2 = fit_monoexp(bvals[m], sig[m])
                entry[f"adc_{name}"] = adc
                entry[f"adc_{name}_r2"] = r2
            # DKI
            d, k, r2 = fit_dki(bvals, sig)
            entry["adc_DKI_D"] = d
            entry["adc_DKI_K"] = k
            entry["adc_DKI_r2"] = r2
            rows.append(entry)

    df = pd.DataFrame(rows)

    # First-moment "spectral ADC" (units: μm²/ms = ×10⁻³ mm²/s).
    # Convert to mm²/s by ×1e-3 so it matches the monoexp ADCs.
    spec_first_moment = (features[MU_COLS].values * DIFF_BINS[None, :]).sum(axis=1) * 1e-3
    fm = pd.DataFrame({"roi_id": features["roi_id"], "adc_spec_M1": spec_first_moment})
    df = df.merge(fm, on="roi_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def loocv_lr_probs(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    n = len(y)
    yp = np.zeros(n)
    for tr, te in LeaveOneOut().split(X):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, random_state=42).fit(
            scaler.transform(X[tr]), y[tr]
        )
        yp[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    return yp


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
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


# Paired DeLong (Sun & Xu 2014 fast midrank)
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = T
    return out


def paired_delong(p1, p2, y) -> tuple[float, float, float]:
    preds = np.vstack([p1, p2])
    pos = y == 1
    neg = y == 0
    m = pos.sum()
    nn = neg.sum()
    V10 = np.zeros((2, m))
    V01 = np.zeros((2, nn))
    aucs = np.zeros(2)
    for k in range(2):
        sp = preds[k][pos]; sn = preds[k][neg]
        TR = _midrank(np.concatenate([sp, sn]))
        TX = _midrank(sp); TY = _midrank(sn)
        V10[k] = (TR[:m] - TX) / nn
        V01[k] = 1.0 - (TR[m:] - TY) / m
        aucs[k] = V10[k].mean()
    S10 = np.cov(V10, ddof=1); S01 = np.cov(V01, ddof=1)
    S = S10 / m + S01 / nn
    delta = aucs[0] - aucs[1]
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    if var <= 0:
        return float(delta), float("nan"), float("nan")
    z = delta / np.sqrt(var)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(delta), float(z), float(p)


def partial_spearman(x, y, z):
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.column_stack([np.ones_like(rz, dtype=float), rz])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    return float(np.corrcoef(rx - Z @ bx, ry - Z @ by)[0, 1])


def partial_spearman_p(rho, n, k=1):
    df = n - 2 - k
    if df <= 0 or not np.isfinite(rho) or abs(rho) >= 1.0:
        return float("nan")
    t = rho * np.sqrt(df / (1.0 - rho * rho))
    return float(2.0 * (1.0 - stats.t.cdf(abs(t), df)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    features = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features.csv: {len(features)} ROIs")

    adc_df = build_adc_table(features)
    adc_df.to_csv(OUT_ROI, index=False)
    print(f"Wrote per-ROI ADC variants: {OUT_ROI} ({len(adc_df)} rows)")

    # Merge ADC variants into features
    df = features.merge(
        adc_df.drop(columns=["zone", "is_tumor"]), on="roi_id", how="left"
    )

    variant_cols = [c for c in adc_df.columns if c.startswith("adc_") and not c.endswith("_r2") and c != "adc_DKI_K"]
    # Include the original 'adc' column from features.csv (sanity check vs std)
    variant_cols = ["adc"] + variant_cols + ["adc_DKI_K"]

    rows: list[dict] = []

    # ---- Task 1: tumor-vs-normal, per zone ----
    for zone in ("pz", "tz"):
        sub = df[df["zone"] == zone].copy()
        y = sub["is_tumor"].astype(int).values
        n = len(sub)
        for vc in variant_cols:
            x = sub[vc].values.astype(float)
            if not np.isfinite(x).all():
                continue
            X = x.reshape(-1, 1)
            p = loocv_lr_probs(X, y)
            auc, lo, hi = bootstrap_auc(y, p)
            rows.append(dict(
                task="detection", zone=zone, n=n, variant=vc,
                auc=auc, ci_lo=lo, ci_hi=hi,
                spearman_rho=np.nan, spearman_p=np.nan,
                partial_rho_mu050=np.nan, partial_p_mu050=np.nan,
                delong_p_vs_std=np.nan,
            ))

    # ---- Task 2 & 3: grading on tumor ROIs with valid GGG ----
    tum = df[(df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)].copy()
    n_tum = len(tum)
    y_bin = (tum["ggg"].values >= 3).astype(int)
    ggg_cont = tum["ggg"].values.astype(float)
    mu050 = tum["nuts_D_0.50"].values.astype(float)

    # Compute LOOCV probs for each variant and the std-ADC baseline
    loocv_probs: dict[str, np.ndarray] = {}
    for vc in variant_cols:
        x = tum[vc].values.astype(float)
        if not np.isfinite(x).all():
            continue
        p = loocv_lr_probs(x.reshape(-1, 1), y_bin)
        loocv_probs[vc] = p

    baseline_probs = loocv_probs.get("adc")  # original std-ADC from features.csv

    for vc in variant_cols:
        x = tum[vc].values.astype(float)
        if not np.isfinite(x).all():
            continue

        rho_uncond, p_uncond = stats.spearmanr(x, ggg_cont)
        rho_partial = partial_spearman(mu050, ggg_cont, x)
        p_partial = partial_spearman_p(rho_partial, n_tum, k=1)

        p = loocv_probs[vc]
        auc, lo, hi = bootstrap_auc(y_bin, p)

        if baseline_probs is not None and vc != "adc":
            _, _, dp = paired_delong(p, baseline_probs, y_bin)
        else:
            dp = np.nan

        rows.append(dict(
            task="grading_binary", zone="pooled", n=n_tum, variant=vc,
            auc=auc, ci_lo=lo, ci_hi=hi,
            spearman_rho=float(rho_uncond), spearman_p=float(p_uncond),
            partial_rho_mu050=rho_partial, partial_p_mu050=p_partial,
            delong_p_vs_std=dp,
        ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_SUM, index=False)
    print(f"Wrote summary: {OUT_SUM} ({len(out)} rows)\n")

    # ---------------------- Print pretty tables ----------------------

    def fmt(v, fmt_spec=".3f", na="—"):
        if v is None or not np.isfinite(v):
            return na
        return f"{v:{fmt_spec}}"

    print("=" * 110)
    print(f"  DETECTION  (tumor-vs-normal, LOOCV LR, single-feature)  •  bootstrap 95% CI (B={N_BOOT})")
    print("=" * 110)
    for zone in ("pz", "tz"):
        sub = out[(out["task"] == "detection") & (out["zone"] == zone)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("auc", ascending=False)
        n = int(sub["n"].iloc[0])
        print(f"\n[{zone}] N={n}")
        print(f"  {'variant':14s}  {'AUC':>5s}  {'95% CI':>16s}")
        for _, r in sub.iterrows():
            ci = f"[{fmt(r['ci_lo'])},{fmt(r['ci_hi'])}]"
            print(f"  {r['variant']:14s}  {fmt(r['auc']):>5s}  {ci:>16s}")

    print("\n" + "=" * 110)
    print(f"  GRADING-CONTINUOUS (Spearman) + GRADING-BINARY (LOOCV AUC + DeLong vs std-ADC)  •  N=29 pooled")
    print("=" * 110)
    sub = out[out["task"] == "grading_binary"].copy()
    sub = sub.sort_values("auc", ascending=False)
    print(f"  {'variant':14s}  {'ρ_cont':>7s}  {'p_cont':>7s}  {'AUC':>5s}  {'95% CI':>16s}  {'ΔAUC':>6s}  {'DeLong p':>9s}  {'partial ρ(μ0.50|var)':>22s}  {'p':>7s}")
    base_auc = sub.loc[sub["variant"] == "adc", "auc"].values
    base_auc = float(base_auc[0]) if len(base_auc) else float("nan")
    for _, r in sub.iterrows():
        dauc = r["auc"] - base_auc if np.isfinite(base_auc) else float("nan")
        ci = f"[{fmt(r['ci_lo'])},{fmt(r['ci_hi'])}]"
        print(
            f"  {r['variant']:14s}  "
            f"{fmt(r['spearman_rho'],'+.3f'):>7s}  "
            f"{fmt(r['spearman_p'],'.4f'):>7s}  "
            f"{fmt(r['auc']):>5s}  "
            f"{ci:>16s}  "
            f"{fmt(dauc,'+.3f'):>6s}  "
            f"{fmt(r['delong_p_vs_std'],'.4f'):>9s}  "
            f"{fmt(r['partial_rho_mu050'],'+.3f'):>22s}  "
            f"{fmt(r['partial_p_mu050'],'.4f'):>7s}"
        )

    # Best per task — explicit "different range for different task?" check
    print("\n" + "=" * 110)
    print("  BEST VARIANT PER TASK  (Patrick's afterthought — is detection best at one range, grading at another?)")
    print("=" * 110)
    det_best = {}
    for zone in ("pz", "tz"):
        s = out[(out["task"] == "detection") & (out["zone"] == zone)].sort_values("auc", ascending=False).head(3)
        det_best[zone] = s
        print(f"\n[detection / {zone}] top 3 by AUC:")
        for _, r in s.iterrows():
            print(f"   {r['variant']:14s}  AUC={r['auc']:.3f}  CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")

    grad_best = out[out["task"] == "grading_binary"].sort_values("auc", ascending=False).head(3)
    print("\n[grading-binary / pooled] top 3 by AUC:")
    for _, r in grad_best.iterrows():
        print(f"   {r['variant']:14s}  AUC={r['auc']:.3f}  CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")

    grad_cont_best = out[out["task"] == "grading_binary"].copy()
    grad_cont_best["abs_rho"] = grad_cont_best["spearman_rho"].abs()
    grad_cont_best = grad_cont_best.sort_values("abs_rho", ascending=False).head(3)
    print("\n[grading-continuous / pooled] top 3 by |ρ|:")
    for _, r in grad_cont_best.iterrows():
        print(f"   {r['variant']:14s}  ρ={r['spearman_rho']:+.3f}  p={r['spearman_p']:.4f}")


if __name__ == "__main__":
    run()
