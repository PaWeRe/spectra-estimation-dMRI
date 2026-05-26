"""
8-feat NUTS-LR coefficient decomposition onto D_vec — Exec 4P1.

Path A' centerpiece test (§10h re-scoped):
    "How much of the LR discriminant is just first-moment projection onto
     the diffusivity grid?"

For each (zone × task), fit standardized LogisticRegression on the 8 NUTS
posterior-mean spectrum bins and extract the coefficient vector w. Then:

  1. cos(w, D_vec) — alignment of the classifier direction with the
     diffusivity vector. High |cos| → discriminant is essentially a
     scalar first-moment projection (i.e., ADC-like). Low |cos| → the
     classifier exploits non-D-vec spectral directions.

  2. cos(w, D_vec) with D=20.0 free-water bin removed (7 bins). The
     D=20 entry dominates the cosine numerically; the 7-bin variant is
     the tissue-restricted analog.

  3. cos(w_T, w_G) — between-task angle of detection (tumor-vs-normal)
     and grading (GGG≥3) coefficient vectors. Low |cos| → detection
     and grading axes are spectrally orthogonal (§10f finding).

Bootstrap percentile 95% CI on all cosines (B=1000, seed=42, paired
resamples for cross-task cos so the bootstrap distributions of w_T and
w_G are jointly conditioned on the same superset of patients).

Endpoints:
- Tumor-vs-normal: y = is_tumor on all 149 ROIs (zone-restricted as
  applicable).
- GGG ≥ 3:        y = (ggg ≥ 3) on tumor ROIs with valid GGG (~29).

Classifier: LogisticRegression(C=1.0), StandardScaler on full sample.
Coefficients reported in both standardized (per-SD) and raw spaces. Raw
coefficients are the natural input for cos(·, D_vec) because spectrum
bins are already on a common volume-fraction scale.

Outputs:
- results/biomarkers/lr_coef_decomp.csv         (per-task rows)
- results/biomarkers/lr_coef_decomp_cross.csv   (cross-task cos rows)
- console summary

Usage:
    uv run python scripts/lr_coef_decomp.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"
OUTPUT_CSV = REPO_ROOT / "results" / "biomarkers" / "lr_coef_decomp.csv"
OUTPUT_CROSS_CSV = REPO_ROOT / "results" / "biomarkers" / "lr_coef_decomp_cross.csv"

DIFFUSIVITIES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]
MU_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
D_VEC = np.array(DIFFUSIVITIES, dtype=float)
D_VEC_NO_FREE = D_VEC[:-1]  # drop D=20.0

LR_C = 1.0
N_BOOTSTRAP = 1000
ALPHA = 0.05
RNG_SEED = 42


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def fit_lr(X: np.ndarray, y: np.ndarray, C: float = LR_C) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit standardized LR. Return (w_std, w_raw, intercept_std, scales).

    w_std lives in standardized feature space; w_raw = w_std / scales
    is the equivalent linear weight on raw features.
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=42).fit(Xs, y)
    w_std = clf.coef_.ravel()
    scales = scaler.scale_.copy()
    # Guard against zero-variance bins
    safe_scales = np.where(scales > 0, scales, 1.0)
    w_raw = w_std / safe_scales
    return w_std, w_raw, float(clf.intercept_[0]), safe_scales


def prepare_tumor_vs_normal(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    sub = df.copy()
    if zone != "pooled":
        sub = sub[sub["zone"] == zone]
    return sub


def prepare_ggg(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    sub = df[(df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)].copy()
    if zone != "pooled":
        sub = sub[sub["zone"] == zone]
    return sub


def bootstrap_coefs(X: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Return (n_boot, n_features) array of raw coefficient vectors.

    Resamples (X, y) with replacement; refits standardized LR each time;
    unscales coefficients with the bootstrap-sample's own SDs.
    """
    n, p = X.shape
    out = np.full((n_boot, p), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        Xb = X[idx]
        try:
            _, w_raw, _, _ = fit_lr(Xb, yb, C=LR_C)
            out[b] = w_raw
        except Exception:
            continue
    return out


def percentile_ci(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.percentile(finite, 100 * alpha / 2)),
        float(np.percentile(finite, 100 * (1 - alpha / 2))),
    )


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features.csv: {len(df)} ROIs\n")

    tasks = {
        "tumor_vs_normal": (prepare_tumor_vs_normal, "is_tumor"),
        "ggg_ge_3": (prepare_ggg, None),  # special y below
    }

    rows: list[dict] = []
    cross_rows: list[dict] = []

    # Cache bootstrap coefficient draws per (zone, task) for cross-task pairing.
    coef_draws: dict[tuple[str, str], np.ndarray] = {}
    point_coefs_raw: dict[tuple[str, str], np.ndarray] = {}
    point_coefs_std: dict[tuple[str, str], np.ndarray] = {}

    for zone in ("pooled", "pz", "tz"):
        for task, (prep, y_col) in tasks.items():
            sub = prep(df, zone)
            n = len(sub)
            if n < 16:
                print(f"  [{zone}/{task}] N={n} too small, skipping")
                continue

            if task == "tumor_vs_normal":
                y = sub["is_tumor"].astype(int).values
            else:
                y = (sub["ggg"].values >= 3).astype(int)

            n_pos = int(y.sum())
            n_neg = int((1 - y).sum())
            if n_pos < 4 or n_neg < 4:
                print(f"  [{zone}/{task}] N={n} pos={n_pos} neg={n_neg} — too imbalanced, skipping")
                continue

            X = sub[MU_COLS].values.astype(float)

            # Point estimate
            w_std, w_raw, b0, scales = fit_lr(X, y)
            point_coefs_raw[(zone, task)] = w_raw
            point_coefs_std[(zone, task)] = w_std

            cos_raw_full = cos_sim(w_raw, D_VEC)
            cos_raw_nofree = cos_sim(w_raw[:-1], D_VEC_NO_FREE)
            cos_std_full = cos_sim(w_std, D_VEC)

            # Bootstrap for CIs
            rng = np.random.default_rng(RNG_SEED + hash((zone, task)) % (2**31))
            boots = bootstrap_coefs(X, y, N_BOOTSTRAP, rng)
            coef_draws[(zone, task)] = boots

            boot_cos_raw_full = np.array([cos_sim(w, D_VEC) for w in boots])
            boot_cos_raw_nofree = np.array([cos_sim(w[:-1], D_VEC_NO_FREE) for w in boots])
            lo_full, hi_full = percentile_ci(boot_cos_raw_full)
            lo_nofree, hi_nofree = percentile_ci(boot_cos_raw_nofree)
            lo_norm, hi_norm = percentile_ci(np.linalg.norm(boots, axis=1))

            row = dict(
                zone=zone, task=task, n=n, n_pos=n_pos, n_neg=n_neg,
                c=LR_C, intercept_std=b0,
                w_norm_raw=float(np.linalg.norm(w_raw)),
                w_norm_raw_ci_lo=lo_norm, w_norm_raw_ci_hi=hi_norm,
                cos_w_Dvec_raw=cos_raw_full,
                cos_w_Dvec_raw_ci_lo=lo_full, cos_w_Dvec_raw_ci_hi=hi_full,
                cos_w_Dvec_raw_nofree=cos_raw_nofree,
                cos_w_Dvec_raw_nofree_ci_lo=lo_nofree, cos_w_Dvec_raw_nofree_ci_hi=hi_nofree,
                cos_w_Dvec_std=cos_std_full,
            )
            for d, w_r, w_s in zip(DIFFUSIVITIES, w_raw, w_std):
                row[f"w_raw_D_{d:.2f}"] = float(w_r)
                row[f"w_std_D_{d:.2f}"] = float(w_s)
            rows.append(row)

            print(f"  [{zone}/{task}] N={n} pos={n_pos} neg={n_neg}  "
                  f"||w||={np.linalg.norm(w_raw):.3f}  "
                  f"cos(w,Dvec)={cos_raw_full:+.3f} [{lo_full:+.3f},{hi_full:+.3f}]  "
                  f"cos(w,Dvec\\D=20)={cos_raw_nofree:+.3f} [{lo_nofree:+.3f},{hi_nofree:+.3f}]")

    # Cross-task cosines per zone (paired bootstrap by row index)
    print("\n" + "-" * 80)
    print("Cross-task: cos(w_T, w_G) per zone (paired bootstrap)")
    print("-" * 80)
    for zone in ("pooled", "pz", "tz"):
        kT = (zone, "tumor_vs_normal")
        kG = (zone, "ggg_ge_3")
        if kT not in point_coefs_raw or kG not in point_coefs_raw:
            continue
        wT = point_coefs_raw[kT]
        wG = point_coefs_raw[kG]
        c_raw = cos_sim(wT, wG)
        c_raw_nofree = cos_sim(wT[:-1], wG[:-1])
        c_std = cos_sim(point_coefs_std[kT], point_coefs_std[kG])

        bT = coef_draws[kT]
        bG = coef_draws[kG]
        # The two tasks have different sample sizes and different bootstrap
        # rngs. Pair by bootstrap index (independent draws, not paired by
        # underlying sample) — gives a reasonable joint distribution of
        # angles without pretending the samples are nested.
        n_b = min(bT.shape[0], bG.shape[0])
        boot_cross_full = np.array([cos_sim(bT[i], bG[i]) for i in range(n_b)])
        boot_cross_nofree = np.array([cos_sim(bT[i, :-1], bG[i, :-1]) for i in range(n_b)])
        lo_full, hi_full = percentile_ci(boot_cross_full)
        lo_nofree, hi_nofree = percentile_ci(boot_cross_nofree)

        cross_rows.append(dict(
            zone=zone, c=LR_C,
            cos_wT_wG_raw=c_raw, cos_wT_wG_raw_ci_lo=lo_full, cos_wT_wG_raw_ci_hi=hi_full,
            cos_wT_wG_raw_nofree=c_raw_nofree,
            cos_wT_wG_raw_nofree_ci_lo=lo_nofree, cos_wT_wG_raw_nofree_ci_hi=hi_nofree,
            cos_wT_wG_std=c_std,
        ))
        print(f"  [{zone}] cos(w_T, w_G) raw={c_raw:+.3f} [{lo_full:+.3f},{hi_full:+.3f}]  "
              f"nofree={c_raw_nofree:+.3f} [{lo_nofree:+.3f},{hi_nofree:+.3f}]  "
              f"std={c_std:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV} ({len(out)} rows)")

    out_cross = pd.DataFrame(cross_rows)
    out_cross.to_csv(OUTPUT_CROSS_CSV, index=False)
    print(f"Wrote {OUTPUT_CROSS_CSV} ({len(out_cross)} rows)")

    return out, out_cross


def summarize(out: pd.DataFrame, out_cross: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print(f"  8-feat NUTS-LR coefficient decomposition  •  C={LR_C}  •  B={N_BOOTSTRAP}  •  α={ALPHA}")
    print("=" * 96)
    print("\nPer-bin raw coefficients w_raw (sign + magnitude) by (zone, task):\n")
    for _, r in out.iterrows():
        bins = "  ".join(f"{r[f'w_raw_D_{d:.2f}']:+.2f}" for d in DIFFUSIVITIES)
        print(f"  [{r['zone']:>6s}/{r['task']:<16s}] {bins}  "
              f"||w||={r['w_norm_raw']:.2f}")

    print("\nInterpretation guide:")
    print("  cos(w, D_vec) ≈ ±1  → discriminant ≈ first-moment of spectrum (ADC-like)")
    print("  cos(w, D_vec) ≈ 0   → discriminant orthogonal to D_vec (non-ADC direction)")
    print("  cos(w_T, w_G) ≈ 0   → detection and grading axes are spectrally separable")


if __name__ == "__main__":
    out, out_cross = run()
    summarize(out, out_cross)
