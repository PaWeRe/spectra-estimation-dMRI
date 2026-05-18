"""
Partial-Spearman test: does spectral signal carry continuous-GGG info
beyond ADC?

Per-bin spectrum-vs-GGG (§10f, 2026-05-17) found μ_D=0.50 ρ=+0.565 and
ADC ρ=−0.546 at pooled N=29 — similar magnitude. The pivotal question
for Path A' framing:

    ρ(spectral_feature, GGG | ADC) — partial Spearman with ADC partialled out.

If significant for any spectral feature → spectrum carries grading info
that ADC misses ("spectrum beats ADC" survives in a partial sense).
If null → spectrum and ADC are mechanistically redundant scalars
for grading at N=29 ("comparable performance, richer mechanism").

Implementation. Partial Spearman ρ(X, Y | Z) is computed as the Pearson
correlation between the ranks of X and Y after each is linearly
residualized on the ranks of Z (the standard non-parametric definition,
matches pingouin.partial_corr(method='spearman')).

Inference:
- Bootstrap percentile 95% CI (B=1000, seed=42; matches §10f sweep).
- Analytical p via t = ρ * sqrt((n - 2 - k) / (1 - ρ^2)),
  df = n - 2 - k, where k = 1 (one covariate partialled).

Features tested: 8 μ_D + 8 σ_D = 16 features. Bonferroni α = 0.05/16.

Outputs:
- results/biomarkers/partial_corr_ggg.csv
- console summary

Usage:
    uv run python scripts/partial_corr_ggg.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"
OUTPUT_CSV = REPO_ROOT / "results" / "biomarkers" / "partial_corr_ggg.csv"

DIFFUSIVITIES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]
MU_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
SIGMA_COLS = [f"nuts_std_D_{d:.2f}" for d in DIFFUSIVITIES]

N_BOOTSTRAP = 1000
ALPHA = 0.05
N_FEATURES_BONF = len(MU_COLS) + len(SIGMA_COLS)  # 16
RNG = np.random.default_rng(42)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """ρ(x, y | z) — rank-based partial correlation.

    Equivalent to pingouin.partial_corr(method='spearman') and to the
    Pearson of the rank-residuals.
    """
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    Z = np.column_stack([np.ones_like(rz, dtype=float), rz])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex = rx - Z @ bx
    ey = ry - Z @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def partial_spearman_pvalue(rho: float, n: int, k: int = 1) -> float:
    """Two-sided analytical p-value for partial correlation.

    t = rho * sqrt((n - 2 - k) / (1 - rho^2)), df = n - 2 - k.
    """
    df = n - 2 - k
    if df <= 0 or not np.isfinite(rho) or abs(rho) >= 1.0:
        return float("nan")
    t = rho * np.sqrt(df / (1.0 - rho * rho))
    return float(2.0 * (1.0 - stats.t.cdf(abs(t), df)))


def partial_spearman_with_ci(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float, float, float]:
    """Returns (rho_partial, p_analytical, ci_lo, ci_hi)."""
    n = len(x)
    rho = partial_spearman(x, y, z)
    p = partial_spearman_pvalue(rho, n)
    rhos = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        xs, ys, zs = x[idx], y[idx], z[idx]
        if (
            np.unique(xs).size < 2
            or np.unique(ys).size < 2
            or np.unique(zs).size < 2
        ):
            continue
        try:
            r = partial_spearman(xs, ys, zs)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(r):
            rhos.append(r)
    if not rhos:
        return rho, p, float("nan"), float("nan")
    return (
        rho,
        p,
        float(np.percentile(rhos, 100 * ALPHA / 2)),
        float(np.percentile(rhos, 100 * (1 - ALPHA / 2))),
    )


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def prepare_ggg_tumors(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    sub = df[(df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)].copy()
    if zone != "pooled":
        sub = sub[sub["zone"] == zone]
    return sub


def run() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features.csv: {len(df)} ROIs")

    rows: list[dict] = []
    feature_cols = MU_COLS + SIGMA_COLS

    for zone in ("pooled", "pz", "tz"):
        sub = prepare_ggg_tumors(df, zone)
        n = len(sub)
        if n < 4:
            print(f"  {zone}: N={n} too small, skipping")
            continue
        adc = sub["adc"].values.astype(float)
        ggg = sub["ggg"].values.astype(float)

        rho_adc, p_adc = spearman(adc, ggg)
        print(
            f"\nZone={zone:6s}  N={n}  ADC ρ(ADC, GGG)={rho_adc:+.3f} p={p_adc:.4f}"
        )

        for col in feature_cols:
            feat = sub[col].values.astype(float)
            rho_uncond, p_uncond = spearman(feat, ggg)
            rho_p, p_p, lo, hi = partial_spearman_with_ci(feat, ggg, adc)
            rows.append(
                {
                    "zone": zone,
                    "n": n,
                    "feature": col,
                    "rho_unconditional": rho_uncond,
                    "p_unconditional": p_uncond,
                    "rho_partial_given_adc": rho_p,
                    "p_partial_given_adc": p_p,
                    "partial_ci_lo": lo,
                    "partial_ci_hi": hi,
                    "rho_adc_vs_ggg": rho_adc,
                    "p_adc_vs_ggg": p_adc,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV} ({len(out)} rows)")
    return out


def summarize(out: pd.DataFrame) -> None:
    bonf = ALPHA / N_FEATURES_BONF
    print("\n" + "=" * 88)
    print(f"PARTIAL SPEARMAN ρ(feature, GGG | ADC)   Bonferroni α = 0.05/{N_FEATURES_BONF} = {bonf:.4f}")
    print("=" * 88)

    for zone in ("pooled", "pz", "tz"):
        sub = out[out["zone"] == zone].copy()
        if sub.empty:
            continue
        n = int(sub["n"].iloc[0])
        rho_adc = sub["rho_adc_vs_ggg"].iloc[0]
        p_adc = sub["p_adc_vs_ggg"].iloc[0]
        print(
            f"\n[{zone}] N={n}   reference: Spearman(ADC, GGG) = {rho_adc:+.3f}  p={p_adc:.4f}"
        )
        print(
            f"{'feature':16s}  {'ρ_uncond':>9s}  {'p_uncond':>9s}  {'ρ_partial':>10s}  {'partial CI':>20s}  {'p_partial':>10s}  flag"
        )
        sub = sub.reindex(sub["p_partial_given_adc"].abs().sort_values().index)
        for _, r in sub.iterrows():
            ci = f"[{r['partial_ci_lo']:+.3f},{r['partial_ci_hi']:+.3f}]"
            flag = (
                "***bonf"
                if r["p_partial_given_adc"] < bonf
                else ("*" if r["p_partial_given_adc"] < ALPHA else "")
            )
            ci_excludes_zero = r["partial_ci_lo"] * r["partial_ci_hi"] > 0
            if ci_excludes_zero and not flag:
                flag = "CI≠0"
            print(
                f"{r['feature']:16s}  {r['rho_unconditional']:+9.3f}  {r['p_unconditional']:9.4f}  "
                f"{r['rho_partial_given_adc']:+10.3f}  {ci:>20s}  {r['p_partial_given_adc']:10.4f}  {flag}"
            )


if __name__ == "__main__":
    out = run()
    summarize(out)
