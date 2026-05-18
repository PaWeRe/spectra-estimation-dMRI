"""
§10f extended: continuous-GGG subset sweep.

Tests whether spectral feature subsets carry continuous tumor-grading
signal. Primary statistic: Spearman ρ of feature (or LR discriminant
score) vs continuous GGG across tumor ROIs.

Motivation (2026-05-17 session, after §10g closed):
- Per-bin Spearman against continuous GGG surfaced D=0.50 (μ ρ=+0.57,
  p=0.001) and D=2.0 (μ ρ=−0.47, p=0.009) as stronger predictors than
  D=0.25 (ρ=+0.44) or D=3.0 (ρ=−0.16).
- This contradicts the tumor-vs-normal finding (where D=0.25, D=3.00
  dominate) — separate "detection" and "grading" axes.
- Need to test (a) whether subsets of intermediate bins together
  outperform single-bin D=0.50, (b) whether posterior σ adds signal,
  (c) whether the pattern survives zone stratification.

Primary task: tumor ROIs with non-zero GGG (N=29: PZ 21, TZ 8).
Primary statistic: Spearman ρ vs continuous GGG (1..5).
For multi-feature subsets: score = LOOCV LR-probability trained on
binary GGG≥3, then Spearman the score vs continuous GGG (uses the
binary task as a dimensionality-reduction trick to get a scalar score).

Outputs: results/biomarkers/ggg_continuous_sweep.csv

Usage:
    uv run python scripts/ggg_continuous_sweep.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"
OUTPUT_CSV = REPO_ROOT / "results" / "biomarkers" / "ggg_continuous_sweep.csv"

DIFFUSIVITIES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]
MU_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
SIGMA_COLS = [f"nuts_std_D_{d:.2f}" for d in DIFFUSIVITIES]

INTERMEDIATE_BINS = [0.5, 0.75, 1.0, 1.5, 2.0]
OUTER_BINS = [0.25, 3.0, 20.0]

N_BOOTSTRAP = 1000
ALPHA = 0.05
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def single_score(values: np.ndarray) -> np.ndarray:
    """Single-feature score is just the values themselves."""
    return values


def loocv_lr_score(X: np.ndarray, y_binary: np.ndarray, C: float = 1.0) -> np.ndarray:
    """LOOCV LR-probability trained on binary GGG>=3. Used as scalar
    discriminant score for multi-feature subsets. Note: LR is binary-
    trained, but the predicted probability is then Spearman-correlated
    against the *continuous* GGG label, which uses more of the
    information than a strict binary endpoint.
    """
    n = len(y_binary)
    yp = np.zeros(n)
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42).fit(
            sc.transform(X[tr]), y_binary[tr]
        )
        yp[te] = clf.predict_proba(sc.transform(X[te]))[0, 1]
    return yp


def spearman_with_ci(
    score: np.ndarray, ggg: np.ndarray, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float, float, float]:
    """Spearman ρ with bootstrap percentile CI. Returns (rho, p, ci_lo, ci_hi).
    Sign-invariant: takes |rho| direction from the point estimate."""
    rho, p = stats.spearmanr(score, ggg)
    if np.isnan(rho):
        return (np.nan, np.nan, np.nan, np.nan)
    n = len(score)
    rhos = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if np.unique(score[idx]).size < 2 or np.unique(ggg[idx]).size < 2:
            continue
        r, _ = stats.spearmanr(score[idx], ggg[idx])
        if not np.isnan(r):
            rhos.append(r)
    if not rhos:
        return (rho, p, np.nan, np.nan)
    return (
        float(rho),
        float(p),
        float(np.percentile(rhos, 100 * ALPHA / 2)),
        float(np.percentile(rhos, 100 * (1 - ALPHA / 2))),
    )


# ---------------------------------------------------------------------------
# Feature set enumeration
# ---------------------------------------------------------------------------

def label_for(cols: list[str]) -> str:
    mu_bins, sig_bins = [], []
    for c in cols:
        d = c.split("_D_")[-1]
        (sig_bins if c.startswith("nuts_std_") else mu_bins).append(d)
    parts = []
    if mu_bins:
        parts.append("mu{" + ",".join(mu_bins) + "}")
    if sig_bins:
        parts.append("sig{" + ",".join(sig_bins) + "}")
    return "+".join(parts)


def enumerate_feature_sets() -> list[tuple[str, str, tuple[str, ...]]]:
    sets: list[tuple[str, str, tuple[str, ...]]] = []
    # Block A: μ singles, pairs, triples + intermediate-only + outer-only + full-8
    for c in MU_COLS:
        sets.append(("A_mu", label_for([c]), (c,)))
    for combo in itertools.combinations(MU_COLS, 2):
        sets.append(("A_mu", label_for(list(combo)), combo))
    for combo in itertools.combinations(MU_COLS, 3):
        sets.append(("A_mu", label_for(list(combo)), combo))
    sets.append(
        (
            "A_mu",
            "mu_intermediate_only{0.50-2.00}",
            tuple(f"nuts_D_{d:.2f}" for d in INTERMEDIATE_BINS),
        )
    )
    sets.append(
        (
            "A_mu",
            "mu_outer_only{0.25,3.00,20.00}",
            tuple(f"nuts_D_{d:.2f}" for d in OUTER_BINS),
        )
    )
    sets.append(("A_mu", "mu_full8", tuple(MU_COLS)))

    # Block B: σ singles, pairs, triples + full-8
    for c in SIGMA_COLS:
        sets.append(("B_sig", label_for([c]), (c,)))
    for combo in itertools.combinations(SIGMA_COLS, 2):
        sets.append(("B_sig", label_for(list(combo)), combo))
    for combo in itertools.combinations(SIGMA_COLS, 3):
        sets.append(("B_sig", label_for(list(combo)), combo))
    sets.append(("B_sig", "sig_full8", tuple(SIGMA_COLS)))

    # Block C: μ+σ targeted joints
    for d, mu, sg in zip(DIFFUSIVITIES, MU_COLS, SIGMA_COLS):
        sets.append(("C_joint", label_for([mu, sg]), (mu, sg)))
    sets.append(
        (
            "C_joint",
            "mu+sig_at_grading_bins{0.50,2.00}",
            ("nuts_D_0.50", "nuts_std_D_0.50", "nuts_D_2.00", "nuts_std_D_2.00"),
        )
    )
    sets.append(
        (
            "C_joint",
            "mu+sig_at_headline_bins{0.25,3.00}",
            ("nuts_D_0.25", "nuts_std_D_0.25", "nuts_D_3.00", "nuts_std_D_3.00"),
        )
    )
    sets.append(("C_joint", "mu+sig_full16", tuple(MU_COLS) + tuple(SIGMA_COLS)))

    return sets


# ---------------------------------------------------------------------------
# Zone preparation
# ---------------------------------------------------------------------------

def prepare_ggg_tumors(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Return tumor ROIs with valid GGG, filtered by zone."""
    sub = df[
        (df["is_tumor"]) & (df["ggg"].notna()) & (df["ggg"] != 0)
    ].copy()
    if zone != "pooled":
        sub = sub[sub["zone"] == zone]
    return sub


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {FEATURES_CSV.relative_to(REPO_ROOT)}: {df.shape}")

    feature_sets = enumerate_feature_sets()
    print(f"Enumerated {len(feature_sets)} feature sets")

    rows = []
    zones = ["pz", "tz", "pooled"]

    # Headline diagnostics
    print("\nZone counts (tumor ROIs with valid GGG):")
    for zone in zones:
        sub = prepare_ggg_tumors(df, zone)
        n = len(sub)
        g = sub["ggg"].values
        n_lo = (g <= 2).sum(); n_hi = (g >= 3).sum()
        print(f"  {zone:6s} N={n}  low={n_lo} hi={n_hi}  range GGG=[{int(g.min())}, {int(g.max())}]")

    # ADC reference
    print("\nADC vs continuous GGG (reference baseline):")
    adc_refs = {}
    for zone in zones:
        sub = prepare_ggg_tumors(df, zone)
        if len(sub) < 5:
            continue
        rho, p, lo, hi = spearman_with_ci(sub["adc"].values, sub["ggg"].values)
        adc_refs[zone] = (rho, lo, hi)
        print(f"  {zone:6s} ρ={rho:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")

    # Sweep
    print("\nRunning sweep ...")
    n_total = len(feature_sets) * len(zones)
    i = 0
    for block, label, cols in feature_sets:
        for zone in zones:
            sub = prepare_ggg_tumors(df, zone)
            if len(sub) < 5:
                continue
            ggg = sub["ggg"].values
            y_bin = (ggg >= 3).astype(int)
            X = sub[list(cols)].values

            # Scalar score
            if X.shape[1] == 1:
                score = single_score(X[:, 0])
            else:
                if len(np.unique(y_bin)) < 2:
                    # All same binary class — fall back to mean of standardized features
                    score = StandardScaler().fit_transform(X).mean(axis=1)
                else:
                    score = loocv_lr_score(X, y_bin, C=1.0)

            rho, p, lo, hi = spearman_with_ci(score, ggg)

            # ADC reference for this zone
            adc_rho, adc_lo, adc_hi = adc_refs.get(zone, (np.nan, np.nan, np.nan))
            # Beats ADC: |ρ| > |adc_rho| AND CIs don't overlap with ADC's
            beats_adc = (
                not np.isnan(rho)
                and not np.isnan(adc_rho)
                and abs(rho) > abs(adc_rho)
                # for orientation, compare absolute magnitudes
            )

            i += 1
            if i % 100 == 0:
                print(f"  {i}/{n_total} ...")

            rows.append(
                {
                    "block": block,
                    "feature_set": label,
                    "n_features": len(cols),
                    "feature_cols": ",".join(cols),
                    "zone": zone,
                    "n": len(sub),
                    "spearman_rho": rho,
                    "spearman_p": p,
                    "rho_ci_lo": lo,
                    "rho_ci_hi": hi,
                    "adc_rho": adc_rho,
                    "abs_rho_gt_abs_adc": beats_adc,
                    "abs_rho": abs(rho) if not np.isnan(rho) else np.nan,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def bonferroni_singles(out: pd.DataFrame, zone: str) -> None:
    """Bonferroni-corrected per-bin Spearman: 16 tests (8 μ + 8 σ) per zone."""
    print(f"\n--- {zone.upper()}: per-bin Spearman, Bonferroni-corrected (16 tests, α=0.05 → p<{0.05/16:.4f}) ---")
    sub = out[(out["zone"] == zone) & (out["n_features"] == 1) & (out["block"].isin(["A_mu", "B_sig"]))]
    sub = sub.sort_values("spearman_p")
    print(
        sub[
            ["feature_set", "spearman_rho", "rho_ci_lo", "rho_ci_hi", "spearman_p"]
        ].to_string(index=False, float_format=lambda x: f"{x:+.4f}")
    )
    survivors = sub[sub["spearman_p"] < 0.05 / 16]
    if not survivors.empty:
        print(f"\n  Bonferroni survivors at α=0.05: {survivors['feature_set'].tolist()}")
    else:
        print("\n  NO Bonferroni survivors at α=0.05 in this zone.")


def top_per_block(out: pd.DataFrame, zone: str, k: int = 8) -> None:
    print(f"\n--- {zone.upper()}: top-{k} by |ρ| across all blocks ---")
    sub = out[out["zone"] == zone].copy().sort_values("abs_rho", ascending=False).head(k)
    print(
        sub[
            [
                "block",
                "feature_set",
                "n_features",
                "spearman_rho",
                "rho_ci_lo",
                "rho_ci_hi",
                "spearman_p",
                "adc_rho",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:+.4f}")
    )


def intermediate_vs_outer(out: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("INTERMEDIATE-ONLY vs OUTER-ONLY vs REFERENCES (continuous-GGG ρ)")
    print("=" * 80)
    labels_of_interest = [
        "mu_intermediate_only{0.50-2.00}",
        "mu_outer_only{0.25,3.00,20.00}",
        "mu{0.25}",
        "mu{0.50}",
        "mu{2.00}",
        "mu{3.00}",
        "mu_full8",
    ]
    for zone in ["pz", "tz", "pooled"]:
        sub = out[(out["zone"] == zone) & (out["feature_set"].isin(labels_of_interest))]
        if sub.empty:
            continue
        print(f"\n--- {zone.upper()} ---")
        for label in labels_of_interest:
            row = sub[sub["feature_set"] == label]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"  {label:40s}  ρ={r['spearman_rho']:+.3f}  CI=[{r['rho_ci_lo']:+.3f},{r['rho_ci_hi']:+.3f}]  p={r['spearman_p']:.4f}"
            )


def headline_table(out: pd.DataFrame) -> None:
    """Compact headline table for the manuscript: best single-bin per zone +
    intermediate-only + outer-only + full-8 + ADC."""
    print("\n" + "=" * 80)
    print("MANUSCRIPT-READY SUMMARY (continuous-GGG Spearman ρ)")
    print("=" * 80)
    print(f"{'Zone':6s} {'Feature set':35s} {'ρ':>8s} {'95% CI':>20s} {'p':>10s}")
    for zone in ["pz", "tz", "pooled"]:
        # Best single μ
        sub = out[(out["zone"] == zone) & (out["block"] == "A_mu") & (out["n_features"] == 1)]
        sub = sub.copy().sort_values("abs_rho", ascending=False)
        # Single bins
        for label in [
            "best_single_μ",
            "mu_intermediate_only{0.50-2.00}",
            "mu_outer_only{0.25,3.00,20.00}",
            "mu_full8",
            "ADC",
        ]:
            if label == "best_single_μ":
                if sub.empty:
                    continue
                r = sub.iloc[0]
                ci = f"[{r['rho_ci_lo']:+.3f},{r['rho_ci_hi']:+.3f}]"
                print(f"{zone:6s} {label + '=' + r['feature_set']:35s} {r['spearman_rho']:+8.3f} {ci:>20s} {r['spearman_p']:>10.4f}")
            elif label == "ADC":
                adc_row = out[(out["zone"] == zone) & (out["block"] == "A_mu") & (out["n_features"] == 1)].iloc[0]
                # We didn't run ADC through sweep — use the adc_rho field
                adc_rho = adc_row["adc_rho"]
                print(f"{zone:6s} {'ADC':35s} {adc_rho:+8.3f} {'(ref)':>20s} {'-':>10s}")
            else:
                row = out[(out["zone"] == zone) & (out["feature_set"] == label)]
                if row.empty:
                    continue
                r = row.iloc[0]
                ci = f"[{r['rho_ci_lo']:+.3f},{r['rho_ci_hi']:+.3f}]"
                print(f"{zone:6s} {label:35s} {r['spearman_rho']:+8.3f} {ci:>20s} {r['spearman_p']:>10.4f}")
        print()


def main() -> int:
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run recompute.py first.", file=sys.stderr)
        return 1

    out = run_sweep()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV.relative_to(REPO_ROOT)}: {out.shape}")

    intermediate_vs_outer(out)

    for zone in ["pz", "tz", "pooled"]:
        bonferroni_singles(out, zone)
        top_per_block(out, zone, k=10)

    headline_table(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
