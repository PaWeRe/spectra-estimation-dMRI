"""
§10g extended: bin-information sweep.

Stronger test of "intermediate bins are uninformative" in NUTS spectra,
extended with two additional questions raised 2026-05-17:

  - Do sub-region selections of the spectrum (e.g. intermediate-only)
    discriminate tumor/normal independently of {D=0.25, D=3.00}?
  - Does adding posterior std (σ_i) as a feature contribute signal beyond
    posterior mean (μ_i)?

Reference classifier: 2-feat NUTS-LR on {μ_D=0.25, μ_D=3.00}.
Decision rule: any subset with ΔAUC > 0.01 AND non-overlapping bootstrap
95% CI vs the reference → flag as Path A revision candidate.

Inputs : results/biomarkers/features.csv
Outputs: results/biomarkers/bin_information_sweep.csv
         (plus printed ranked summary per zone)

Usage:
    uv run python scripts/bin_information_sweep.py
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "results" / "biomarkers" / "features.csv"
OUTPUT_CSV = REPO_ROOT / "results" / "biomarkers" / "bin_information_sweep.csv"

DIFFUSIVITIES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0]
MU_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
SIGMA_COLS = [f"nuts_std_D_{d:.2f}" for d in DIFFUSIVITIES]

REFERENCE_FEATURES = ("nuts_D_0.25", "nuts_D_3.00")
INTERMEDIATE_BINS = [0.5, 0.75, 1.0, 1.5, 2.0]
OUTER_BINS = [0.25, 3.0, 20.0]

C_VALUES = [0.1, 1.0, 10.0]
N_BOOTSTRAP = 1000
ALPHA = 0.05  # 95 % CI
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Core LOOCV + bootstrap
# ---------------------------------------------------------------------------

def loocv_auc(X: np.ndarray, y: np.ndarray, C: float) -> tuple[float, np.ndarray]:
    """LOOCV logistic regression. Returns (AUC, y_pred_proba).

    For single-feature inputs, uses raw values as the score (no training)
    so the AUC matches `raw_rank_auc` and isn't distorted by fitting a
    1-D LR. Inverts predictions if AUC < 0.5.
    """
    n, p = X.shape
    if p == 1:
        scores = X[:, 0]
        auc = roc_auc_score(y, scores)
        if auc < 0.5:
            scores = -scores
            auc = 1.0 - auc
        return auc, scores

    y_pred = np.zeros(n)
    for train_idx, test_idx in LeaveOneOut().split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(Xtr, y[train_idx])
        y_pred[test_idx] = clf.predict_proba(Xte)[0, 1]

    auc = roc_auc_score(y, y_pred)
    if auc < 0.5:
        y_pred = 1.0 - y_pred
        auc = 1.0 - auc
    return auc, y_pred


def bootstrap_auc_ci(
    y: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI on AUC. Returns (ci_lower, ci_upper)."""
    rng = rng or RNG
    n = len(y)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], y_pred[idx]))
    if not aucs:
        return (np.nan, np.nan)
    return (
        float(np.percentile(aucs, 100 * alpha / 2)),
        float(np.percentile(aucs, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Subset enumeration
# ---------------------------------------------------------------------------

def label_for(cols: Iterable[str]) -> str:
    """Compact label like 'mu{0.25,3.00}' or 'mu{0.25}+sig{3.00}'."""
    mu_bins, sig_bins = [], []
    for c in cols:
        d = c.split("_D_")[-1]
        if c.startswith("nuts_std_"):
            sig_bins.append(d)
        else:
            mu_bins.append(d)
    parts = []
    if mu_bins:
        parts.append("mu{" + ",".join(mu_bins) + "}")
    if sig_bins:
        parts.append("sig{" + ",".join(sig_bins) + "}")
    return "+".join(parts)


def enumerate_feature_sets() -> list[tuple[str, str, tuple[str, ...]]]:
    """Return list of (block, label, feature_cols).

    Block A: μ-only subsets.
    Block B: σ-only subsets.
    Block C: μ+σ targeted joints (NOT exhaustive 16-pair sweep).
    """
    sets: list[tuple[str, str, tuple[str, ...]]] = []

    # --------------------- Block A: μ only ----------------------------
    # Singles
    for c in MU_COLS:
        sets.append(("A_mu", label_for([c]), (c,)))
    # Pairs
    for combo in itertools.combinations(MU_COLS, 2):
        sets.append(("A_mu", label_for(combo), combo))
    # Triples
    for combo in itertools.combinations(MU_COLS, 3):
        sets.append(("A_mu", label_for(combo), combo))
    # Intermediate-only
    inter_cols = tuple(f"nuts_D_{d:.2f}" for d in INTERMEDIATE_BINS)
    sets.append(("A_mu", "mu_intermediate_only{0.50-2.00}", inter_cols))
    # Outer-only
    outer_cols = tuple(f"nuts_D_{d:.2f}" for d in OUTER_BINS)
    sets.append(("A_mu", "mu_outer_only{0.25,3.00,20.00}", outer_cols))
    # Full 8
    sets.append(("A_mu", "mu_full8", tuple(MU_COLS)))

    # --------------------- Block B: σ only ----------------------------
    for c in SIGMA_COLS:
        sets.append(("B_sig", label_for([c]), (c,)))
    for combo in itertools.combinations(SIGMA_COLS, 2):
        sets.append(("B_sig", label_for(combo), combo))
    for combo in itertools.combinations(SIGMA_COLS, 3):
        sets.append(("B_sig", label_for(combo), combo))
    sets.append(("B_sig", "sig_full8", tuple(SIGMA_COLS)))

    # --------------------- Block C: μ+σ targeted ----------------------
    # Per-bin {μ_i, σ_i} pairs — tests if posterior width adds signal *at the same bin*
    for d, mu, sg in zip(DIFFUSIVITIES, MU_COLS, SIGMA_COLS):
        sets.append(("C_joint", label_for((mu, sg)), (mu, sg)))
    # Headline-bin 4-feat: {μ_0.25, σ_0.25, μ_3.00, σ_3.00}
    sets.append(
        (
            "C_joint",
            "mu+sig_at_headline_bins",
            ("nuts_D_0.25", "nuts_std_D_0.25", "nuts_D_3.00", "nuts_std_D_3.00"),
        )
    )
    # Full 16
    sets.append(("C_joint", "mu+sig_full16", tuple(MU_COLS) + tuple(SIGMA_COLS)))

    return sets


# ---------------------------------------------------------------------------
# Zone preparation
# ---------------------------------------------------------------------------

def prepare_zone(df: pd.DataFrame, zone: str) -> tuple[pd.DataFrame, np.ndarray]:
    if zone == "pooled":
        sub = df.copy()
    else:
        sub = df[df["zone"] == zone].copy()
    y = sub["is_tumor"].astype(int).values
    return sub, y


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {FEATURES_CSV.relative_to(REPO_ROOT)}: {df.shape}")

    feature_sets = enumerate_feature_sets()
    print(f"Enumerated {len(feature_sets)} feature sets across A/B/C blocks")
    by_block = {}
    for b, _, _ in feature_sets:
        by_block[b] = by_block.get(b, 0) + 1
    for b, n in by_block.items():
        print(f"  {b}: {n}")

    rows = []
    zones = ["pz", "tz", "pooled"]

    # Compute reference AUC + CI per zone × C
    print("\nReference: 2-feat NUTS-LR on", REFERENCE_FEATURES)
    ref_table = {}
    for zone in zones:
        sub, y = prepare_zone(df, zone)
        X_ref = sub[list(REFERENCE_FEATURES)].values
        for C in C_VALUES:
            auc, y_pred = loocv_auc(X_ref, y, C=C)
            lo, hi = bootstrap_auc_ci(y, y_pred)
            ref_table[(zone, C)] = (auc, lo, hi)
            print(f"  {zone:6s} C={C:5.1f}  AUC={auc:.4f}  CI=[{lo:.3f},{hi:.3f}]  n={len(y)}")

    # Run all subsets
    print("\nRunning sweep ...")
    n_total = len(feature_sets) * len(zones) * len(C_VALUES)
    i = 0
    for block, label, cols in feature_sets:
        for zone in zones:
            sub, y = prepare_zone(df, zone)
            if len(np.unique(y)) < 2:
                continue
            X = sub[list(cols)].values
            for C in C_VALUES:
                i += 1
                if i % 200 == 0:
                    print(f"  {i}/{n_total} ...")
                auc, y_pred = loocv_auc(X, y, C=C)
                lo, hi = bootstrap_auc_ci(y, y_pred)
                ref_auc, ref_lo, ref_hi = ref_table[(zone, C)]
                delta = auc - ref_auc
                non_overlap_above = lo > ref_hi
                non_overlap_below = hi < ref_lo
                rows.append(
                    {
                        "block": block,
                        "feature_set": label,
                        "n_features": len(cols),
                        "feature_cols": ",".join(cols),
                        "zone": zone,
                        "C": C,
                        "n": len(y),
                        "auc": auc,
                        "auc_ci_lo": lo,
                        "auc_ci_hi": hi,
                        "ref_auc": ref_auc,
                        "delta_vs_ref": delta,
                        "ci_strictly_above_ref": non_overlap_above,
                        "ci_strictly_below_ref": non_overlap_below,
                        "candidate_revision": (delta > 0.01) and non_overlap_above,
                    }
                )

    out = pd.DataFrame(rows)
    return out


def print_top_per_zone(out: pd.DataFrame, k: int = 12) -> None:
    print("\n" + "=" * 80)
    print(f"TOP-{k} FEATURE SETS PER ZONE (C=1.0)")
    print("=" * 80)
    for zone in ["pz", "tz", "pooled"]:
        sub = out[(out["zone"] == zone) & (out["C"] == 1.0)].copy()
        sub = sub.sort_values("auc", ascending=False).head(k)
        print(f"\n--- {zone.upper()} ---")
        print(
            sub[
                [
                    "block",
                    "feature_set",
                    "n_features",
                    "auc",
                    "auc_ci_lo",
                    "auc_ci_hi",
                    "delta_vs_ref",
                    "ci_strictly_above_ref",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )


def print_candidate_revisions(out: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("CANDIDATE REVISIONS (ΔAUC > 0.01 AND CI strictly above reference)")
    print("=" * 80)
    cand = out[out["candidate_revision"]].copy()
    if cand.empty:
        print("\n  NONE — reference {D=0.25, D=3.00} is not beaten by any subset.")
        print("  Path A headline stands. §10g closed.")
        return
    cand = cand.sort_values(["zone", "C", "delta_vs_ref"], ascending=[True, True, False])
    print(
        cand[
            [
                "block",
                "feature_set",
                "zone",
                "C",
                "auc",
                "auc_ci_lo",
                "auc_ci_hi",
                "ref_auc",
                "delta_vs_ref",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


def print_intermediate_test(out: pd.DataFrame) -> None:
    """Specifically report the intermediate-only test result."""
    print("\n" + "=" * 80)
    print("§10g HEADLINE TEST: intermediate-only LR vs reference")
    print("=" * 80)
    sub = out[
        (out["feature_set"] == "mu_intermediate_only{0.50-2.00}") & (out["C"] == 1.0)
    ]
    for _, r in sub.iterrows():
        verdict = "REVISES Path A" if r["candidate_revision"] else "consistent with Path A"
        print(
            f"  {r['zone']:6s}  AUC={r['auc']:.4f} CI=[{r['auc_ci_lo']:.3f},{r['auc_ci_hi']:.3f}]  "
            f"ref={r['ref_auc']:.4f}  Δ={r['delta_vs_ref']:+.4f}  → {verdict}"
        )


def print_pair_triple_check(out: pd.DataFrame) -> None:
    """§10g.1 + §10g.2 as originally written: pairs/triples containing D=0.25."""
    print("\n" + "=" * 80)
    print("§10g.1/.2 PAIR & TRIPLE SWEEPS containing μ_D=0.25 (C=1.0)")
    print("=" * 80)
    for zone in ["pz", "tz"]:
        print(f"\n--- {zone.upper()} ---")
        sub = out[(out["zone"] == zone) & (out["C"] == 1.0) & (out["block"] == "A_mu")]

        # pairs containing 0.25
        pair_rows = sub[
            sub["feature_set"].str.startswith("mu{0.25,")
            & (sub["n_features"] == 2)
        ].sort_values("auc", ascending=False)
        if not pair_rows.empty:
            print("  Pairs {D=0.25, D=k}:")
            print(
                pair_rows[
                    [
                        "feature_set",
                        "auc",
                        "auc_ci_lo",
                        "auc_ci_hi",
                        "delta_vs_ref",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )

        # triples containing {0.25, 3.00}
        trip_rows = sub[
            (sub["n_features"] == 3)
            & sub["feature_set"].str.contains("0.25")
            & sub["feature_set"].str.contains("3.00")
        ].sort_values("auc", ascending=False)
        if not trip_rows.empty:
            print("  Triples {D=0.25, D=3.00, D=k}:")
            print(
                trip_rows[
                    [
                        "feature_set",
                        "auc",
                        "auc_ci_lo",
                        "auc_ci_hi",
                        "delta_vs_ref",
                    ]
                ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )


def main() -> int:
    if not FEATURES_CSV.exists():
        print(f"ERROR: {FEATURES_CSV} not found. Run recompute.py first.", file=sys.stderr)
        return 1

    out = run_sweep()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV.relative_to(REPO_ROOT)}: {out.shape}")

    print_pair_triple_check(out)
    print_intermediate_test(out)
    print_top_per_zone(out)
    print_candidate_revisions(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
