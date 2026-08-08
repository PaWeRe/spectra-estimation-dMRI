"""Zone-confound check for the grade-dependent spectral shifts (Discussion, Fig 7).

Figure 7 pools the peripheral and transition zones, because the Gleason grade groups
(n = 8-12) are too small to subdivide. Stephan raised the obvious objection at review:
the two zones have different NORMAL baselines (Results), so a pooled normal reference
could in principle manufacture the apparent grade trend.

This script produces the two checks quoted in the Discussion:

  1. Direction of the pooling bias. PZ normal tissue carries more free water and less
     restricted mass than TZ normal tissue, while the graded tumors are PZ-dominated
     (21 of 29). A zone-matched baseline would therefore make both trends LARGER, so
     pooling is conservative rather than favourable.

  2. Within-zone replication. Restricted to the peripheral zone alone (n = 21 graded
     tumors), the restricted bin still rises monotonically with grade and the
     glandular-epithelial bin (D = 2.0) still falls, the latter with a nominally
     significant within-zone rank correlation.

It also prints the zone x grade cross-tabulation, which documents the residual
imbalance (higher grades are relatively more often transition-zone) that is the stated
reason the grade reading is kept descriptive.

Input:  results/biomarkers/features.csv  (written by biomarkers/recompute.py)
Output: stdout table + results/biomarkers/zone_grade_check.csv

Usage:  uv run python scripts/zone_grade_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO / "results" / "biomarkers" / "features.csv"
OUT_CSV = REPO / "results" / "biomarkers" / "zone_grade_check.csv"

D = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 20.00]
NUTS = [f"nuts_D_{d:.2f}" for d in D]


def grade_group(ggg: float) -> str:
    """Collapse Gleason Grade Group to the three bins used in Figure 7."""
    if ggg == 1:
        return "GGG=1"
    if ggg == 2:
        return "GGG=2"
    return "GGG>=3"


def main() -> None:
    feat = pd.read_csv(FEATURES_CSV)
    feat["zone"] = feat["zone"].str.upper().str[:2]

    normal = feat[feat["is_tumor"] == 0]
    tumor = feat[feat["is_tumor"] == 1].copy()
    tumor["ggg"] = pd.to_numeric(tumor["ggg"], errors="coerce")
    graded = tumor[tumor["ggg"] >= 1].copy()
    graded["grp"] = graded["ggg"].map(grade_group)

    print("=== Zone x grade cross-tabulation (graded tumors) ===")
    print(pd.crosstab(graded["grp"], graded["zone"], margins=True))
    rho_zone, p_zone = spearmanr(graded["ggg"], (graded["zone"] == "PZ").astype(int))
    print(f"\nSpearman(GGG, zone==PZ) = {rho_zone:+.2f}  (P = {p_zone:.3f})")
    print("Normal baseline zone split: "
          f"PZ n={(normal.zone == 'PZ').sum()}, TZ n={(normal.zone == 'TZ').sum()}")

    rows = {}
    rows["normal_pooled"] = normal[NUTS].mean()
    for z in ("PZ", "TZ"):
        rows[f"normal_{z}"] = normal[normal.zone == z][NUTS].mean()
    for grp in ("GGG=1", "GGG=2", "GGG>=3"):
        rows[f"pooled_{grp}"] = graded[graded.grp == grp][NUTS].mean()
        for z in ("PZ", "TZ"):
            sub = graded[(graded.grp == grp) & (graded.zone == z)]
            rows[f"{z}_{grp} (n={len(sub)})"] = sub[NUTS].mean()

    table = pd.DataFrame(rows).T
    table.columns = [f"D={d}" for d in D]
    print("\n=== Group-mean NUTS spectra: pooled vs split by zone ===")
    print(table.round(3).to_string())

    print("\n=== Check 1: direction of the pooling bias ===")
    for lbl, col in (("restricted D=0.25", "nuts_D_0.25"),
                     ("free water D=3.00", "nuts_D_3.00")):
        pz = normal[normal.zone == "PZ"][col].mean()
        tz = normal[normal.zone == "TZ"][col].mean()
        pooled = normal[col].mean()
        print(f"  {lbl}: normal PZ={pz:.3f}  TZ={tz:.3f}  pooled={pooled:.3f}")
    print("  Graded tumors are PZ-dominated "
          f"({(graded.zone == 'PZ').sum()}/{len(graded)}), so a PZ-matched baseline "
          "would show a LARGER free-water drop and a LARGER restricted rise.")

    print("\n=== Check 2: within-zone rank correlation of each bin with grade ===")
    corr_rows = []
    for lbl, sub in (("pooled", graded),
                     ("PZ", graded[graded.zone == "PZ"]),
                     ("TZ", graded[graded.zone == "TZ"])):
        print(f"\n  [{lbl}] n={len(sub)}")
        for d, col in zip(D, NUTS):
            rho, p = spearmanr(sub["ggg"], sub[col])
            print(f"    D={d:<6} rho={rho:+.2f}  P={p:.3f}")
            corr_rows.append({"subset": lbl, "n": len(sub), "D": d,
                              "spearman_rho": rho, "p_value": p})

    print("\nNOTE: eight bins are tested per subset without multiplicity correction, "
          "and n is small; these are descriptive, as stated in the Discussion.")

    out = pd.concat(
        [table.reset_index().rename(columns={"index": "group"}).assign(kind="group_mean"),
         pd.DataFrame(corr_rows).assign(kind="spearman_vs_grade")],
        ignore_index=True,
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
