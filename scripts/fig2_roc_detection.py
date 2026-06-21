"""
Figure 2 (v3): Tumor-detection ROC curves for PZ and TZ, demonstrating that
the spectral classifier matches ADC and that detection collapses onto the two
outer compartments (D=0.25 restricted, D=3.0 free-water).

Narrative (Pillar 1 -- "the collapse"):
  - The 2-feature {D=0.25, D=3.0} spectral LR is as good as, or better than,
    the full 8-feature LR, and lands on ADC.
  - The MIRROR ablation -- a 6-feature LR on the inner bins with the two outer
    compartments REMOVED -- drops well below (AUC ~0.78-0.82). The inner bins
    are individually informative but redundant once the outer pair is present
    ("redundancy, not uselessness"). The three orange curves are the SAME
    spectral classifier on different bin subsets: solid = all 8, dashed = 2
    outer (sits on ADC), dotted = 6 inner (sits low).
  - The single outer components (D=0.25 alone, D=3.0 alone) are already strong;
    intermediate components help only a little; the degenerate free-water dump
    bin (D=20) sits on the chance diagonal -- it is irrelevant to detection.
  => Detection needs only the two outer compartments, which is exactly why a
     single scalar (ADC) is hard to beat.

TWO ESTIMATION REGIMES IN THIS FIGURE (read carefully before editing):
  (a) The THICK reference curves -- ADC, 8-bin spectral LR, 2-bin {0.25,3.0}
      spectral LR -- are trained/evaluated through the SAME leave-one-out CV
      logistic-regression pipeline (apples-to-apples; the multivariate LRs pay
      the same out-of-sample tax ADC does).
  (b) The THIN single-component curves are plotted as RAW single-feature ROC in
      tumor-positive ("max") orientation -- NOT through the LOOCV-LR pipeline.
      Rationale (Stefan 2026-06-03): a near-zero-signal feature (esp. the D=20
      dump bin, but also weak intermediate bins) cannot be reliably ORIENTED by
      a held-out logistic regression, so the LOOCV-LR ROC for such a feature
      dips BELOW the chance diagonal (AUC ~ 0.0-0.22) -- an alarming artifact of
      the held-out fit, not of the feature. A single monotonic feature has no
      genuine orientation ambiguity (tumor is either higher or lower in that
      bin), so the honest descriptor is its raw single-feature ROC in the
      orientation that puts tumor-positive. This keeps every single-bin curve
      at/above chance and lets the D=20 bin show its true behaviour: a curve
      that hugs the diagonal == carries no detection signal. A caption clause is
      needed to state that single-bin curves are raw (not LOOCV-LR) ROC.

Estimator: NUTS posterior-mean features for ALL spectral curves (MAP is
near-identical per Figs 1 & 3; MAP numbers go to Table 1). NUTS is shown here
because it is the diagnostic-grade estimator (per-bin uncertainty + slightly
better reconstruction). The legend states "(NUTS)" explicitly.

Quantitative AUC [95% CI] numbers live in Table 1 -- the panels carry only an
identity legend on top (panels are too small for ~5 curves' worth of numbers).
This script PRINTS the full ranked AUC table so Table 1 can be kept in sync.

Layout: 1 row x 2 cols (PZ left, TZ right). Shared paper_style.

Output:
  paper/figures/fig2_v3.png
  paper/figures/fig2_v3.pdf
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = "/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI"
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from spectra_estimation_dmri.visualization.paper_style import (  # noqa: E402
    apply_style, COLORS,
)

FEATURES_CSV = os.path.join(REPO_ROOT, "results/biomarkers/features.csv")
OUT_DIR = os.path.join(REPO_ROOT, "paper/figures")

DIFFUSIVITIES = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 20.00]
D_LABELS = ["0.25", "0.50", "0.75", "1.00", "1.50", "2.00", "3.00", "20.0"]
NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
OUTER_IDX = [0, 6]  # D=0.25 and D=3.00 -> the "collapse" pair
INNER_IDX = [1, 2, 3, 4, 5, 7]  # the six inner bins -> the mirror ablation
D20_IDX = len(DIFFUSIVITIES) - 1  # index 7 -> D=20 free-water dump bin
# Zone-dependent weak single bins shown explicitly in BOTH panels so the reader
# can see WHICH bins are near-random in EACH zone (Patrick 2026-06-05). The
# weakest non-D=20 single bin is NOT the same in the two zones (raw max-
# orientation single-feature ROC AUC, verified by the printed table):
#   * D=2.00 (index 5) is weak in PZ only  (PZ 0.587, TZ 0.792)
#   * D=1.50 (index 4) is weak in TZ only  (PZ 0.833, TZ 0.545 -- near-chance,
#                                           ~as random as D=20's TZ 0.545)
#   * D=20.0 (index 7) is near-chance in BOTH zones (0.519 / 0.545)
# Both zone-specific weak bins are drawn in BOTH panels alongside D=20 so the
# zone-dependence of the detection extremes is visible.
WEAK_PZ_IDX = 5  # D=2.00 -> weak single bin in PZ
WEAK_TZ_IDX = 4  # D=1.50 -> weak single bin in TZ
LR_C = 1.0
N_BOOT = 2000
RNG = np.random.RandomState(42)

# --- Typography: shared paper style (legend == title size; Stefan) ---
apply_style("grid")

# --- Palette (shared paper_style COLORS) ---
COLOR_ADC = COLORS["adc"]          # black  -> ADC reference
COLOR_NUTS = COLORS["nuts"]        # orange -> NUTS spectral classifier
COLOR_D025 = COLORS["restricted"]  # teal  -> D=0.25 single (restricted)
COLOR_D300 = COLORS["freewater"]   # brown -> D=3.0  single (free-water)
COLOR_OTHER = COLORS["muted"]      # faint grey -> intermediate single bins
COLOR_D20 = "#c2185b"              # magenta -> D=20 dump bin (distinct; shown
#                                    deliberately to demonstrate irrelevance)
COLOR_WEAK_PZ = "#b8860b"          # dark goldenrod -> D=2.0, weak single bin
#                                    in PZ (distinct hue, far from magenta)
COLOR_WEAK_TZ = "#3949ab"          # indigo -> D=1.5, weak single bin in TZ
#                                    (distinct hue, not reused elsewhere in
#                                    this figure: cf. teal/brown/goldenrod/
#                                    magenta/grey/black/orange)


def loocv_pred(X, y, C=LR_C):
    """Leave-one-out held-out predicted P(tumor). X is (n, p)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(C=C, max_iter=2000, random_state=42, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        y_pred[te] = clf.predict_proba(Xte)[0, 1]
    return y_pred


def bootstrap_auc_ci(y, y_pred, n_boot=N_BOOT, alpha=0.05, rng=RNG):
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) > 1:
            aucs.append(roc_auc_score(y[idx], y_pred[idx]))
    return (np.percentile(aucs, 100 * alpha / 2),
            np.percentile(aucs, 100 * (1 - alpha / 2)))


def curve_for(X, y):
    """LOOCV-LR ROC for a (single- or multi-feature) design matrix.

    Used for the THICK reference curves only (ADC, 8-bin LR, 2-bin LR)."""
    pred = loocv_pred(X, y)
    auc = roc_auc_score(y, pred)
    fpr, tpr, _ = roc_curve(y, pred)
    lo, hi = bootstrap_auc_ci(y, pred)
    return {"fpr": fpr, "tpr": tpr, "auc": auc, "ci": (lo, hi)}


def raw_curve_for(x, y):
    """RAW single-feature ROC in tumor-positive (max) orientation.

    A single feature has no genuine orientation ambiguity: tumor is either
    higher or lower in that bin. We orient the score so tumor is positive
    (flip sign if the as-is AUC < 0.5), which guarantees the curve sits
    at/above the chance diagonal. This is the honest single-bin descriptor
    and avoids the sub-diagonal artifact a held-out LR produces for weak /
    near-zero-signal bins (esp. the D=20 dump bin). 'flipped' records whether
    tumor is LOWER in that bin (e.g. free-water bins, which tumors deplete)."""
    x = np.asarray(x, dtype=float)
    auc_raw = roc_auc_score(y, x)
    flipped = auc_raw < 0.5
    score = -x if flipped else x
    auc = roc_auc_score(y, score)  # == max(auc_raw, 1 - auc_raw)
    fpr, tpr, _ = roc_curve(y, score)
    lo, hi = bootstrap_auc_ci(y, score)
    return {"fpr": fpr, "tpr": tpr, "auc": auc, "ci": (lo, hi),
            "flipped": bool(flipped)}


def build_zone(df_zone):
    """Return all curves for one zone.

    Thick references (adc/lr8/lr2) use the LOOCV-LR pipeline; the single
    components (`comp`) use raw single-feature ROC in max orientation."""
    y = df_zone["is_tumor"].astype(int).values
    X_all = df_zone[NUTS_COLS].values
    X_2 = df_zone[[NUTS_COLS[i] for i in OUTER_IDX]].values
    X_6 = df_zone[[NUTS_COLS[i] for i in INNER_IDX]].values
    adc = df_zone["adc"].values

    out = {
        "n": len(y),
        "adc": curve_for(adc, y),
        "lr8": curve_for(X_all, y),
        "lr2": curve_for(X_2, y),
        "lr6": curve_for(X_6, y),   # mirror ablation: inner bins, outer removed
        "comp": [raw_curve_for(df_zone[NUTS_COLS[i]].values, y)
                 for i in range(len(DIFFUSIVITIES))],
    }
    return out


def plot_panel(ax, res, title, show_ylabel=True):
    n = res["n"]

    # --- the intermediate single components: faint grey bundle ---
    # (D=0.5, 0.75, 1.0). Raw single-feature ROC in max orientation.
    # Shows that the middle bins help only a little -- they cluster modestly
    # above the diagonal, between the strong outer pair and the weak bins.
    # The zone-specific weak bins D=2.0 (WEAK_PZ_IDX) + D=1.5 (WEAK_TZ_IDX)
    # and the dump bin D=20 (D20_IDX) are drawn separately below.
    HIGHLIGHT = set(OUTER_IDX) | {D20_IDX, WEAK_PZ_IDX, WEAK_TZ_IDX}
    for i, c in enumerate(res["comp"]):
        if i in HIGHLIGHT:
            continue
        ax.plot(c["fpr"], c["tpr"], color=COLOR_OTHER,
                linewidth=1.2, alpha=0.55, zorder=2)

    # --- the weakest single bins, shown deliberately to display the
    # detection extremes and their zone-dependence (Patrick 2026-06-05):
    # D=20 free-water dump bin (near-zero signal; near-chance in BOTH zones),
    # D=2.0 (weak in PZ only) and D=1.5 (weak in TZ only). Both zone-specific
    # weak bins appear in BOTH panels so the reader sees which bins are near-
    # random for each zone. All plotted raw (not LOOCV-LR, which would dip
    # below chance -- see module docstring). ---
    cw_pz = res["comp"][WEAK_PZ_IDX]
    ax.plot(cw_pz["fpr"], cw_pz["tpr"], color=COLOR_WEAK_PZ,
            linewidth=2.0, alpha=0.95, zorder=3.3)
    cw_tz = res["comp"][WEAK_TZ_IDX]
    ax.plot(cw_tz["fpr"], cw_tz["tpr"], color=COLOR_WEAK_TZ,
            linewidth=2.0, alpha=0.95, zorder=3.4)
    c20 = res["comp"][D20_IDX]
    ax.plot(c20["fpr"], c20["tpr"], color=COLOR_D20,
            linewidth=2.0, alpha=0.95, zorder=3.5)

    # --- the two outer single components, highlighted ---
    c025 = res["comp"][0]
    c300 = res["comp"][6]
    ax.plot(c025["fpr"], c025["tpr"], color=COLOR_D025,
            linewidth=2.0, alpha=0.95, zorder=4)
    ax.plot(c300["fpr"], c300["tpr"], color=COLOR_D300,
            linewidth=2.0, alpha=0.95, zorder=4)

    # --- thick reference curves (LOOCV-LR; NUTS features) ---
    # Three orange curves = the SAME spectral classifier on different bin
    # subsets: solid = all 8, dashed = 2 outer (sits on ADC), dotted = 6 inner
    # (mirror ablation -- sits well below: redundancy, not uselessness).
    ax.plot(res["lr6"]["fpr"], res["lr6"]["tpr"], color=COLOR_NUTS,
            linewidth=2.6, linestyle=":", zorder=5)
    ax.plot(res["lr8"]["fpr"], res["lr8"]["tpr"], color=COLOR_NUTS,
            linewidth=2.8, linestyle="-", zorder=5.5)
    ax.plot(res["lr2"]["fpr"], res["lr2"]["tpr"], color=COLOR_NUTS,
            linewidth=2.8, linestyle="--", zorder=6)
    ax.plot(res["adc"]["fpr"], res["adc"]["tpr"], color=COLOR_ADC,
            linewidth=3.0, linestyle="-", zorder=7)

    ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1.0, zorder=1)
    ax.set_xlabel("False positive rate")
    # Right (TZ) panel shares the y-axis; drop its redundant title (Stephan 2026-06-19).
    if show_ylabel:
        ax.set_ylabel("True positive rate")
    ax.set_title(f"{title}  (n = {n})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.5)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feats = pd.read_csv(FEATURES_CSV)

    zones = [("pz", "PZ: Tumor Detection"), ("tz", "TZ: Tumor Detection")]
    results = {z: build_zone(feats[feats["zone"] == z].copy()) for z, _ in zones}

    # ---------- printed AUC table (keep Table 1 in sync) ----------
    # NOTE on AUC semantics, mirroring the plotted curves:
    #   * ADC / 8-bin LR / 2-bin LR  -> LOOCV-LR AUC (out-of-sample)
    #   * single D=...               -> RAW single-feature AUC in max
    #                                   (tumor-positive) orientation
    print("\n=== Fig 2  AUC ranking (C = %.3g, NUTS features) ===" % LR_C)
    print("    refs = LOOCV-LR ; single bins = raw max-orientation ROC")
    for zkey, zlabel in zones:
        r = results[zkey]
        rows = [("ADC (LOOCV-LR)", r["adc"], ""),
                ("8-bin LR (LOOCV)", r["lr8"], ""),
                ("2-bin {0.25,3.0} LR (LOOCV)", r["lr2"], ""),
                ("6-inner LR (LOOCV)", r["lr6"], "")]
        rows += [(f"single D={D_LABELS[i]} (raw)", r["comp"][i],
                  " [tumor LOW]" if r["comp"][i].get("flipped") else "")
                 for i in range(len(DIFFUSIVITIES))]
        rows.sort(key=lambda kv: kv[1]["auc"], reverse=True)
        print(f"\n  {zlabel}  (n = {r['n']})")
        print(f"    {'curve':<30s} {'AUC':>6s}   {'95% CI':>16s}")
        for name, c, note in rows:
            lo, hi = c["ci"]
            print(f"    {name:<30s} {c['auc']:.3f}   "
                  f"[{lo:.3f}, {hi:.3f}]{note}")

    # ---------- sanity check: every PLOTTED single-bin curve at/above chance --
    # Raw max-orientation guarantees AUC >= 0.5 by construction; this confirms
    # it (and surfaces the D=20 dump bin's near-chance AUC, the figure's point).
    print("\n=== Plotted single-bin AUCs (raw, max orientation) ===")
    for zkey, zlabel in zones:
        r = results[zkey]
        aucs = [(D_LABELS[i], r["comp"][i]["auc"])
                for i in range(len(DIFFUSIVITIES))]
        mlabel, mauc = min(aucs, key=lambda kv: kv[1])
        d20 = r["comp"][D20_IDX]["auc"]
        flag = "OK" if mauc >= 0.50 - 1e-9 else "WARNING (< 0.50)"
        print(f"    {zlabel:<22s}  min single AUC = {mauc:.3f} (D={mlabel}) "
              f"[{flag}] ; D=20 AUC = {d20:.3f} (should hug 0.5)")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.0), sharex=True, sharey=True)
    for i, (ax, (zkey, zlabel)) in enumerate(zip(axes, zones)):
        plot_panel(ax, results[zkey], zlabel, show_ylabel=(i == 0))

    # Shared identity legend on top (no AUC numbers -> Table 1). Concise
    # labels only: no legend title row, no "(NUTS)"/"(LOOCV-LR)" qualifiers,
    # no ":irrelevant" suffix, and no "chance" entry (the dashed diagonal is
    # self-evident). Qualifiers (estimator, raw vs LOOCV-LR) go in the
    # manuscript caption. Nine entries (added the TZ-weak bin D=1.5 alongside
    # the PZ-weak bin D=2.0) laid out as three rows of three (ncol=3) to keep
    # the legend width close to the two side-by-side panels (Patrick
    # 2026-06-05). The two zone-specific weak bins are annotated so the reader
    # sees the zone-dependence at a glance.
    # Ten entries laid out as two rows of five (ncol=5). The three spectral
    # entries are consecutive (8 / 2-outer / 6-inner) so the reader reads them
    # as one classifier on three bin subsets.
    legend_handles = [
        Line2D([0], [0], color=COLOR_ADC, lw=3.0, label="ADC"),
        Line2D([0], [0], color=COLOR_NUTS, lw=2.8, ls="-",
               label="Spectral, 8 bins"),
        Line2D([0], [0], color=COLOR_NUTS, lw=2.8, ls="--",
               label="Spectral, 2 outer bins"),
        Line2D([0], [0], color=COLOR_NUTS, lw=2.6, ls=":",
               label="Spectral, 6 inner bins"),
        Line2D([0], [0], color=COLOR_D025, lw=2.0, label="D = 0.25"),
        Line2D([0], [0], color=COLOR_D300, lw=2.0, label="D = 3.0"),
        Line2D([0], [0], color=COLOR_WEAK_PZ, lw=2.0,
               label="D = 2.0"),
        Line2D([0], [0], color=COLOR_WEAK_TZ, lw=2.0,
               label="D = 1.5"),
        Line2D([0], [0], color=COLOR_D20, lw=2.0,
               label="D = 20"),
        Line2D([0], [0], color=COLOR_OTHER, lw=1.6, alpha=0.7,
               label="Other single bins"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.3, handlelength=1.9, handletextpad=0.5)

    # Two legend rows above the panels.
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.85])
    png = os.path.join(OUT_DIR, "fig2_v3.png")
    pdf = os.path.join(OUT_DIR, "fig2_v3.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
