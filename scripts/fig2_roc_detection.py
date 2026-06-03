"""
Figure 2 (v1): Tumor-detection ROC curves for PZ and TZ, demonstrating that
the spectral classifier matches ADC and that detection collapses onto the two
outer compartments (D=0.25 restricted, D=3.0 free-water).

Narrative (Pillar 1 -- "the collapse"):
  - The 2-feature {D=0.25, D=3.0} spectral LR is as good as, or better than,
    the full 8-feature LR, and lands on ADC.
  - The single outer components (D=0.25 alone, D=3.0 alone) are already strong;
    intermediate components are weak.
  => Detection needs only the two outer compartments, which is exactly why a
     single scalar (ADC) is hard to beat.

APPLES-TO-APPLES FAIRNESS FIX (the central methodological point of this figure):
  Every curve -- ADC, each single component, the 2-feature, and the 8-feature
  classifier -- is evaluated through the SAME leave-one-out CV pipeline. This
  removes the in-sample (raw-rank) vs out-of-sample (LOOCV) asymmetry that made
  raw single features spuriously "beat" the trained multivariate classifiers.
  A single monotonic feature barely overfits, so its LOOCV AUC is ~ its raw-rank
  AUC; a multivariate LR pays a real overfitting tax at n ~ 68-81. Putting them
  on the same footing turns "raw beats trained" into the figure's thesis: the
  intermediate bins add variance, not signal, at this sample size.

Estimator: NUTS posterior-mean features for ALL spectral curves (MAP is
near-identical per Figs 1 & 3; MAP numbers go to Table 1). NUTS is shown here
because it is the diagnostic-grade estimator (per-bin uncertainty + slightly
better reconstruction).

Quantitative AUC [95% CI] numbers live in Table 1 -- the panels carry only an
identity legend on top (panels are too small for ~5 curves' worth of numbers).
This script PRINTS the full ranked AUC table so Table 1 can be kept in sync.

Layout: 1 row x 2 cols (PZ, TZ). Matches Fig 3 typography & conventions.

Output:
  paper/figures/fig2_v1.png
  paper/figures/fig2_v1.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = "/Users/PWR/Documents/Professional/Papers/Paper3/code/spectra-estimation-dMRI"
FEATURES_CSV = os.path.join(REPO_ROOT, "results/biomarkers/features.csv")
OUT_DIR = os.path.join(REPO_ROOT, "paper/figures")

DIFFUSIVITIES = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 20.00]
D_LABELS = ["0.25", "0.50", "0.75", "1.00", "1.50", "2.00", "3.00", "20.0"]
NUTS_COLS = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
OUTER_IDX = [0, 6]  # D=0.25 and D=3.00 -> the "collapse" pair
LR_C = 1.0
N_BOOT = 2000
RNG = np.random.RandomState(42)

# --- Typography: match the fig3 / fig_roc 2x2 scale ---
mpl.rcParams.update({
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 17,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})

# --- Palette (project convention) ---
COLOR_ADC = "#000000"        # black  -> ADC reference
COLOR_NUTS = "#ff7f0e"       # orange -> NUTS spectral classifier
# Two outer single components highlighted; teal/brown are NOT load-bearing
# elsewhere (orange=NUTS, green=MAP, blue=normal, red=tumor, purple=ID).
COLOR_D025 = "#17becf"       # teal  -> D=0.25 single (restricted)
COLOR_D300 = "#8c564b"       # brown -> D=3.0  single (free-water)
COLOR_OTHER = "#9e9e9e"      # grey  -> the other 6 single components (bundle)


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
    """LOOCV ROC for a (single- or multi-feature) design matrix."""
    pred = loocv_pred(X, y)
    auc = roc_auc_score(y, pred)
    fpr, tpr, _ = roc_curve(y, pred)
    lo, hi = bootstrap_auc_ci(y, pred)
    return {"fpr": fpr, "tpr": tpr, "auc": auc, "ci": (lo, hi)}


def build_zone(df_zone):
    """Return all LOOCV curves for one zone."""
    y = df_zone["is_tumor"].astype(int).values
    X_all = df_zone[NUTS_COLS].values
    X_2 = df_zone[[NUTS_COLS[i] for i in OUTER_IDX]].values
    adc = df_zone["adc"].values

    out = {
        "n": len(y),
        "adc": curve_for(adc, y),
        "lr8": curve_for(X_all, y),
        "lr2": curve_for(X_2, y),
        "comp": [curve_for(df_zone[NUTS_COLS[i]].values, y)
                 for i in range(len(DIFFUSIVITIES))],
    }
    return out


def plot_panel(ax, res, title):
    n = res["n"]

    # --- the 5 intermediate single components: faint grey bundle ---
    # (D=0.5, 0.75, 1.0, 1.5, 2.0). Shows transparently that single
    # intermediate bins are unreliable detectors, without the 8-line
    # spaghetti. The degenerate free-water bin D=20 is DROPPED from the
    # bundle entirely: it carries essentially zero tumor signal, so the
    # LOOCV-LR orients it near-randomly per fold, yielding an AUC that
    # sits at the bottom-right (AUC<<0.5) and reads as a spurious red
    # flag. It is excluded from the plot only; the 8-feature classifier
    # is still trained on all bins including D=20.
    D20_IDX = len(DIFFUSIVITIES) - 1  # index 7 -> D=20 free-water artifact
    for i, c in enumerate(res["comp"]):
        if i in OUTER_IDX or i == D20_IDX:
            continue
        ax.plot(c["fpr"], c["tpr"], color=COLOR_OTHER,
                linewidth=1.1, alpha=0.5, zorder=2)

    # --- the two outer single components, highlighted ---
    c025 = res["comp"][0]
    c300 = res["comp"][6]
    ax.plot(c025["fpr"], c025["tpr"], color=COLOR_D025,
            linewidth=1.9, alpha=0.95, zorder=3)
    ax.plot(c300["fpr"], c300["tpr"], color=COLOR_D300,
            linewidth=1.9, alpha=0.95, zorder=3)

    # --- thick reference curves ---
    ax.plot(res["lr8"]["fpr"], res["lr8"]["tpr"], color=COLOR_NUTS,
            linewidth=2.8, linestyle="-", zorder=4)
    ax.plot(res["lr2"]["fpr"], res["lr2"]["tpr"], color=COLOR_NUTS,
            linewidth=2.8, linestyle="--", zorder=5)
    ax.plot(res["adc"]["fpr"], res["adc"]["tpr"], color=COLOR_ADC,
            linewidth=3.0, linestyle="-", zorder=6)

    ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=1.0, zorder=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title}  (n = {n})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.5)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feats = pd.read_csv(FEATURES_CSV)

    zones = [("pz", "PZ: tumor detection"), ("tz", "TZ: tumor detection")]
    results = {z: build_zone(feats[feats["zone"] == z].copy()) for z, _ in zones}

    # ---------- printed AUC table (keep Table 1 in sync) ----------
    print("\n=== Fig 2  LOOCV AUC ranking (C = %.3g, NUTS features) ===" % LR_C)
    for zkey, zlabel in zones:
        r = results[zkey]
        rows = [("ADC", r["adc"]),
                ("8-feature LR", r["lr8"]),
                ("2-feature {0.25,3.0} LR", r["lr2"])]
        rows += [(f"single D={D_LABELS[i]}", r["comp"][i])
                 for i in range(len(DIFFUSIVITIES))]
        rows.sort(key=lambda kv: kv[1]["auc"], reverse=True)
        print(f"\n  {zlabel}  (n = {r['n']})")
        print(f"    {'curve':<26s} {'AUC':>6s}   {'95% CI':>16s}")
        for name, c in rows:
            lo, hi = c["ci"]
            print(f"    {name:<26s} {c['auc']:.3f}   [{lo:.3f}, {hi:.3f}]")

    # ---------- red-flag check: minimum *plotted* grey-curve AUC ----------
    # The grey bundle now excludes the outer pair (OUTER_IDX) and the
    # degenerate D=20 free-water bin. Confirm no remaining grey curve is
    # below ~0.40.
    D20_IDX = len(DIFFUSIVITIES) - 1
    print("\n=== Min plotted grey (intermediate single-bin) AUC per zone ===")
    print("    (grey bundle excludes outer pair D=0.25/3.0 and D=20)")
    for zkey, zlabel in zones:
        r = results[zkey]
        grey = [(D_LABELS[i], r["comp"][i]["auc"])
                for i in range(len(DIFFUSIVITIES))
                if i not in OUTER_IDX and i != D20_IDX]
        mlabel, mauc = min(grey, key=lambda kv: kv[1])
        flag = "OK" if mauc >= 0.40 else "WARNING (< 0.40)"
        print(f"    {zlabel:<22s}  min grey AUC = {mauc:.3f} "
              f"(D={mlabel})  [{flag}]")

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), sharex=True, sharey=True)
    for ax, (zkey, zlabel) in zip(axes, zones):
        plot_panel(ax, results[zkey], zlabel)

    # shared identity legend on top (no AUC numbers -> Table 1).
    # Ordered as two logical groups of 3 (ncol=3): top row = references/
    # classifiers, bottom row = single-component lines.
    legend_handles = [
        Line2D([0], [0], color=COLOR_ADC, lw=3.0, label="ADC"),
        Line2D([0], [0], color=COLOR_NUTS, lw=2.8, ls="--",
               label="Spectral LR, 2 bins (D=0.25, 3.0)"),
        Line2D([0], [0], color=COLOR_NUTS, lw=2.8, ls="-",
               label="Spectral LR, 8 bins"),
        Line2D([0], [0], color=COLOR_D025, lw=2.0, label="D=0.25 single"),
        Line2D([0], [0], color=COLOR_D300, lw=2.0, label="D=3.0 single"),
        Line2D([0], [0], color=COLOR_OTHER, lw=1.6, alpha=0.7,
               label="Other single bins"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, 1.0),
               fontsize=14, columnspacing=1.6, handlelength=2.4)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    png = os.path.join(OUT_DIR, "fig2_v2.png")
    pdf = os.path.join(OUT_DIR, "fig2_v2.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
