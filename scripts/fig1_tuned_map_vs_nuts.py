"""
Figure 1 (v3): tuned-MAP (lambda=1e-3) vs NUTS spectra on 8 representative ROIs.

v3 changes vs v2:
    * MAP is now recomputed via the corrected solver: NNLS on the augmented
      system [U; sqrt(lambda) I] R = [s/S0; 0], i.e. the constrained ridge MAP.
      v2 used the OLD buggy "project unconstrained Gaussian MAP onto non-negative
      orthant" recipe, which is NOT equivalent whenever the unconstrained
      optimum is infeasible (Sandy 2026-05-25 counter-example, fixed
      2026-05-26 in src/spectra_estimation_dmri/biomarkers/recompute.py).
    * Larger fonts matching Fig 2 conventions: xtick=17, ytick=17, legend=15,
      title=15, axis labels=17.
    * CV-coloring legend moved outside the plot panels, on the right side as a
      compact vertical column.

ROIs: re-using the v2 picks for continuity (PZ-N: new58, new10; PZ-T: new07,
new63; TZ-N: new35, new30; TZ-T: new44, new01) — these were chosen as the two
ROIs per (zone x class) cell closest to the per-cell median in the 2-D NUTS
{R(D=0.25), R(D=3.00)} feature space.

Saves:
    paper/figures/fig1_v3.png  (300 dpi)
    paper/figures/fig1_v3.pdf
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.optimize import nnls

# Make the project's source importable when run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from spectra_estimation_dmri.biomarkers.recompute import (  # noqa: E402
    DIFFUSIVITIES,
    build_design_matrix,
    compute_spectra_id,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAMBDA_TUNED = 1e-3  # F1 tuned regulariser (matches recompute.py RIDGE_STRENGTH)
NC_DIR = ROOT / "results" / "inference_bwh_backup"
FEATURES_CSV = ROOT / "results" / "biomarkers" / "features.csv"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hard-coded ROI picks reused from v2 for continuity.
FIXED_PICKS: dict[tuple[str, bool], list[str]] = {
    ("pz", False): ["new58_pz_normal", "new10_pz_normal"],
    ("pz", True):  ["new07_pz_tumor",  "new63_pz_tumor"],
    ("tz", False): ["new35_tz_normal", "new30_tz_normal"],
    ("tz", True):  ["new44_tz_tumor",  "new01_tz_tumor"],
}

# Style: match Fig 2 conventions
mpl.rcParams.update({
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.labelsize": 17,
    "axes.titlesize": 15,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})


# CV -> colour bands
def cv_color(cv: float) -> str:
    if not np.isfinite(cv):
        return "#bdbdbd"
    if cv < 0.4:
        return "#2ca02c"  # green
    if cv < 0.6:
        return "#ffd92f"  # yellow
    if cv < 0.8:
        return "#ff7f0e"  # orange
    return "#d62728"      # red


# ---------------------------------------------------------------------------
# MAP fit at tuned lambda — CORRECTED constrained ridge solver.
# Matches src/spectra_estimation_dmri/biomarkers/recompute.py::compute_map_spectrum
# ---------------------------------------------------------------------------

def fit_map_tuned(signal: np.ndarray, U: np.ndarray, lam: float = LAMBDA_TUNED) -> np.ndarray:
    """argmin_{R>=0} ||U R - s/S0||^2 + lam ||R||^2 via NNLS on augmented system."""
    S0 = signal[0] if signal[0] > 0 else 1.0
    s_norm = signal / S0
    n_d = U.shape[1]
    U_aug = np.vstack([U, np.sqrt(lam) * np.eye(n_d)])
    s_aug = np.concatenate([s_norm, np.zeros(n_d)])
    spectrum, _ = nnls(U_aug, s_aug)
    s = spectrum.sum()
    return spectrum / s if s > 0 else spectrum


# ---------------------------------------------------------------------------
# NUTS loading helper: returns normalised per-bin posterior mean and std
# ---------------------------------------------------------------------------

def load_nuts(roi: dict) -> tuple[np.ndarray, np.ndarray]:
    sid = compute_spectra_id(roi["signal"], roi["b_values"], roi["snr"])
    nc_path = NC_DIR / f"{sid}.nc"
    idata = az.from_netcdf(str(nc_path))
    var_names = [f"diff_{d:.2f}" for d in DIFFUSIVITIES]
    cols = []
    for vn in var_names:
        cols.append(idata.posterior[vn].values.flatten())
    samples = np.column_stack(cols)  # (N, 8) unnormalised
    row_sums = samples.sum(axis=1, keepdims=True)
    samples_norm = samples / np.maximum(row_sums, 1e-10)
    return samples_norm.mean(axis=0), samples_norm.std(axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def panel(ax, diffs, map_R, nuts_R, nuts_std, title):
    """Side-by-side bars: MAP (left) + NUTS (right) per bin, coloured by NUTS CV."""
    x = np.arange(len(diffs))
    w = 0.4

    cvs = np.divide(
        nuts_std, nuts_R, out=np.full_like(nuts_R, np.nan, dtype=float), where=nuts_R > 1e-8
    )
    colors = [cv_color(c) for c in cvs]

    # MAP bars: filled with CV colour, white hatched to distinguish from NUTS
    ax.bar(
        x - w / 2,
        map_R,
        width=w,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        hatch="///",
    )
    # NUTS bars: filled with CV colour, no hatch + error bars
    ax.bar(
        x + w / 2,
        nuts_R,
        width=w,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        yerr=nuts_std,
        ecolor="black",
        capsize=2.5,
        error_kw=dict(elinewidth=1.0),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in diffs])
    ax.set_xlabel(r"D ($\mu$m$^2$/ms)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 0.65)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)


def make_figure(features: pd.DataFrame, rois_by_id: dict[str, dict],
                chosen: dict[tuple[str, bool], list[str]],
                U: np.ndarray):
    """4 rows (PZ-N, PZ-T, TZ-N, TZ-T) x 2 cols. Each panel overlays MAP + NUTS
    side-by-side bars. CV-colour legend lives outside on the right."""
    row_order = [("pz", False), ("pz", True), ("tz", False), ("tz", True)]
    row_labels = {
        ("pz", False): "PZ normal",
        ("pz", True): "PZ tumour",
        ("tz", False): "TZ normal",
        ("tz", True): "TZ tumour",
    }

    # gridspec: 2 columns of panels + 1 narrow column for legend on right.
    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        width_ratios=[1.0, 1.0, 0.30],
        hspace=0.55, wspace=0.28,
        left=0.07, right=0.97, top=0.94, bottom=0.05,
    )

    axes = np.empty((4, 2), dtype=object)
    for r in range(4):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c])

    chosen_ids: list[str] = []

    for r, key in enumerate(row_order):
        ids = chosen[key]
        for c, roi_id in enumerate(ids):
            roi = rois_by_id[roi_id]
            map_R = fit_map_tuned(roi["signal"], U, lam=LAMBDA_TUNED)
            nuts_mean, nuts_std = load_nuts(roi)

            feat_row = features[features.roi_id == roi_id].iloc[0]
            # features.csv ADC is in mm^2/s (= 1e-3 * um^2/ms);  convert.
            adc_um2_ms = feat_row["adc"] * 1e3
            title = (
                f"{row_labels[key]}  |  {roi_id}\n"
                rf"ADC = {adc_um2_ms:.2f} $\mu$m$^2$/ms"
            )
            panel(axes[r, c], DIFFUSIVITIES, map_R, nuts_mean, nuts_std, title)
            chosen_ids.append(roi_id)

    # Legend column on the right.
    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.axis("off")

    cv_handles = [
        mpatches.Patch(facecolor="#2ca02c", edgecolor="black", label="CV < 0.4"),
        mpatches.Patch(facecolor="#ffd92f", edgecolor="black", label="0.4 – 0.6"),
        mpatches.Patch(facecolor="#ff7f0e", edgecolor="black", label="0.6 – 0.8"),
        mpatches.Patch(facecolor="#d62728", edgecolor="black", label="CV > 0.8"),
    ]
    style_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///",
                       label=r"MAP ($\lambda$=1e-3)"),
        mpatches.Patch(facecolor="white", edgecolor="black",
                       label=r"NUTS (mean $\pm$ std)"),
    ]

    leg1 = legend_ax.legend(
        handles=cv_handles,
        title="NUTS posterior CV",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        frameon=True,
        fontsize=15,
        title_fontsize=15,
        borderaxespad=0.0,
    )
    legend_ax.add_artist(leg1)
    legend_ax.legend(
        handles=style_handles,
        title="Estimator",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.55),
        frameon=True,
        fontsize=15,
        title_fontsize=15,
        borderaxespad=0.0,
    )

    return fig, chosen_ids


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    features = pd.read_csv(FEATURES_CSV)
    rois = load_dataset()
    rois_by_id = {r["roi_id"]: r for r in rois}
    U = build_design_matrix()

    chosen = FIXED_PICKS
    print("=== Representative ROIs (re-used from v2 for continuity) ===")
    for k, v in chosen.items():
        print(f"  {k}: {v}")

    fig, chosen_ids = make_figure(features, rois_by_id, chosen, U)

    out_png = OUT_DIR / "fig1_v3.png"
    out_pdf = OUT_DIR / "fig1_v3.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Chosen ROI ids: {chosen_ids}")


if __name__ == "__main__":
    main()
