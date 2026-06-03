"""
Figure 7 (v2): direction-independence / trace-averaging validation at the
WHOLE-ROI level, for one representative patient.

Co-author instruction (meeting): show 1 patient (one normal + one tumour ROI),
3 per-direction MAP spectra + the trace-averaged spectrum per panel, and report
aggregate per-direction statistics over all patients in text (no second figure).
Separate MAP and NUTS.

Data source
-----------
Stephan's XPhase ROI exports in
    src/spectra_estimation_dmri/data/diffusion-spectrum-analysis/*.dat
Each .dat has a numbered table (NR 1..46) whose 2nd column "MEAN" is the
ROI-mean intensity for each of the 46 acquired images.

Decoded layout (confirmed empirically, 2026-06-03 — see report):
    * Image 1 (NR=1)        -> scanner reference (brightest); DROPPED, exactly as
                               scripts/direction_comparison.py drops the brightest
                               binary image for patient 8640.
    * Images 2..46 (45)     -> 3 CONTIGUOUS blocks of 15. Block index = gradient
                               direction (dir 1, 2, 3).
    * Within each 15-block  -> b-values are stored DESCENDING (first entry = b=3500,
                               last entry = b=0). We reverse each block so index 0
                               is b=0 and index 14 is b=3500, matching the canonical
                               B_VALUES order used everywhere else in the project.

Validation: trace-average of the three reversed blocks, normalised by b=0,
correlates r=0.977 with the canonical signal_decays.json decay for the same
patient/ROI (9283 normal_pz). The forward (un-reversed) ordering gives r=-0.41,
so the descending-b interpretation is unambiguous.

MAP solver: imported verbatim from
    spectra_estimation_dmri.biomarkers.recompute.compute_map_spectrum
(constrained ridge NNLS on the augmented system). RIDGE_STRENGTH there is the
tuned lambda=1e-3 used for every other paper MAP figure (fig1/fig3/...). NOTE:
CLAUDE.md / configs/prior/ridge.yaml still list 0.1 as the nominal value; the
canonical paper pipeline uses 1e-3. We follow recompute.py so Fig 7 is
consistent with the other figures. Flip USE_LAMBDA_0p1 to True to reproduce at
0.1 instead.

Outputs
-------
    paper/figures/fig_directions_v2.{png,pdf}     -- the 2-panel figure
    results/direction_comparison/fig7_roi_direction_cv_MAP.csv  -- per-bin CV table
    (printed to stdout) aggregate per-bin direction-CV table for the text.

Usage:
    uv run python scripts/fig7_directions_roi.py
"""

from __future__ import annotations

import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from spectra_estimation_dmri.biomarkers.recompute import (  # noqa: E402
    DIFFUSIVITIES,
    build_design_matrix,
    compute_map_spectrum,
    RIDGE_STRENGTH,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAT_DIR = ROOT / "src" / "spectra_estimation_dmri" / "data" / "diffusion-spectrum-analysis"
OUT_FIG_DIR = ROOT / "paper" / "figures"
OUT_CSV_DIR = ROOT / "results" / "direction_comparison"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)

N_DIRECTIONS = 3
N_BVALUES = 15

# Representative patient with a matched same-zone (PZ) normal + tumour pair.
PATIENT = "9283-Series12-Slice6"
NORMAL_ROI_FILE = f"{PATIENT}-NormalPZ.dat"
TUMOR_ROI_FILE = f"{PATIENT}-TumorPZ.dat"

# Style: match Fig 1 / Fig 2 conventions.
mpl.rcParams.update({
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.labelsize": 17,
    "axes.titlesize": 15,
    "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})

DIR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
DIR_LABELS = ["Direction 1", "Direction 2", "Direction 3"]
TRACE_COLOR = "black"


# ---------------------------------------------------------------------------
# .dat loader
# ---------------------------------------------------------------------------
def load_dat_mean_column(path) -> np.ndarray:
    """Return the 46-entry MEAN column (one ROI-mean intensity per acquired image)."""
    vals = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            parts = s.split()
            try:
                nr = int(parts[0])
            except (ValueError, IndexError):
                continue
            if 1 <= nr <= 46 and len(parts) >= 2:
                vals.append(float(parts[1]))
    arr = np.asarray(vals, dtype=float)
    if arr.size != 46:
        raise ValueError(f"{path}: expected 46 MEAN rows, got {arr.size}")
    return arr


def decode_roi_directions(path):
    """Decode a .dat MEAN column into a per-direction signal-decay matrix.

    Returns
    -------
    dir_decays : (N_DIRECTIONS, N_BVALUES) array, b=0..b=3500 ascending, raw (not normalised).
    trace_decay : (N_BVALUES,) array, direction-averaged raw decay.
    """
    arr = load_dat_mean_column(path)
    # arr[0] = scanner reference -> dropped
    rest = arr[1:]  # 45 = 3 dirs x 15 b-values
    blocks = rest.reshape(N_DIRECTIONS, N_BVALUES)
    # Within each block b is DESCENDING (b=3500 first). Reverse to b=0..b=3500.
    dir_decays = blocks[:, ::-1].copy()
    trace_decay = dir_decays.mean(axis=0)
    return dir_decays, trace_decay


def normalize_by_b0(decay: np.ndarray) -> np.ndarray:
    s0 = decay[0] if decay[0] > 0 else 1.0
    return decay / s0


# ---------------------------------------------------------------------------
# MAP per direction + trace, for one ROI
# ---------------------------------------------------------------------------
def map_spectra_for_roi(path, U):
    """MAP spectra for the 3 directions + trace average of one ROI.

    Spectra are returned normalised to sum to 1 (fractions), matching the
    presentation in the other spectrum figures.
    """
    dir_decays, trace_decay = decode_roi_directions(path)

    dir_spectra = np.zeros((N_DIRECTIONS, len(DIFFUSIVITIES)))
    for d in range(N_DIRECTIONS):
        # compute_map_spectrum normalises by signal[0] internally -> pass raw decay.
        sp = compute_map_spectrum(dir_decays[d], U)
        tot = sp.sum()
        dir_spectra[d] = sp / tot if tot > 0 else sp

    sp_tr = compute_map_spectrum(trace_decay, U)
    tot = sp_tr.sum()
    trace_spectrum = sp_tr / tot if tot > 0 else sp_tr

    return dir_decays, trace_decay, dir_spectra, trace_spectrum


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_figure(normal, tumor, out_stem):
    """Two-panel figure: per-direction MAP spectra + trace, normal & tumour ROI."""
    x = np.arange(len(DIFFUSIVITIES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    panels = [("(a) Normal PZ", normal), ("(b) Tumour PZ", tumor)]
    for ax, (label, res) in zip(axes, panels):
        dir_spectra = res["dir_spectra"]
        trace_spectrum = res["trace_spectrum"]
        for d in range(N_DIRECTIONS):
            ax.plot(x, dir_spectra[d], "o-", color=DIR_COLORS[d], lw=2,
                    ms=7, alpha=0.9, label=DIR_LABELS[d])
        ax.plot(x, trace_spectrum, "s-", color=TRACE_COLOR, lw=2.6, ms=8,
                label="Trace-averaged", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:g}" for d in DIFFUSIVITIES], rotation=45)
        ax.set_xlabel(r"Diffusivity $D$ ($\mu$m$^2$/ms)")
        ax.grid(axis="y", alpha=0.3)
        # Per-panel annotation of which ROI (in-axes, small — not a figure title).
        ax.text(0.97, 0.95, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=15, fontweight="bold")

    axes[0].set_ylabel("Spectral fraction")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = f"{out_stem}.png"
    pdf = f"{out_stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {png}")
    print(f"[INFO] Saved figure: {pdf}")


# ---------------------------------------------------------------------------
# Aggregate per-bin direction-CV over ALL ROIs
# ---------------------------------------------------------------------------
def aggregate_direction_cv(U):
    """Per-bin CV of the per-direction MAP spectrum, averaged over all 13 ROIs.

    For each ROI: fit 3 per-direction MAP spectra (each summing to 1), then per
    diffusivity bin compute CV = std/mean across the 3 directions. Average that
    per-bin CV over all ROIs.

    Returns a DataFrame with the per-ROI rows and prints the aggregate table.
    """
    files = sorted(glob.glob(str(DAT_DIR / "*.dat")))
    rows = []
    for path in files:
        name = os.path.basename(path).replace(".dat", "")
        try:
            dir_decays, trace_decay = decode_roi_directions(path)
        except ValueError as exc:
            print(f"[WARN] skipping {name}: {exc}")
            continue
        dir_spectra = np.zeros((N_DIRECTIONS, len(DIFFUSIVITIES)))
        for d in range(N_DIRECTIONS):
            sp = compute_map_spectrum(dir_decays[d], U)
            tot = sp.sum()
            dir_spectra[d] = sp / tot if tot > 0 else sp
        mean = dir_spectra.mean(axis=0)
        std = dir_spectra.std(axis=0)  # population std across 3 directions
        cv = np.divide(std, mean, out=np.full_like(mean, np.nan), where=mean > 1e-8) * 100.0
        for j, D in enumerate(DIFFUSIVITIES):
            rows.append({
                "roi": name,
                "diffusivity": D,
                "mean_fraction": mean[j],
                "std_fraction": std[j],
                "cv_pct": cv[j],
            })
    df = pd.DataFrame(rows)

    agg = (df.groupby("diffusivity")
             .agg(mean_cv_pct=("cv_pct", "mean"),
                  median_cv_pct=("cv_pct", "median"),
                  n_rois_with_signal=("cv_pct", lambda s: int(np.isfinite(s).sum())),
                  mean_fraction=("mean_fraction", "mean"))
             .reset_index())
    return df, agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("FIGURE 7 v2 — direction independence at whole-ROI level")
    print("=" * 72)
    print(f"MAP solver: recompute.compute_map_spectrum (ridge NNLS, lambda={RIDGE_STRENGTH})")
    U = build_design_matrix()

    # --- representative patient figure ---
    normal_path = DAT_DIR / NORMAL_ROI_FILE
    tumor_path = DAT_DIR / TUMOR_ROI_FILE
    print(f"\nPatient: {PATIENT}")
    print(f"  normal ROI: {NORMAL_ROI_FILE}")
    print(f"  tumour ROI: {TUMOR_ROI_FILE}")

    nd, ntr, nsp, ntsp = map_spectra_for_roi(normal_path, U)
    td, ttr, tsp, ttsp = map_spectra_for_roi(tumor_path, U)
    normal = {"dir_spectra": nsp, "trace_spectrum": ntsp}
    tumor = {"dir_spectra": tsp, "trace_spectrum": ttsp}

    make_figure(normal, tumor, str(OUT_FIG_DIR / "fig_directions_v2"))

    # --- aggregate per-bin direction-CV over all ROIs ---
    print("\n[Aggregate] per-bin direction-CV across all ROIs (MAP)")
    df, agg = aggregate_direction_cv(U)
    csv_path = OUT_CSV_DIR / "fig7_roi_direction_cv_MAP.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] per-ROI CV table saved: {csv_path}")
    n_rois = df["roi"].nunique()
    print(f"\nAggregate per-bin direction-CV (MAP), averaged over {n_rois} ROIs:")
    print(f"  {'D (um2/ms)':<12}{'mean CV%':<12}{'median CV%':<14}{'n ROIs':<8}{'mean frac':<10}")
    for _, r in agg.iterrows():
        print(f"  {r['diffusivity']:<12g}{r['mean_cv_pct']:<12.1f}"
              f"{r['median_cv_pct']:<14.1f}{int(r['n_rois_with_signal']):<8d}"
              f"{r['mean_fraction']:<10.3f}")

    finite = agg[np.isfinite(agg["mean_cv_pct"])]
    lo = finite.loc[finite["mean_cv_pct"].idxmin()]
    hi = finite.loc[finite["mean_cv_pct"].idxmax()]
    print(f"\n  Lowest direction-CV:  D={lo['diffusivity']:g} ({lo['mean_cv_pct']:.1f}%)")
    print(f"  Highest direction-CV: D={hi['diffusivity']:g} ({hi['mean_cv_pct']:.1f}%)")
    print(f"  Overall mean per-bin direction-CV: {finite['mean_cv_pct'].mean():.1f}%")
    print("\n  NOTE: MAP only (NUTS not run for these .dat ROIs — no .nc files exist "
          "for them; they are not part of the 149-ROI gold-standard set).")


if __name__ == "__main__":
    main()
