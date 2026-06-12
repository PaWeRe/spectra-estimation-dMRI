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
import argparse
import re
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
from spectra_estimation_dmri.visualization.paper_style import (  # noqa: E402
    apply_style,
    COLORS,
    DLABELS,
    set_diff_xaxis,
    top_legend,
    DIRECTION_COLORS,
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

# --- v3 (supplementary) selection: 4 patients x 2 ROIs, 2x4 grid -------------
# Stefan + Patrick picks. Columns = the 4 patients (relabelled 1-4; real IDs are
# arbitrary picks so they are NOT shown). Row 1 = first ROI of each pair, row 2
# = second ROI. Each entry is (.dat-file stem, zone, tissue-word-for-title).
V3_PATIENTS = [
    {  # column 1
        "row1": ("9675-Series9-Slice3-NormalPZ", "PZ", "normal"),
        "row2": ("9675-Series9-Slice3-TumorTZ", "TZ", "tumour"),
    },
    {  # column 2
        "row1": ("9322-Series11-Slice7-NormalPZ", "PZ", "normal"),
        "row2": ("9322-Series11-Slice7-TumorTZ", "TZ", "tumour"),
    },
    {  # column 3
        "row1": ("9283-Series12-Slice6-NormalTZ", "TZ", "normal"),
        "row2": ("9283-Series12-Slice6-TumorPZ", "PZ", "tumour"),
    },
    {  # column 4
        "row1": ("10203-Series9-Slice6-NormalPZ", "PZ", "normal"),
        "row2": ("10203-Series9-Slice6-TumorTZ", "TZ", "tumour"),
    },
]
V3_DIR_LABELS = ["Direction 1", "Direction 2", "Direction 3"]

# --- v4 (supplementary) selection: 4 patients x 2 zones, 4x2 grid ------------
# Same 8 ROIs as v3, but relaid out to enforce the manuscript-wide PZ-LEFT /
# TZ-RIGHT convention (matching Fig 1 + Fig 3). ROWS = the 4 patients (1-4),
# COLUMNS = zone (left = PZ, right = TZ). Each entry is the (.dat-file stem,
# tissue-word-for-title) for that patient's selected ROI in that zone; the
# tissue (normal/tumour) varies per cell and is shown in the panel title.
# Note patient 3 (9283) is reversed vs the others: tumour PZ / normal TZ.
V4_PATIENTS = [
    {  # row 1 -- patient 1 (9675): normal PZ, tumour TZ
        "PZ": ("9675-Series9-Slice3-NormalPZ", "normal"),
        "TZ": ("9675-Series9-Slice3-TumorTZ", "tumour"),
    },
    {  # row 2 -- patient 2 (9322): normal PZ, tumour TZ
        "PZ": ("9322-Series11-Slice7-NormalPZ", "normal"),
        "TZ": ("9322-Series11-Slice7-TumorTZ", "tumour"),
    },
    {  # row 3 -- patient 3 (9283): tumour PZ, normal TZ  (REVERSED)
        "PZ": ("9283-Series12-Slice6-TumorPZ", "tumour"),
        "TZ": ("9283-Series12-Slice6-NormalTZ", "normal"),
    },
    {  # row 4 -- patient 4 (10203): normal PZ, tumour TZ
        "PZ": ("10203-Series9-Slice6-NormalPZ", "normal"),
        "TZ": ("10203-Series9-Slice6-TumorTZ", "tumour"),
    },
]
V4_ZONES = ["PZ", "TZ"]  # column order: PZ left, TZ right
V4_DIR_LABELS = ["Direction 1", "Direction 2", "Direction 3"]

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
# v3 supplementary figure: 2 rows x 4 columns (4 patients x 2 ROIs)
# ---------------------------------------------------------------------------
def make_figure_v3(U, out_stem):
    """2x4 grid of per-direction MAP spectra + trace-averaged reference.

    Columns = 4 patients (relabelled 1-4). Row 1 = first ROI of each pair,
    row 2 = second ROI. Style follows the shared paper_style contract (Fig-1
    consistent): no angled tick labels (DLABELS), top legend, DIRECTION_COLORS
    for the 3 encoding directions, COLORS['truth'] for the trace-averaged
    reference. No corner CV annotation, no figure suptitle.

    Returns a list of (column_index, roi_stem, missing_or_ok) for reporting.
    """
    x = np.arange(len(DIFFUSIVITIES))
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.4 * nrows),
                             sharey=True)

    status = []
    for c, patient in enumerate(V3_PATIENTS):
        for r, row_key in enumerate(("row1", "row2")):
            ax = axes[r, c]
            stem, zone, tissue = patient[row_key]
            path = DAT_DIR / f"{stem}.dat"
            try:
                _, _, dir_spectra, trace_spectrum = map_spectra_for_roi(path, U)
                status.append((c + 1, stem, "ok"))
            except (ValueError, FileNotFoundError) as exc:
                ax.text(0.5, 0.5, f"{stem}\nMISSING:\n{exc}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="#b22222")
                ax.set_title(f"Patient {c + 1} — {tissue} {zone}")
                status.append((c + 1, stem, f"FAILED: {exc}"))
                continue

            for d in range(N_DIRECTIONS):
                ax.plot(x, dir_spectra[d], "o-", color=DIRECTION_COLORS[d], lw=2,
                        ms=6, alpha=0.9, label=V3_DIR_LABELS[d])
            ax.plot(x, trace_spectrum, "s-", color=COLORS["truth"], lw=2.6, ms=7,
                    label="Trace-averaged", zorder=5)

            set_diff_xaxis(ax, label=(r == nrows - 1), rotation=0)
            ax.set_title(f"Patient {c + 1} — {tissue} {zone}")
            ax.grid(axis="y", alpha=0.3)

    for r in range(nrows):
        axes[r, 0].set_ylabel("spectral fraction")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=4, y=1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = f"{out_stem}.png"
    pdf = f"{out_stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure: {png}")
    print(f"[INFO] Saved figure: {pdf}")
    return status


# ---------------------------------------------------------------------------
# v4 supplementary figure: 4 rows x 2 columns (4 patients x 2 zones)
# ---------------------------------------------------------------------------
def _tissue_disp(tissue: str) -> str:
    """Title-case, American-spelling tissue label for panel titles
    (Stephan 2026-06-12): 'tumour'/'normal' -> 'Tumor'/'Normal'."""
    return "Tumor" if tissue.lower().startswith("tum") else "Normal"


def make_figure_v4(U, out_stem):
    """4x2 grid of per-direction MAP spectra + trace-averaged reference.

    ROWS = the 4 patients (relabelled 1-4). COLUMNS = zone (left = peripheral
    zone PZ, right = transition zone TZ), enforcing the manuscript-wide
    PZ-LEFT / TZ-RIGHT convention (Fig 1 + Fig 3). The tissue (normal/tumour)
    varies per cell and is reported in the panel title, e.g. "Patient 1 --
    normal PZ" / "Patient 1 -- tumour TZ".

    Style follows the shared paper_style contract: no angled tick labels
    (DLABELS), single top legend, DIRECTION_COLORS for the 3 encoding
    directions, COLORS['truth'] for the trace-averaged reference. No corner CV
    annotation, no figure suptitle. Spacing mirrors Fig 1 (small legend->row1
    gap, even row-to-row gaps); the figure is tall (4 rows) and is destined for
    the SI.

    Returns a list of (patient_index, zone, roi_stem, status) for reporting.
    """
    x = np.arange(len(DIFFUSIVITIES))
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 16.0), sharey=True)

    status = []
    for r, patient in enumerate(V4_PATIENTS):
        for c, zone in enumerate(V4_ZONES):
            ax = axes[r, c]
            stem, tissue = patient[zone]
            path = DAT_DIR / f"{stem}.dat"
            try:
                _, _, dir_spectra, trace_spectrum = map_spectra_for_roi(path, U)
                status.append((r + 1, zone, stem, "ok"))
            except (ValueError, FileNotFoundError) as exc:
                ax.text(0.5, 0.5, f"{stem}\nMISSING:\n{exc}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="#b22222")
                ax.set_title(f"Patient {r + 1} — {_tissue_disp(tissue)} {zone}")
                status.append((r + 1, zone, stem, f"FAILED: {exc}"))
                continue

            for d in range(N_DIRECTIONS):
                ax.plot(x, dir_spectra[d], "o-", color=DIRECTION_COLORS[d], lw=2,
                        ms=6, alpha=0.9, label=V4_DIR_LABELS[d])
            ax.plot(x, trace_spectrum, "s-", color=COLORS["truth"], lw=2.6, ms=7,
                    label="Trace-averaged", zorder=5)

            # x-tick labels only on the bottom row (Stephan 2026-06-12: don't
            # repeat them per row); x-axis title also on the bottom row only.
            bottom = (r == nrows - 1)
            set_diff_xaxis(ax, label=bottom, ticklabels=bottom, rotation=0)
            ax.set_title(f"Patient {r + 1} — {_tissue_disp(tissue)} {zone}")
            ax.grid(axis="y", alpha=0.3)

    for r in range(nrows):
        axes[r, 0].set_ylabel("Spectral fraction")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, handles, labels, ncol=4, y=0.985)

    # Mirror Fig 1's visual feel: small gap from the top legend to row 1, even
    # row-to-row gaps. Tall 4-row figure so values differ from Fig 1's 2-row;
    # leave headroom so the legend clears the row-1 panel titles.
    fig.subplots_adjust(top=0.915, bottom=0.05, left=0.09, right=0.97,
                        hspace=0.45, wspace=0.18)
    png = f"{out_stem}.png"
    pdf = f"{out_stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[INFO] Saved figure: {png}")
    print(f"[INFO] Saved figure: {pdf}")
    return status


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
# Exploration helpers: verify the directional decode + list ALL ROIs
# ---------------------------------------------------------------------------
def parse_roi_label(name: str) -> dict:
    """Pull patient / zone / tissue out of a .dat basename.

    e.g. '9283-Series12-Slice6-TumorPZ' -> {patient:'9283', tissue:'Tumor',
    zone:'PZ', short:'9283 Tumor PZ'}.
    """
    patient = name.split("-")[0]
    m = re.search(r"(Normal|Tumor)(PZ|TZ)$", name)
    tissue = m.group(1) if m else "?"
    zone = m.group(2) if m else "?"
    return {
        "patient": patient,
        "tissue": tissue,
        "zone": zone,
        "short": f"{patient} {tissue} {zone}",
    }


def verify_all_rois(U):
    """Sanity-check the directional decode for EVERY .dat ROI.

    For each ROI prints:
      * n MEAN rows (must be 46) and the 3x15 block split,
      * whether each per-direction block is monotonic in b (raw ascending =
        b descending, the assumed layout),
      * the spread of the three b=0 estimates (last entry of each raw block;
        these are nominally the same image, so a small spread is a decode
        sanity check),
      * the trace ADC (b<=1000) as a physical-plausibility check,
      * trace-decay correlation of the forward vs reversed ordering is implied
        by monotonicity (reversed gives a decreasing S(b), the physical one).
    """
    files = sorted(glob.glob(str(DAT_DIR / "*.dat")))
    print("\n" + "=" * 104)
    print("DIRECTIONAL DECODE VERIFICATION (all ROIs)")
    print("=" * 104)
    print(f"{'ROI':<34}{'#rows':<7}{'trend rho':<11}{'ADC rev':<10}"
          f"{'ADC fwd':<10}{'b0 spread%':<12}{'dir-CV%':<9}")
    print("-" * 104)
    from spectra_estimation_dmri.biomarkers.recompute import compute_adc

    summary = []
    for path in files:
        name = os.path.basename(path).replace(".dat", "")
        try:
            arr = load_dat_mean_column(path)
        except ValueError as exc:
            print(f"{name:<34}FAILED: {exc}")
            continue
        rest = arr[1:]
        blocks = rest.reshape(N_DIRECTIONS, N_BVALUES)  # raw, b assumed descending
        # Trend: raw block should rank-increase (b descending). rho ~ +1 confirms
        # an ordered ramp up to noise (robust to single-step dips).
        order = np.arange(N_BVALUES)
        rhos = []
        for d in range(N_DIRECTIONS):
            ranks = np.argsort(np.argsort(blocks[d]))
            rhos.append(np.corrcoef(order, ranks)[0, 1])
        rho_min = min(rhos)
        # b=0 = last entry of each raw block; spread across directions (should be small).
        b0s = blocks[:, -1]
        b0_spread = (b0s.max() - b0s.min()) / b0s.mean() * 100.0
        dir_decays = blocks[:, ::-1]                    # reversed -> b ascending (physical)
        trace_decay = dir_decays.mean(axis=0)
        adc_rev = compute_adc(trace_decay) * 1000.0          # -> um^2/ms (physical)
        adc_fwd = compute_adc(blocks.mean(axis=0)) * 1000.0  # un-reversed -> should be < 0

        dir_spectra = np.zeros((N_DIRECTIONS, len(DIFFUSIVITIES)))
        for d in range(N_DIRECTIONS):
            sp = compute_map_spectrum(dir_decays[d], U)
            tot = sp.sum()
            dir_spectra[d] = sp / tot if tot > 0 else sp
        m = dir_spectra.mean(axis=0)
        s = dir_spectra.std(axis=0)
        cv = np.divide(s, m, out=np.full_like(m, np.nan), where=m > 1e-3) * 100.0
        mean_cv = np.nanmean(cv)

        warn = "  <-- big b0 spread" if b0_spread > 15 else ""
        print(f"{name:<34}{int(arr.size):<7}{rho_min:<11.3f}{adc_rev:<10.3f}"
              f"{adc_fwd:<+10.3f}{b0_spread:<12.1f}{mean_cv:<9.1f}{warn}")
        info = parse_roi_label(name)
        info.update(name=name, rho_min=rho_min, adc_rev=adc_rev, adc_fwd=adc_fwd,
                    b0_spread=b0_spread, mean_cv=mean_cv)
        summary.append(info)
    print("-" * 104)
    print("trend rho = min over 3 dirs of rank-corr(b-position, signal) for the raw block;")
    print("            ~+1 => clean ramp, confirming the 3x15 contiguous-block layout.")
    print("ADC rev   = monoexp ADC (b<=1000) of the REVERSED (physical) trace, um^2/ms (PZ normal ~1.5-2).")
    print("ADC fwd   = ADC of the UN-reversed ordering; NEGATIVE confirms b is stored descending.")
    print("b0 spread = (max-min)/mean of the 3 per-direction b=0 estimates, % (should be small).")
    print("dir-CV    = mean over bins of std/mean of the 3 per-direction MAP fractions.")
    return summary


def make_all_roi_grid(U, out_stem):
    """Grid of EVERY ROI: 3 per-direction MAP spectra + trace, one panel each.

    Lets Patrick eyeball all candidates and choose the representative ROI(s)
    for the final Fig 7. Panels are ordered Normal-first then Tumour, PZ before
    TZ, grouped by patient, and titled with patient/tissue/zone + mean dir-CV.
    """
    files = sorted(glob.glob(str(DAT_DIR / "*.dat")))
    items = []
    for path in files:
        name = os.path.basename(path).replace(".dat", "")
        info = parse_roi_label(name)
        info["path"] = path
        items.append(info)
    # Normal before Tumour, PZ before TZ, then patient.
    items.sort(key=lambda i: (i["tissue"] != "Normal", i["zone"], i["patient"]))

    x = np.arange(len(DIFFUSIVITIES))
    ncols = 4
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.7 * nrows),
                             sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, info in enumerate(items):
        ax = axes_flat[idx]
        dir_decays, trace_decay = decode_roi_directions(info["path"])
        dir_spectra = np.zeros((N_DIRECTIONS, len(DIFFUSIVITIES)))
        for d in range(N_DIRECTIONS):
            sp = compute_map_spectrum(dir_decays[d], U)
            tot = sp.sum()
            dir_spectra[d] = sp / tot if tot > 0 else sp
        sp_tr = compute_map_spectrum(trace_decay, U)
        tot = sp_tr.sum()
        trace_spectrum = sp_tr / tot if tot > 0 else sp_tr

        m = dir_spectra.mean(axis=0)
        s = dir_spectra.std(axis=0)
        cv = np.divide(s, m, out=np.full_like(m, np.nan), where=m > 1e-3) * 100.0
        mean_cv = np.nanmean(cv)

        for d in range(N_DIRECTIONS):
            ax.plot(x, dir_spectra[d], "o-", color=DIR_COLORS[d], lw=1.6, ms=5,
                    alpha=0.9, label=DIR_LABELS[d])
        ax.plot(x, trace_spectrum, "s-", color=TRACE_COLOR, lw=2.2, ms=6,
                label="Trace-averaged", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:g}" for d in DIFFUSIVITIES], rotation=45, fontsize=11)
        tissue_color = "#b22222" if info["tissue"] == "Tumor" else "#1a5fb4"
        ax.set_title(f"{info['short']}", fontsize=13, color=tissue_color,
                     fontweight="bold")
        ax.text(0.97, 0.97, f"dir-CV {mean_cv:.0f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="#555")
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(items), len(axes_flat)):
        axes_flat[j].axis("off")

    for r in range(nrows):
        axes_flat[r * ncols].set_ylabel("Spectral fraction")
    for c in range(ncols):
        bottom = (nrows - 1) * ncols + c
        if bottom < len(axes_flat):
            axes_flat[bottom].set_xlabel(r"Diffusivity $D$ ($\mu$m$^2$/ms)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=14)
    fig.suptitle("All directional ROIs — per-direction MAP spectra + trace "
                 "(red title = tumour, blue = normal)", fontsize=15, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n[INFO] Saved all-ROI spectrum grid: {out_stem}.png / .pdf")


def make_all_roi_decay_grid(U, out_stem):
    """Grid of EVERY ROI: the 3 per-direction NORMALISED signal decays + trace.

    This is the most direct visual check that the directional encoding decoded
    correctly: the three decays should overlap tightly and fall monotonically
    from 1.0 at b=0. Any block mis-assignment shows up as a decay that does not
    decrease or that sits far from the other two.
    """
    from spectra_estimation_dmri.biomarkers.recompute import B_VALUES_S_MM2
    b = B_VALUES_S_MM2
    files = sorted(glob.glob(str(DAT_DIR / "*.dat")))
    items = []
    for path in files:
        name = os.path.basename(path).replace(".dat", "")
        info = parse_roi_label(name)
        info["path"] = path
        items.append(info)
    items.sort(key=lambda i: (i["tissue"] != "Normal", i["zone"], i["patient"]))

    ncols = 4
    nrows = (len(items) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.7 * nrows),
                             sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, info in enumerate(items):
        ax = axes_flat[idx]
        dir_decays, trace_decay = decode_roi_directions(info["path"])
        for d in range(N_DIRECTIONS):
            ax.semilogy(b, normalize_by_b0(dir_decays[d]), "o-", color=DIR_COLORS[d],
                        lw=1.4, ms=4, alpha=0.85, label=DIR_LABELS[d])
        ax.semilogy(b, normalize_by_b0(trace_decay), "s-", color=TRACE_COLOR,
                    lw=2.0, ms=5, label="Trace-averaged", zorder=5)
        tissue_color = "#b22222" if info["tissue"] == "Tumor" else "#1a5fb4"
        ax.set_title(f"{info['short']}", fontsize=13, color=tissue_color,
                     fontweight="bold")
        ax.grid(alpha=0.3, which="both")

    for j in range(len(items), len(axes_flat)):
        axes_flat[j].axis("off")
    for r in range(nrows):
        axes_flat[r * ncols].set_ylabel(r"$S/S_0$")
    for c in range(ncols):
        bottom = (nrows - 1) * ncols + c
        if bottom < len(axes_flat):
            axes_flat[bottom].set_xlabel(r"$b$ (s/mm$^2$)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=14)
    fig.suptitle("All directional ROIs — per-direction signal decays + trace "
                 "(tight overlap + monotonic fall = correct decode)",
                 fontsize=15, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved all-ROI decay grid: {out_stem}.png / .pdf")


def explore_main():
    """Verify the decode for every ROI and emit the all-ROI grids."""
    print("=" * 72)
    print("FIGURE 7 — EXPLORE: verify decode + list ALL ROIs")
    print("=" * 72)
    U = build_design_matrix()
    verify_all_rois(U)
    make_all_roi_grid(U, str(OUT_FIG_DIR / "fig7_explore_all_rois_spectra"))
    make_all_roi_decay_grid(U, str(OUT_FIG_DIR / "fig7_explore_all_rois_decays"))
    print("\nDone. Inspect the two grids to choose the representative ROI(s).")


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

    # --- v3 supplementary 2x4 figure (4 patients x 2 ROIs) ---
    # Apply the shared paper_style contract (Fig-1 consistent) for v3.
    print("\n[v3] supplementary 2x4 directional figure (4 patients x 2 ROIs)")
    apply_style("grid")
    v3_status = make_figure_v3(U, str(OUT_FIG_DIR / "fig_directions_v3"))
    print("[v3] panel decode status:")
    for col, stem, st in v3_status:
        print(f"     col {col}: {stem:<34} {st}")
    n_ok = sum(1 for _, _, st in v3_status if st == "ok")
    print(f"[v3] {n_ok}/{len(v3_status)} ROIs decoded successfully.")

    # --- v4 supplementary 4x2 figure (4 patients x 2 zones, PZ-left/TZ-right) ---
    print("\n[v4] supplementary 4x2 directional figure (PZ-left / TZ-right)")
    v4_status = make_figure_v4(U, str(OUT_FIG_DIR / "fig_directions_v4"))
    print("[v4] panel decode status:")
    for pat, zone, stem, st in v4_status:
        print(f"     patient {pat} {zone}: {stem:<34} {st}")
    n_ok4 = sum(1 for _, _, _, st in v4_status if st == "ok")
    print(f"[v4] {n_ok4}/{len(v4_status)} ROIs decoded successfully.")

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explore", action="store_true",
                        help="Verify the decode for every ROI and emit all-ROI "
                             "grids (selection aid) instead of the final 2-panel "
                             "Fig 7.")
    cli = parser.parse_args()
    if cli.explore:
        explore_main()
    else:
        main()
