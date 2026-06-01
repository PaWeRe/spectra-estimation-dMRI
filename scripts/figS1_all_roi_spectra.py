"""Supplementary atlas S1: per-ROI NUTS posterior spectra as box plots.

Each panel is ONE ROI's full NUTS posterior over the 8-bin diffusivity spectrum,
drawn as box plots (median line, IQR box, 5th-95th percentile whiskers, no
fliers -- the posteriors are continuous and unimodal, so matplotlib "outliers"
are merely tail draws). The box FACE is coloured by the within-ROI per-bin
coefficient of variation (CV = posterior std / posterior mean; purple sequential,
matching the Fig 4 identifiability scheme). The box width is the *absolute*
posterior spread; the colour is the *relative* spread, so a low-mean outer bin
can show a narrow box yet a dark (poorly-identified-relative-to-its-size) colour.
The tuned-MAP point estimate (lambda = 1e-3) is overlaid as a green x.

Reproducibility: panel titles carry the PUBLIC identifiers Patrick released --
patient id (key into data/bwh/signal_decays.json) + zone/tissue + Gleason
score / GGG (data/bwh/metadata.csv). A companion key CSV (figS1_roi_key.csv)
lists the exact anatomical_region + SNR per panel so any reader can trace a
panel back to the released signal decays.

Multi-page PDF, 2 columns/page, grouped by zone x tissue, ordered by ADC within
each group. Embed via \\includepdf (needs \\usepackage{pdfpages}).

Posterior draws are read directly from results/inference_bwh_backup/*.nc (the
.nc filename is the MD5 spectra_id, reproduced via recompute.compute_spectra_id).

Outputs (paper/figures/):
    figS1_all_roi_spectra.pdf          -- multi-page atlas
    figS1_all_roi_spectra_preview.png  -- first page, for quick review
    figS1_roi_key.csv                  -- panel -> public data key (authors/reviewers)

Usage:
    uv run python scripts/figS1_all_roi_spectra.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from spectra_estimation_dmri.biomarkers.recompute import compute_spectra_id
from spectra_estimation_dmri.visualization.identifiability import (
    cv_color, cv_hatch, cv_legend_handles)

REPO = Path(__file__).resolve().parents[1]
FEAT = REPO / "results" / "biomarkers" / "features.csv"
SIGNAL_JSON = REPO / "src" / "spectra_estimation_dmri" / "data" / "bwh" / "signal_decays.json"
METADATA_CSV = REPO / "src" / "spectra_estimation_dmri" / "data" / "bwh" / "metadata.csv"
NC_DIR = REPO / "results" / "inference_bwh_backup"
OUT = REPO / "paper" / "figures"

D = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
DLAB = ["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "20"]
DIFF_VARS = [f"diff_{d:.2f}" for d in D]  # var names inside the .nc posteriors
MAPC = [f"map_D_{d:.2f}" for d in D]      # tuned-MAP columns in features.csv

MAP_MARK = "#2ca02c"   # green x, matches the MAP colour convention
NCOLS, NROWS = 2, 4    # 8 ROIs / page -- taller panels = more y-axis resolution
                       # (supplementary is not space-limited; more pages is fine)
YMAX = 0.95            # normalised fractions; 95th pct of the lumen bin can be high

# Identifiability (per-bin posterior CV) colour + hatch are shared with Fig 4
# via spectra_estimation_dmri.visualization.identifiability (colourblind-safe).

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 14, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 11,
    "hatch.linewidth": 0.6,
})


# ---------------------------------------------------------------------------
# Data loading: posterior draws + public-data identifiers, joined on roi_id.
# ---------------------------------------------------------------------------
def load_rois() -> list[dict]:
    """Build one record per ROI with posterior draws + public identifiers.

    Returns dicts keyed by roi_id ("{patient}_{zone}_{tissue}") carrying:
        samples (n_draws, 8) normalised NUTS posterior, anatomical_region, snr.
    Mirrors recompute.load_dataset's roi_id / snr / spectra_id conventions so the
    .nc lookup and the features.csv join both line up.
    """
    import arviz as az

    signal_data = json.load(open(SIGNAL_JSON))
    meta = pd.read_csv(METADATA_CSV, dtype=str).set_index("patient_id")

    records: dict[str, dict] = {}
    for patient_id, patient_rois in signal_data.items():
        m = meta.loc[patient_id] if patient_id in meta.index else None
        gs = (m["gs"] if m is not None else None)
        targets = (m["targets"] if m is not None else None)
        ggg = int(targets) if (isinstance(targets, str) and targets.isdigit()) else None

        for roi in patient_rois.values():
            anat = roi["anatomical_region"]
            if "pz" in anat:
                zone = "pz"
            elif "tz" in anat:
                zone = "tz"
            else:
                continue  # "Neglected!"
            is_tumor = "tumor" in anat
            signal = np.array(roi["signal_values"])
            b_values = roi["b_values"]
            snr = float(np.sqrt(roi["v_count"] / 16) * 150)
            roi_id = f"{patient_id}_{zone}_{'tumor' if is_tumor else 'normal'}"

            nc_path = NC_DIR / f"{compute_spectra_id(signal, b_values, snr)}.nc"
            if not nc_path.exists():
                print(f"  [WARN] missing .nc for {roi_id}: {nc_path.name}")
                continue
            idata = az.from_netcdf(nc_path)
            samp = np.column_stack(
                [idata.posterior[v].values.flatten() for v in DIFF_VARS])
            samp = samp / np.maximum(samp.sum(axis=1, keepdims=True), 1e-10)

            records[roi_id] = {
                "roi_id": roi_id,
                "patient": patient_id,
                "zone": zone,
                "is_tumor": is_tumor,
                "anatomical_region": anat,
                "snr": snr,
                "gs": gs,
                "ggg": ggg,
                "samples": samp,
            }
    return records


def panel_title(rec: dict) -> str:
    """Public reproducibility id: patient + zone/tissue (+ Gleason/GGG for tumor)."""
    tissue = "tumor" if rec["is_tumor"] else "normal"
    base = f"{rec['patient']} · {rec['zone'].upper()} {tissue}"
    if not rec["is_tumor"]:
        return base
    gs = rec["gs"]
    gs_str = gs if (isinstance(gs, str) and "+" in gs and gs != "0+0") else None
    ggg = rec["ggg"]
    has_ggg = ggg is not None and ggg >= 1
    if gs_str and has_ggg:
        return f"{base} · {gs_str} (GGG {ggg})"
    if gs_str:
        return f"{base} · {gs_str}"
    if has_ggg:
        return f"{base} · GGG {ggg}"
    return f"{base} · ungraded"


def draw_panel(ax, rec, mapvec, show_x, show_y) -> None:
    samp = rec["samples"]
    pos = np.arange(1, len(D) + 1)
    mean = samp.mean(axis=0)
    std = samp.std(axis=0)
    cv = np.divide(std, mean, out=np.full_like(mean, np.nan), where=mean > 1e-8)

    bp = ax.boxplot(
        samp, positions=pos, widths=0.62, whis=(5, 95), showfliers=False,
        patch_artist=True, showmeans=True, meanline=True,
        medianprops=dict(color="black", linewidth=1.1),
        meanprops=dict(color="0.25", linestyle="--", linewidth=0.9),
        whiskerprops=dict(color="0.4", linewidth=0.8),
        capprops=dict(color="0.4", linewidth=0.8),
        boxprops=dict(linewidth=0.5),
    )
    for patch, c in zip(bp["boxes"], cv):
        patch.set_facecolor(cv_color(c))
        patch.set_edgecolor("black")
        patch.set_hatch(cv_hatch(c))

    ax.scatter(pos, mapvec, marker="x", s=24, c=MAP_MARK, linewidths=1.4, zorder=6)

    ax.set_ylim(0, YMAX)
    ax.set_xlim(0.4, len(D) + 0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(DLAB if show_x else [])
    ax.set_yticks(np.arange(0, YMAX + 0.01, 0.2))
    # NB: with sharey=True, set_yticklabels([]) on an inner axis propagates to the
    # shared group and blanks the left column too; let sharey auto-hide instead.
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_title(panel_title(rec), fontsize=11)


def legend_handles():
    handles = cv_legend_handles()
    handles += [
        Line2D([0], [0], color="black", lw=1.3, label="median"),
        Line2D([0], [0], color="0.25", lw=1.0, linestyle="--", label="mean"),
        Line2D([0], [0], marker="x", color=MAP_MARK, linestyle="None",
               markersize=8, markeredgewidth=1.5, label=r"MAP ($\lambda=10^{-3}$)"),
    ]
    return handles


def make_page(pdf, chunk, mapchunk, header, preview_path=None):
    n = len(chunk)
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(8.5, 11.0), squeeze=False,
                             sharey=True)
    # Reserve top band for legend+header and left for the shared y-label so the
    # fixed-size PDF page never clips them (the v1 bug).
    fig.subplots_adjust(left=0.105, right=0.975, top=0.835, bottom=0.075,
                        hspace=0.46, wspace=0.10)
    for k in range(NROWS * NCOLS):
        r, c = k // NCOLS, k % NCOLS
        ax = axes[r][c]
        if k >= n:
            ax.axis("off")
            continue
        is_bottom = (k + NCOLS >= n) or (r == NROWS - 1)
        draw_panel(ax, chunk[k], mapchunk[k], show_x=is_bottom, show_y=(c == 0))

    fig.legend(handles=legend_handles(), loc="upper center", ncol=4, frameon=True,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.945),
               title="NUTS posterior   (box = IQR, whiskers = 5–95th pct; "
                     "box colour = per-bin CV / identifiability)",
               title_fontsize=10)
    fig.text(0.105, 0.988, header, ha="left", va="top", fontsize=11,
             color="0.35", fontstyle="italic")
    fig.supxlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)", fontsize=14, y=0.018)
    fig.supylabel(r"spectral fraction $R_j$", fontsize=14, x=0.022)
    pdf.savefig(fig)  # no bbox_inches -> fixed 8.5x11 page; nothing clipped
    if preview_path is not None:
        fig.savefig(preview_path, dpi=160)
    plt.close(fig)


def main() -> None:
    records = load_rois()
    feat = pd.read_csv(FEAT).set_index("roi_id")

    cats = [
        ("pz", False, "peripheral zone · normal"),
        ("pz", True, "peripheral zone · tumor"),
        ("tz", False, "transition zone · normal"),
        ("tz", True, "transition zone · tumor"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "figS1_all_roi_spectra.pdf"
    preview = OUT / "figS1_all_roi_spectra_preview.png"
    per = NROWS * NCOLS
    key_rows = []
    n_pages = 0

    with PdfPages(pdf_path) as pdf:
        for z, t, header in cats:
            group = [r for r in records.values()
                     if r["zone"] == z and r["is_tumor"] == t]
            # order by ADC (descending) for a readable gradient, like v1
            group.sort(
                key=lambda r: feat.loc[r["roi_id"], "adc"]
                if r["roi_id"] in feat.index else np.nan,
                reverse=True)
            pages_in_cat = int(np.ceil(len(group) / per))
            for start in range(0, len(group), per):
                chunk = group[start:start + per]
                mapchunk = [
                    feat.loc[r["roi_id"], MAPC].to_numpy(float)
                    if r["roi_id"] in feat.index else np.full(len(D), np.nan)
                    for r in chunk
                ]
                pg = start // per + 1
                hdr = f"{header}   ({len(group)} ROIs · page {pg}/{pages_in_cat})"
                make_page(pdf, chunk, mapchunk, hdr,
                          preview_path=preview if n_pages == 0 else None)
                n_pages += 1

            for r in group:
                adc = feat.loc[r["roi_id"], "adc"] if r["roi_id"] in feat.index else np.nan
                key_rows.append({
                    "panel_title": panel_title(r),
                    "roi_id": r["roi_id"],
                    "patient_id": r["patient"],
                    "anatomical_region": r["anatomical_region"],
                    "zone": r["zone"],
                    "tissue": "tumor" if r["is_tumor"] else "normal",
                    "gs": r["gs"],
                    "ggg": r["ggg"],
                    "snr": round(r["snr"], 1),
                    "adc": round(float(adc), 5) if np.isfinite(adc) else np.nan,
                    "n_draws": r["samples"].shape[0],
                })

    key_path = OUT / "figS1_roi_key.csv"
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    print(f"wrote {pdf_path.relative_to(REPO)} ({n_pages} pages)")
    print(f"wrote {preview.relative_to(REPO)} (page 1)")
    print(f"wrote {key_path.relative_to(REPO)} ({len(key_rows)} ROIs)")


if __name__ == "__main__":
    main()
