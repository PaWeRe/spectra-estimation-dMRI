"""Supplementary subset figures (Stephan 2026-06-19).

A representative 8-ROI subset -- 2 zones x {Normal, GGG1, GGG3, GGG5} -- shown
two complementary ways, sharing the SAME ROIs so the reader can cross-reference:

  (1) figS_subset_atlas.pdf       -- NUTS posterior spectra as box plots (2x4
                                     grid: rows = zone, cols = grade ladder).
  (2) figS_subset_convergence.pdf -- NUTS sampler diagnostics per ROI: trace +
                                     autocorrelation, ALL 8 fractions overlaid
                                     per panel (the old Gibbs-atlas style), so
                                     the wandering / "fraction switching" of the
                                     unidentifiable intermediate bins is visible
                                     while the well-identified outer bins stay
                                     pinned. One page per zone (4 ROIs x 2).

This supersedes the full 149-ROI atlas (Patrick 2026-06-19: subsample to a
representative set rather than publish every case). NUTS only.

Posterior draws (4 chains x 2000) are read from results/inference_bwh_backup/*.nc
(filename = MD5 spectra_id, via recompute.compute_spectra_id), the same lookup as
scripts/figS1_all_roi_spectra.py.

Outputs (paper/figures/):
    figS_subset_atlas.{pdf,png}
    figS_subset_convergence.{pdf,png}        (pdf = 2 pages; png = PZ page preview)
    figS_subset_convergence_tz.png           (TZ page preview)
    figS_subset_key.csv                      (panel -> public data key)

Usage:
    uv run python scripts/fig_si_subset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import arviz as az
import matplotlib.pyplot as plt
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
DIFF_VARS = [f"diff_{d:.2f}" for d in D]
MAPC = [f"map_D_{d:.2f}" for d in D]
# 8-fraction palette: consistent with Fig 1 panel (c); avoids orange/grey clashes.
FRAC_COLORS = ["#1f77b4", "#17becf", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]
MAP_MARK = "#2ca02c"

# Locked subset (Patrick 2026-06-19). Rows = zone; cols = Normal/GGG1/GGG3/GGG5.
SUBSET_PZ = ["new52_pz_normal", "new61_pz_tumor", "new02_pz_tumor", "new03_pz_tumor"]
SUBSET_TZ = ["new37_tz_normal", "new20_tz_tumor", "new01_tz_tumor", "new45_tz_tumor"]
SUBSET = SUBSET_PZ + SUBSET_TZ

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 12, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "hatch.linewidth": 0.6,
})


def disp_pid(pid: str) -> str:
    """new52 -> P52 (Stephan 2026-06-19); the raw id stays in the key CSV."""
    return ("P" + pid[3:]) if pid.startswith("new") else pid


def panel_label(rec: dict) -> str:
    pid = disp_pid(rec["patient"])
    tissue = "Tumor" if rec["is_tumor"] else "Normal"
    base = f"{pid} · {rec['zone'].upper()} {tissue}"
    if rec["is_tumor"] and rec["ggg"]:
        base += f" · GGG {rec['ggg']}"
    return base


def load_subset() -> dict:
    """Load only the 8 subset ROIs with per-chain posterior draws + metadata."""
    signal_data = json.load(open(SIGNAL_JSON))
    meta = pd.read_csv(METADATA_CSV, dtype=str).set_index("patient_id")
    feat = pd.read_csv(FEAT).set_index("roi_id")

    recs: dict[str, dict] = {}
    for patient_id, patient_rois in signal_data.items():
        m = meta.loc[patient_id] if patient_id in meta.index else None
        gs = m["gs"] if m is not None else None
        targets = m["targets"] if m is not None else None
        ggg = int(targets) if (isinstance(targets, str) and targets.isdigit()) else None

        for roi in patient_rois.values():
            anat = roi["anatomical_region"]
            zone = "pz" if "pz" in anat else ("tz" if "tz" in anat else None)
            if zone is None:
                continue
            is_tumor = "tumor" in anat
            roi_id = f"{patient_id}_{zone}_{'tumor' if is_tumor else 'normal'}"
            if roi_id not in SUBSET:
                continue

            signal = np.array(roi["signal_values"])
            b_values = roi["b_values"]
            snr = float(np.sqrt(roi["v_count"] / 16) * 150)
            nc_path = NC_DIR / f"{compute_spectra_id(signal, b_values, snr)}.nc"
            idata = az.from_netcdf(nc_path)

            # per-chain draws: (chain, draw, 8), normalised within each draw
            ch = np.stack([idata.posterior[v].values for v in DIFF_VARS], axis=-1)
            ch = ch / np.maximum(ch.sum(axis=-1, keepdims=True), 1e-10)

            rh = az.rhat(idata)
            rhat = max(float(rh[v].values) for v in DIFF_VARS)
            mapvec = (feat.loc[roi_id, MAPC].to_numpy(float)
                      if roi_id in feat.index else np.full(len(D), np.nan))
            adc = float(feat.loc[roi_id, "adc"]) if roi_id in feat.index else np.nan

            recs[roi_id] = dict(
                roi_id=roi_id, patient=patient_id, zone=zone, is_tumor=is_tumor,
                anat=anat, snr=snr, gs=gs, ggg=ggg, rhat=rhat, mapvec=mapvec,
                adc=adc, chains=ch, samples=ch.reshape(-1, len(D)))

    missing = [r for r in SUBSET if r not in recs]
    if missing:
        raise SystemExit(f"missing subset ROIs (no .nc?): {missing}")
    return recs


# ---------------------------------------------------------------------------
# Figure 1: subset spectral atlas (2x4 box plots).
# ---------------------------------------------------------------------------
def make_atlas(recs: dict) -> None:
    pos = np.arange(1, len(D) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.4), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.115,
                        hspace=0.50, wspace=0.10)

    for ri, row_ids in enumerate([SUBSET_PZ, SUBSET_TZ]):
        for ci, roi_id in enumerate(row_ids):
            rec = recs[roi_id]
            ax = axes[ri][ci]
            samp = rec["samples"]
            mean, std = samp.mean(0), samp.std(0)
            cv = np.divide(std, mean, out=np.full_like(mean, np.nan), where=mean > 1e-8)
            bp = ax.boxplot(
                samp, positions=pos, widths=0.62, whis=(5, 95), showfliers=False,
                patch_artist=True, showmeans=True, meanline=True,
                medianprops=dict(color="black", linewidth=1.1),
                meanprops=dict(color="0.25", linestyle="--", linewidth=0.9),
                whiskerprops=dict(color="0.4", linewidth=0.8),
                capprops=dict(color="0.4", linewidth=0.8),
                boxprops=dict(linewidth=0.5))
            for patch, c in zip(bp["boxes"], cv):
                patch.set_facecolor(cv_color(c))
                patch.set_edgecolor("black")
                patch.set_hatch(cv_hatch(c))
            ax.scatter(pos, rec["mapvec"], marker="x", s=22, c=MAP_MARK,
                       linewidths=1.3, zorder=6)
            ax.set_ylim(0, 0.95)
            ax.set_xlim(0.4, len(D) + 0.6)
            ax.set_xticks(pos)
            ax.set_xticklabels(DLAB if ri == 1 else [], fontsize=8)
            ax.set_yticks(np.arange(0, 0.96, 0.2))
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_title(f"{panel_label(rec)}\nSNR {rec['snr']:.0f}", fontsize=10)

    handles = cv_legend_handles() + [
        Line2D([0], [0], color="black", lw=1.3, label="median"),
        Line2D([0], [0], color="0.25", lw=1.0, linestyle="--", label="mean"),
        Line2D([0], [0], marker="x", color=MAP_MARK, linestyle="None",
               markersize=8, markeredgewidth=1.5, label=r"MAP ($\lambda=10^{-3}$)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.99),
               title="NUTS posterior   (box = IQR, whiskers = 5–95th pct;  "
                     "box colour = per-bin CV / identifiability)",
               title_fontsize=10, fontsize=10)
    fig.supxlabel(r"Diffusivity $D$ ($\mu$m$^2$/ms)", fontsize=13, y=0.02)
    fig.supylabel(r"Spectral fraction $R_j$", fontsize=13, x=0.02)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figS_subset_atlas.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figS_subset_atlas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figS_subset_atlas.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure 2: subset convergence (trace + autocorrelation), one page per zone.
# ---------------------------------------------------------------------------
def make_convergence(recs: dict, max_lag: int = 120) -> None:
    pages = [(SUBSET_PZ, "Peripheral Zone", "figS_subset_convergence.png"),
             (SUBSET_TZ, "Transition Zone", "figS_subset_convergence_tz.png")]
    with PdfPages(OUT / "figS_subset_convergence.pdf") as pdf:
        for zone_ids, zlabel, preview in pages:
            fig, axes = plt.subplots(4, 2, figsize=(8.5, 11.0), squeeze=False)
            fig.subplots_adjust(left=0.09, right=0.975, top=0.90, bottom=0.06,
                                hspace=0.62, wspace=0.24)
            for r, roi_id in enumerate(zone_ids):
                rec = recs[roi_id]
                ch = rec["chains"]               # (chain, draw, 8)
                nchain, ndraw, _ = ch.shape
                ax_tr, ax_ac = axes[r][0], axes[r][1]

                # --- trace: 4 chains concatenated, all 8 fractions overlaid ---
                concat = ch.reshape(-1, len(D))
                xs = np.arange(concat.shape[0])
                for j in range(len(D)):
                    ax_tr.plot(xs, concat[:, j], color=FRAC_COLORS[j], lw=0.35,
                               alpha=0.8)
                for bnd in range(1, nchain):
                    ax_tr.axvline(bnd * ndraw, color="0.7", lw=0.6, ls=":")
                ax_tr.set_xlim(0, concat.shape[0])
                ax_tr.set_ylim(0, max(0.6, float(concat.max()) * 1.05))
                ax_tr.set_ylabel("Fraction", fontsize=9)
                ax_tr.set_title(
                    f"{panel_label(rec)}   ·   SNR {rec['snr']:.0f}   ·   "
                    rf"$\hat{{R}}_{{\max}}$ {rec['rhat']:.3f}",
                    fontsize=9.5, loc="left")
                ax_tr.tick_params(labelsize=8)
                if r == len(zone_ids) - 1:
                    ax_tr.set_xlabel("Draw (4 chains concatenated)", fontsize=9)

                # --- autocorrelation: per-chain ACF averaged, all 8 overlaid ---
                for j in range(len(D)):
                    acf = np.mean([az.autocorr(ch[c, :, j])[:max_lag]
                                   for c in range(nchain)], axis=0)
                    ax_ac.plot(np.arange(max_lag), acf, color=FRAC_COLORS[j],
                               lw=1.0, alpha=0.85)
                ax_ac.axhline(0, color="0.6", lw=0.6)
                ax_ac.set_xlim(0, max_lag)
                ax_ac.set_ylim(-0.15, 1.02)
                ax_ac.set_ylabel("Autocorr.", fontsize=9)
                ax_ac.set_title("Autocorrelation", fontsize=9.5, loc="left")
                ax_ac.tick_params(labelsize=8)
                if r == len(zone_ids) - 1:
                    ax_ac.set_xlabel("Lag (draws)", fontsize=9)

            handles = [Line2D([0], [0], color=FRAC_COLORS[j], lw=2.2,
                              label=rf"$D$ = {DLAB[j]}") for j in range(len(D))]
            fig.legend(handles=handles, loc="upper center", ncol=8, frameon=True,
                       framealpha=0.95, bbox_to_anchor=(0.5, 0.955), fontsize=9,
                       columnspacing=1.0, handletextpad=0.4)
            fig.suptitle(f"NUTS convergence — {zlabel} subset "
                         "(trace + autocorrelation, all 8 fractions)",
                         y=0.992, fontsize=12)
            pdf.savefig(fig)
            fig.savefig(OUT / preview, dpi=150, bbox_inches="tight")
            plt.close(fig)
    print("wrote figS_subset_convergence.pdf (2 pages) + previews")


def write_key(recs: dict) -> None:
    rows = []
    for roi_id in SUBSET:
        r = recs[roi_id]
        rows.append(dict(
            panel=panel_label(r), display_id=disp_pid(r["patient"]),
            roi_id=roi_id, raw_patient_id=r["patient"],
            anatomical_region=r["anat"], zone=r["zone"],
            tissue="tumor" if r["is_tumor"] else "normal",
            gs=r["gs"], ggg=r["ggg"], snr=round(r["snr"], 1),
            adc=round(r["adc"], 5) if np.isfinite(r["adc"]) else np.nan,
            rhat_max=round(r["rhat"], 4), n_draws=r["samples"].shape[0]))
    path = OUT / "figS_subset_key.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path.name} ({len(rows)} ROIs)")


def main() -> None:
    recs = load_subset()
    make_atlas(recs)
    make_convergence(recs)
    write_key(recs)


if __name__ == "__main__":
    main()
