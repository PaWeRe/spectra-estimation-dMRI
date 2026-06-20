"""Supplementary subset figures (Patrick 2026-06-20 redesign).

Two complementary, single-page SI figures, each doing the job it is best at:

  (1) figS_subset_atlas.pdf       -- NUTS posterior spectra as box plots, PAIRED
                                     PER PATIENT: portrait grid, 2 cols
                                     (Normal | Tumor) x 6 rows (one patient each),
                                     spanning the grade ladder GGG1/3/5 in both
                                     zones. The reader reads each row left->right
                                     to see the within-patient tumor-vs-normal
                                     spectral shift directly. Box colour = per-bin
                                     posterior CV (identifiability), NO hatch
                                     (colour-only, like Fig 6).
  (2) figS_subset_convergence.pdf -- NUTS sampler diagnostics on 3 tumours chosen
                                     to span the SNR range (low / mid / high), so
                                     the effect of noise on mixing is visible.
                                     Per ROI: trace (a single chain, poorly-id
                                     bins bold / well-id bins faded so the
                                     "fraction switching" of unidentifiable bins
                                     stands out) + zoomed autocorrelation. One
                                     page (3 rows x 2).

NUTS only. Posterior draws (4 chains x 2000) are read from
results/inference_bwh_backup/*.nc (filename = MD5 spectra_id, via
recompute.compute_spectra_id), the same lookup as scripts/figS1_all_roi_spectra.py.

Outputs (paper/figures/):
    figS_subset_atlas.{pdf,png}              (1 page, portrait)
    figS_subset_convergence.{pdf,png}        (1 page, portrait)
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
from matplotlib.lines import Line2D

from spectra_estimation_dmri.biomarkers.recompute import compute_spectra_id
from spectra_estimation_dmri.visualization.identifiability import (
    cv_color, cv_legend_handles)

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

# --- S2 atlas: 6 same-patient pairs, grade ladder x zone (Patrick 2026-06-20) ---
# Each row = one patient; col 0 = its Normal ROI, col 1 = its Tumor ROI.
ATLAS_PAIRS = [  # (patient, zone), ordered PZ GGG1/3/5 then TZ GGG1/3/5
    ("new54", "pz"), ("new02", "pz"), ("new03", "pz"),
    ("new20", "tz"), ("new01", "tz"), ("new45", "tz"),
]
ATLAS_ROWS = [(f"{p}_{z}_normal", f"{p}_{z}_tumor") for p, z in ATLAS_PAIRS]

# --- S3 convergence: 3 tumours spanning SNR (low/mid/high) (Patrick 2026-06-20) ---
CONV_ROIS = ["new50_pz_tumor",   # SNR 188  (low,  GGG2 3+4)
             "new44_tz_tumor",   # SNR 467  (mid,  GGG2 3+4)
             "new55_pz_tumor"]   # SNR 1229 (high, GGG4 4+4)

# Union of all ROIs that must be loaded (atlas pairs + convergence tumours).
SUBSET = list(dict.fromkeys([r for row in ATLAS_ROWS for r in row] + CONV_ROIS))

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.labelsize": 12, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "hatch.linewidth": 0.6,
})


def disp_pid(pid: str) -> str:
    """new52 -> P52 (Stephan 2026-06-19); the raw id stays in the key CSV."""
    return ("P" + pid[3:]) if pid.startswith("new") else pid


def panel_label(rec: dict, with_grade: bool = True) -> str:
    """One-line subtitle, '|'-separated (Patrick 2026-06-20), e.g.
    'P02 | PZ Tumor | GGG3 (4+3) | SNR 369'."""
    pid = disp_pid(rec["patient"])
    tissue = "Tumor" if rec["is_tumor"] else "Normal"
    parts = [pid, f"{rec['zone'].upper()} {tissue}"]
    if with_grade and rec["is_tumor"] and rec["ggg"]:
        gs = rec.get("gs")
        gs_txt = f" ({gs})" if isinstance(gs, str) and gs and gs != "nan" else ""
        parts.append(f"GGG{rec['ggg']}{gs_txt}")
    parts.append(f"SNR {rec['snr']:.0f}")
    return "  |  ".join(parts)


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
# Figure S2: patient-paired spectral atlas (portrait, 2 cols x 6 rows).
# col 0 = Normal, col 1 = Tumor, both from the same patient (direct comparison).
# ---------------------------------------------------------------------------
def make_atlas(recs: dict) -> None:
    pos = np.arange(1, len(D) + 1)
    nrows = len(ATLAS_ROWS)
    # Narrower than tall so the panels are less elongated (Patrick 2026-06-20);
    # width-fit to \textwidth in LaTeX still leaves it within one page.
    fig, axes = plt.subplots(nrows, 2, figsize=(8.6, 2.15 * nrows + 1.2),
                             sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.905, bottom=0.05,
                        hspace=0.42, wspace=0.08)

    for ri, (n_id, t_id) in enumerate(ATLAS_ROWS):
        for ci, roi_id in enumerate((n_id, t_id)):
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
            # Colour-only identifiability encoding (no hatch); matches Fig 6
            # (Patrick 2026-06-20: drop the "muster" / hatch pattern).
            for patch, c in zip(bp["boxes"], cv):
                patch.set_facecolor(cv_color(c))
                patch.set_edgecolor("black")
            ax.scatter(pos, rec["mapvec"], marker="x", s=20, c=MAP_MARK,
                       linewidths=1.2, zorder=6)
            ax.set_ylim(0, 0.95)
            ax.set_xlim(0.4, len(D) + 0.6)
            ax.set_xticks(pos)
            ax.set_xticklabels(DLAB if ri == nrows - 1 else [], fontsize=8)
            ax.set_yticks(np.arange(0, 0.96, 0.2))
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.set_title(panel_label(rec), fontsize=9.5)

    handles = cv_legend_handles(hatch=False) + [
        Line2D([0], [0], color="black", lw=1.3, label="median"),
        Line2D([0], [0], color="0.25", lw=1.0, linestyle="--", label="mean"),
        Line2D([0], [0], marker="x", color=MAP_MARK, linestyle="None",
               markersize=8, markeredgewidth=1.5, label=r"MAP ($\lambda=10^{-3}$)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.995),
               title="NUTS posterior   ·   box = IQR, whiskers 5–95th pct   ·   "
                     "box colour = per-bin CV (identifiability)",
               title_fontsize=9.5, fontsize=9.5)
    fig.supxlabel(r"Diffusivity $D$ ($\mu$m$^2$/ms)", fontsize=12, y=0.018)
    fig.supylabel(r"Spectral fraction $R_j$", fontsize=12, x=0.018)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figS_subset_atlas.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figS_subset_atlas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figS_subset_atlas.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure S3: NUTS convergence on 3 tumours spanning SNR (single page, 3 rows x 2).
# Trace = ONE chain. Highlighted (bold) = the well-determined restricted bin
# D=0.25 (the stable anchor) + the 2 largest-spread bins (the anti-correlated
# "switching" pair along the posterior ridge); other bins faded. The ACF panel
# keeps all 8 fractions at uniform intensity (Patrick 2026-06-20).
# ---------------------------------------------------------------------------
def make_convergence(recs: dict, max_lag: int = 15,
                     trace_window: int = 300) -> None:
    rois = CONV_ROIS
    nrows = len(rois)
    # Compact (no overarching title; tighter rows) so it fits one Overleaf page.
    fig, axes = plt.subplots(nrows, 2, figsize=(9.5, 2.4 * nrows + 0.9),
                             squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.93, bottom=0.085,
                        hspace=0.42, wspace=0.22)
    for r, roi_id in enumerate(rois):
        rec = recs[roi_id]
        ch = rec["chains"]                       # (chain, draw, 8)
        nchain, ndraw, _ = ch.shape
        ax_tr, ax_ac = axes[r][0], axes[r][1]

        # Highlight set: D=0.25 (index 0, the well-determined anchor -- show its
        # stability) + the 2 bins with the largest absolute posterior spread (the
        # anti-correlated pair that trades weight along the collinear ridge).
        std = rec["samples"].std(0)
        wanderers = [j for j in np.argsort(-std) if j != 0][:2]
        bold = np.zeros(len(D), dtype=bool)
        bold[0] = True
        bold[wanderers] = True

        # --- trace: a SINGLE chain, first `trace_window` draws so individual
        # excursions and the weight-swapping are legible (full-chain mixing is
        # certified by the ACF + R-hat). Faded bins first, bold on top.
        nshow = min(trace_window, ndraw)
        chain0 = ch[0, :nshow]                    # (nshow, 8)
        xs = np.arange(nshow)
        for j in np.argsort(bold.astype(int)):    # False (faded) then True (bold)
            ax_tr.plot(xs, chain0[:, j], color=FRAC_COLORS[j],
                       lw=1.0 if bold[j] else 0.4,
                       alpha=0.9 if bold[j] else 0.18,
                       zorder=3 if bold[j] else 1)
        ax_tr.set_xlim(0, nshow)
        ax_tr.set_ylim(0, max(0.6, float(chain0.max()) * 1.05))
        ax_tr.set_ylabel(r"Spectral fraction $R_j$", fontsize=9)
        ax_tr.set_title(
            panel_label(rec) + rf"   |   $\hat{{R}}_{{\max}}$ {rec['rhat']:.3f}",
            fontsize=9.5, loc="left")
        ax_tr.tick_params(labelsize=8)
        if r == nrows - 1:
            ax_tr.set_xlabel(f"Draw (single chain, first {nshow} of 2000)",
                             fontsize=9)

        # --- autocorrelation: all 8 fractions, uniform intensity (Patrick
        # 2026-06-20), per-chain ACF averaged, zoomed to a short lag window. ---
        for j in range(len(D)):
            acf = np.mean([az.autocorr(ch[c, :, j])[:max_lag]
                           for c in range(nchain)], axis=0)
            ax_ac.plot(np.arange(max_lag), acf, color=FRAC_COLORS[j],
                       lw=1.0, alpha=0.85)
        ax_ac.axhline(0, color="0.6", lw=0.6)
        ax_ac.set_xlim(0, max_lag - 1)
        ax_ac.set_ylim(-0.15, 1.02)
        ax_ac.set_ylabel("Autocorrelation", fontsize=9)   # no repeating title
        ax_ac.tick_params(labelsize=8)
        if r == nrows - 1:
            ax_ac.set_xlabel("Sample Lag", fontsize=9)

    handles = [Line2D([0], [0], color=FRAC_COLORS[j], lw=2.2,
                      label=rf"$D$ = {DLAB[j]}") for j in range(len(D))]
    fig.legend(handles=handles, loc="upper center", ncol=8, frameon=True,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.995), fontsize=9,
               columnspacing=1.0, handletextpad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figS_subset_convergence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figS_subset_convergence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote figS_subset_convergence.{pdf,png} (1 page)")


def write_key(recs: dict) -> None:
    atlas_ids = {r for row in ATLAS_ROWS for r in row}
    rows = []
    for roi_id in SUBSET:
        r = recs[roi_id]
        used = [tag for tag, ids in (("S2_atlas", atlas_ids),
                                     ("S3_convergence", set(CONV_ROIS)))
                if roi_id in ids]
        rows.append(dict(
            panel=panel_label(r), used_in="+".join(used),
            display_id=disp_pid(r["patient"]),
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
