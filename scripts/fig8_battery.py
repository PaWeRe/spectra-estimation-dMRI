"""
Figure 8 (v6, MAIN) — Simulation recovery BATTERY with an SNR dimension
=======================================================================

A focused simulation-recovery battery that establishes the joint Bayesian
inference recovers a variety of ground-truth spectra ACROSS noise levels,
rebuilt with the CURRENT paper's real inference pipeline (no configuration
drift) and the locked paper style.

Lead-author redesign (2026-06-05, v6):
  * 4 representative ground truths:
      1. normal-like   — cohort-mean NORMAL NUTS spectrum
      2. tumour-like   — cohort-mean TUMOUR NUTS spectrum
      3. bimodal       — peaks at D=0.25 and D=3.0
      4. delta (δ)     — all mass at D=0.75 (most ill-conditioned bin)
  * SNR dimension WIDENED to the extremes (we also apply this voxel-wise where
    per-pixel SNR is low): LOW=50 and HIGH=1000. True noise σ_true = 1/SNR.
  * LAYOUT = 5 ROWS x 2 COLUMNS. Columns = SNR (LEFT col = SNR 50, RIGHT col =
    SNR 1000), consistent across all rows.
      - Rows 1–4 = the 4 ground-truth spectra recovery: black truth marks +
        orange NUTS box plots + green-x tuned-MAP; R-hat in each subtitle.
      - Row 5 = NOISE INFERENCE as BOX PLOTS: bottom-LEFT = σ̂ box plots at
        SNR=50 (one box per spectrum, ~12 noise realizations each) with a
        horizontal line at true σ=1/50; bottom-RIGHT = σ̂ box plots at
        SNR=1000 with a line at true σ=1/1000. Noise boxes use a DISTINCT
        colour from the recovery panels. Per-SNR split avoids the previous
        single-panel axis-compression problem.
  * ONE figure-level legend on TOP only — NO legends inside any subplot.
    Everything (truth / NUTS / MAP / noise-box identity / true-σ line) is
    folded into the top legend.

Methodology fidelity:
  - NUTS : spectra_estimation_dmri.inference.nuts.NUTSSampler (configs/inference/nuts.yaml)
           2000 draws / 200 tune / 4 chains / target_accept 0.95.
           sigma ~ HalfCauchy is JOINTLY INFERRED (HalfCauchy beta = 1/SNR).
  - MAP  : biomarkers.recompute.compute_map_spectrum (tuned ridge lambda = 1e-3).
  - prior: configs/prior/ridge.yaml (strength 0.1 -> sigma_R = 1/sqrt(0.1) = 3.162).
  - NO Gibbs (the Gibbs-vs-NUTS comparison is DEFERRED for this submission).

Caching / resumability
----------------------
  results/simulation/fig8_battery_snr.npz   (cache; resumable)
  results/simulation/fig8_battery_snr_nc/    (per-condition NUTS .nc)
The cache stores per-(spectrum, SNR) recovery results and per-realization
sigma_hat lists keyed by (spectrum, SNR). On --regen we reuse everything
already present FOR THE CURRENT SNR LEVELS and only sample the MISSING
conditions / realizations, saving after EACH run so a crash/resume is cheap.
If a cached condition was sampled at a DIFFERENT SNR than the current target
(e.g. legacy 75/600), it is re-sampled at the new SNR.

Outputs
-------
  paper/figures/fig8_v6.png  (300 dpi)
  paper/figures/fig8_v6.pdf
(fig8_v5 is left intact.)

Usage
-----
  uv run python scripts/fig8_battery.py --regen --quick   # fast pipeline check
  uv run python scripts/fig8_battery.py --regen           # full (real configs)
  uv run python scripts/fig8_battery.py                   # --replot from cache
"""

import os
import io
import sys
import argparse
import time
import contextlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, "src")

from spectra_estimation_dmri.visualization.paper_style import (
    apply_style, COLORS, DIFFUSIVITIES as DIFF_STYLE, DLABELS, set_diff_xaxis,
    top_legend,
)

# ----------------------------------------------------------------- constants
B_VALUES_MS = np.array([0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75,
                        2., 2.25, 2.5, 2.75, 3., 3.25, 3.5])
DIFFUSIVITIES = np.asarray(DIFF_STYLE)               # locked grid from paper_style
N_D = len(DIFFUSIVITIES)

# Two SNR levels WIDENED to the extremes (v6). True noise σ_true = 1/SNR.
SNR_LEVELS = {"low": 50, "high": 1000}
PRIOR_STRENGTH = 0.1           # configs/prior/ridge.yaml -> sigma_R = 3.162

# Number of independent noise realizations per (spectrum, SNR) for the sigma
# box plots, and the REDUCED NUTS settings used for those runs.
N_SIGMA_REALIZATIONS = 12
SIGMA_DRAWS = 1000
SIGMA_CHAINS = 2

FEATURES_CSV = "results/biomarkers/features.csv"
CACHE_NPZ = "results/simulation/fig8_battery_snr.npz"
NC_DIR = "results/simulation/fig8_battery_snr_nc"
OUT_PNG = "paper/figures/fig8_v6.png"
OUT_PDF = "paper/figures/fig8_v6.pdf"

# Locked colours from paper_style
C_TRUTH = COLORS["truth"]
C_NUTS = COLORS["nuts"]
C_MAP = COLORS["map"]
# Distinct colour for the noise (sigma) box plots — NOT a reserved estimator
# colour (orange/green/black). Use the slate blue hue (unused here; Gibbs is
# excluded from this figure so there is no clash).
C_NOISE = COLORS["gibbs"]      # slate blue

# 4 representative ground truths (panel/cache order) + pretty labels.
GT_ORDER = ["normal", "tumor", "bimodal", "delta"]
GT_LABELS = {
    "normal": "normal-like",
    "tumor": "tumour-like",
    "bimodal": "bimodal",
    "delta": r"$\delta$ ($D{=}0.75$)",
}
# Compact x-tick labels for the noise row (full names collide at 18pt).
GT_LABELS_SHORT = {
    "normal": "normal",
    "tumor": "tumour",
    "bimodal": "bimodal",
    "delta": r"$\delta$",
}


# ----------------------------------------------------------------- helpers
def build_U():
    return np.exp(-np.outer(B_VALUES_MS, DIFFUSIVITIES))


def normalize(R):
    s = R.sum()
    return R / s if s > 0 else R


def normalize_rows(S):
    rs = S.sum(axis=1, keepdims=True)
    return S / np.maximum(rs, 1e-12)


def define_ground_truths():
    """The 4 representative ground-truth spectra (each normalized to sum 1).

    Cohort-mean normal/tumour from the NUTS feature table; bimodal + delta from
    the robustness menu.
    """
    df = pd.read_csv(FEATURES_CSV)
    cols = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
    normal = normalize(df[df["is_tumor"] == 0][cols].mean().values.astype(float))
    tumor = normalize(df[df["is_tumor"] == 1][cols].mean().values.astype(float))

    # bimodal: peaks at D=0.25 (bin 0) and D=3.0 (bin 6), small fill, no free water
    bimodal = np.zeros(N_D)
    bimodal[0] = 0.4
    bimodal[6] = 0.4
    bimodal[1:6] = 0.04
    bimodal = normalize(bimodal)

    # delta at D=0.75 (bin 2): most ill-conditioned region
    delta = np.zeros(N_D)
    delta[int(np.argmin(np.abs(DIFFUSIVITIES - 0.75)))] = 1.0

    return {"normal": normal, "tumor": tumor, "bimodal": bimodal, "delta": delta}


def simulate_signal(R_true_norm, U, snr, rng):
    mu = U @ R_true_norm
    return mu + rng.normal(0.0, 1.0 / snr, size=mu.shape)


def box_stats(samples):
    q05, q25, med, q75, q95 = np.percentile(samples, [5, 25, 50, 75, 95], axis=0)
    return dict(whislo=q05, q1=q25, med=med, q3=q75, whishi=q95,
                mean=samples.mean(axis=0))


def rhat_from_nc(path):
    """Per-bin R-hat from a saved .nc with diff_* variables -> array of R-hat."""
    import arviz as az
    idata = az.from_netcdf(path)
    var_names = [f"diff_{d:.2f}" for d in DIFFUSIVITIES]
    rhat_ds = az.rhat(idata, var_names=var_names)
    return np.array([float(rhat_ds[v].values) for v in var_names])


def _load_store():
    if os.path.exists(CACHE_NPZ):
        return dict(np.load(CACHE_NPZ, allow_pickle=True))
    return {}


def _save_store(store):
    os.makedirs(os.path.dirname(CACHE_NPZ), exist_ok=True)
    np.savez(CACHE_NPZ, **store)


# ----------------------------------------------------------------- generation
def _build_nuts_cfg(snr, n_draws, n_chains, tune, target_accept, seed):
    """Build a NUTS run config whose HalfCauchy noise prior is matched to SNR.

    The HalfCauchy beta is set from dataset.snr (= 1/SNR) — the honest setup,
    since in the real pipeline the per-ROI SNR is known. sampler_snr=None lets
    the sampler fall back to dataset.snr for the prior scale.
    """
    from omegaconf import OmegaConf
    base_ds = {"diff_values": DIFFUSIVITIES.tolist(), "snr": snr,
               "spectrum_pair": None}
    return OmegaConf.create({
        "seed": seed, "dataset": base_ds,
        "inference": {"name": "nuts", "n_iter": n_draws, "tune": tune,
                      "n_chains": n_chains, "target_accept": target_accept,
                      "sampler_snr": None, "init": "map"}})


def generate(quick=False, only=None, target_accept=0.95, tune=None,
             do_recovery=True, do_sigma=True):
    """Resumable NUTS recovery + tuned-MAP battery across (spectrum, SNR).

    Reuses everything already present in the cache FOR THE CURRENT SNR LEVELS;
    only samples the MISSING (or SNR-mismatched) conditions / realizations.
    Saves the cache after EACH run.

    only : optional list of GT keys to (re)sample (recovery only).
    """
    from spectra_estimation_dmri.models.prob_model import ProbabilisticModel
    from spectra_estimation_dmri.data.data_models import SignalDecay
    from spectra_estimation_dmri.inference.nuts import NUTSSampler
    from spectra_estimation_dmri.biomarkers.recompute import compute_map_spectrum
    from omegaconf import OmegaConf

    os.makedirs(NC_DIR, exist_ok=True)
    U = build_U()
    gts = define_ground_truths()
    keys_to_run = only if only else GT_ORDER

    if quick:
        rec_draws, rec_tune, rec_chains = 300, 200, 2
        sig_draws, sig_chains, n_real = 300, 2, 4
    else:
        rec_draws, rec_tune, rec_chains = 2000, 200, 4      # NUTS = nuts.yaml
        sig_draws, sig_chains, n_real = SIGMA_DRAWS, SIGMA_CHAINS, N_SIGMA_REALIZATIONS
    rec_tune = tune if tune is not None else rec_tune

    prior_cfg = OmegaConf.create({"type": "ridge", "strength": PRIOR_STRENGTH})
    model = ProbabilisticModel(data_snr=None, prior_config=prior_cfg,
                               b_values=B_VALUES_MS.tolist(),
                               diffusivities=DIFFUSIVITIES.tolist())

    def make_sd(y, snr):
        return SignalDecay(patient="sim", signal_values=y.tolist(),
                           b_values=B_VALUES_MS.tolist(), a_region="sim", snr=snr)

    def run_quiet(fn):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()

    store = _load_store()
    store.update({"D": DIFFUSIVITIES, "gt_keys": np.array(GT_ORDER),
                  "snr_low": SNR_LEVELS["low"], "snr_high": SNR_LEVELS["high"]})
    # persist the true spectra (cheap, idempotent)
    for key in GT_ORDER:
        store[f"{key}_R_true"] = normalize(gts[key].astype(float))

    t_start = time.time()

    # ---------------- recovery panels: one realization per (spectrum, SNR) ----
    if do_recovery:
        for key in keys_to_run:
            R_true = normalize(gts[key].astype(float))
            for snr_name, snr in SNR_LEVELS.items():
                tag = f"{key}_{snr_name}"
                # SNR-aware resume: a cached condition is only valid if its
                # stored sigma_true matches the current 1/SNR target.
                cached_complete = (
                    (f"{tag}_nuts_med" in store) and (f"{tag}_map_R" in store)
                    and (f"{tag}_nuts_rhat_max" in store))
                snr_matches = (
                    f"{tag}_sigma_true" in store
                    and np.isclose(float(store[f"{tag}_sigma_true"]), 1.0 / snr))
                if cached_complete and snr_matches:
                    print(f"[skip] recovery {tag} already cached (SNR={snr})")
                    continue
                print(f"\n=== recovery '{key}' @ SNR={snr} ({snr_name}) ===")
                t0 = time.time()
                rng = np.random.default_rng(
                    20260604 + (abs(hash(key)) % 10000) + snr)
                y0 = simulate_signal(R_true, U, snr, rng)
                sd0 = make_sd(y0, snr)
                cfg = _build_nuts_cfg(snr, rec_draws, rec_chains, rec_tune,
                                      target_accept, seed=42)
                nuts0 = run_quiet(lambda: NUTSSampler(model, sd0, cfg).run(
                    show_progress=False, save_dir=NC_DIR, unique_hash=f"nuts_{tag}"))
                map0 = normalize(compute_map_spectrum(y0, U))

                nuts_samp = normalize_rows(np.array(nuts0.spectrum_samples))
                st = box_stats(nuts_samp)
                for fld, arr in st.items():
                    store[f"{tag}_nuts_{fld}"] = arr
                store[f"{tag}_map_R"] = map0
                store[f"{tag}_sigma_true"] = 1.0 / snr
                store[f"{tag}_sigma_hat"] = 1.0 / nuts0.sampler_snr

                rhat = rhat_from_nc(os.path.join(NC_DIR, f"nuts_{tag}.nc"))
                store[f"{tag}_nuts_rhat"] = rhat
                store[f"{tag}_nuts_rhat_max"] = float(rhat.max())
                _save_store(store)  # save after each run (resumable)
                print(f"    done in {time.time()-t0:.0f}s | "
                      f"sigma_hat={store[f'{tag}_sigma_hat']:.5f} "
                      f"(true {1.0/snr:.5f}) | max R-hat={rhat.max():.3f}")

    # ---------------- sigma calibration: many realizations (reduced draws) ----
    if do_sigma:
        for key in keys_to_run:
            R_true = normalize(gts[key].astype(float))
            for snr_name, snr in SNR_LEVELS.items():
                tag = f"{key}_{snr_name}"
                sig_key = f"{tag}_sigma_real"
                # SNR-aware resume: discard cached realizations sampled at a
                # different SNR (e.g. legacy 75/600).
                snr_matches = (
                    f"{tag}_sigma_real_snr" in store
                    and np.isclose(float(store[f"{tag}_sigma_real_snr"]), snr))
                existing = list(store.get(sig_key, np.array([]))) \
                    if (sig_key in store and snr_matches) else []
                if not snr_matches:
                    # drop stale realizations from a previous SNR
                    store[sig_key] = np.array([], dtype=float)
                need = n_real - len(existing)
                if need <= 0:
                    print(f"[skip] sigma {tag}: {len(existing)} realizations "
                          f"cached (SNR={snr})")
                    continue
                print(f"\n=== sigma realizations '{key}' @ SNR={snr} "
                      f"({snr_name}) : need {need} more ===")
                for r in range(len(existing), n_real):
                    t0 = time.time()
                    rng = np.random.default_rng(
                        99000000 + (abs(hash(key)) % 10000) + snr * 1000 + r)
                    y = simulate_signal(R_true, U, snr, rng)
                    sd = make_sd(y, snr)
                    cfg = _build_nuts_cfg(snr, sig_draws, sig_chains, 200,
                                          target_accept, seed=1000 + r)
                    # save_dir required: the result model rejects a null path.
                    # we only need sampler_snr, so overwrite a single scratch .nc.
                    sig_nc_dir = os.path.join(NC_DIR, "sigma_scratch")
                    nuts = run_quiet(lambda: NUTSSampler(model, sd, cfg).run(
                        show_progress=False, save_dir=sig_nc_dir,
                        unique_hash=f"sig_{tag}"))
                    sig_hat = 1.0 / nuts.sampler_snr
                    existing.append(sig_hat)
                    store[sig_key] = np.array(existing, dtype=float)
                    store[f"{tag}_sigma_real_snr"] = float(snr)
                    store[f"{tag}_sigma_true"] = 1.0 / snr
                    _save_store(store)  # save after each realization
                    print(f"    real {r}: sigma_hat={sig_hat:.5f} "
                          f"(true {1.0/snr:.5f}) | {time.time()-t0:.0f}s")

    store["wall_time_s"] = float(time.time() - t_start)
    _save_store(store)
    print(f"\nWrote cache {CACHE_NPZ}")
    rmaxes = [store[f"{k}_{s}_nuts_rhat_max"]
              for k in GT_ORDER for s in SNR_LEVELS
              if f"{k}_{s}_nuts_rhat_max" in store]
    if rmaxes:
        print(f"max R-hat across all recovery panels = {max(rmaxes):.3f}")
    print(f"this invocation wall-time = {store['wall_time_s']:.0f}s")


# ----------------------------------------------------------------- plotting
def _bxp_dicts(d, tag):
    out = []
    for j in range(N_D):
        out.append(dict(
            whislo=float(d[f"{tag}_nuts_whislo"][j]),
            q1=float(d[f"{tag}_nuts_q1"][j]),
            med=float(d[f"{tag}_nuts_med"][j]),
            q3=float(d[f"{tag}_nuts_q3"][j]),
            whishi=float(d[f"{tag}_nuts_whishi"][j]),
            mean=float(d[f"{tag}_nuts_mean"][j]),
            fliers=[], label=""))
    return out


def _recovery_panel(ax, d, key, snr_name, snr, show_ylabel, show_xlabel):
    tag = f"{key}_{snr_name}"
    x = np.arange(N_D)
    bw = 0.5
    ax.bxp(_bxp_dicts(d, tag), positions=x, widths=bw,
           showfliers=False, showmeans=False, patch_artist=True,
           medianprops=dict(color="black", lw=1.1),
           whiskerprops=dict(color="0.4", lw=0.9),
           capprops=dict(color="0.4", lw=0.9),
           boxprops=dict(facecolor=C_NUTS, edgecolor="black",
                         linewidth=0.6, alpha=0.7),
           zorder=3)
    # ground truth: discrete per-bin level marks (NOT a connected line)
    ax.hlines(d[f"{key}_R_true"], x - 0.44, x + 0.44, color=C_TRUTH, lw=2.6,
              zorder=5)
    # tuned-MAP point estimate: green "x"
    ax.scatter(x, d[f"{tag}_map_R"], marker="x", s=70, c=C_MAP, linewidths=2.0,
               zorder=6)

    set_diff_xaxis(ax, label=show_xlabel, rotation=0)
    ax.set_xlim(-0.65, N_D - 0.35)
    if show_ylabel:
        ax.set_ylabel("spectral fraction")

    rhat_max = float(d[f"{tag}_nuts_rhat_max"])
    label = GT_LABELS[key]
    title = f"{label}, SNR={snr} — $\\hat{{R}}$={rhat_max:.2f}"
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.grid(True, axis="y", alpha=0.25)


def _noise_panel(ax, d, snr_name, snr, show_ylabel):
    """Box plots of inferred sigma_hat across noise realizations, one box per
    spectrum, with a horizontal line at the true sigma = 1/SNR."""
    bxp_list = []
    positions = []
    n_real_seen = []
    for i, key in enumerate(GT_ORDER):
        tag = f"{key}_{snr_name}"
        sig_key = f"{tag}_sigma_real"
        if sig_key not in d:
            continue
        vals = np.asarray(d[sig_key], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q05, q1, med, q3, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        bxp_list.append(dict(whislo=q05, q1=q1, med=med, q3=q3, whishi=q95,
                             mean=float(vals.mean()), fliers=[], label=""))
        positions.append(i)
        n_real_seen.append(vals.size)

    if bxp_list:
        ax.bxp(bxp_list, positions=positions, widths=0.5,
               showfliers=False, showmeans=False, patch_artist=True,
               medianprops=dict(color="black", lw=1.1),
               whiskerprops=dict(color="0.4", lw=0.9),
               capprops=dict(color="0.4", lw=0.9),
               boxprops=dict(facecolor=C_NOISE, edgecolor="black",
                             linewidth=0.6, alpha=0.7),
               zorder=3)

    sig_true = 1.0 / snr
    ax.axhline(sig_true, ls="--", lw=2.0, color=C_TRUTH, zorder=4)

    ax.set_xticks(range(len(GT_ORDER)))
    ax.set_xticklabels([GT_LABELS_SHORT[k] for k in GT_ORDER], rotation=0)
    ax.set_xlim(-0.65, len(GT_ORDER) - 0.35)
    if show_ylabel:
        ax.set_ylabel(r"inferred $\hat{\sigma}$")

    n_real = max(n_real_seen) if n_real_seen else N_SIGMA_REALIZATIONS
    # Descriptive subplot title matching the recovery-panel style; the meaning
    # of n (independent noise realizations) lives in the figure caption.
    title = rf"noise inference, SNR={snr} — $n = {n_real}$"
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.grid(True, axis="y", alpha=0.25)
    # headroom so the true-sigma line and boxes are not flush with the frame
    ymax = sig_true
    for key in GT_ORDER:
        sig_key = f"{key}_{snr_name}_sigma_real"
        if sig_key in d:
            vals = np.asarray(d[sig_key], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ymax = max(ymax, float(np.percentile(vals, 95)))
    ax.set_ylim(0, ymax * 1.18)


def plot():
    d = dict(np.load(CACHE_NPZ, allow_pickle=True))
    apply_style("grid")

    # Layout: 5 rows x 2 columns. Columns = SNR (LEFT = low, RIGHT = high).
    # Rows 1-4 = recovery; row 5 = noise box plots split by SNR.
    fig = plt.figure(figsize=(14, 24))
    gs = fig.add_gridspec(
        5, 2, height_ratios=[1, 1, 1, 1, 1],
        hspace=0.40, wspace=0.30,
        left=0.085, right=0.985, top=0.945, bottom=0.045)

    snr_items = [("low", SNR_LEVELS["low"]), ("high", SNR_LEVELS["high"])]

    # Per-row shared y-limit (over both SNR columns) so the SNR comparison is
    # fair within a spectrum; spiky spectra get their own scale automatically.
    for r, key in enumerate(GT_ORDER):
        ymax = float(d[f"{key}_R_true"].max())
        for snr_name, _ in snr_items:
            tag = f"{key}_{snr_name}"
            ymax = max(ymax, float(d[f"{tag}_nuts_whishi"].max()))
        ymax *= 1.12
        for c, (snr_name, snr) in enumerate(snr_items):
            ax = fig.add_subplot(gs[r, c])
            _recovery_panel(ax, d, key, snr_name, snr,
                            show_ylabel=(c == 0),
                            show_xlabel=False)
            ax.set_ylim(0, ymax)

    # ---- row 5: noise inference box plots, split by SNR ----
    for c, (snr_name, snr) in enumerate(snr_items):
        ax_n = fig.add_subplot(gs[4, c])
        _noise_panel(ax_n, d, snr_name, snr, show_ylabel=(c == 0))

    # ---- ONE figure-level legend on top (NO in-panel legends) ----
    handles = [
        Line2D([0], [0], color=C_TRUTH, lw=2.6, label="ground truth"),
        Patch(facecolor=C_NUTS, edgecolor="black", alpha=0.7,
              label="NUTS posterior"),
        Line2D([0], [0], color=C_MAP, marker="x", ms=11, lw=0, mew=2.2,
               label=r"tuned MAP ($\lambda=10^{-3}$)"),
        Patch(facecolor=C_NOISE, edgecolor="black", alpha=0.7,
              label=r"inferred $\hat{\sigma}$ (noise row)"),
        Line2D([0], [0], ls="--", lw=2.0, color=C_TRUTH,
               label=r"true $\sigma=1/\mathrm{SNR}$"),
    ]
    top_legend(fig, handles, [h.get_label() for h in handles],
               ncol=5, y=0.992)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}\nWrote {OUT_PDF}")

    rmaxes = [float(d[f"{k}_{s}_nuts_rhat_max"])
              for k in GT_ORDER for s in SNR_LEVELS
              if f"{k}_{s}_nuts_rhat_max" in d]
    if rmaxes:
        print(f"[caption] max R-hat across all panels = {max(rmaxes):.3f}")
    print("[caption] sigma_hat/sigma_true (median over realizations):")
    for key in GT_ORDER:
        for snr_name, snr in snr_items:
            sig_key = f"{key}_{snr_name}_sigma_real"
            if sig_key not in d:
                continue
            vals = np.asarray(d[sig_key], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            med = float(np.median(vals))
            ratio = med / (1.0 / snr)
            print(f"    {GT_LABELS[key]:<22} SNR={snr:>4}: "
                  f"sigma_hat={med:.5f} sigma_true={1.0/snr:.5f} "
                  f"ratio={ratio:.3f} (n={vals.size})")
    if "wall_time_s" in d:
        print(f"[info] cached total sampling wall-time (last invocation) = "
              f"{float(d['wall_time_s']):.0f}s")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true",
                    help="re-run NUTS sampling + tuned-MAP and refresh the cache "
                         "(resumable: only samples missing/SNR-mismatched conditions)")
    ap.add_argument("--replot", action="store_true",
                    help="re-plot from cache only (no sampling); this is the default")
    ap.add_argument("--quick", action="store_true",
                    help="fast NUTS settings for a pipeline smoke-test")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated GT keys to (re)sample (e.g. --only delta)")
    ap.add_argument("--no-recovery", dest="do_recovery", action="store_false",
                    help="skip the recovery-panel sampling")
    ap.add_argument("--no-sigma", dest="do_sigma", action="store_false",
                    help="skip the sigma-calibration realization sampling")
    ap.add_argument("--target-accept", dest="target_accept", type=float,
                    default=0.95, help="NUTS target_accept (raise for hard GTs)")
    ap.add_argument("--tune", type=int, default=None,
                    help="NUTS tuning steps (raise for hard GTs)")
    args = ap.parse_args()
    if args.regen:
        only = args.only.split(",") if args.only else None
        generate(quick=args.quick, only=only,
                 target_accept=args.target_accept, tune=args.tune,
                 do_recovery=args.do_recovery, do_sigma=args.do_sigma)
    plot()
