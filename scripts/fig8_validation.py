"""
Figure 8 (new, MAIN) — Method validation on simulated ground truth
===================================================================

Replaces the old Fisher/CRLB Fig 8 (dissolved to supplementary). Establishes
that the joint Bayesian inference recovers BOTH the spectrum and the noise
level on simulated data where the truth is known, and that the gradient-based
sampler (NUTS) is the right tool — the coordinate-wise Gibbs sampler we tried
first mixes too slowly on this correlated geometry.

IMPORTANT — methodology fidelity:
  This script drives the figure with the SAME pipeline classes used to produce
  the cohort .nc files and MAP features, so there is no configuration drift:
    - NUTS  : spectra_estimation_dmri.inference.nuts.NUTSSampler   (configs/inference/nuts.yaml)
    - Gibbs : spectra_estimation_dmri.inference.gibbs.GibbsSamplerClean (configs/inference/gibbs.yaml)
    - MAP   : biomarkers.recompute.compute_map_spectrum (tuned ridge lambda=1e-3)
    - prior : configs/prior/ridge.yaml (strength=0.1 -> sigma_R = 1/sqrt(0.1) = 3.162)
  NUTS infers sigma (HalfCauchy, beta=1/SNR); Gibbs fixes sigma=1/SNR (as deployed).

Layout (2 rows):
  Top row — spectrum recovery on three ground truths at cohort-median SNR=303,
  with the SAME noisy realisation fed to all three estimators:
    (a) normal-like  (cohort-mean NORMAL NUTS spectrum)
    (b) tumour-like  (cohort-mean TUMOUR NUTS spectrum)
    (c) delta stress (all mass at one intermediate bin)
    Truth = black line; NUTS / Gibbs posteriors = box plots (median/IQR/5-95%);
    tuned-MAP = green diamond. Wide intermediate-bin spread + the smeared delta
    make the identifiability limit (F8) and coverage caveat (F6) visible, and
    show recovery is non-trivial (rebuts the "you only recover what you put in"
    circularity worry).

  Bottom row —
    (d) joint NOISE recovery: NUTS posterior-mean sigma across noise
        realisations per ground truth, vs true sigma = 1/SNR. Noise level is
        INFERRED, not assumed.
    (e) sampler mixing on the delta stress case: per-bin effective sample size
        (log axis), NUTS vs Gibbs. Gibbs's ESS collapses on the correlated
        grid. Max R-hat annotated for both.

Outputs
-------
  results/simulation/fig8_validation.npz   (cache)
  paper/figures/fig8_v2.png  (300 dpi)
  paper/figures/fig8_v2.pdf

Usage
-----
  uv run python scripts/fig8_validation.py --regen --quick   # fast pipeline check
  uv run python scripts/fig8_validation.py --regen           # full (uses real configs)
  uv run python scripts/fig8_validation.py                   # re-plot from cache
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

# ----------------------------------------------------------------- constants
B_VALUES_MS = np.array([0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75,
                        2., 2.25, 2.5, 2.75, 3., 3.25, 3.5])
DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
D_LABELS = [f"{d:g}" for d in DIFFUSIVITIES]
N_D = len(DIFFUSIVITIES)

SNR = 303                      # cohort-median SNR (F10); true sigma = 1/SNR
TRUE_SIGMA = 1.0 / SNR
PRIOR_STRENGTH = 0.1           # configs/prior/ridge.yaml -> sigma_R = 3.162
DELTA_BIN = 2                  # delta stress at D=0.75 (most ill-conditioned region)
CONV_KEY = "delta"             # spectrum used for the mixing panel (e)

FEATURES_CSV = "results/biomarkers/features.csv"
CACHE_NPZ = "results/simulation/fig8_validation.npz"
NC_DIR = "results/simulation/fig8_nc"
OUT_PNG = "paper/figures/fig8_v2.png"
OUT_PDF = "paper/figures/fig8_v2.pdf"
OUT_PNG_V3 = "paper/figures/fig8_v3.png"
OUT_PDF_V3 = "paper/figures/fig8_v3.pdf"

# Reuse the EXACT colours/markers from Fig 1 and the supplementary S1 atlas so
# we don't introduce new style elements:
#   NUTS = orange solid box (Fig 1 NUTS_COLOR), tuned-MAP = green "x" (S1 atlas),
#   ground truth = black. Gibbs (new 3rd method) = a distinct cool colour.
C_TRUTH = "#1a1a1a"
C_NUTS = "#ff7f0e"   # Fig 1 NUTS_COLOR
C_MAP = "#2ca02c"    # Fig 1 / S1 MAP colour
C_GIBBS = "#4c72b0"  # 3rd method — distinct cool colour (slate blue)


# ----------------------------------------------------------------- helpers
def build_U():
    return np.exp(-np.outer(B_VALUES_MS, DIFFUSIVITIES))


def normalize(R):
    s = R.sum()
    return R / s if s > 0 else R


def normalize_rows(S):
    rs = S.sum(axis=1, keepdims=True)
    return S / np.maximum(rs, 1e-12)


def cohort_ground_truths():
    """Tumour-like / normal-like GTs from cohort-mean NUTS spectra + delta."""
    df = pd.read_csv(FEATURES_CSV)
    cols = [f"nuts_D_{d:.2f}" for d in DIFFUSIVITIES]
    normal = normalize(df[df["is_tumor"] == 0][cols].mean().values)
    tumor = normalize(df[df["is_tumor"] == 1][cols].mean().values)
    dlt = np.zeros(N_D); dlt[DELTA_BIN] = 1.0
    return {"normal": normal.astype(float), "tumor": tumor.astype(float),
            "delta": dlt}


def simulate_signal(R_true_norm, U, snr, rng):
    mu = U @ R_true_norm
    return mu + rng.normal(0.0, 1.0 / snr, size=mu.shape)


def box_stats(samples):
    """Per-bin box-plot stats from posterior samples (N, 8)."""
    q05, q25, med, q75, q95 = np.percentile(samples, [5, 25, 50, 75, 95], axis=0)
    return dict(whislo=q05, q1=q25, med=med, q3=q75, whishi=q95,
                mean=samples.mean(axis=0))


def rhat_ess_from_nc(path):
    """Per-bin (rhat, ess) from a saved .nc with diff_* variables."""
    import arviz as az
    idata = az.from_netcdf(path)
    var_names = [f"diff_{d:.2f}" for d in DIFFUSIVITIES]
    rhat_ds = az.rhat(idata, var_names=var_names)
    ess_ds = az.ess(idata, var_names=var_names)
    rhat = np.array([float(rhat_ds[v].values) for v in var_names])
    ess = np.array([float(ess_ds[v].values) for v in var_names])
    n_samples = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    return rhat, ess, n_samples


# ----------------------------------------------------------------- generation
def generate(quick=False):
    from omegaconf import OmegaConf
    from spectra_estimation_dmri.models.prob_model import ProbabilisticModel
    from spectra_estimation_dmri.data.data_models import SignalDecay
    from spectra_estimation_dmri.inference.nuts import NUTSSampler
    from spectra_estimation_dmri.inference.gibbs import GibbsSamplerClean
    from spectra_estimation_dmri.biomarkers.recompute import compute_map_spectrum

    os.makedirs(os.path.dirname(CACHE_NPZ), exist_ok=True)
    os.makedirs(NC_DIR, exist_ok=True)
    U = build_U()
    gts = cohort_ground_truths()

    if quick:
        n_reps, n_draws, tune, n_chains = 3, 300, 200, 2
        g_iter, g_burn, g_chains = 2000, 500, 2
    else:
        n_reps, n_draws, tune, n_chains = 25, 2000, 200, 4   # NUTS = nuts.yaml
        g_iter, g_burn, g_chains = 100000, 10000, 4          # Gibbs = gibbs.yaml

    prior_cfg = OmegaConf.create({"type": "ridge", "strength": PRIOR_STRENGTH})
    model = ProbabilisticModel(data_snr=SNR, prior_config=prior_cfg,
                               b_values=B_VALUES_MS.tolist(),
                               diffusivities=DIFFUSIVITIES.tolist())
    base_ds = {"diff_values": DIFFUSIVITIES.tolist(), "snr": SNR,
               "spectrum_pair": None}
    cfg_nuts = OmegaConf.create({
        "seed": 42, "dataset": base_ds,
        "inference": {"name": "nuts", "n_iter": n_draws, "tune": tune,
                      "n_chains": n_chains, "target_accept": 0.95,
                      "sampler_snr": None, "init": "map"}})
    cfg_gibbs = OmegaConf.create({
        "seed": 42, "dataset": base_ds,
        "inference": {"name": "gibbs", "n_iter": g_iter, "burn_in": g_burn,
                      "n_chains": g_chains, "init": "map", "sampler_snr": None}})

    def make_sd(y):
        return SignalDecay(patient="sim", signal_values=y.tolist(),
                           b_values=B_VALUES_MS.tolist(), a_region="sim",
                           snr=SNR)

    def run_quiet(fn):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()

    store = {"D": DIFFUSIVITIES, "true_sigma": TRUE_SIGMA, "snr": SNR,
             "gt_keys": np.array(list(gts.keys())), "conv_key": CONV_KEY}

    for key, R_true in gts.items():
        R_true = normalize(R_true.astype(float))
        store[f"{key}_R_true"] = R_true
        print(f"\n=== GT '{key}' : recovery (NUTS+Gibbs+MAP) + {n_reps} noise reps ===")
        t0 = time.time()
        rng = np.random.default_rng(20260531 + abs(hash(key)) % 1000)
        sigma_hats = []

        # rep 0 = representative recovery realisation, fed to ALL estimators
        y0 = simulate_signal(R_true, U, SNR, rng)
        sd0 = make_sd(y0)
        nuts0 = run_quiet(lambda: NUTSSampler(model, sd0, cfg_nuts).run(
            show_progress=False, save_dir=NC_DIR, unique_hash=f"nuts_{key}"))
        gibbs0 = run_quiet(lambda: GibbsSamplerClean(model, sd0, cfg_gibbs).run(
            show_progress=False, save_dir=NC_DIR, unique_hash=f"gibbs_{key}"))
        map0 = normalize(compute_map_spectrum(y0, U))

        nuts_samp = normalize_rows(np.array(nuts0.spectrum_samples))
        gibbs_samp = normalize_rows(np.array(gibbs0.spectrum_samples))
        for nm, st in [("nuts", box_stats(nuts_samp)),
                       ("gibbs", box_stats(gibbs_samp))]:
            for fld, arr in st.items():
                store[f"{key}_{nm}_{fld}"] = arr
        store[f"{key}_map_R"] = map0
        sigma_hats.append(1.0 / nuts0.sampler_snr)

        # remaining noise reps: NUTS only (sigma posterior mean per realisation)
        for rep in range(1, n_reps):
            y = simulate_signal(R_true, U, SNR, rng)
            spec = run_quiet(lambda: NUTSSampler(model, make_sd(y), cfg_nuts).run(
                show_progress=False, save_dir=NC_DIR,
                unique_hash=f"nuts_{key}_noisetmp"))
            sigma_hats.append(1.0 / spec.sampler_snr)
        store[f"{key}_sigma_hats"] = np.array(sigma_hats)

        # per-bin R-hat / ESS / sample-count from the saved recovery .nc files
        nr, ne, nn = rhat_ess_from_nc(os.path.join(NC_DIR, f"nuts_{key}.nc"))
        gr, ge, gn = rhat_ess_from_nc(os.path.join(NC_DIR, f"gibbs_{key}.nc"))
        store[f"{key}_nuts_rhat"] = nr; store[f"{key}_nuts_ess"] = ne
        store[f"{key}_gibbs_rhat"] = gr; store[f"{key}_gibbs_ess"] = ge
        store[f"{key}_nuts_n"] = nn; store[f"{key}_gibbs_n"] = gn
        print(f"    done in {time.time()-t0:.0f}s | "
              f"sigma_hat median={np.median(sigma_hats):.5f} (true {TRUE_SIGMA:.5f}) | "
              f"NUTS maxR^={nr.max():.3f} Gibbs maxR^={gr.max():.3f} | "
              f"eff/draw min: NUTS { (ne/nn).min():.1e} vs Gibbs {(ge/gn).min():.1e}")

    np.savez(CACHE_NPZ, **store)
    print(f"\nWrote cache {CACHE_NPZ}")


# ----------------------------------------------------------------- plotting
def _bxp_dicts(d, key, method):
    out = []
    for j in range(N_D):
        out.append(dict(
            whislo=float(d[f"{key}_{method}_whislo"][j]),
            q1=float(d[f"{key}_{method}_q1"][j]),
            med=float(d[f"{key}_{method}_med"][j]),
            q3=float(d[f"{key}_{method}_q3"][j]),
            whishi=float(d[f"{key}_{method}_whishi"][j]),
            mean=float(d[f"{key}_{method}_mean"][j]),
            fliers=[], label=""))
    return out


def _recovery_panel(ax, d, key, title, show_ylabel):
    x = np.arange(N_D)
    off, bw = 0.22, 0.38
    # NUTS box (left) + Gibbs box (right) — same box style as Fig 1
    # (orange solid, median = black, neutral whiskers); Gibbs in a cool colour.
    for method, pos, fc in [("nuts", x - off, C_NUTS), ("gibbs", x + off, C_GIBBS)]:
        ax.bxp(_bxp_dicts(d, key, method), positions=pos, widths=bw,
               showfliers=False, showmeans=False, patch_artist=True,
               medianprops=dict(color="black", lw=1.1),
               whiskerprops=dict(color="0.4", lw=0.9),
               capprops=dict(color="0.4", lw=0.9),
               boxprops=dict(facecolor=fc, edgecolor="black",
                             linewidth=0.6, alpha=0.7),
               zorder=3)
    # ground truth: discrete per-bin level marks (NOT a connected line)
    ax.hlines(d[f"{key}_R_true"], x - 0.44, x + 0.44, color=C_TRUTH, lw=2.6,
              zorder=5)
    # tuned-MAP point estimate: green "x" (matches S1 individual-spectra atlas)
    ax.scatter(x, d[f"{key}_map_R"], marker="x", s=80, c=C_MAP, linewidths=2.0,
               zorder=6)
    ax.set_xticks(x); ax.set_xticklabels(D_LABELS)
    ax.set_xlim(-0.65, N_D - 0.35)
    ax.set_xlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
    if show_ylabel:
        ax.set_ylabel("spectral fraction")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot():
    d = dict(np.load(CACHE_NPZ, allow_pickle=True))
    true_sigma = float(d["true_sigma"])
    conv_key = str(d["conv_key"])

    mpl.rcParams.update({
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "axes.labelsize": 20, "axes.titlesize": 17, "legend.fontsize": 17,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(19, 11.5))
    gs = fig.add_gridspec(2, 6, hspace=0.36, wspace=0.70,
                          left=0.085, right=0.985, top=0.88, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    # ---- top row: recovery ----
    _recovery_panel(ax_a, d, "normal", "(a)  normal-like spectrum", True)
    _recovery_panel(ax_b, d, "tumor", "(b)  tumour-like spectrum", False)
    _recovery_panel(ax_c, d, "delta", r"(c)  $\delta$ stress test (concentrated)", False)
    ymax_real = max(d["normal_nuts_whishi"].max(), d["normal_R_true"].max(),
                    d["tumor_nuts_whishi"].max(), d["tumor_R_true"].max())
    ax_a.set_ylim(0, ymax_real * 1.12); ax_b.set_ylim(0, ymax_real * 1.12)
    ax_c.set_ylim(0, 1.06)

    # shared, BOXED top legend (general for all panels); box stats explained
    # in the caption, not here.
    rec_handles = [
        Line2D([0], [0], color=C_TRUTH, lw=2.6, label="ground truth"),
        Patch(facecolor=C_NUTS, edgecolor="black", alpha=0.7,
              label="NUTS posterior"),
        Patch(facecolor=C_GIBBS, edgecolor="black", alpha=0.7,
              label="Gibbs posterior"),
        Line2D([0], [0], color=C_MAP, marker="x", ms=11, lw=0, mew=2.2,
               label=r"tuned MAP ($\lambda=10^{-3}$)"),
    ]
    fig.legend(handles=rec_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=True, fontsize=16)

    # ---- (d) noise recovery ----
    keys = ["normal", "tumor", "delta"]
    labels = ["normal-\nlike", "tumour-\nlike", r"$\delta$ stress"]
    data = [d[f"{k}_sigma_hats"] for k in keys]
    pos = np.arange(len(keys))
    parts = ax_d.violinplot(data, positions=pos, showmeans=False,
                            showextrema=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_NUTS); pc.set_edgecolor("black")
        pc.set_alpha(0.40); pc.set_linewidth(0.7)
    for i, vals in enumerate(data):
        jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.16
        ax_d.scatter(pos[i] + jit, vals, s=22, color=C_NUTS, edgecolor="black",
                     linewidth=0.3, zorder=3, alpha=0.85)
    ax_d.axhline(true_sigma, color=C_TRUTH, ls="--", lw=2.0, zorder=2)
    ax_d.text(len(keys) - 0.5, true_sigma, r"  true $\sigma=1/\mathrm{SNR}$",
              ha="right", va="bottom", fontsize=15, color=C_TRUTH,
              style="italic")
    ax_d.set_xticks(pos); ax_d.set_xticklabels(labels)
    ax_d.set_ylabel(r"inferred noise $\hat{\sigma}$")
    ax_d.set_xlabel("ground-truth spectrum")
    ax_d.set_title("(d)  joint noise recovery (SNR=303)", loc="left",
                   fontweight="bold", pad=8)
    ax_d.grid(True, axis="y", alpha=0.25)

    # ---- (e) sampler mixing on the delta stress case ----
    # ESS per draw (efficiency): run-length-independent, so the 45x-longer
    # Gibbs run is compared fairly to NUTS. Gibbs efficiency collapses.
    nuts_eff = d[f"{conv_key}_nuts_ess"] / float(d[f"{conv_key}_nuts_n"])
    gibbs_eff = d[f"{conv_key}_gibbs_ess"] / float(d[f"{conv_key}_gibbs_n"])
    x = np.arange(N_D); w = 0.4
    ax_e.bar(x - w/2, nuts_eff, w, color=C_NUTS,
             edgecolor="black", linewidth=0.6)
    ax_e.bar(x + w/2, gibbs_eff, w, color=C_GIBBS,
             edgecolor="black", linewidth=0.6)
    ax_e.set_yscale("log")
    ax_e.set_xticks(x); ax_e.set_xticklabels(D_LABELS)
    ax_e.set_xlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
    ax_e.set_ylabel("ESS per draw (efficiency)")
    ax_e.set_title(r"(e)  sampler mixing ($\delta$ stress)", loc="left",
                   fontweight="bold", pad=8)
    ax_e.grid(True, which="both", axis="y", alpha=0.2)
    # No in-panel legend/annotations: NUTS/Gibbs are colour-keyed to the top
    # legend; the max R-hat values go in the caption. Print them for the caption.
    print(f"[caption] max R-hat ({conv_key}):  "
          f"NUTS {d[f'{conv_key}_nuts_rhat'].max():.2f}   "
          f"Gibbs {d[f'{conv_key}_gibbs_rhat'].max():.2f}")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}\nWrote {OUT_PDF}")
    plt.close(fig)


# ----------------------------------------------------------------- plotting (v3: trimmed 2x2)
def _recovery_panel_v3(ax, d, key, title, show_ylabel):
    """Recovery panel for the trimmed figure: NUTS posterior box + tuned-MAP
    point estimate vs ground truth. NO Gibbs."""
    x = np.arange(N_D)
    bw = 0.5
    # NUTS posterior box (orange), centred on each bin — same box style as Fig 1.
    ax.bxp(_bxp_dicts(d, key, "nuts"), positions=x, widths=bw,
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
    ax.scatter(x, d[f"{key}_map_R"], marker="x", s=80, c=C_MAP, linewidths=2.0,
               zorder=6)
    ax.set_xticks(x); ax.set_xticklabels(D_LABELS)
    ax.set_xlim(-0.65, N_D - 0.35)
    ax.set_xlabel(r"diffusivity $D$ ($\mu$m$^2$/ms)")
    if show_ylabel:
        ax.set_ylabel("spectral fraction")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.grid(True, axis="y", alpha=0.25)


def plot_v3():
    """Trimmed 2x2 figure: (a,b,c) recovery (NUTS + tuned-MAP only, no Gibbs)
    and (d) joint noise recovery. Re-plotted entirely from the cache."""
    d = dict(np.load(CACHE_NPZ, allow_pickle=True))
    true_sigma = float(d["true_sigma"])

    mpl.rcParams.update({
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "axes.labelsize": 20, "axes.titlesize": 17, "legend.fontsize": 17,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(14, 13.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30,
                          left=0.095, right=0.985, top=0.90, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- (a,b,c) recovery (NUTS posterior + tuned-MAP vs truth) ----
    _recovery_panel_v3(ax_a, d, "normal", "(a)  normal-like spectrum", True)
    _recovery_panel_v3(ax_b, d, "tumor", "(b)  tumour-like spectrum", True)
    _recovery_panel_v3(ax_c, d, "delta",
                       r"(c)  $\delta$ stress test (concentrated)", True)
    ymax_real = max(d["normal_nuts_whishi"].max(), d["normal_R_true"].max(),
                    d["tumor_nuts_whishi"].max(), d["tumor_R_true"].max())
    ax_a.set_ylim(0, ymax_real * 1.12); ax_b.set_ylim(0, ymax_real * 1.12)
    ax_c.set_ylim(0, 1.06)

    # shared top legend: NUTS=orange, MAP=green, truth=black only.
    rec_handles = [
        Line2D([0], [0], color=C_TRUTH, lw=2.6, label="ground truth"),
        Patch(facecolor=C_NUTS, edgecolor="black", alpha=0.7,
              label="NUTS posterior"),
        Line2D([0], [0], color=C_MAP, marker="x", ms=11, lw=0, mew=2.2,
               label=r"tuned MAP ($\lambda=10^{-3}$)"),
    ]
    fig.legend(handles=rec_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=True, fontsize=17)

    # ---- (d) joint noise recovery ----
    keys = ["normal", "tumor", "delta"]
    labels = ["normal-\nlike", "tumour-\nlike", r"$\delta$ stress"]
    data = [d[f"{k}_sigma_hats"] for k in keys]
    pos = np.arange(len(keys))
    parts = ax_d.violinplot(data, positions=pos, showmeans=False,
                            showextrema=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_NUTS); pc.set_edgecolor("black")
        pc.set_alpha(0.40); pc.set_linewidth(0.7)
    for i, vals in enumerate(data):
        jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.16
        ax_d.scatter(pos[i] + jit, vals, s=22, color=C_NUTS, edgecolor="black",
                     linewidth=0.3, zorder=3, alpha=0.85)
    ax_d.axhline(true_sigma, color=C_TRUTH, ls="--", lw=2.0, zorder=2)
    ax_d.text(len(keys) - 0.5, true_sigma, r"  true $\sigma=1/\mathrm{SNR}$",
              ha="right", va="bottom", fontsize=15, color=C_TRUTH,
              style="italic")
    ax_d.set_xticks(pos); ax_d.set_xticklabels(labels)
    ax_d.set_ylabel(r"inferred noise $\hat{\sigma}$")
    ax_d.set_xlabel("ground-truth spectrum")
    ax_d.set_title("(d)  joint noise recovery (SNR=303)", loc="left",
                   fontweight="bold", pad=8)
    ax_d.grid(True, axis="y", alpha=0.25)

    fig.savefig(OUT_PNG_V3, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF_V3, bbox_inches="tight")
    print(f"Wrote {OUT_PNG_V3}\nWrote {OUT_PDF_V3}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--v3", action="store_true",
                    help="re-plot the trimmed 2x2 figure (fig8_v3) from cache")
    args = ap.parse_args()
    if args.regen:
        generate(quick=args.quick)
    if args.v3:
        plot_v3()
    else:
        plot()
