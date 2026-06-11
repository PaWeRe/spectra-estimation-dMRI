"""Fig 3 (production) — simulation recovery battery, bar-chart redesign.

Design approved by Patrick 2026-06-10 (prototype fig3_proto). Each (ground-truth,
SNR) panel shows 8 bins x 3 bars: ground truth (near-black reference),
tuned-MAP, NUTS posterior mean; error bars = +/-1 SD across noise realizations.
A bottom row shows noise-sigma recovery (true sigma vs MAP residual sigma-hat vs
NUTS-inferred sigma-hat) per SNR.

HONESTY / FAITHFULNESS to the real BWH analysis:
  * Each fit uses the EXACT nuts.yaml config (2000 draws, 200 tune, 4 chains,
    target_accept 0.95) and the corrected constrained-ridge MAP -- identical to
    the deployed pipeline, so recovery is not understated by under-sampling.
  * max R-hat is recorded per fit; fits with R-hat > 1.01 are flagged.
  * SNR 300 = cohort-median SNR (the real operating point); 50 / 1000 bracket it
    (pixel-wise low end / ideal).
  * Realistic GTs are the cohort-mean NUTS normal / tumour spectra (real shapes).

Resumable: per-rep posterior-mean spectra + sigma-hats + R-hat accumulate in a
compact npz cache (saved after EVERY rep -> crash/resume safe). One representative
.nc per (GT,SNR) is kept for inspection; the per-rep summaries in the cache are
the reproducible record of the figure.

Main figure  -> paper/figures/fig3_v7.{png,pdf}  : GTs normal, tumor, uniform, delta
SI figure    -> paper/figures/figS_recovery.{png,pdf} : full 6-GT battery

Usage:
  uv run python scripts/fig3_recovery_battery.py --quick        # fast smoke test
  uv run python scripts/fig3_recovery_battery.py --reps 100     # full resumable run
  uv run python scripts/fig3_recovery_battery.py --replot       # figures from cache
"""
import os, io, sys, time, argparse, contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, "src")
from spectra_estimation_dmri.visualization.paper_style import (
    apply_style, COLORS, DIFFUSIVITIES, DLABELS,
)

# ----------------------------------------------------------------- constants
B_VALUES_MS = np.array([0., .25, .5, .75, 1., 1.25, 1.5, 1.75,
                        2., 2.25, 2.5, 2.75, 3., 3.25, 3.5])
D = np.asarray(DIFFUSIVITIES); N_D = len(D)
SNRS = [50, 300, 1000]
PRIOR_STRENGTH = 0.1
FEATURES_CSV = "results/biomarkers/features.csv"
CACHE = "results/simulation/fig3_recovery_battery.npz"
NC_DIR = "results/simulation/fig3_battery_nc"

# Ground truths: slug -> display label. Main = first 4; SI = all 6.
GT_MAIN = ["normal", "tumor", "uniform", "delta"]
GT_ALL = GT_MAIN + ["inverse", "bimodal"]
GT_LABELS = {
    "normal": "Normal-like", "tumor": "Tumor-like", "uniform": "Uniform",
    "delta": r"$\delta$ ($D$=0.75)", "inverse": "Inverse tumor",
    "bimodal": "Bimodal",
}
C_TRUTH, C_MAP, C_NUTS = "#a6a6a6", COLORS["map"], COLORS["nuts"]  # truth = light grey reference
PANEL_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

U = np.exp(-np.outer(B_VALUES_MS, D))
def norm(R):
    s = R.sum(); return R / s if s > 0 else R


def ground_truths():
    df = pd.read_csv(FEATURES_CSV)
    cols = [f"nuts_D_{d:.2f}" for d in D]
    tumor = norm(df[df.is_tumor == 1][cols].mean().values.astype(float))
    normal = norm(df[df.is_tumor == 0][cols].mean().values.astype(float))
    uniform = norm(np.ones(N_D))
    delta = np.zeros(N_D); delta[int(np.argmin(np.abs(D - 0.75)))] = 1.0
    inverse = norm(tumor[::-1].copy())                 # reversed tumour (robustness menu)
    bimodal = np.zeros(N_D); bimodal[0] = 0.4; bimodal[6] = 0.4; bimodal[1:6] = 0.04
    bimodal = norm(bimodal)
    return {"normal": normal, "tumor": tumor, "uniform": uniform,
            "delta": delta, "inverse": inverse, "bimodal": bimodal}


# ----------------------------------------------------------------- inference
def nuts_cfg(snr, draws, tune, chains):
    return OmegaConf.create({
        "seed": 42,
        "dataset": {"diff_values": D.tolist(), "snr": snr, "spectrum_pair": None},
        "inference": {"name": "nuts", "n_iter": draws, "tune": tune,
                      "n_chains": chains, "target_accept": 0.95,
                      "sampler_snr": None, "init": "map"}})


def rhat_max_from_nc(path):
    import arviz as az
    idata = az.from_netcdf(path)
    vs = [f"diff_{d:.2f}" for d in D]
    r = az.rhat(idata, var_names=vs)
    return float(np.max([float(r[v].values) for v in vs]))


def quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def load_cache():
    return dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}


def save_cache(store):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **store)


def n_done(store, slug, snr):
    k = f"{slug}__{snr}__nuts"
    return len(store[k]) if k in store else 0


def run(reps, draws, tune, chains):
    from spectra_estimation_dmri.models.prob_model import ProbabilisticModel
    from spectra_estimation_dmri.data.data_models import SignalDecay
    from spectra_estimation_dmri.inference.nuts import NUTSSampler
    from spectra_estimation_dmri.biomarkers.recompute import compute_map_spectrum

    os.makedirs(NC_DIR, exist_ok=True)
    gts = ground_truths()
    model = ProbabilisticModel(
        data_snr=None,
        prior_config=OmegaConf.create({"type": "ridge", "strength": PRIOR_STRENGTH}),
        b_values=B_VALUES_MS.tolist(), diffusivities=D.tolist())
    store = load_cache()
    for slug in GT_ALL:                              # persist true spectra (idempotent)
        for snr in SNRS:
            store[f"{slug}__{snr}__Rtrue"] = norm(gts[slug].astype(float))
    save_cache(store)

    def _app(k, v):
        cur = list(store[k]) if k in store else []
        cur.append(v); store[k] = np.array(cur)

    rhat_warn = 0
    # ROUND-ROBIN over reps: every condition advances together, so an interim
    # --replot is balanced and an interruption leaves a usable, even figure.
    for r in range(reps):
        for slug in GT_ALL:
            Rt = norm(gts[slug].astype(float))
            for snr in SNRS:
                key = f"{slug}__{snr}"
                if n_done(store, slug, snr) > r:
                    continue                          # this rep already cached
                t0 = time.time()
                rng = np.random.default_rng(7_000_000 + abs(hash(slug)) % 9999
                                            + snr * 1000 + r)
                y = U @ Rt + rng.normal(0, 1.0 / snr, size=U.shape[0])
                sd = SignalDecay(patient="sim", signal_values=y.tolist(),
                                 b_values=B_VALUES_MS.tolist(), a_region="sim", snr=snr)
                uh = f"{key}_rep0" if r == 0 else f"{key}_scratch"  # keep rep0 .nc only
                res = quiet(lambda: NUTSSampler(model, sd, nuts_cfg(snr, draws, tune, chains))
                            .run(show_progress=False, save_dir=NC_DIR, unique_hash=uh))
                ns = np.array(res.spectrum_samples)
                ns = ns / ns.sum(axis=1, keepdims=True)
                mR = compute_map_spectrum(y, U)
                S0 = y[0] if y[0] > 0 else 1.0
                rmax = rhat_max_from_nc(os.path.join(NC_DIR, f"{uh}.nc"))
                rhat_warn += int(rmax > 1.01)
                _app(f"{key}__nuts", ns.mean(axis=0))
                _app(f"{key}__map", norm(mR))
                _app(f"{key}__sig_nuts", 1.0 / res.sampler_snr)
                _app(f"{key}__sig_map", float(np.sqrt(np.mean((U @ mR - y / S0) ** 2))))
                _app(f"{key}__rhat", rmax)
                save_cache(store)                     # resume-safe: save every rep
                print(f"  round {r+1}/{reps} {key} | {time.time()-t0:.0f}s | "
                      f"max R-hat={rmax:.3f}")
    print(f"\nDONE. fits with R-hat>1.01: {rhat_warn}")


# ----------------------------------------------------------------- plotting
def build_figure(store, gt_slugs, out_stem):
    n_gt = len(gt_slugs)
    n_reps = min(len(store[f"{s}__{snr}__nuts"]) for s in gt_slugs for snr in SNRS)
    # n_gt recovery rows + a thin spacer + a noise-sigma row. The spacer gives the
    # sigma row extra separation while the recovery rows stay tight.
    fig = plt.figure(figsize=(18.5, 3.0 * n_gt + 3.6))
    gs = fig.add_gridspec(n_gt + 2, 3, height_ratios=[1] * n_gt + [0.30, 0.95],
                          hspace=0.28, wspace=0.22,
                          left=0.07, right=0.975, top=0.955, bottom=0.045)
    x = np.arange(N_D); w = 0.27
    for ci, snr in enumerate(SNRS):
        for ri, slug in enumerate(gt_slugs):
            ax = fig.add_subplot(gs[ri, ci])
            Rt = store[f"{slug}__{snr}__Rtrue"]
            mp = store[f"{slug}__{snr}__map"]; nt = store[f"{slug}__{snr}__nuts"]
            rh = store.get(f"{slug}__{snr}__rhat", np.array([np.nan]))
            ax.bar(x - w, Rt, w, color=C_TRUTH, label="Ground truth")
            ax.bar(x, mp.mean(0), w, yerr=mp.std(0), color=C_MAP, alpha=0.85,
                   ecolor="0.3", capsize=2, error_kw=dict(lw=0.8), label="Tuned MAP")
            ax.bar(x + w, nt.mean(0), w, yerr=nt.std(0), color=C_NUTS, alpha=0.9,
                   ecolor="0.3", capsize=2, error_kw=dict(lw=0.8),
                   label="NUTS (posterior mean)")
            ax.set_xticks(x)
            if ri == n_gt - 1:
                ax.set_xticklabels(DLABELS, fontsize=14)   # 8 values: keep compact
            else:
                ax.set_xticklabels([])
            ax.set_ylim(0, max(0.5, float(Rt.max()) * 1.22))
            if ci == 0:
                ax.set_ylabel("Spectral fraction $R_j$")
            else:
                ax.tick_params(labelleft=False)          # within-row shared y-scale
            if ri == n_gt - 1:
                ax.set_xlabel(r"Diffusivity $D$ ($\mu$m$^2$/ms)")
            ax.set_title(f"({PANEL_LETTERS[ri]}) {GT_LABELS[slug]}  |  SNR {snr}",
                         fontsize=18)
            ax.text(0.03, 0.95, rf"$\hat{{R}}_{{\max}}\,{np.nanmax(rh):.2f}$",
                    transform=ax.transAxes, fontsize=10, color="0.45",
                    ha="left", va="top")
            ax.grid(axis="y", alpha=0.25, lw=0.5)
        # noise-sigma row (x-axis = ground-truth letters A-F)
        axs = fig.add_subplot(gs[n_gt + 1, ci])
        xs = np.arange(n_gt)
        st = 1.0 / snr
        axs.axhline(st, color="0.35", ls="--", lw=1.8, label=r"true $\sigma=1/$SNR")
        axs.errorbar(xs - 0.10, [store[f"{s}__{snr}__sig_map"].mean() for s in gt_slugs],
                     yerr=[store[f"{s}__{snr}__sig_map"].std() for s in gt_slugs],
                     fmt="s", color=C_MAP, capsize=3, ms=7, label=r"MAP residual $\hat\sigma$")
        axs.errorbar(xs + 0.10, [store[f"{s}__{snr}__sig_nuts"].mean() for s in gt_slugs],
                     yerr=[store[f"{s}__{snr}__sig_nuts"].std() for s in gt_slugs],
                     fmt="o", color=C_NUTS, capsize=3, ms=7, label=r"NUTS $\hat\sigma$")
        axs.set_xticks(xs); axs.set_xticklabels(PANEL_LETTERS[:n_gt], fontsize=18)
        axs.set_xlim(-0.5, n_gt - 0.5); axs.set_ylim(0, st * 1.9)
        axs.set_xlabel("Ground truth (A--F)", fontsize=20)
        if ci == 0:
            axs.set_ylabel(r"Noise $\sigma$")
        axs.set_title(rf"Noise $\sigma$ Recovery  |  SNR {snr}", fontsize=18)
        axs.grid(axis="y", alpha=0.25, lw=0.5)
        if ci == 2:
            axs.legend(loc="upper right", fontsize=12, framealpha=0.95)

    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 0.999), frameon=True,
               framealpha=0.95, fontsize=18)
    fig.text(0.975, 0.999, f"n = {n_reps} realizations / condition",
             ha="right", va="top", fontsize=14, color="0.3")
    fig.savefig(f"paper/figures/{out_stem}.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"paper/figures/{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved paper/figures/{out_stem}.png/.pdf  (n_reps={n_reps})")


def make_figures(quick=False):
    apply_style("grid")
    store = load_cache()
    # report rep counts + convergence
    reps = [n_done(store, s, snr) for s in GT_ALL for snr in SNRS]
    rmax = max((store[f"{s}__{snr}__rhat"].max()
                for s in GT_ALL for snr in SNRS
                if f"{s}__{snr}__rhat" in store), default=float("nan"))
    print(f"rep counts: min={min(reps)} max={max(reps)} | global max R-hat={rmax:.3f}")
    suffix = "_quick" if quick else ""
    # Single main figure = the full 6-ground-truth battery (Patrick 2026-06-11:
    # no separate 4-GT main + SI; the battery IS Fig 3, full page).
    build_figure(store, GT_ALL, f"fig3_v7{suffix}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 reps, reduced draws")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    if args.quick:
        # Separate cache + .nc dir so reduced-sampling smoke-test reps NEVER
        # contaminate the honest full-config production cache.
        CACHE = CACHE.replace(".npz", "_quick.npz")
        NC_DIR = NC_DIR + "_quick"
    if args.replot:
        make_figures(quick=args.quick)
    elif args.quick:
        run(reps=2, draws=200, tune=150, chains=2)
        make_figures(quick=True)
    else:
        run(reps=args.reps, draws=2000, tune=200, chains=4)
        make_figures(quick=False)
