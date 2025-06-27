import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os


# TODO: get all the az.function() into the already defined data structures themselves if possible, so that we can simplify everything and maybe don't even need this diagnostics.py
def plot_trace(idata, var_name="R", save_path=None, wandb_run=None, **kwargs):
    fig = az.plot_trace(idata, var_names=[var_name], **kwargs)
    if save_path:
        plt.savefig(save_path)
        if wandb_run is not None:
            import wandb

            wandb_run.log({os.path.basename(save_path): wandb.Image(save_path)})
    plt.close()
    return save_path


def plot_autocorr(idata, var_name="R", save_path=None, wandb_run=None, **kwargs):
    fig = az.plot_autocorr(idata, var_names=[var_name], **kwargs)
    if save_path:
        plt.savefig(save_path)
        if wandb_run is not None:
            import wandb

            wandb_run.log({os.path.basename(save_path): wandb.Image(save_path)})
    plt.close()
    return save_path


def plot_posterior(idata, var_name="R", save_path=None, wandb_run=None, **kwargs):
    fig = az.plot_posterior(idata, var_names=[var_name], **kwargs)
    if save_path:
        plt.savefig(save_path)
        if wandb_run is not None:
            import wandb

            wandb_run.log({os.path.basename(save_path): wandb.Image(save_path)})
    plt.close()
    return save_path


def plot_boxplot_comparison(
    true_spectrum,
    map_estimate,
    gibbs_samples,
    diffusivities,
    title=None,
    save_path=None,
    wandb_run=None,
):
    """
    Compare true spectrum, MAP estimate, and Gibbs posterior mean in a boxplot.
    true_spectrum: 1D array
    map_estimate: 1D array
    gibbs_samples: 2D array (samples x components)
    diffusivities: 1D array
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(diffusivities)
    positions = np.arange(n)
    # Boxplot for Gibbs samples
    ax.boxplot(
        gibbs_samples,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="blue"),
    )
    # MAP estimate
    ax.plot(positions, map_estimate, "ro-", label="MAP/NNLS", linewidth=2)
    # True spectrum
    ax.vlines(
        positions,
        0,
        true_spectrum,
        colors="green",
        alpha=0.8,
        linewidth=2,
        label="True Spectrum",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{d:.2f}" for d in diffusivities], rotation=45)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Diffusivity")
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        if wandb_run is not None:
            import wandb

            wandb_run.log({os.path.basename(save_path): wandb.Image(save_path)})
    plt.close()
    return save_path


def run_all_diagnostics(
    idata_gibbs, idata_map, true_spectrum, diffusivities, title=None, wandb_run=None
):
    """
    Run all diagnostics and comparison plots for a given experiment.
    idata_gibbs: ArviZ InferenceData from Gibbs
    idata_map: ArviZ InferenceData from MAP/NNLS
    true_spectrum: 1D array
    diffusivities: 1D array
    """
    # Use Hydra's run dir
    run_dir = os.getcwd()
    trace_path = os.path.join(run_dir, "trace.png")
    autocorr_path = os.path.join(run_dir, "autocorr.png")
    posterior_path = os.path.join(run_dir, "posterior.png")
    boxplot_path = os.path.join(run_dir, "boxplot_comparison.png")
    plot_trace(idata_gibbs, var_name="R", save_path=trace_path, wandb_run=wandb_run)
    plot_autocorr(
        idata_gibbs, var_name="R", save_path=autocorr_path, wandb_run=wandb_run
    )
    plot_posterior(
        idata_gibbs, var_name="R", save_path=posterior_path, wandb_run=wandb_run
    )
    # Extract samples
    gibbs_samples = idata_gibbs.posterior["R"].values.reshape(-1, len(diffusivities))
    map_estimate = idata_map.posterior["R"].values.reshape(-1, len(diffusivities))[0]
    plot_boxplot_comparison(
        true_spectrum,
        map_estimate,
        gibbs_samples,
        diffusivities,
        title=title,
        save_path=boxplot_path,
        wandb_run=wandb_run,
    )
    return {
        "trace": trace_path,
        "autocorr": autocorr_path,
        "posterior": posterior_path,
        "boxplot": boxplot_path,
    }
