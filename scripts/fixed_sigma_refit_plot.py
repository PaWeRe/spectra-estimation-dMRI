"""
Visualize fixed-σ refit results: per-ROI posterior spectra for the three
σ modes (free, fixed_formula, fixed_residual).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/biomarkers"
df = pd.read_csv(os.path.join(OUTPUT_DIR, "fixed_sigma_refit.csv"))

rois = df["roi_id"].unique()
fig, axes = plt.subplots(1, len(rois), figsize=(3.4 * len(rois), 3.2), sharey=True)
if len(rois) == 1:
    axes = [axes]

mode_styles = {
    "free":           {"color": "#1f77b4", "label": "σ free (HalfCauchy)"},
    "fixed_formula":  {"color": "#2ca02c", "label": "σ = formula"},
    "fixed_residual": {"color": "#d62728", "label": "σ = MAP-residual"},
}
mode_offsets = {"free": -0.18, "fixed_formula": 0.0, "fixed_residual": 0.18}

for ax, roi in zip(axes, rois):
    sub = df[df["roi_id"] == roi]
    x_base = np.arange(8)
    for mode, style in mode_styles.items():
        m = sub[sub["mode"] == mode].sort_values("bin")
        x = x_base + mode_offsets[mode]
        ax.errorbar(x, m["R_mean"].values,
                    yerr=[m["R_mean"].values - m["R_q05"].values,
                          m["R_q95"].values - m["R_mean"].values],
                    fmt="o", color=style["color"], capsize=2, ms=4,
                    label=style["label"])
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{d:.2g}" for d in sub.sort_values("bin")["D"].unique()],
                       rotation=45, fontsize=8)
    ax.set_xlabel("D (μm²/ms)")
    ax.set_title(roi, fontsize=9)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Posterior R (90% CI)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Spectra under three σ assumptions — 5 representative ROIs", y=1.02)
fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "fixed_sigma_spectra.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved {out}")
