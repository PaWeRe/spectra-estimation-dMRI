"""Shared paper-figure style for the MRM manuscript.

Single source of truth for fonts, colours, diffusivity-bin labels, and legend
placement so every main + supplementary figure is visually consistent (Stefan
2026-06-03 figure review). Import this in each figure script, call
``apply_style(<preset>)`` once at the top, and use the ``COLORS`` /
``DIFFUSIVITIES`` / ``DLABELS`` constants plus the small helpers below.

Conventions (LOCKED — do not deviate per-figure):
  * Legend font size == panel-title font size (Stefan, repeated).
  * Legend lives on TOP (a figure-level legend), never an in-figure suptitle
    (the caption is the title -- MRM convention).
  * NO angled tick labels.
  * PZ always LEFT (or top), TZ always RIGHT (or bottom). Fig 1 sets this.
  * Diffusivity bins + tick labels are identical across every figure.

Identifiability / posterior-CV colouring is NOT here: it lives in
``visualization.identifiability`` (purple sequential + hatch). Import that
module for any CV-coded bars/boxes so Fig 4 and the S1 atlas stay identical.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

# --------------------------------------------------------------------------- #
# Diffusivity grid (identical across all figures; consistent with Fig 1).
# --------------------------------------------------------------------------- #
DIFFUSIVITIES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 20.0])
# Canonical tick labels == Fig 1's rendering (the reference figure). 2-decimal
# for the resolved bins, "20.0" for the dump bin. Use these verbatim everywhere.
DLABELS = ["0.25", "0.50", "0.75", "1.00", "1.50", "2.00", "3.00", "20.0"]
DIFF_UNIT = r"$\mu$m$^2$/ms"
DIFF_AXIS_LABEL = rf"diffusivity $D$ ({DIFF_UNIT})"

# --------------------------------------------------------------------------- #
# Colour palette (LOCKED; matches the figures.tex header convention).
# --------------------------------------------------------------------------- #
COLORS = {
    # tissue
    "normal": "#1f77b4",        # blue
    "tumor": "#d62728",         # red
    "normal_dark": "#0b3d61",   # misclassified-normal (Fig 6)
    "tumor_dark": "#7f1416",    # misclassified-tumor  (Fig 6)
    # estimators / reference
    "nuts": "#ff7f0e",          # orange
    "map": "#2ca02c",           # green
    "truth": "#1a1a1a",         # near-black ground truth / reference
    "adc": "#000000",           # black ADC reference
    "gibbs": "#4c72b0",         # slate blue (SI only)
    "crlb": "#8c8c8c",          # grey unconstrained CRLB bar
    "crlb_bayes": "#3a3a3a",    # charcoal van-Trees / Bayesian CRLB bar
    # single-component highlights (Fig 2 ROC)
    "restricted": "#17becf",    # teal,  D = 0.25
    "freewater": "#8c564b",     # brown, D = 3.0
    "muted": "#9e9e9e",         # faint grey bundle (other single bins)
    # misc
    "sensitivity": "#3a3a3a",   # charcoal ADC-sensitivity diamonds (Fig 4)
}

# New, previously-unused hues for figures that must NOT reuse the reserved
# colours above (Stefan): directional-spectra lines (Fig 7/SI directional) and
# pixel-wise score / uncertainty maps (Fig 9). Red/blue are reserved for
# restricted/free-water fraction maps; purple->yellow is the uncertainty ramp
# (avoids white, which clashes with the anatomical background).
DIRECTION_COLORS = ["#9467bd", "#e377c2", "#bcbd22"]  # 3 encoding directions
SCORE_CMAP = "PuOr"          # diverging, decision-boundary-centred score maps
UNCERTAINTY_CMAP = "viridis"  # purple->yellow, high-contrast, avoids white

# --------------------------------------------------------------------------- #
# Font presets. Legend == title in BOTH presets (Stefan).
# --------------------------------------------------------------------------- #
FONTS_GRID = {  # 2xN multi-panel grids (Figs 1, 3, 4, 6, 8, ...)
    "axes.labelsize": 20,
    "axes.titlesize": 17,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
}
FONTS_SINGLE = {  # single / wide panels (Fig 5)
    "axes.labelsize": 19,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
}

_BASE = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "pdf.fonttype": 42,  # editable text in the PDF
    "ps.fonttype": 42,
}


def apply_style(preset: str = "grid") -> None:
    """Set global rcParams. ``preset`` in {"grid", "single"}.

    Call once at the top of a figure script. Replaces any per-script font
    rcParams so sizing is uniform across the manuscript.
    """
    fonts = FONTS_SINGLE if preset == "single" else FONTS_GRID
    mpl.rcParams.update({**_BASE, **fonts})


def set_diff_xaxis(ax, label: bool = True, rotation: int = 0) -> None:
    """Standard diffusivity x-axis: integer tick positions, fixed labels, no
    rotation (no angled text)."""
    ax.set_xticks(range(len(DIFFUSIVITIES)))
    ax.set_xticklabels(DLABELS, rotation=rotation)
    if label:
        ax.set_xlabel(DIFF_AXIS_LABEL)


def top_legend(fig, handles, labels, ncol: int | None = None, y: float = 0.99):
    """Figure-level legend on top, centred. ncol defaults to len(handles)."""
    return fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=ncol or len(handles), frameon=True, framealpha=0.95,
    )
