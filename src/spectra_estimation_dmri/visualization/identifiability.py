"""Shared per-bin identifiability (posterior CV) colour + hatch scheme.

Used by BOTH Fig. 4 (LR-coefficient bars) and the S1 atlas (box faces) so the
identifiability encoding is identical across the manuscript. The four CV bands
use a single-hue purple sequential ramp (colourblind-safe, avoids the
tumor=red / normal=blue / NUTS=orange / MAP=green conventions) PLUS redundant
hatching, so the bands stay distinguishable in grayscale and for colour-vision
deficiency. Hatch "busyness" increases with CV: a clean light bin is well
identified; a dark cross-hatched bin is poorly identified.

CV = posterior std / posterior mean (within-ROI, per bin).
"""

from __future__ import annotations

import numpy as np
import matplotlib.patches as mpatches

# (upper bound, fill colour, hatch) -- light/clean -> dark/busy.
CV_BANDS = [
    (0.4, "#f2f0f7", ""),       # well identified
    (0.6, "#cbc9e2", ".."),
    (0.8, "#9e9ac8", "//"),
    (np.inf, "#6a51a3", "xx"),  # poorly identified
]
CV_LABELS = ["CV < 0.4", "0.4–0.6", "0.6–0.8", "CV > 0.8"]
CV_NAN_COLOR = "#d9d9d9"


def cv_color(cv: float) -> str:
    if not np.isfinite(cv):
        return CV_NAN_COLOR
    for hi, c, _ in CV_BANDS:
        if cv < hi:
            return c
    return CV_BANDS[-1][1]


def cv_hatch(cv: float) -> str:
    if not np.isfinite(cv):
        return ""
    for hi, _, h in CV_BANDS:
        if cv < hi:
            return h
    return CV_BANDS[-1][2]


def cv_legend_handles(hatch: bool = True):
    """Four band patches for a figure legend.

    ``hatch=True`` (default) keeps the colour+hatch swatches used by the S1
    atlas; ``hatch=False`` returns colour-only swatches (Fig. 6 / sensitivity,
    per Stephan 2026-06-12 — colour encodes CV, no hatching)."""
    return [
        mpatches.Patch(facecolor=c, edgecolor="black",
                       hatch=(h if hatch else None), label=lab)
        for (_, c, h), lab in zip(CV_BANDS, CV_LABELS)
    ]
