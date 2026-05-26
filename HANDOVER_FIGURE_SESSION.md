# Handover — Figure Regeneration Session

> **2026-05-25 update — the figure *list* below is partially superseded.** See
> `PROJECT_STATE.md` §6 for the current figure plan. Several figures are slated
> for cuts (Fig 5, Fig 8 → SI), heavy revision (Fig 4 — given F4b that the
> ADC-sensitivity ≈ inverted-LR-coef r ≈ −0.98 was a regulariser artifact), or
> new additions (F-new-1 MAP λ-sweep panel, F-new-2 axis separation). The
> figure-regen *sub-tasks* in this doc (font bump, color conventions, marker
> sizes, NUTS pixel overlay, anonymisation) remain accurate and useful — treat
> this file as the checklist for when figure regen actually begins, not as the
> figure list itself.

**Status when this hand-off was written:** May 2026, manuscript text is locked-in
after addressing all of Stephan's `(@comment stephan: ...)` markers from commit
`a7aa22d`. The Stephan-comments commit + Kuczera-style code-availability
sentence are at `6ed81cf`, pushed to `origin/main`. Patrick has also added
`applied_math_approaches.txt` and `notes_manuscript_2nd_pass.txt` in commit
`a4104b5` covering separate "second pass" content (mostly Methods-section
expansion on Gibbs/VI/NUTS history) — **out of scope for the figure session**;
revisit once figures are done.

## Goal of this session

Execute the 9 `% TODO(figure-regen)` items left in `paper/sections/figures.tex`
and `paper/sections/supporting.tex`. Patrick will sit with the agent and
iterate live on visual choices — fonts, colors, legend placement, panel
layouts.

## House rules

- All numbers in the manuscript already match `results/biomarkers/*.csv`
  (verified during the previous text pass). When regenerating figures, the
  underlying data must remain unchanged; only visual encoding changes.
- Color convention used throughout the paper:
  - `blue` = normal tissue, `red` = tumor
  - `orange` = NUTS, `green` = MAP, `gray`/`black` = CRLB or ground truth
  - For figures where MAP/NUTS bars sit next to LR-coefficient bars (Fig 4),
    pick a non-blue/non-red palette so it doesn't collide with the
    tumor/normal convention.
- MRM caption style: lowercase summary sentence at the start, **no titles, no
  bold lead-in**. Already enforced in the LaTeX captions; figure scripts
  must not bake titles into the image itself either.
- Use `uv run python ...`, never `pip`.

## Data sources (do NOT regenerate; all precomputed)

- `src/spectra_estimation_dmri/data/bwh/signal_decays.json` — 56 patients ×
  2–4 ROIs each, **trace-averaged** (no per-direction). 149 ROIs total.
- `src/spectra_estimation_dmri/data/bwh/metadata.csv` — GGG labels
- `results/inference_bwh_backup/*.nc` — 149 NUTS posteriors (one per ROI)
- `results/biomarkers/features.csv` — MAP + NUTS fractions, ADC, metadata
  for all 149 ROIs (single source of truth for paper numbers)
- `results/biomarkers/{auc_table,adc_discriminant,adc_sensitivity,
  identifiability,map_nuts_comparison}.csv` — derived tables
- `results/pixelwise/nuts_results.npz` — **pixel-wise NUTS posterior**
  already done (146 voxels, posterior mean+std, σ, SNR, R̂, coords, mask,
  diffusivities). Use this for Fig 9 NUTS panels — no re-running needed.
- `results/pixelwise_all_fast.npz` — pixel-wise MAP (used by current Fig 9)
- `8640-sl6-bin/000*.bin` — per-direction binary signal data for the
  pixelwise patient (only place per-direction data lives)

## Plot script → figure mapping

| Figure | Source script | Output file |
|---|---|---|
| Fig 1 spectra | `scripts/generate_paper_figures.py` | `paper/figures/fig_spectra_combined.pdf` |
| Fig 2 ROC | `scripts/generate_paper_figures.py` | `paper/figures/fig_roc.pdf` |
| Fig 3 ADC vs discriminant | `scripts/generate_paper_figures.py` | `paper/figures/fig_adc_discriminant.pdf` |
| Fig 4 sensitivity | `scripts/generate_paper_figures.py` | `paper/figures/fig_sensitivity.pdf` |
| Fig 5 MAP vs NUTS | `scripts/generate_paper_figures.py` | `paper/figures/fig_map_nuts.pdf` |
| Fig 6 uncertainty | `scripts/generate_paper_figures.py` | `paper/figures/fig_uncertainty.pdf` |
| Fig 7 directions | `scripts/direction_comparison.py` (and `generate_paper_figures.py`) | `paper/figures/fig_directions.png` |
| Fig 8 Fisher | `scripts/generate_fisher_figure.py` | `paper/figures/fig_fisher.pdf` |
| Fig 9 pixelwise | `scripts/generate_paper_figures.py` | `paper/figures/fig_pixelwise_v2.pdf` |
| Fig S1 NUTS trace | `scripts/generate_paper_figures.py` | `paper/figures/fig_trace_simulation.pdf` |
| Fig S2 robustness | `scripts/generate_paper_figures.py` | `paper/figures/fig_robustness.png` |

To find each figure's plotting function, grep for the output filename inside
the script.

## TODO items (all 9, ordered by suggested attack)

### 1. Global font enlargement (~+50%)

Stephan flagged this on every figure. The cleanest solution is one
`matplotlib` rcParams block at the top of `scripts/generate_paper_figures.py`
and `scripts/generate_fisher_figure.py` — e.g.:

```python
plt.rcParams.update({
    "font.size": 14,            # was likely 9-10
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
```

Verify per-figure that nothing overflows after the bump. Tighten layout
with `fig.tight_layout()` or `bbox_inches='tight'` on save.

### 2. Fig 4 — color palette + sensitivity sign convention

- Currently: `dark green` (sensitivity, $-\partial\text{ADC}/\partial R$) +
  `purple` (LR coefficients).
- Stephan: blue/red are reserved for normal/tumor; pick a non-conflicting
  pair. Suggestion (matches conventions elsewhere): `green` for ADC
  sensitivity (matches MAP convention since both are "classical" estimates)
  and `orange` for LR coefficients (matches NUTS convention since LR is the
  "Bayesian-like" optimal classifier). Or pick a neutral teal/charcoal pair
  if that overloads MAP/NUTS associations.
- Sign convention is now explicit in the Results text: paper reports raw
  $\partial\text{ADC}/\partial R_{0.25} = -1.59$, figure plots
  $-\partial\text{ADC}/\partial R$ to align visually with LR coefficients.
  **Action:** verify the figure axis label still says
  $-\partial\text{ADC}/\partial R_j$ after regen so the caption stays
  truthful.

### 3. Fig 6 — enlarge crosses, reduce saturation

- Current crosses for misclassified samples are too small / over-saturated.
- Increase `markersize` for the cross marker (e.g., 12 → 18) and dim its
  alpha or use a desaturated red.

### 4. Fig 7 — direction comparison rework (⚠ awaits Stephan's data)

This is the figure Stephan struggled with. He wants a whole-ROI direction-
wise comparison (one normal + one tumor), not arbitrary voxel selections.

**Patrick's task BEFORE this session:** find the per-direction signal decays
Stephan sent (likely in an old email or shared folder). The trace-level
`signal_decays.json` has trace-averaged decays only. Per-direction data
exists only in the pixelwise patient's `8640-sl6-bin/*.bin` binaries.

**Two options for the session:**
- **(a)** If Stephan's per-direction decays surface: regenerate with one
  normal ROI + one tumor ROI from his data, drop the per-voxel panel.
- **(b)** If not: build a per-ROI panel by aggregating per-direction signals
  across a normal-tissue cluster and a tumor cluster within
  `8640-sl6-bin/`. Keep the existing voxel panel as a smaller secondary
  panel, or push to supplementary.
- Either way: anonymize ("supplementary patient" not "patient 8640"), drop
  voxel counts, drop "randomly selected" framing.

### 5. Fig 8(b) — legend placement

Currently the legend on the Fisher figure's panel (b) overlaps the bars
and the improvement-factor labels. Move the legend to a free area in the
middle of the panel (`loc='center'` or explicit `bbox_to_anchor`).

### 6. Fig 9 — add NUTS pixel maps + feature-importance recolor

- Current: MAP-only spectral fraction maps + ADC + LR coefficients +
  discriminant + uncertainty.
- Stephan: also show NUTS pixel maps. Data is precomputed in
  `results/pixelwise/nuts_results.npz` (key `spectrum_mean` shape (146,8),
  `spectrum_std` shape (146,8), `coords` shape (146,2), `mask` shape
  (64,64)).
- Suggested layout: 2 rows of 4 panels — top row MAP, bottom row NUTS,
  shared with ADC + discriminant + uncertainty as a third row. Or stack
  MAP/NUTS for D=0.25 and D=3.0 side by side.
- Feature-importance vector (panel D) currently uses red→blue diverging.
  Pick a colormap that doesn't overload the tumor/normal palette
  (e.g., `coolwarm` is the conflict — try `RdBu_r` flipped, `PiYG`, or
  `BrBG`).
- Anonymize: no patient ID anywhere on the figure.

### 7. Fig S2 robustness — recolor + optional MAP

- Recolor: NUTS posterior in `orange` (matching Fig 1), ground truth in
  `black`. Currently NUTS is `red` and truth is `blue`.
- Optionally add MAP recovery as a `green` overlay for direct comparison.
  Patrick to decide whether the visual gets too cluttered.

## Verification after each regen

After regenerating a figure:
1. Open the PDF/PNG, check fonts didn't overflow and colors render correctly.
2. Re-run the cross-check script in `Bash` to make sure no number drifted
   (this is unlikely since data didn't change, but worth a sanity check):
   ```bash
   uv run python -c "
   import pandas as pd
   df = pd.read_csv('results/biomarkers/features.csv')
   for d in [0.25, 3.00]:
       for zone in ['pz','tz']:
           for tum in [True, False]:
               sub = df[(df.zone==zone) & (df.is_tumor==tum)]
               print(f'{zone} {\"tumor\" if tum else \"normal\"} D={d}: MAP={sub[f\"map_D_{d:.2f}\"].mean()*100:.1f}%, NUTS={sub[f\"nuts_D_{d:.2f}\"].mean()*100:.1f}%')
   "
   ```
3. After all figures look good in a single coherent batch, commit with
   message `Regenerate figures per Stephan's review feedback (font, color,
   legend, NUTS pixel maps)` and push.

## What NOT to do

- **Don't** edit the manuscript LaTeX text in this session unless a figure
  caption needs to be retuned to match a panel re-layout. The text is
  locked.
- **Don't** rerun NUTS sampling. Both the per-ROI (`results/inference_bwh_backup/`)
  and pixel-wise (`results/pixelwise/nuts_results.npz`) posteriors are
  already saved and authoritative.
- **Don't** touch `notes_manuscript_2nd_pass.txt` or
  `applied_math_approaches.txt`. Patrick will direct that work in a
  separate session about Methods-section expansion.
- **Don't** push without showing Patrick the regenerated figures first.

## Key memory entries to skim at session start

- `MEMORY.md` — index
- `project_stephan_review_round_20260507.md` — what was done in the text pass
- `feedback_coauthor_meeting_20260329.md` — Sandy/Stephan figure feedback
- `user_patrick.md` — working style preferences

## Quick orientation prompt for the next session

> "We're picking up the figure-regeneration pass for the MRM manuscript on
> prostate spectral DWI. Read `HANDOVER_FIGURE_SESSION.md` for the full
> brief. Stephan's per-direction signal decays for Fig 7: [Patrick will
> say "found, here:" or "still missing"]. Start with TODO 1 (global
> fonts) so we can iterate."
