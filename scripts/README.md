# `scripts/` — what each script is for

Every script is standalone: run it from the repository root with
`uv run python scripts/<name>.py`. They read the cached analysis outputs in
`results/biomarkers/` (regenerate those first with
`uv run python -m spectra_estimation_dmri.biomarkers.recompute`) and write to
`paper/figures/` or back into `results/biomarkers/`.

## Manuscript figures

| Script | Produces | Manuscript |
|---|---|---|
| `generate_fisher_figure.py` | `fig_fisher_v2` | Figure 1 — Fisher information, CRLB decomposition, component decays |
| `fig3_recovery_battery.py` | `fig3_v7` | Figure 2 — simulation recovery, 6 ground truths × 3 SNR |
| `generate_paper_figures.py` | `fig1_v4` | Figure 3 — cohort spectra by tissue type and zone |
| `fig2_roc_detection.py` | `fig2_v3` | Figure 4 — detection ROC curves |
| `fig3_adc_vs_discriminant.py` | `fig3_v4` | Figure 5 — ADC vs spectral discriminant |
| `fig4_lr_coefs_and_sensitivity.py` | `fig4_std_v4` | Figure 6 — per-bin weights + ADC sensitivity |
| `plot_spectrum_by_ggg.py` | `fig5_v5` | Figure 7 — spectrum by Gleason Grade Group |
| `fig6_uncertainty_classifier.py` | `fig6_v2` | Figure 8 — posterior propagated through the classifier |
| `fig9_pixelwise.py` | `fig9_v2` | Figure 9 — pixel-wise spectral maps |
| `fig_si_subset.py` | `figS_subset_atlas`, `figS_subset_convergence` | Figures S1, S2 |
| `fig7_directions_roi.py` | `fig_directions_v4` | Figure S3 — direction independence |

Figure file names keep their historical stems; the printed figure number comes
from the float order in `paper/sections/figures.tex`.

## Analyses quoted in the manuscript

| Script | Produces | Used for |
|---|---|---|
| `zone_grade_check.py` | `zone_grade_check.csv` | Discussion — zone-confound check on the grade trends (Figure 7 pools zones) |
| `adc_sensitivity_at_tuned_lambda.py` | `adc_sens_vs_lr_tuned_lambda.csv` | Discussion — ADC sensitivity vs classifier weights across λ |
| `map_lambda_bwh.py` | `map_lambda_bwh*.csv` | Methods — cohort-level ridge λ sweep |
| `adc_variants_sweep.py` | `adc_variants*.csv` | Methods — ADC b-value-range choice |
| `two_feature_lr_vs_adc.py` | `two_feature_lr_vs_adc.csv` | Results — two-fraction model vs ADC |
| `snr_diagnostic.py` | `snr_comparison.csv` | Methods — inferred per-ROI SNR vs closed-form estimate |
| `figS1_all_roi_spectra.py` | full 149-ROI atlas | superseded in the manuscript by the 5-pair subset (`fig_si_subset.py`), kept because the Supporting Information refers to it |

## Supporting / exploratory

Not cited in the manuscript; retained because they document choices made along
the way and are cheap to re-run.

`bin_information_sweep.py` (per-bin information vs grid choice) ·
`classifier_comparison.py` (logistic vs tree ensembles) ·
`lr_coef_decomp.py`, `plot_lr_weights_per_bin.py`,
`plot_lr_weights_vs_adc_sensitivity.py` (earlier views of Figure 6) ·
`partial_corr_ggg.py`, `ggg_continuous_sweep.py` (grade correlations) ·
`fixed_sigma_refit.py`, `fixed_sigma_refit_plot.py`, `wider_prior_check.py`
(prior/noise sensitivity) · `map_lambda_sweep.py`, `regrid_robustness.py`,
`robustness_test.py`, `simulation_study.py` (simulation robustness) ·
`direction_comparison.py` (early version of Figure S3) ·
`pixel_wise_heatmap.py` (early version of Figure 9) ·
`fig1_tuned_map_vs_nuts.py`, `fig8_battery.py`, `fig8_simulation_and_crlb.py`,
`fig8_validation.py`, `flagship_figure.py`, `fnew1_promoted_simulation.py`
(superseded figure drafts) ·
`generate_prostate_signal_decay.py`,
`generate_combined_prostate_signal_snr_spectrum.py`,
`generate_snr_posterior_ismrm.py` (conference-poster figures) ·
`run_bwh_biomarker_analysis.py`, `verify_findings.py` (legacy entry points,
superseded by `biomarkers/recompute.py`) ·
`plot_lambda_sweep.py` (plotting helper).
