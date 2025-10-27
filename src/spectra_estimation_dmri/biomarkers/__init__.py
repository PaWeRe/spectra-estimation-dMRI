"""
Biomarker analysis module for cancer prediction using diffusivity spectra.

Provides:
- Feature extraction with MC uncertainty propagation
- ADC baseline computation
- Classification with LOOCV
- Statistical comparison and visualization
"""

from .features import (
    extract_spectrum_features,
    extract_mc_features,
    extract_features_from_dataset,
    get_feature_names,
)

from .adc_baseline import (
    compute_adc,
    compute_adc_from_signal_decay,
    extract_adc_features,
    create_adc_feature_vector,
)

from .mc_classification import (
    prepare_classification_data,
    loocv_predictions,
    mc_predictions,
    compute_metrics,
    delong_test,
    bootstrap_auc_ci,
    evaluate_feature_set,
)

from .biomarker_viz import (
    plot_roc_curves,
    create_auc_table,
    plot_prediction_uncertainty,
    plot_feature_importance,
    create_summary_report,
)

__all__ = [
    # Features
    "extract_spectrum_features",
    "extract_mc_features",
    "extract_features_from_dataset",
    "get_feature_names",
    # ADC
    "compute_adc",
    "compute_adc_from_signal_decay",
    "extract_adc_features",
    "create_adc_feature_vector",
    # Classification
    "prepare_classification_data",
    "loocv_predictions",
    "mc_predictions",
    "compute_metrics",
    "delong_test",
    "bootstrap_auc_ci",
    "evaluate_feature_set",
    # Visualization
    "plot_roc_curves",
    "create_auc_table",
    "plot_prediction_uncertainty",
    "plot_feature_importance",
    "create_summary_report",
]
