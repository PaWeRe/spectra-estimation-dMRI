"""
ISMRM Abstract Figure Exports

Creates publication-ready figures for ISMRM abstract submissions:
- 3:2 aspect ratio (e.g., 900×600 pixels)
- PNG format
- <1MB file size
- High-quality but optimized
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import roc_curve


def export_ismrm_plot(
    fig_generator,
    output_path: str,
    width_px: int = 900,
    height_px: Optional[int] = None,
    dpi: int = 150,
) -> None:
    """
    Export a plot in ISMRM-compatible format.

    Args:
        fig_generator: Function that returns a matplotlib figure
        output_path: Path for PNG output
        width_px: Width in pixels (default 900 for 900×600)
        height_px: Height in pixels (default None = 3:2 aspect ratio)
        dpi: DPI for rendering (default 150 for good quality)
    """
    # Default 3:2 aspect ratio, or custom if specified
    if height_px is None:
        height_px = int(width_px * 2 / 3)

    # Convert to inches for matplotlib
    width_in = width_px / dpi
    height_in = height_px / dpi

    # Generate figure
    fig = fig_generator(figsize=(width_in, height_in))

    # Save as PNG with optimization
    # NOTE: Do NOT use bbox_inches="tight" - it overrides gridspec spacing!
    fig.savefig(
        output_path,
        dpi=dpi,
        format="png",
        pil_kwargs={"optimize": True, "quality": 85},
    )
    plt.close(fig)

    # Check file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(
        f"[ISMRM] Saved {output_path} ({width_px}×{height_px}px, {file_size_mb:.2f} MB)"
    )

    if file_size_mb > 1.0:
        print(f"[WARNING] File size exceeds 1MB limit")


def create_ismrm_averaged_spectra(
    regions: Dict[str, List],
    output_dir: str,
) -> None:
    """
    Create ISMRM-compatible averaged spectra figure with shared y-axis range.

    Args:
        regions: Dictionary from group_spectra_by_region()
        output_dir: Directory to save PNG
    """

    def fig_generator(figsize):
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
        axes = axes.flatten()

        region_names = {
            "normal_pz": "Normal PZ",
            "normal_tz": "Normal TZ",
            "tumor_pz": "Tumor PZ",
            "tumor_tz": "Tumor TZ",
        }

        diffusivities = None

        # First pass: collect all data to determine shared y-axis range
        all_data = []
        region_data = {}

        for idx, (region_key, spectra_list) in enumerate(regions.items()):
            if idx >= 4 or len(spectra_list) == 0:
                continue

            # Get diffusivities
            if diffusivities is None:
                diffusivities = spectra_list[0].diffusivities

            # Collect samples - handle variable sample counts
            all_samples = []
            for spectrum in spectra_list:
                arrays = spectrum.as_numpy()
                samples = arrays["spectrum_samples"]
                if samples is not None:
                    all_samples.append(samples)

            if len(all_samples) > 0:
                # Find minimum number of samples across all spectra
                min_n_samples = min(s.shape[0] for s in all_samples)

                # Truncate all to same length
                all_samples_truncated = [s[:min_n_samples, :] for s in all_samples]

                # Average samples across spectra
                avg_samples = np.mean(np.array(all_samples_truncated), axis=0)

                # Normalize
                avg_samples_norm = avg_samples / (
                    np.sum(avg_samples, axis=1, keepdims=True) + 1e-10
                )

                # Store for plotting
                region_data[idx] = (region_key, spectra_list, avg_samples_norm)
                all_data.append(avg_samples_norm)

        # Calculate shared y-axis range across all regions
        if len(all_data) > 0:
            all_values = np.concatenate(all_data)
            y_min = np.percentile(all_values, 1)
            y_max = np.percentile(all_values, 99)
            y_range = y_max - y_min
            shared_ylim = [
                max(0, y_min - 0.1 * y_range),
                min(1.0, y_max + 0.1 * y_range),
            ]
        else:
            shared_ylim = [0, 1]

        # Second pass: plot with shared y-axis
        for idx, (region_key, spectra_list, avg_samples_norm) in region_data.items():
            ax = axes[idx]

            # Plot
            bp = ax.boxplot(
                avg_samples_norm,
                showfliers=False,
                manage_ticks=False,
                showmeans=True,
                meanline=True,
            )

            # Apply shared y-axis limits
            ax.set_ylim(shared_ylim)

            # Grid and styling
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"{region_names[region_key]} (n={len(spectra_list)})",
                fontsize=14,
                fontweight="bold",
            )

            # X-axis labels (bottom row only)
            if idx >= 2:
                ax.set_xlabel(
                    r"Diffusivity ($\mu$m$^2$/ms)", fontsize=13, fontweight="bold"
                )
                str_diffs = [f"{d:.2f}" for d in diffusivities]
                ax.set_xticks(
                    np.arange(1, len(str_diffs) + 1),
                    labels=str_diffs,
                    rotation=45,
                    fontsize=11,
                )

            # Y-axis labels (left column only)
            if idx % 2 == 0:
                ax.set_ylabel("Relative Fraction", fontsize=13, fontweight="bold")

            # Make tick labels larger for all subplots
            ax.tick_params(axis="both", which="major", labelsize=11)

        plt.tight_layout()
        return fig

    output_path = os.path.join(output_dir, "averaged_spectra_ismrm.png")
    export_ismrm_plot(fig_generator, output_path, width_px=1800, height_px=1800)


def create_ismrm_roc_curve(
    results_list: List[Dict],
    output_path: str,
    title: str = "ROC Curve",
    highlight_features: Optional[List[str]] = None,
) -> None:
    """
    Create ISMRM-compatible ROC curve figure.

    Args:
        results_list: List of classification results
        output_path: Path for PNG output
        title: Plot title
        highlight_features: List of feature names to emphasize (thicker lines)
    """

    def fig_generator(figsize):
        fig, ax = plt.subplots(figsize=figsize)

        # Filter valid results
        valid_results = [
            r
            for r in results_list
            if r is not None and not np.isnan(r.get("metrics", {}).get("auc", np.nan))
        ]

        if not valid_results:
            ax.text(0.5, 0.5, "No valid results", ha="center", va="center")
            return fig

        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))

        # Plot curves
        for i, result in enumerate(valid_results):
            y_true = result["y_true"]
            y_pred_proba = result["y_pred_proba"]
            feature_name = result["feature_name"]
            auc = result["metrics"]["auc"]

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

            # Highlight certain features
            linewidth = (
                3 if (highlight_features and feature_name in highlight_features) else 2
            )
            alpha = (
                1.0
                if (highlight_features and feature_name in highlight_features)
                else 0.7
            )

            label = f"{feature_name} (AUC={auc:.3f})"
            ax.plot(fpr, tpr, color=colors[i], lw=linewidth, label=label, alpha=alpha)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")

        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        return fig

    export_ismrm_plot(fig_generator, output_path)


def create_ismrm_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: str,
    title: str = "Feature Importance",
) -> None:
    """
    Create ISMRM-compatible feature importance figure.

    Args:
        feature_names: List of feature names
        importances: Feature importance values (LR coefficients)
        output_path: Path for PNG output
        title: Plot title
    """

    def fig_generator(figsize):
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by absolute importance
        sorted_idx = np.argsort(np.abs(importances))[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_importances = np.abs(importances[sorted_idx])

        # Limit to top 8 features for readability
        n_show = min(8, len(sorted_names))
        sorted_names = sorted_names[:n_show]
        sorted_importances = sorted_importances[:n_show]

        y_pos = np.arange(n_show)
        ax.barh(y_pos, sorted_importances, color="steelblue", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.set_xlabel("Absolute Coefficient", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()  # Highest importance on top

        plt.tight_layout()
        return fig

    export_ismrm_plot(fig_generator, output_path)


def create_ismrm_roc_tumor_detection(
    results_dict: Dict[str, List[Dict]],
    output_path: str,
) -> None:
    """
    Create ROC curves for PZ and TZ tumor detection (1x2 layout).
    Only includes key features: ADC, Full LR, D 0.25, D 3.00.

    Args:
        results_dict: Dictionary with results for each task
        output_path: Path for PNG output
    """

    def fig_generator(figsize):
        # Create 1x2 subplots for PZ and TZ
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        tasks = [
            ("tumor_vs_normal_pz", "PZ: Tumor Detection", 0),
            ("tumor_vs_normal_tz", "TZ: Tumor Detection", 1),
        ]

        # Only include these key features
        key_features = ["ADC (baseline)", "Full LR", "D_0.25", "D_3.00"]

        for task_key, title, ax_idx in tasks:
            ax = axes[ax_idx]

            if task_key not in results_dict:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(title, fontsize=11, fontweight="bold")
                continue

            results_list = results_dict[task_key]

            # Filter: only include key features
            valid_results = [
                r
                for r in results_list
                if r is not None
                and not np.isnan(r.get("metrics", {}).get("auc", np.nan))
                and r["feature_name"] in key_features
            ]

            if not valid_results:
                ax.text(0.5, 0.5, "No valid results", ha="center", va="center")
                ax.set_title(title, fontsize=11, fontweight="bold")
                continue

            # Sort by AUC descending
            valid_results = sorted(
                valid_results, key=lambda x: x["metrics"]["auc"], reverse=True
            )

            # Color mapping for consistency
            color_map = {
                "ADC (baseline)": "#1f77b4",  # Blue
                "Full LR": "#ff7f0e",  # Orange
                "D_0.25": "#2ca02c",  # Green
                "D_3.00": "#d62728",  # Red
            }

            # Plot curves
            for result in valid_results:
                y_true = result["y_true"]
                y_pred_proba = result["y_pred_proba"]
                feature_name = result["feature_name"]
                auc = result["metrics"]["auc"]

                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

                # Simplified labels
                if feature_name.startswith("D_"):
                    d_val = feature_name.split("_")[1]
                    label = f"D {d_val} ({auc:.2f})"
                elif feature_name == "ADC (baseline)":
                    label = f"ADC ({auc:.2f})"
                elif feature_name == "Full LR":
                    label = f"Full LR ({auc:.2f})"
                else:
                    label = f"{feature_name} ({auc:.2f})"

                color = color_map.get(feature_name, "gray")
                lw = 2.5

                ax.plot(fpr, tpr, color=color, lw=lw, label=label, alpha=0.9)

            # Diagonal reference
            ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random")

            ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
            ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
            ax.grid(alpha=0.3)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        return fig

    export_ismrm_plot(fig_generator, output_path, width_px=1200)


def create_ismrm_roc_ggg(
    results_dict: Dict[str, List[Dict]],
    output_path: str,
) -> None:
    """
    Create ROC curve for GGG grading (single plot).
    Only includes key features: ADC, Full LR, D 0.25, D 3.00.

    Args:
        results_dict: Dictionary with results for each task
        output_path: Path for PNG output
    """

    def fig_generator(figsize):
        fig, ax = plt.subplots(figsize=figsize)

        task_key = "ggg"

        if task_key not in results_dict:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title("GGG(PZ) 1-2 vs 3-5", fontsize=12, fontweight="bold")
            return fig

        results_list = results_dict[task_key]

        # Only include these key features
        key_features = ["ADC (baseline)", "Full LR", "D_0.25", "D_3.00"]

        # Filter: only include key features
        valid_results = [
            r
            for r in results_list
            if r is not None
            and not np.isnan(r.get("metrics", {}).get("auc", np.nan))
            and r["feature_name"] in key_features
        ]

        if not valid_results:
            ax.text(0.5, 0.5, "No valid results", ha="center", va="center")
            ax.set_title("GGG(PZ) 1-2 vs 3-5", fontsize=12, fontweight="bold")
            return fig

        # Sort by AUC descending
        valid_results = sorted(
            valid_results, key=lambda x: x["metrics"]["auc"], reverse=True
        )

        # Color mapping for consistency
        color_map = {
            "ADC (baseline)": "#1f77b4",  # Blue
            "Full LR": "#ff7f0e",  # Orange
            "D_0.25": "#2ca02c",  # Green
            "D_3.00": "#d62728",  # Red
        }

        # Plot curves
        for result in valid_results:
            y_true = result["y_true"]
            y_pred_proba = result["y_pred_proba"]
            feature_name = result["feature_name"]
            auc = result["metrics"]["auc"]

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

            # Simplified labels
            if feature_name.startswith("D_"):
                d_val = feature_name.split("_")[1]
                label = f"D {d_val} ({auc:.2f})"
            elif feature_name == "ADC (baseline)":
                label = f"ADC ({auc:.2f})"
            elif feature_name == "Full LR":
                label = f"Full LR ({auc:.2f})"
            else:
                label = f"{feature_name} ({auc:.2f})"

            color = color_map.get(feature_name, "gray")
            lw = 2.5

            ax.plot(fpr, tpr, color=color, lw=lw, label=label, alpha=0.9)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random")

        ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        ax.set_title("GGG(PZ) 1-2 vs 3-5", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        return fig

    export_ismrm_plot(fig_generator, output_path, width_px=900)


def create_ismrm_roc_combined(
    results_dict: Dict[str, List[Dict]],
    output_path: str,
) -> None:
    """
    Create combined ROC curves for PZ, TZ, and GGG in a single figure (2x2 layout).
    Includes ADC baseline and ALL single diffusivity bin features (raw univariate predictors).

    Args:
        results_dict: Dictionary with results for each task
        output_path: Path for PNG output
    """

    def fig_generator(figsize):
        # Create figure with custom spacing for tighter layout
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=figsize)

        # Create gridspec for custom spacing
        # ========================================================================
        # TUNING PARAMETERS - Optimized for 1:1 (square) aspect ratio
        # ========================================================================
        # wspace: Horizontal gap between columns (0.0-1.0, smaller = tighter)
        #         Negative values make plots overlap and appear closer
        # hspace: Vertical gap between rows (0.0-1.0)
        # left/right/top/bottom: Margins as fraction of figure (0.0-1.0)
        # ========================================================================
        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            wspace=0.15,
            hspace=0.15,
            left=0.06,  # ← Left margin
            right=0.97,  # ← Right margin
            top=0.97,  # ← Top margin
            bottom=0.06,  # ← Bottom margin - adequate for square format
        )

        # Create axes from gridspec
        axes = [
            fig.add_subplot(gs[0, 0]),  # Top-left
            fig.add_subplot(gs[0, 1]),  # Top-right
            fig.add_subplot(gs[1, 0]),  # Bottom-left
            fig.add_subplot(gs[1, 1]),
        ]  # Bottom-right

        tasks = [
            ("tumor_vs_normal_pz", "PZ: Tumor Detection", 0),
            ("tumor_vs_normal_tz", "TZ: Tumor Detection", 1),
            ("ggg", "GGG(PZ) 1-2 vs 3-5", 2),
        ]

        # Include ADC (raw), all single bin features (raw), and Full LR (trained LOOCV classifier)
        key_features = [
            "ADC (baseline)",
            "Full LR",
            "D_0.25",
            "D_0.50",
            "D_0.75",
            "D_1.00",
            "D_1.50",
            "D_2.00",
            "D_3.00",
            "D_20.00",
        ]

        # Color palette for all features
        colors_list = plt.cm.tab10(np.linspace(0, 1, 10))
        color_map = {
            "ADC (baseline)": colors_list[0],
            "Full LR": colors_list[1],  # Orange - stands out as the multivariate model
            "D_0.25": colors_list[2],
            "D_0.50": colors_list[3],
            "D_0.75": colors_list[4],
            "D_1.00": colors_list[5],
            "D_1.50": colors_list[6],
            "D_2.00": colors_list[7],
            "D_3.00": colors_list[8],
            "D_20.00": colors_list[9],
        }

        for task_key, title, ax_idx in tasks:
            ax = axes[ax_idx]

            if task_key not in results_dict:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.axis("off")
                continue

            results_list = results_dict[task_key]

            # Filter: only include key features
            valid_results = [
                r
                for r in results_list
                if r is not None
                and not np.isnan(r.get("metrics", {}).get("auc", np.nan))
                and r["feature_name"] in key_features
            ]

            if not valid_results:
                ax.text(0.5, 0.5, "No valid results", ha="center", va="center")
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.axis("off")
                continue

            # Sort by AUC descending
            valid_results = sorted(
                valid_results, key=lambda x: x["metrics"]["auc"], reverse=True
            )

            # Get sample counts from first result (all results have same y_true for a given task)
            y_true_first = valid_results[0]["y_true"]
            n_positive = int(np.sum(y_true_first))
            n_negative = int(len(y_true_first) - n_positive)

            # Add counts to title
            if task_key == "ggg":
                title_with_counts = (
                    f"{title} (n={n_positive} grade 3-5, n={n_negative} grade 1-2)"
                )
            else:  # PZ or TZ tumor detection
                title_with_counts = (
                    f"{title} (n={n_positive} tumor, n={n_negative} normal)"
                )

            # Plot curves
            for result in valid_results:
                y_true = result["y_true"]
                y_pred_proba = result["y_pred_proba"]
                feature_name = result["feature_name"]
                auc = result["metrics"]["auc"]
                auc_ci_lower = result["metrics"].get("auc_ci_lower", np.nan)
                auc_ci_upper = result["metrics"].get("auc_ci_upper", np.nan)

                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

                # Simplified labels with 95% CI
                if feature_name.startswith("D_"):
                    d_val = feature_name.split("_")[1]
                    base_label = f"D {d_val}"
                elif feature_name == "ADC (baseline)":
                    base_label = "ADC"
                elif feature_name == "Full LR":
                    base_label = "Full LR"
                else:
                    base_label = feature_name

                # Add AUC with CI if available
                if not np.isnan(auc_ci_lower) and not np.isnan(auc_ci_upper):
                    label = f"{base_label} {auc:.2f} [{auc_ci_lower:.2f}-{auc_ci_upper:.2f}]"
                else:
                    label = f"{base_label} ({auc:.2f})"

                color = color_map.get(feature_name, "gray")
                lw = 1.8  # Slightly thinner lines for more features

                ax.plot(fpr, tpr, color=color, lw=lw, label=label, alpha=0.85)

            # Diagonal reference
            ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.35)

            # Axis labels - only on left column and bottom row for space efficiency
            if ax_idx in [0, 2]:  # Left column
                ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
            if ax_idx in [2, 3]:  # Bottom row
                ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")

            ax.set_title(title_with_counts, fontsize=12, fontweight="bold")
            # Two-column legend for more features - larger font for readability
            ax.legend(loc="lower right", fontsize=10, framealpha=0.95, ncol=2)
            ax.grid(alpha=0.3)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_aspect("equal", adjustable="box")  # Force 1:1 aspect ratio

        # Hide the 4th subplot (bottom-right)
        axes[3].axis("off")

        # No need for tight_layout - using gridspec spacing instead
        return fig

    # Use 1:1 aspect ratio (square) for better arrangement of 3 plots
    # 1800×1800px provides good resolution and space
    export_ismrm_plot(fig_generator, output_path, width_px=1800, height_px=1800)


def create_ismrm_combined_feature_importance(
    results_dict: Dict[str, List[Dict]],
    output_path: str,
    diffusivity_order: List[float] = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 20.00],
) -> None:
    """
    Create combined feature importance for PZ, TZ, GGG(PZ) as separate subplots in 2x2 layout.

    Each subplot shows one task with positive and negative coefficients side-by-side.
    Sorted by diffusivity for comparison across tasks.

    Args:
        results_dict: Dictionary with results for each task
        output_path: Path for PNG output
        diffusivity_order: Order of diffusivities for x-axis
    """

    def fig_generator(figsize):
        # Create figure with custom spacing for tighter layout
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=figsize)

        # Create gridspec for custom spacing - optimized for 1:1 square format
        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            wspace=0.15,  # Horizontal spacing
            hspace=0.18,  # Vertical spacing - tighter for square format
            left=0.06,
            right=0.98,
            top=0.96,
            bottom=0.08,
        )

        # Create axes from gridspec
        axes = [
            fig.add_subplot(gs[0, 0]),  # Top-left
            fig.add_subplot(gs[0, 1]),  # Top-right
            fig.add_subplot(gs[1, 0]),  # Bottom-left
            fig.add_subplot(gs[1, 1]),
        ]  # Bottom-right

        tasks = [
            ("tumor_vs_normal_pz", "PZ: Tumor Detection", 0),
            ("tumor_vs_normal_tz", "TZ: Tumor Detection", 1),
            ("ggg", "GGG(PZ) 1-2 vs 3-5", 2),
        ]

        # Collect feature importance for Full LR model
        importance_data = {}

        for task_key, task_label, _ in tasks:
            if task_key not in results_dict:
                continue

            # Find Full LR result
            full_lr_result = next(
                (
                    r
                    for r in results_dict[task_key]
                    if r is not None and r.get("feature_name") == "Full LR"
                ),
                None,
            )

            if full_lr_result and "feature_importance" in full_lr_result:
                feature_names = full_lr_result["feature_names_ordered"]
                importances = full_lr_result["feature_importance"]

                importance_data[task_key] = {"label": task_label, "features": {}}
                for fname, imp in zip(feature_names, importances):
                    # Extract diffusivity value from feature name (e.g., "D_0.25" -> 0.25)
                    if fname.startswith("D_"):
                        try:
                            diff_val = float(fname.split("_")[1])
                            importance_data[task_key]["features"][diff_val] = imp
                        except:
                            pass

        # Find global max coefficient value across all tasks for consistent y-axis
        global_max = 0
        for task_key in importance_data:
            task_coeffs = importance_data[task_key]["features"].values()
            if task_coeffs:
                task_max = max(abs(c) for c in task_coeffs)
                global_max = max(global_max, task_max)

        # Add 10% padding to the top
        y_max = global_max * 1.1

        # Plot each task in its own subplot
        for task_key, task_label, ax_idx in tasks:
            ax = axes[ax_idx]

            if task_key not in importance_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                ax.set_title(task_label, fontsize=10, fontweight="bold")
                ax.axis("off")
                continue

            task_data = importance_data[task_key]["features"]

            # Sort diffusivities by absolute coefficient magnitude (descending)
            sorted_items = sorted(
                task_data.items(), key=lambda x: abs(x[1]), reverse=True
            )
            sorted_diffs = [d for d, _ in sorted_items]
            sorted_values = [v for _, v in sorted_items]

            # Prepare x-axis with sorted diffusivities
            x_labels = [f"D {d:.2f}" for d in sorted_diffs]
            x_pos = np.arange(len(sorted_diffs))
            values_arr = np.array(sorted_values)

            # Create color array: blue for positive, red for negative
            colors = [
                "#2E86AB" if v >= 0 else "#D7263D" for v in values_arr
            ]  # Blue for positive, Red for negative

            # Plot centered bars with absolute values
            bar_width = 0.6
            bars = ax.bar(
                x_pos,
                np.abs(values_arr),  # Show absolute values
                width=bar_width,
                color=colors,
                alpha=0.85,
            )

            # Add legend manually (since we don't have separate bar calls)
            if ax_idx == 0:
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor="#2E86AB", alpha=0.85, label="Positive"),
                    Patch(facecolor="#D7263D", alpha=0.85, label="Negative"),
                ]
                ax.legend(
                    handles=legend_elements,
                    loc="upper right",
                    fontsize=8,
                    framealpha=0.95,
                )

            # Styling
            ax.set_title(task_label, fontsize=11, fontweight="bold")

            # Only show x-label on bottom row
            if ax_idx in [2, 3]:
                ax.set_xlabel(r"Diffusivity Bin ($\mu$m$^2$/ms)", fontsize=10)

            # Only show y-label on left column
            if ax_idx in [0, 2]:
                ax.set_ylabel("Absolute LR Coefficient", fontsize=10)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            # Set y-axis to start at 0 with consistent scale across all subplots
            ax.set_ylim(bottom=0, top=y_max)

        # Hide the 4th subplot (bottom-right)
        axes[3].axis("off")

        # No need for tight_layout - using gridspec spacing instead
        return fig

    # Use 1:1 aspect ratio (square) - 1500×1500px for consistency with ROC plots
    export_ismrm_plot(fig_generator, output_path, width_px=1500, height_px=1500)


def create_ismrm_snr_posterior(
    spectra_dataset,
    output_path: str,
    max_realizations: int = 10,
) -> None:
    """
    Create ISMRM-compatible SNR posterior plot for simulation studies.

    Shows joint inference of SNR and spectrum with:
    - Boxplot of SNR posteriors across realizations
    - Ground truth SNR line
    - Bias statistics

    Args:
        spectra_dataset: DiffusivitySpectraDataset (must contain simulation data)
        output_path: Path for PNG output
        max_realizations: Maximum number of realizations to plot
    """
    import arviz as az
    import os

    def fig_generator(figsize):
        fig, ax = plt.subplots(figsize=figsize)

        # Filter to NUTS spectra with sigma inference
        spectra_with_sigma = []
        for spectrum in spectra_dataset.spectra[:max_realizations]:
            if spectrum.inference_method == "nuts" and spectrum.inference_data:
                if os.path.exists(spectrum.inference_data):
                    try:
                        idata = az.from_netcdf(spectrum.inference_data)
                        if "sigma" in idata.posterior:
                            spectra_with_sigma.append(spectrum)
                    except:
                        pass

        if not spectra_with_sigma:
            ax.text(
                0.5,
                0.5,
                "No SNR inference data available\n(Requires simulation with NUTS)",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("SNR Posterior Inference", fontsize=12, fontweight="bold")
            return fig

        # Extract SNR posteriors
        all_snr_samples = []
        for spectrum in spectra_with_sigma:
            idata = az.from_netcdf(spectrum.inference_data)
            sigma_samples = idata.posterior["sigma"].values.flatten()
            snr_samples = 1.0 / sigma_samples
            all_snr_samples.append(snr_samples)

        # Get true SNR if available
        true_snr = spectra_with_sigma[0].data_snr
        n_realizations = len(spectra_with_sigma)

        # Create boxplot
        bp = ax.boxplot(
            all_snr_samples,
            showfliers=False,
            showmeans=True,
            meanline=False,
            meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
            medianprops=dict(color="blue", linewidth=2),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        # Add ground truth line
        if true_snr:
            ax.axhline(
                true_snr,
                color="darkgreen",
                linestyle="--",
                linewidth=2.5,
                label="Ground Truth",
                zorder=10,
            )

            # Calculate bias
            snr_means = [np.mean(samples) for samples in all_snr_samples]
            overall_mean = np.mean(snr_means)
            bias = overall_mean - true_snr
            rel_error = (bias / true_snr) * 100

            # Add statistics box
            stats_text = (
                f"Est: {overall_mean:.0f}\nBias: {bias:+.0f} ({rel_error:+.1f}%)"
            )
            ax.text(
                0.98,
                0.02,
                stats_text,
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=9,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=1.5,
                ),
                zorder=15,
            )

        # Styling
        ax.set_xlabel("Realization", fontsize=11, fontweight="bold")
        ax.set_ylabel("SNR", fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, n_realizations + 1))
        ax.set_xticklabels([f"{i}" for i in range(1, n_realizations + 1)])
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(
            "Joint SNR + Spectrum Inference (NUTS)", fontsize=12, fontweight="bold"
        )

        plt.tight_layout()
        return fig

    export_ismrm_plot(fig_generator, output_path)


def create_ismrm_snr_and_spectrum_combined(
    spectra_dataset,
    output_path: str,
    max_realizations: int = 10,
    realization_idx: int = 0,
    width_px: int = 1800,
    height_px: int = 900,
    dpi: int = 150,
) -> None:
    """
    Create ISMRM-compatible combined SNR + Spectrum posterior plot.

    Two subplots side-by-side:
    - Left: SNR posterior across realizations (boxplot)
    - Right: Spectrum posterior for one realization (boxplot per diffusivity)

    Args:
        spectra_dataset: DiffusivitySpectraDataset (must contain simulation data)
        output_path: Path for PNG output
        max_realizations: Maximum number of realizations to include
        realization_idx: Which realization to show in spectrum subplot (default: 0 = first)
        width_px: Width in pixels (default 1800 for 2:1 aspect ratio)
        height_px: Height in pixels (default 900 for 2:1 aspect, giving square subplots)
        dpi: DPI for rendering (default 150)
    """
    import arviz as az
    import os

    def fig_generator(figsize):
        # Create 1x2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Filter to NUTS spectra with sigma inference
        spectra_with_sigma = []
        for spectrum in spectra_dataset.spectra[:max_realizations]:
            if spectrum.inference_method == "nuts" and spectrum.inference_data:
                if os.path.exists(spectrum.inference_data):
                    try:
                        idata = az.from_netcdf(spectrum.inference_data)
                        if "sigma" in idata.posterior:
                            spectra_with_sigma.append(spectrum)
                    except:
                        pass

        if not spectra_with_sigma:
            ax1.text(0.5, 0.5, "No SNR inference data", ha="center", va="center")
            ax2.text(0.5, 0.5, "No spectrum data", ha="center", va="center")
            return fig

        # === LEFT SUBPLOT: SNR Posterior ===
        all_snr_samples = []
        for spectrum in spectra_with_sigma:
            idata = az.from_netcdf(spectrum.inference_data)
            sigma_samples = idata.posterior["sigma"].values.flatten()
            snr_samples = 1.0 / sigma_samples
            all_snr_samples.append(snr_samples)

        true_snr = spectra_with_sigma[0].data_snr
        n_realizations = len(spectra_with_sigma)

        # Create SNR boxplot
        bp1 = ax1.boxplot(
            all_snr_samples,
            showfliers=False,
            showmeans=True,
            meanline=False,
            meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
            medianprops=dict(color="blue", linewidth=2),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        # Add ground truth SNR line
        if true_snr:
            ax1.axhline(
                true_snr,
                color="darkgreen",
                linestyle="--",
                linewidth=2.5,
                label="Ground Truth",
                zorder=10,
            )

            # Calculate bias
            snr_means = [np.mean(samples) for samples in all_snr_samples]
            overall_mean = np.mean(snr_means)
            bias = overall_mean - true_snr
            rel_error = (bias / true_snr) * 100

            # Add statistics box (TOP LEFT)
            stats_text = f"Estimated SNR: {overall_mean:.1f}\nBias: {bias:+.1f} ({rel_error:+.1f}%)"
            ax1.text(
                0.02,
                0.98,
                stats_text,
                transform=ax1.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=1.0,
                ),
                zorder=15,
            )

        # SNR subplot styling
        ax1.set_xlabel("Realization", fontsize=10, fontweight="bold")
        ax1.set_ylabel("SNR", fontsize=10, fontweight="bold")
        ax1.set_xticks(range(1, n_realizations + 1))
        ax1.set_xticklabels([f"{i}" for i in range(1, n_realizations + 1)], fontsize=8)
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title(
            "Joint Inference: SNR Across Multiple Realizations",
            fontsize=10,
            fontweight="bold",
        )

        # === RIGHT SUBPLOT: Spectrum Posterior (like posterior_shape) ===
        # Use the specified realization
        if realization_idx >= len(spectra_with_sigma):
            realization_idx = 0

        spectrum = spectra_with_sigma[realization_idx]

        # Get spectrum samples
        if spectrum.spectrum_samples is not None:
            samples = np.array(spectrum.spectrum_samples)
            diffusivities = np.array(spectrum.diffusivities)
            n_diff = len(diffusivities)

            # Create boxplot for spectrum (same style as posterior_shape)
            bp2 = ax2.boxplot(
                samples,
                showfliers=False,
                showmeans=True,
                meanline=True,
                labels=[f"{d:.2f}" for d in diffusivities],
                boxprops=dict(linewidth=1.5),
                medianprops=dict(linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
            )

            # Add MAP initialization if available
            if spectrum.spectrum_init is not None:
                x_positions = np.arange(1, n_diff + 1)
                ax2.plot(
                    x_positions,
                    spectrum.spectrum_init,
                    "r*",
                    label="Initial R (MAP)",
                    markersize=8,
                )

            # Add true spectrum if available
            if spectrum.true_spectrum is not None:
                true_spectrum = np.array(spectrum.true_spectrum)
                x_positions = np.arange(1, n_diff + 1)
                ax2.vlines(
                    x_positions,
                    0,
                    true_spectrum,
                    colors="blue",
                    linewidth=2.5,
                    label="True Spectrum",
                )

            # Spectrum subplot styling
            ax2.set_xlabel(
                r"Diffusivity ($\mu$m$^2$/ms)", fontsize=10, fontweight="bold"
            )
            ax2.set_ylabel("Relative Fraction", fontsize=10, fontweight="bold")
            ax2.set_xticklabels(
                [f"{d:.2f}" for d in diffusivities], rotation=45, fontsize=8
            )
            ax2.grid(True, alpha=0.3)
            ax2.set_title(
                f"Joint Inference: Spectrum (Realization {realization_idx+1})",
                fontsize=10,
                fontweight="bold",
            )

            if spectrum.spectrum_init is not None or spectrum.true_spectrum is not None:
                ax2.legend(fontsize=8, loc="best")
        else:
            ax2.text(0.5, 0.5, "No spectrum samples", ha="center", va="center")

        plt.tight_layout()
        return fig

    # Use 2:1 aspect ratio (1800×900) so each subplot is square
    export_ismrm_plot(
        fig_generator, output_path, width_px=width_px, height_px=height_px, dpi=dpi
    )


def create_ismrm_prostate_and_signal_decay(
    signal_decay_dataset,
    prostate_image_path: str,
    output_path: str,
    patient_id: Optional[str] = None,
    width_px: int = 1800,
    height_px: int = 900,
    dpi: int = 150,
) -> None:
    """
    Create ISMRM-compatible figure with prostate MRI image and signal decay curves.

    Two subplots side-by-side:
    - Left: Prostate MRI annotation image
    - Right: Signal decay curves (normal vs tumor tissue from same patient)

    Args:
        signal_decay_dataset: SignalDecayDataset with BWH patient data
        prostate_image_path: Path to prostate MRI annotation image
        output_path: Path for PNG output
        patient_id: Specific patient to plot (if None, picks a random patient with both normal and tumor)
        width_px: Width in pixels (default 1800 for 2:1 aspect ratio)
        height_px: Height in pixels (default 900 for 2:1 aspect, giving square subplots)
        dpi: DPI for rendering (default 150)
    """
    from PIL import Image
    import random

    # Find patient with both tumor and normal tissue
    selected_patient_id = patient_id
    if selected_patient_id is None:
        # Find all patients with both tumor and normal samples
        from collections import defaultdict

        patient_samples = defaultdict(list)
        for sample in signal_decay_dataset.samples:
            patient_samples[sample.patient].append(sample)

        # Filter patients with both tumor and normal
        valid_patients = []
        for pid, samples in patient_samples.items():
            has_tumor = any(s.is_tumor for s in samples)
            has_normal = any(not s.is_tumor for s in samples)
            if has_tumor and has_normal:
                valid_patients.append(pid)

        if valid_patients:
            # Pick random patient
            selected_patient_id = random.choice(valid_patients)
            print(f"[ISMRM] Selected patient: {selected_patient_id}")

    def fig_generator(figsize):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # === LEFT SUBPLOT: Prostate MRI Image ===
        if os.path.exists(prostate_image_path):
            img = Image.open(prostate_image_path)
            ax1.imshow(img)
            ax1.axis("off")
            ax1.set_title(
                "Prostate MRI with ROI Annotations", fontsize=11, fontweight="bold"
            )
        else:
            ax1.text(0.5, 0.5, "Image not found", ha="center", va="center")
            ax1.axis("off")

        # === RIGHT SUBPLOT: Signal Decay ===
        if selected_patient_id is None:
            ax2.text(
                0.5,
                0.5,
                "No patient with both\ntumor and normal tissue",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax2.set_title("Signal Decay Comparison", fontsize=11, fontweight="bold")
            return fig

        # Get samples for this patient
        patient_samples_list = [
            s for s in signal_decay_dataset.samples if s.patient == selected_patient_id
        ]

        # Separate tumor and normal
        tumor_samples = [s for s in patient_samples_list if s.is_tumor]
        normal_samples = [s for s in patient_samples_list if not s.is_tumor]

        if not tumor_samples or not normal_samples:
            ax2.text(
                0.5,
                0.5,
                f"Patient {selected_patient_id} missing\ntumor or normal tissue",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax2.set_title("Signal Decay Comparison", fontsize=11, fontweight="bold")
            return fig

        # Pick one sample from each (prioritize different zones if available)
        tumor_sample = tumor_samples[0]
        normal_sample = normal_samples[0]

        # Get signal decay data
        tumor_signal, tumor_b = tumor_sample.as_numpy()
        normal_signal, normal_b = normal_sample.as_numpy()

        # Normalize to b=0 (first value)
        tumor_signal_norm = tumor_signal / tumor_signal[0]
        normal_signal_norm = normal_signal / normal_signal[0]

        # Plot signal decays
        ax2.plot(
            tumor_b,
            tumor_signal_norm,
            "o-",
            color="#d62728",
            linewidth=2.5,
            markersize=7,
            label="Tumor Tissue",
            alpha=0.9,
        )
        ax2.plot(
            normal_b,
            normal_signal_norm,
            "o-",
            color="#1f77b4",
            linewidth=2.5,
            markersize=7,
            label="Normal Tissue",
            alpha=0.9,
        )

        # Styling
        ax2.set_xlabel("b-value (s/mm²)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Normalized Signal", fontsize=11, fontweight="bold")
        ax2.set_title("Signal Decay Comparison", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc="upper right", framealpha=0.95)
        ax2.set_xlim([tumor_b[0] - 50, tumor_b[-1] + 50])
        ax2.set_ylim([0, 1.05])

        # Add patient info text box
        tumor_region = tumor_sample.a_region.upper()
        normal_region = normal_sample.a_region.upper()
        ggg = tumor_sample.ggg if tumor_sample.ggg is not None else "N/A"

        info_text = f"Patient: {selected_patient_id}\nTumor: {tumor_region} | Normal: {normal_region}\nGGG(PZ): {ggg}"
        ax2.text(
            0.02,
            0.02,
            info_text,
            transform=ax2.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor="black",
                linewidth=1.0,
            ),
        )

        plt.tight_layout()
        return fig

    export_ismrm_plot(
        fig_generator, output_path, width_px=width_px, height_px=height_px, dpi=dpi
    )


def create_ismrm_combined_prostate_signal_snr_spectrum(
    spectra_dataset,
    signal_decay_dataset,
    prostate_image_path: str,
    output_path: str,
    patient_id: Optional[str] = None,
    max_realizations: int = 10,
    realization_idx: int = 0,
    width_px: int = 1800,
    height_px: int = 1800,
    dpi: int = 150,
) -> None:
    """
    Create ISMRM-compatible 2x2 combined figure with:
    - Top-left: Prostate MRI annotation image
    - Top-right: Signal decay curves (scatter only, no lines)
    - Bottom-left: SNR posterior across realizations
    - Bottom-right: Spectrum posterior for one realization

    Args:
        spectra_dataset: DiffusivitySpectraDataset (must contain simulation data with NUTS)
        signal_decay_dataset: SignalDecayDataset with BWH patient data
        prostate_image_path: Path to prostate MRI annotation image
        output_path: Path for PNG output
        patient_id: Specific patient to plot (if None, picks a random patient with both normal and tumor)
        max_realizations: Maximum number of realizations to include in SNR plot
        realization_idx: Which realization to show in spectrum subplot (default: 0 = first)
        width_px: Width in pixels (default 1800 for 1:1 aspect ratio)
        height_px: Height in pixels (default 1800 for 1:1 aspect ratio)
        dpi: DPI for rendering (default 150)
    """
    from PIL import Image
    import random
    from collections import defaultdict
    import arviz as az

    # Find patient with both tumor and normal tissue
    selected_patient_id = patient_id
    if selected_patient_id is None:
        patient_samples = defaultdict(list)
        for sample in signal_decay_dataset.samples:
            patient_samples[sample.patient].append(sample)

        # Filter patients with both tumor and normal
        valid_patients = []
        for pid, samples in patient_samples.items():
            has_tumor = any(s.is_tumor for s in samples)
            has_normal = any(not s.is_tumor for s in samples)
            if has_tumor and has_normal:
                valid_patients.append(pid)

        if valid_patients:
            selected_patient_id = random.choice(valid_patients)
            print(f"[ISMRM] Selected patient: {selected_patient_id}")

    def fig_generator(figsize):
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=figsize)

        # Create gridspec for 2x2 layout with custom spacing (tighter than before)
        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            wspace=0.15,  # Reduced from 0.25
            hspace=0.15,  # Reduced from 0.25
            left=0.06,
            right=0.98,
            top=0.97,
            bottom=0.06,
        )

        # Create axes
        ax_prostate = fig.add_subplot(gs[0, 0])  # Top-left
        ax_signal = fig.add_subplot(gs[0, 1])  # Top-right
        ax_snr = fig.add_subplot(gs[1, 0])  # Bottom-left
        ax_spectrum = fig.add_subplot(gs[1, 1])  # Bottom-right

        # === TOP-LEFT: Prostate MRI Image ===
        if os.path.exists(prostate_image_path):
            img = Image.open(prostate_image_path)
            ax_prostate.imshow(img)
            ax_prostate.axis("off")
            ax_prostate.set_title(
                "Prostate MRI with ROI Annotations", fontsize=10, fontweight="bold"
            )
        else:
            ax_prostate.text(0.5, 0.5, "Image not found", ha="center", va="center")
            ax_prostate.axis("off")

        # === TOP-RIGHT: Signal Decay (SCATTER ONLY, NO LINES) ===
        if selected_patient_id is not None:
            # Get samples for this patient
            patient_samples_list = [
                s
                for s in signal_decay_dataset.samples
                if s.patient == selected_patient_id
            ]

            # Separate tumor and normal
            tumor_samples = [s for s in patient_samples_list if s.is_tumor]
            normal_samples = [s for s in patient_samples_list if not s.is_tumor]

            if tumor_samples and normal_samples:
                # Pick one sample from each
                tumor_sample = tumor_samples[0]
                normal_sample = normal_samples[0]

                # Get signal decay data
                tumor_signal, tumor_b = tumor_sample.as_numpy()
                normal_signal, normal_b = normal_sample.as_numpy()

                # Normalize to b=0 (first value)
                tumor_signal_norm = tumor_signal / tumor_signal[0]
                normal_signal_norm = normal_signal / normal_signal[0]

                # Plot signal decays (SCATTER ONLY - no lines)
                ax_signal.scatter(
                    tumor_b,
                    tumor_signal_norm,
                    color="#d62728",
                    s=80,
                    label="Tumor Tissue",
                    alpha=0.9,
                    zorder=3,
                )
                ax_signal.scatter(
                    normal_b,
                    normal_signal_norm,
                    color="#1f77b4",
                    s=80,
                    label="Normal Tissue",
                    alpha=0.9,
                    zorder=3,
                )

                # Styling
                ax_signal.set_xlabel("b-value (s/mm²)", fontsize=10, fontweight="bold")
                ax_signal.set_ylabel(
                    "Normalized Signal", fontsize=10, fontweight="bold"
                )
                ax_signal.set_title(
                    "Signal Decay Comparison", fontsize=10, fontweight="bold"
                )
                ax_signal.grid(True, alpha=0.3)
                ax_signal.legend(fontsize=9, loc="upper right", framealpha=0.95)
                ax_signal.set_xlim([tumor_b[0] - 50, tumor_b[-1] + 50])
                ax_signal.set_ylim([0, 1.05])

                # Add patient info text box
                tumor_region = tumor_sample.a_region.upper()
                normal_region = normal_sample.a_region.upper()
                ggg = tumor_sample.ggg if tumor_sample.ggg is not None else "N/A"

                info_text = f"Patient: {selected_patient_id}\nTumor: {tumor_region} | Normal: {normal_region}\nGGG: {ggg}"
                ax_signal.text(
                    0.02,
                    0.02,
                    info_text,
                    transform=ax_signal.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=7,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="black",
                        linewidth=1.0,
                    ),
                )
            else:
                ax_signal.text(
                    0.5,
                    0.5,
                    f"Patient {selected_patient_id}\nmissing tumor or normal",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                ax_signal.set_title(
                    "Signal Decay Comparison", fontsize=10, fontweight="bold"
                )
        else:
            ax_signal.text(
                0.5,
                0.5,
                "No patient with both\ntumor and normal tissue",
                ha="center",
                va="center",
                fontsize=9,
            )
            ax_signal.set_title(
                "Signal Decay Comparison", fontsize=10, fontweight="bold"
            )

        # === BOTTOM-LEFT: SNR Posterior ===
        # Filter to NUTS spectra with sigma inference
        spectra_with_sigma = []
        for spectrum in spectra_dataset.spectra[:max_realizations]:
            if spectrum.inference_method == "nuts" and spectrum.inference_data:
                if os.path.exists(spectrum.inference_data):
                    try:
                        idata = az.from_netcdf(spectrum.inference_data)
                        if "sigma" in idata.posterior:
                            spectra_with_sigma.append(spectrum)
                    except:
                        pass

        if spectra_with_sigma:
            # Extract SNR posteriors
            all_snr_samples = []
            for spectrum in spectra_with_sigma:
                idata = az.from_netcdf(spectrum.inference_data)
                sigma_samples = idata.posterior["sigma"].values.flatten()
                snr_samples = 1.0 / sigma_samples
                all_snr_samples.append(snr_samples)

            # Get true SNR if available
            true_snr = spectra_with_sigma[0].data_snr
            n_realizations = len(spectra_with_sigma)

            # Create boxplot
            bp = ax_snr.boxplot(
                all_snr_samples,
                showfliers=False,
                showmeans=True,
                meanline=False,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
                medianprops=dict(color="blue", linewidth=2),
                boxprops=dict(linewidth=1.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
            )

            # Add ground truth line
            if true_snr:
                ax_snr.axhline(
                    true_snr,
                    color="darkgreen",
                    linestyle="--",
                    linewidth=2.5,
                    label="Ground Truth",
                    zorder=10,
                )

                # Calculate bias
                snr_means = [np.mean(samples) for samples in all_snr_samples]
                overall_mean = np.mean(snr_means)
                bias = overall_mean - true_snr
                rel_error = (bias / true_snr) * 100

                # Add statistics box (TOP LEFT)
                stats_text = f"Estimated SNR: {overall_mean:.1f}\nBias: {bias:+.1f} ({rel_error:+.1f}%)"
                ax_snr.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax_snr.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="black",
                        linewidth=1.0,
                    ),
                    zorder=15,
                )

            # Styling
            ax_snr.set_xlabel("Realization", fontsize=10, fontweight="bold")
            ax_snr.set_ylabel("SNR", fontsize=10, fontweight="bold")
            ax_snr.set_xticks(range(1, n_realizations + 1))
            ax_snr.set_xticklabels(
                [f"{i}" for i in range(1, n_realizations + 1)], fontsize=8
            )
            ax_snr.grid(True, alpha=0.3, axis="y")
            if true_snr:
                ax_snr.legend(loc="upper right", fontsize=8)
            ax_snr.set_title(
                "Joint Inference: SNR Across Multiple Realizations",
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax_snr.text(
                0.5,
                0.5,
                "No SNR inference data\n(Requires simulation with NUTS)",
                ha="center",
                va="center",
                fontsize=9,
            )
            ax_snr.set_title(
                "Joint Inference: SNR Across Multiple Realizations",
                fontsize=10,
                fontweight="bold",
            )

        # === BOTTOM-RIGHT: Spectrum Posterior ===
        if spectra_with_sigma and realization_idx < len(spectra_with_sigma):
            spectrum = spectra_with_sigma[realization_idx]

            # Get spectrum samples
            if spectrum.spectrum_samples is not None:
                samples = np.array(spectrum.spectrum_samples)
                diffusivities = np.array(spectrum.diffusivities)
                n_diff = len(diffusivities)

                # Create boxplot for spectrum
                bp2 = ax_spectrum.boxplot(
                    samples,
                    showfliers=False,
                    showmeans=True,
                    meanline=True,
                    labels=[f"{d:.2f}" for d in diffusivities],
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                )

                # Add MAP initialization if available
                if spectrum.spectrum_init is not None:
                    x_positions = np.arange(1, n_diff + 1)
                    ax_spectrum.plot(
                        x_positions,
                        spectrum.spectrum_init,
                        "r*",
                        label="Initial R (MAP)",
                        markersize=8,
                    )

                # Add true spectrum if available
                if spectrum.true_spectrum is not None:
                    true_spectrum = np.array(spectrum.true_spectrum)
                    x_positions = np.arange(1, n_diff + 1)
                    ax_spectrum.vlines(
                        x_positions,
                        0,
                        true_spectrum,
                        colors="blue",
                        linewidth=2.5,
                        label="True Spectrum",
                    )

                # Spectrum subplot styling
                ax_spectrum.set_xlabel(
                    r"Diffusivity ($\mu$m$^2$/ms)", fontsize=10, fontweight="bold"
                )
                ax_spectrum.set_ylabel(
                    "Relative Fraction", fontsize=10, fontweight="bold"
                )
                ax_spectrum.set_xticklabels(
                    [f"{d:.2f}" for d in diffusivities], rotation=45, fontsize=9
                )
                ax_spectrum.grid(True, alpha=0.3)
                ax_spectrum.set_title(
                    f"Joint Inference: Spectrum (Realization {realization_idx+1})",
                    fontsize=10,
                    fontweight="bold",
                )

                if (
                    spectrum.spectrum_init is not None
                    or spectrum.true_spectrum is not None
                ):
                    ax_spectrum.legend(fontsize=8, loc="upper left")
            else:
                ax_spectrum.text(
                    0.5, 0.5, "No spectrum samples", ha="center", va="center"
                )
        else:
            ax_spectrum.text(0.5, 0.5, "No spectrum data", ha="center", va="center")

        return fig

    export_ismrm_plot(
        fig_generator, output_path, width_px=width_px, height_px=height_px, dpi=dpi
    )


def create_all_ismrm_exports(
    spectra_dataset,
    results_dict: Dict[str, List[Dict]],
    regions: Dict[str, List],
    output_dir: str,
    include_snr: bool = False,
    signal_decay_dataset=None,
    prostate_image_path: Optional[str] = None,
) -> None:
    """
    Create all ISMRM-compatible exports for abstract submission.

    Args:
        spectra_dataset: DiffusivitySpectraDataset
        results_dict: Biomarker classification results
        regions: Regional spectra groupings
        output_dir: Directory to save exports
        include_snr: Include SNR inference validation plot (for simulations)
        signal_decay_dataset: SignalDecayDataset (for prostate + signal decay figure)
        prostate_image_path: Path to prostate MRI image (for prostate + signal decay figure)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("CREATING ISMRM ABSTRACT FIGURES")
    print("=" * 60)
    print("Requirements: 900×600px PNG (3:2), <1MB, simple/uncluttered")

    # Determine number of figures
    n_figures = 3
    if include_snr:
        n_figures += 1
    if signal_decay_dataset is not None and prostate_image_path is not None:
        n_figures += 1

    print(f"Creating {n_figures} figures for abstract submission")

    fig_num = 1

    # 1. Regional spectra - shows tissue differences with adaptive y-scaling
    print(f"\n[{fig_num}/{n_figures}] Regional spectra (4 subplots)...")
    create_ismrm_averaged_spectra(regions, output_dir)
    fig_num += 1

    # 2. Combined ROC curves - PZ, TZ, and GGG in single figure
    print(f"\n[{fig_num}/{n_figures}] Combined ROC curves (PZ, TZ, GGG(PZ))...")
    output_path = os.path.join(output_dir, "roc_combined_ismrm.png")
    create_ismrm_roc_combined(results_dict, output_path)
    fig_num += 1

    # 3. LOOCV-based feature importance - actual features used in CV
    print(f"\n[{fig_num}/{n_figures}] LOOCV Feature Importance (actual CV features)...")
    output_path = os.path.join(output_dir, "loocv_feature_importance_ismrm.png")
    from spectra_estimation_dmri.biomarkers.biomarker_viz import (
        create_loocv_feature_importance_plot,
    )

    create_loocv_feature_importance_plot(results_dict, output_path)
    fig_num += 1

    # Also create old combined feature importance for comparison (optional)
    print(f"  [Optional] Creating full-dataset feature importance for comparison...")
    output_path_old = os.path.join(output_dir, "combined_feature_importance_ismrm.png")
    create_ismrm_combined_feature_importance(results_dict, output_path_old)

    # 4. SNR + Spectrum combined (optional, for simulation studies)
    if include_snr:
        print(f"\n[{fig_num}/{n_figures}] SNR + Spectrum combined inference...")
        output_path = os.path.join(output_dir, "snr_and_spectrum_combined_ismrm.png")
        try:
            create_ismrm_snr_and_spectrum_combined(
                spectra_dataset,
                output_path,
                max_realizations=10,
                realization_idx=0,
            )
        except Exception as e:
            print(f"  [WARNING] Could not create SNR + Spectrum plot: {e}")
            print(f"  (This is expected for BWH data - requires simulation)")
        fig_num += 1

    # 5. Prostate MRI + Signal Decay (optional, for BWH data)
    if signal_decay_dataset is not None and prostate_image_path is not None:
        print(f"\n[{fig_num}/{n_figures}] Prostate MRI + Signal Decay...")
        output_path = os.path.join(output_dir, "prostate_signal_decay_ismrm.png")
        try:
            create_ismrm_prostate_and_signal_decay(
                signal_decay_dataset,
                prostate_image_path,
                output_path,
            )
        except Exception as e:
            print(f"  [WARNING] Could not create Prostate + Signal Decay plot: {e}")
            import traceback

            traceback.print_exc()
        fig_num += 1

    print(f"\n✓ ISMRM figures saved to: {output_dir}/")
    print(f"  {n_figures} figure(s) created:")
    print(
        f"    1. averaged_spectra_ismrm.png - Regional tissue differences (1800×1800px)"
    )
    print(
        f"    2. roc_combined_ismrm.png - ROC curves (PZ, TZ, GGG(PZ) combined, 1800×1800px)"
    )
    print(
        f"    3. combined_feature_importance_ismrm.png - Feature analysis (all tasks, 1500×1500px)"
    )
    if include_snr:
        print(
            f"    4. snr_and_spectrum_combined_ismrm.png - SNR inference validation (1800×900px)"
        )
    if signal_decay_dataset is not None and prostate_image_path is not None:
        print(
            f"    {fig_num-1}. prostate_signal_decay_ismrm.png - Prostate MRI + Signal Decay (1800×900px)"
        )
    print(f"\nAll figures use consistent square dimensions for publication")
