"""
Publication-quality plotting functions for EEG replication.

Produces research-standard figures matching AAAI 2024 style:
Figures 1-5: Performance vs Explainability curves (Random, Oracle, Learned EEG)
Figure 6: EEG vs Best Component Model across the 5 datasets
Figure 7: Paper vs Replication Metrics Comparison
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Consistent color scheme matching the paper's style
COLOR_RANDOM = "#2b5c8f"    # Deep Blue
COLOR_ORACLE = "#d95f02"    # Orange / Amber
COLOR_EEG = "#2ca02c"       # Green
COLOR_GB = "#7570b3"        # Purple
COLOR_BB = "#e7298a"        # Magenta


def plot_performance_vs_explainability(
    dataset_name: str,
    q_grid: np.ndarray,
    perf_random: np.ndarray,
    perf_oracle: np.ndarray,
    perf_eeg: np.ndarray,
    perf_gb: float,
    perf_bb: float,
    output_path: str,
    perf_eeg_std: Optional[np.ndarray] = None,
    perf_random_std: Optional[np.ndarray] = None,
    perf_oracle_std: Optional[np.ndarray] = None,
) -> None:
    """
    Plot Explainability (q) vs Sufficient Performance trade-off curve.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Plot curves
    ax.plot(q_grid, perf_oracle, label="Oracle Allocation", color=COLOR_ORACLE, linewidth=2.0)
    if perf_oracle_std is not None:
        ax.fill_between(q_grid, perf_oracle - perf_oracle_std, perf_oracle + perf_oracle_std,
                        color=COLOR_ORACLE, alpha=0.15)

    ax.plot(q_grid, perf_eeg, label="Learned EEG", color=COLOR_EEG, linewidth=2.2)
    if perf_eeg_std is not None:
        ax.fill_between(q_grid, perf_eeg - perf_eeg_std, perf_eeg + perf_eeg_std,
                        color=COLOR_EEG, alpha=0.2)

    ax.plot(q_grid, perf_random, label="Random Allocation", color=COLOR_RANDOM, linestyle="--", linewidth=1.8)
    if perf_random_std is not None:
        ax.fill_between(q_grid, perf_random - perf_random_std, perf_random + perf_random_std,
                        color=COLOR_RANDOM, alpha=0.15)

    # Component model baselines
    ax.axhline(perf_gb, color=COLOR_GB, linestyle=":", alpha=0.7, label=f"Glass-Box ({perf_gb*100:.1f}%)")
    ax.axhline(perf_bb, color=COLOR_BB, linestyle=":", alpha=0.7, label=f"Black-Box ({perf_bb*100:.1f}%)")

    ax.set_xlabel("Explainability Level $q$ (Proportion Allocated to Glass-Box)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Sufficient Ensemble Performance", fontsize=11, fontweight="medium")
    ax.set_title(f"{dataset_name} — Performance vs. Explainability Trade-Off", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_eeg_vs_component_models(
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Figure 6: Bar comparison of Glass-Box, Black-Box, and Max EEG across datasets.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    datasets = summary_df["dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.25

    gb_vals = [summary_df[summary_df["dataset"] == d]["GB_Perf"].mean() for d in datasets]
    bb_vals = [summary_df[summary_df["dataset"] == d]["BB_Perf"].mean() for d in datasets]
    eeg_vals = [summary_df[summary_df["dataset"] == d]["Max_Acc"].mean() for d in datasets]

    rects1 = ax.bar(x - width, gb_vals, width, label="Glass-Box Model", color=COLOR_GB, alpha=0.85)
    rects2 = ax.bar(x, bb_vals, width, label="Black-Box Model", color=COLOR_BB, alpha=0.85)
    rects3 = ax.bar(x + width, eeg_vals, width, label="Learned EEG (Max)", color=COLOR_EEG, alpha=0.85)

    ax.set_ylabel("Performance (%)", fontsize=11, fontweight="medium")
    ax.set_title("Component Model Performance vs. Learned EEG Across Datasets", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10, rotation=15)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_paper_vs_replication_comparison(
    comparison_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Figure 7: Scatter plot comparing Original Paper values vs. Our Replication values.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Filter numeric values
    df_clean = comparison_df.dropna(subset=["Paper", "Ours"]).copy()

    # Metrics to highlight with markers
    metrics = df_clean["Metric"].unique()
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "<"]

    for idx, m in enumerate(metrics):
        sub = df_clean[df_clean["Metric"] == m]
        ax.scatter(
            sub["Paper"],
            sub["Ours"],
            label=m,
            marker=markers[idx % len(markers)],
            s=70,
            alpha=0.85,
        )

    # 45-degree identity line (perfect agreement)
    lims = [-25, 105]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Identity (Perfect Match)")

    ax.set_xlabel("Original Paper Reported (%)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Our Replication Result (%)", fontsize=11, fontweight="medium")
    ax.set_title("Original Paper vs. Replication Metric Agreement", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
