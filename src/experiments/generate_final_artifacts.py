"""
Generate all Phase 1 tables and figures from completed raw results.

Reads all JSON summaries and curve CSVs from results/raw/,
computes cross-replicate statistics, generates Tables 1-4,
and plots Figures 1-7.
"""

import os
import glob
import json
import numpy as np
import pandas as pd

from src.data import DATASET_REGISTRY, load_dataset
from src.experiments.aggregation import (
    build_table1_dataset_characteristics,
    aggregate_replicate_results,
)
from src.utils.plotting import (
    plot_performance_vs_explainability,
    plot_eeg_vs_component_models,
    plot_paper_vs_replication_comparison,
)
from src.utils.logging import setup_logger


def generate_all_artifacts(
    raw_dir: str = "results/raw",
    tables_dir: str = "tables",
    figures_dir: str = "figures",
    processed_dir: str = "results/processed",
):
    logger = setup_logger("generate_final_artifacts")
    logger.info("=== Generating All Phase 1 Replication Artifacts ===")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("results/paper_comparison", exist_ok=True)
    os.makedirs(os.path.join(figures_dir, "performance_explainability"), exist_ok=True)
    os.makedirs(os.path.join(figures_dir, "dataset_comparisons"), exist_ok=True)
    os.makedirs(os.path.join(figures_dir, "paper_comparison"), exist_ok=True)

    # 1. Load all raw JSON run summaries
    json_pattern = os.path.join(raw_dir, "*_summary.json")
    summary_files = sorted(glob.glob(json_pattern))
    logger.info(f"Found {len(summary_files)} run summaries in {raw_dir}")

    all_runs = []
    for fpath in summary_files:
        with open(fpath, "r", encoding="utf-8") as f:
            all_runs.append(json.load(f))

    datasets = sorted(list(set(r["dataset"] for r in all_runs)))
    logger.info(f"Datasets represented: {datasets}")
    for d in datasets:
        n_seeds = len([r for r in all_runs if r["dataset"] == d])
        logger.info(f"  {d}: {n_seeds} seeds")

    # 2. Build Table 1: Dataset Characteristics
    logger.info("Generating Table 1...")
    meta_dict = {}
    for d_name in datasets:
        _, _, meta = load_dataset(d_name)
        meta_dict[d_name] = meta
    table1_df = build_table1_dataset_characteristics(meta_dict, output_dir=tables_dir)
    print("\n--- TABLE 1: Dataset Characteristics ---")
    print(table1_df.to_string(index=False))

    # 3. Aggregate into Tables 2, 3, 4
    logger.info("Aggregating into Tables 2, 3, 4...")
    runs_df, table2_df, table3_df, table4_df = aggregate_replicate_results(
        all_runs=all_runs,
        output_dir=processed_dir,
        tables_dir=tables_dir,
    )

    print("\n--- TABLE 2: Component Model Performance ---")
    print(table2_df.to_string(index=False))

    print("\n--- TABLE 3: Consolidated EEG Metrics (Mean +/- Std) ---")
    print(table3_df.to_string(index=False))

    print("\n--- TABLE 4: Paper vs Replication Comparison ---")
    print(table4_df.to_string(index=False))

    # 4. Generate Figures 1-5 (Trade-off curves)
    fig_mapping = {
        "Wine": (os.path.join(figures_dir, "performance_explainability", "fig1_wine_tradeoff.png"), "Figure 1"),
        "Bank": (os.path.join(figures_dir, "performance_explainability", "fig2_bank_tradeoff.png"), "Figure 2"),
        "PolR": (os.path.join(figures_dir, "performance_explainability", "fig3_polr_tradeoff.png"), "Figure 3"),
        "SuperconductR": (os.path.join(figures_dir, "performance_explainability", "fig4_superconductr_tradeoff.png"), "Figure 4"),
        "BrazilianHousesR": (os.path.join(figures_dir, "performance_explainability", "fig5_brazilianhousesr_tradeoff.png"), "Figure 5"),
    }

    for d_name in datasets:
        curve_pattern = os.path.join(raw_dir, f"{d_name}_seed*_curve.csv")
        curve_files = sorted(glob.glob(curve_pattern))
        if not curve_files:
            continue

        curve_list = [pd.read_csv(cf) for cf in curve_files]
        q_grid = curve_list[0]["q"].to_numpy()

        eeg_stack = np.array([c["perf_eeg"].to_numpy() for c in curve_list])
        rand_stack = np.array([c["perf_random"].to_numpy() for c in curve_list])
        orac_stack = np.array([c["perf_oracle"].to_numpy() for c in curve_list])

        mean_eeg = np.mean(eeg_stack, axis=0)
        std_eeg = np.std(eeg_stack, axis=0) if len(curve_list) > 1 else None

        mean_rand = np.mean(rand_stack, axis=0)
        std_rand = np.std(rand_stack, axis=0) if len(curve_list) > 1 else None

        mean_orac = np.mean(orac_stack, axis=0)
        std_orac = np.std(orac_stack, axis=0) if len(curve_list) > 1 else None

        d_runs = [r for r in all_runs if r["dataset"] == d_name]
        mean_gb = np.mean([r["metrics"]["GB_Perf"] / 100.0 for r in d_runs])
        mean_bb = np.mean([r["metrics"]["BB_Perf"] / 100.0 for r in d_runs])

        fig_path, fig_title = fig_mapping.get(
            d_name, (os.path.join(figures_dir, f"{d_name}_curve.png"), d_name)
        )
        plot_performance_vs_explainability(
            dataset_name=d_name,
            q_grid=q_grid,
            perf_random=mean_rand,
            perf_oracle=mean_orac,
            perf_eeg=mean_eeg,
            perf_gb=mean_gb,
            perf_bb=mean_bb,
            output_path=fig_path,
            perf_eeg_std=std_eeg,
            perf_random_std=std_rand,
            perf_oracle_std=std_orac,
        )
        logger.info(f"Saved {fig_title} to {fig_path}")

    # 5. Generate Figure 6: Bar chart comparison
    fig6_path = os.path.join(figures_dir, "dataset_comparisons", "fig6_eeg_vs_component_models.png")
    plot_eeg_vs_component_models(runs_df, fig6_path)
    logger.info(f"Saved Figure 6 to {fig6_path}")

    # 6. Generate Figure 7: Paper vs Replication scatter
    fig7_path = os.path.join(figures_dir, "paper_comparison", "fig7_paper_vs_replication.png")
    plot_paper_vs_replication_comparison(table4_df, fig7_path)
    logger.info(f"Saved Figure 7 to {fig7_path}")

    logger.info("\n=== All Artifacts Generated Successfully ===")
    return {
        "table1": table1_df,
        "table2": table2_df,
        "table3": table3_df,
        "table4": table4_df,
    }


if __name__ == "__main__":
    generate_all_artifacts()
