"""
Full Phase 1 Replication Experiment Orchestrator.

Runs all 5 datasets across 5 replicate seeds (0, 1, 2, 3, 4),
aggregates results, generates all 7 publication figures, and outputs Tables 1 to 4.
"""

import os
import sys
import argparse
import yaml
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.utils.logging import setup_logger
from src.utils.plotting import (
    plot_performance_vs_explainability,
    plot_eeg_vs_component_models,
    plot_paper_vs_replication_comparison,
)
from src.data import DATASET_REGISTRY, load_dataset
from src.experiments.run_single import run_single_dataset
from src.experiments.aggregation import (
    build_table1_dataset_characteristics,
    aggregate_replicate_results,
)


def run_full_phase1(
    config_path: str = "configs/phase1.yaml",
    datasets: List[str] = None,
    seeds: List[int] = None,
    q_step: float = None,
    random_repetitions: int = None,
) -> Dict[str, Any]:
    """
    Run full Phase 1 replication pipeline.
    """
    logger = setup_logger("phase1_full_experiment")
    logger.info("=================================================================")
    logger.info("PHASE 1: REPLICATION OF AAAI 2024 EEG PAPER")
    logger.info("=================================================================")

    # Load configuration
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    target_datasets = datasets or [d["name"] for d in cfg.get("datasets", [
        {"name": "Wine"}, {"name": "Bank"}, {"name": "PolR"},
        {"name": "SuperconductR"}, {"name": "BrazilianHousesR"}
    ])]
    target_seeds = seeds if seeds is not None else cfg.get("experiment", {}).get("seeds", [0, 1, 2, 3, 4])
    step = q_step if q_step is not None else cfg.get("experiment", {}).get("q_grid", {}).get("step", 0.01)
    n_rand = random_repetitions if random_repetitions is not None else cfg.get("experiment", {}).get("random_repetitions", 10)

    logger.info(f"Target Datasets: {target_datasets}")
    logger.info(f"Replicate Seeds: {target_seeds}")
    logger.info(f"q Grid Step: {step}")

    # Build Table 1: Dataset Metadata
    logger.info("Collecting dataset metadata for Table 1...")
    dataset_meta = {}
    for d_name in target_datasets:
        _, _, meta = load_dataset(d_name)
        dataset_meta[d_name] = meta
    table1_df = build_table1_dataset_characteristics(dataset_meta, output_dir="tables")
    logger.info(f"Table 1 generated:\n{table1_df.to_string(index=False)}")

    # Run experiments across datasets and seeds
    all_runs = []
    curves_by_dataset = {d: [] for d in target_datasets}

    total_experiments = len(target_datasets) * len(target_seeds)
    exp_idx = 0

    for d_name in target_datasets:
        for seed in target_seeds:
            exp_idx += 1
            logger.info(f"\n>>> Running Experiment [{exp_idx}/{total_experiments}]: {d_name} (Seed {seed}) <<<")
            res = run_single_dataset(
                dataset_name=d_name,
                seed=seed,
                q_step=step,
                random_repetitions=n_rand,
                output_dir="results/raw",
                logger=logger,
            )
            all_runs.append(res["run_summary"])
            curves_by_dataset[d_name].append(res["curve_df"])

    # Aggregate results into Tables 2, 3, 4
    logger.info("\nAggregating replicate results...")
    runs_df, table2_df, table3_df, table4_df = aggregate_replicate_results(
        all_runs=all_runs,
        output_dir="results/processed",
        tables_dir="tables",
    )

    logger.info(f"\nTable 2 (Component Models):\n{table2_df.to_string(index=False)}")
    logger.info(f"\nTable 3 (EEG Metrics Summary):\n{table3_df.to_string(index=False)}")
    logger.info(f"\nTable 4 (Paper vs Ours Comparison):\n{table4_df.to_string(index=False)}")

    # Generate Figures 1-5: Performance vs Explainability curves per dataset
    logger.info("\nGenerating publication figures 1 to 5...")
    fig_mapping = {
        "Wine": ("figures/performance_explainability/fig1_wine_tradeoff.png", "Figure 1"),
        "Bank": ("figures/performance_explainability/fig2_bank_tradeoff.png", "Figure 2"),
        "PolR": ("figures/performance_explainability/fig3_polr_tradeoff.png", "Figure 3"),
        "SuperconductR": ("figures/performance_explainability/fig4_superconductr_tradeoff.png", "Figure 4"),
        "BrazilianHousesR": ("figures/performance_explainability/fig5_brazilianhousesr_tradeoff.png", "Figure 5"),
    }

    for d_name, curve_list in curves_by_dataset.items():
        if not curve_list:
            continue
        # Average curves across replicates
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

        # Mean component baselines
        d_runs = [r for r in all_runs if r["dataset"] == d_name]
        mean_gb = np.mean([r["metrics"]["GB_Perf"] / 100.0 for r in d_runs])
        mean_bb = np.mean([r["metrics"]["BB_Perf"] / 100.0 for r in d_runs])

        fig_path, fig_title = fig_mapping.get(d_name, (f"figures/{d_name}_curve.png", d_name))
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

    # Generate Figure 6: Bar chart comparison
    logger.info("Generating Figure 6 (EEG vs Components)...")
    fig6_path = "figures/dataset_comparisons/fig6_eeg_vs_component_models.png"
    plot_eeg_vs_component_models(runs_df, fig6_path)
    logger.info(f"Saved Figure 6 to {fig6_path}")

    # Generate Figure 7: Paper vs Replication scatter
    logger.info("Generating Figure 7 (Paper vs Replication)...")
    fig7_path = "figures/paper_comparison/fig7_paper_vs_replication.png"
    plot_paper_vs_replication_comparison(table4_df, fig7_path)
    logger.info(f"Saved Figure 7 to {fig7_path}")

    logger.info("\n=================================================================")
    logger.info("PHASE 1 REPLICATION EXECUTION FINISHED SUCCESSFULLY")
    logger.info("All tables saved in tables/ and results/processed/")
    logger.info("All figures saved in figures/")
    logger.info("=================================================================")

    return {
        "runs_df": runs_df,
        "table1": table1_df,
        "table2": table2_df,
        "table3": table3_df,
        "table4": table4_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Phase 1 EEG Replication")
    parser.add_argument("--config", type=str, default="configs/phase1.yaml", help="Path to config")
    parser.add_argument("--datasets", nargs="+", default=None, help="Subset of datasets to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Replicate seeds")
    parser.add_argument("--q_step", type=float, default=None, help="q grid step")

    args = parser.parse_args()
    run_full_phase1(
        config_path=args.config,
        datasets=args.datasets,
        seeds=args.seeds,
        q_step=args.q_step,
    )
