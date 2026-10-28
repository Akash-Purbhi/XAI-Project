# -*- coding: utf-8 -*-
"""
Script 07: Master SHAP Meta-Explainability Benchmark Runner (Gap 6)
Runs TreeSHAP attribution across diverse classification and regression datasets,
aggregates meta-feature share (Domain vs Uncertainty Disagreement), and produces
publication-ready cross-dataset summary figures.
"""

import os
import sys
from pathlib import Path
import argparse
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Add repository root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec6 = importlib.util.spec_from_file_location("shap_exp", str(PROJECT_ROOT / "scripts" / "06_analyze_allocator_shap.py"))
shap_exp = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(shap_exp)

# All 24 benchmark datasets covering 12 Classification and 12 Regression domains
DEFAULT_SHAP_DATASETS = [
    # 12 Classification Datasets
    'Wine',
    'Phoneme',
    'KDDIPUMS',
    'EyeMovements',
    'Pol',
    'Bank',
    'MagicTelescope',
    'Credit',
    'House16H',
    'California',
    'Electricity',
    'Jannis',
    # 12 Regression Datasets
    'Wine_R',
    'CPU_R',
    'Isolet_R',
    'BrazilianHouses_R',
    'Sulfur_R',
    'Ailerons_R',
    'MiamiHousing_R',
    'Pol_R',
    'Elevators_R',
    'BikeSharingDemand_R',
    'Fifa_R',
    'California_R'
]


def aggregate_all_shap_results(results_dir="results/shap", out_csv="results/shap_meta_summary.csv", out_fig="results/figures/gap6_meta_explainability_summary.png"):
    base_path = Path(results_dir)
    if not base_path.exists():
        print(f"[ERROR] Directory {results_dir} does not exist.")
        return

    summaries = []
    importance_dfs = []

    for d in base_path.iterdir():
        if d.is_dir():
            meta_file = d / "meta_share_summary.csv"
            imp_file = d / "shap_importance_comparison.csv"
            if meta_file.exists():
                df_m = pd.read_csv(meta_file)
                summaries.append(df_m)
            if imp_file.exists():
                df_i = pd.read_csv(imp_file)
                df_i['Dataset'] = d.name
                importance_dfs.append(df_i)

    if not summaries:
        print("[WARN] No SHAP meta summaries found.")
        return

    master_meta_df = pd.concat(summaries, ignore_index=True)
    master_meta_df.to_csv(out_csv, index=False)
    print(f"[OK] Master SHAP meta summary saved to {out_csv}")

    # Plot Cross-Dataset Meta-Feature Share
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    datasets = master_meta_df['Dataset'].tolist()
    fig_height = max(6.5, len(datasets) * 0.38)
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_height))
    plt.rcParams.update({'font.size': 10})

    # Plot 1: Stacked Bar Chart of Feature Type Share
    domain_share = master_meta_df['Original_Domain_Feature_Share'].to_numpy()
    meta_share = master_meta_df['Original_Disagreement_Confidence_Share'].to_numpy()

    y_pos = np.arange(len(datasets))
    axes[0].barh(y_pos, meta_share, label='Model Disagreement & Predictions', color='#2b6cb0', alpha=0.85)
    axes[0].barh(y_pos, domain_share, left=meta_share, label='Raw Domain Features (X)', color='#e0a800', alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(datasets, fontsize=9)
    axes[0].set_xlabel("Attribution Share of Routing Decision (%)", fontsize=11)
    axes[0].set_title("Who Drives Routing: Model Disagreement vs. Domain Features?", fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Plot 2: Top Drivers Across Datasets
    if importance_dfs:
        all_imp = pd.concat(importance_dfs, ignore_index=True)
        top_driver_counts = all_imp.groupby('Feature')['Original_Mean_SHAP'].mean().sort_values(ascending=False).head(10).iloc[::-1]
        
        y_pos2 = np.arange(len(top_driver_counts))
        axes[1].barh(y_pos2, top_driver_counts.values, color='#48bb78', alpha=0.85)
        axes[1].set_yticks(y_pos2)
        axes[1].set_yticklabels(top_driver_counts.index, fontsize=10)
        axes[1].set_xlabel("Mean |SHAP Value| Across Benchmark Datasets", fontsize=11)
        axes[1].set_title("Universal Top Drivers of Model Allocation", fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[OK] Cross-dataset SHAP summary figure saved to {out_fig}")

    print("\n" + "=" * 80)
    print("META-EXPLAINABILITY BENCHMARK FINDINGS (GAP 6)")
    print("=" * 80)
    print(master_meta_df[['Dataset', 'Problem_Type', 'Original_Disagreement_Confidence_Share', 'Original_Domain_Feature_Share', 'Top_Driver_1']].to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Master SHAP Benchmark Runner")
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_SHAP_DATASETS, help="Datasets to evaluate")
    parser.add_argument('--use_gpu', action='store_true', default=True, help="Use GPU acceleration")
    parser.add_argument('--quick_test', action='store_true', help="Run on Wine and Phoneme only")
    parser.add_argument('--force', action='store_true', help="Re-run datasets even if results exist")
    args = parser.parse_args()

    targets = ['Wine', 'Phoneme'] if args.quick_test else args.datasets
    print("=" * 75)
    print(f"MASTER SHAP META-EXPLAINABILITY BENCHMARK: {len(targets)} DATASETS")
    print(f"Datasets: {', '.join(targets)}")
    print("=" * 75)

    start_time = time()
    completed = []

    for idx, ds in enumerate(targets, 1):
        out_meta = Path(f"results/shap/{ds}/meta_share_summary.csv")
        out_imp = Path(f"results/shap/{ds}/shap_importance_comparison.csv")
        if not args.force and out_meta.exists() and out_imp.exists():
            print(f"\n[{idx}/{len(targets)}] {ds} already completed. Skipping (use --force to overwrite)...")
            completed.append((ds, 0.0))
            continue

        print(f"\n[{idx}/{len(targets)}] Running SHAP for {ds}...")
        t0 = time()
        try:
            shap_exp.run_shap_analysis(dataset_name=ds, use_gpu=args.use_gpu)
            elapsed = time() - t0
            completed.append((ds, elapsed))
        except Exception as e:
            print(f"[ERROR] Failed {ds}: {e}")

    # Aggregate & Plot
    print("\n" + "=" * 75)
    print("Aggregating SHAP Findings across all evaluated datasets...")
    aggregate_all_shap_results()
    total_min = (time() - start_time) / 60.0
    print(f"\nAll SHAP benchmarks finished in {total_min:.2f} minutes ({len(completed)}/{len(targets)} datasets).")


if __name__ == "__main__":
    main()
