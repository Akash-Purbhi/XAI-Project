# -*- coding: utf-8 -*-
"""
Script 04: Aggregate Cross-Dataset Results & Generate Gap 3 Multi-Panel Figures
Collects stability metrics from all evaluated datasets in results/stability/
and exports publication-grade comparative visualizations and master CSV.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def aggregate_all_stability_results(results_dir="results/stability", out_csv="results/stability_master.csv", out_fig="results/figures/gap3_cross_dataset_stability.png"):
    base_path = Path(results_dir)
    if not base_path.exists():
        print(f"[ERROR] Directory {results_dir} does not exist.")
        return

    dataset_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not dataset_dirs:
        print(f"[WARN] No dataset results found in {results_dir}")
        return

    all_detailed = []
    all_summaries = []

    for d in dataset_dirs:
        det_file = d / "detailed_per_seed_results.csv"
        if det_file.exists():
            df = pd.read_csv(det_file)
            all_detailed.append(df)

    if not all_detailed:
        print("[WARN] No detailed_per_seed_results.csv found.")
        return

    master_df = pd.concat(all_detailed, ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    master_df.to_csv(out_csv, index=False)
    print(f"[OK] Master stability results saved to {out_csv} ({len(master_df)} rows)")

    # Aggregate across datasets
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # 1. Classification datasets
    cls_df = master_df[master_df['Problem_Type'] == 'classification']
    reg_df = master_df[master_df['Problem_Type'] == 'regression']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    plt.rcParams.update({'font.size': 11})

    # Plot 1: Classification AFR vs Epsilon
    if not cls_df.empty:
        cls_grp = cls_df.groupby(['Epsilon', 'Method'])['AFR'].mean().unstack()
        cls_std = cls_df.groupby(['Epsilon', 'Method'])['AFR'].std().unstack()
        
        if 'Original EEG' in cls_grp.columns:
            axes[0].errorbar(cls_grp.index, cls_grp['Original EEG'], yerr=cls_std['Original EEG'], fmt='o--', color='#d9534f', label='Original EEG (Baseline)', capsize=4, linewidth=2)
        if 'SR-EEG (Proposed)' in cls_grp.columns:
            axes[0].errorbar(cls_grp.index, cls_grp['SR-EEG (Proposed)'], yerr=cls_std['SR-EEG (Proposed)'], fmt='s-', color='#2b6cb0', label='SR-EEG (Proposed Robust)', capsize=4, linewidth=2.5)
        
        n_cls = cls_df['Dataset'].nunique()
        axes[0].set_title(f"Classification Datasets (N={n_cls}): AFR vs Noise ε", fontweight='bold', fontsize=12)
        axes[0].set_xlabel("Noise Perturbation Magnitude ε")
        axes[0].set_ylabel("Allocation Flip Rate (%) [Lower is Better]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # Plot 2: Regression AFR vs Epsilon
    if not reg_df.empty:
        reg_grp = reg_df.groupby(['Epsilon', 'Method'])['AFR'].mean().unstack()
        reg_std = reg_df.groupby(['Epsilon', 'Method'])['AFR'].std().unstack()
        
        if 'Original EEG' in reg_grp.columns:
            axes[1].errorbar(reg_grp.index, reg_grp['Original EEG'], yerr=reg_std['Original EEG'], fmt='o--', color='#d9534f', label='Original EEG (Baseline)', capsize=4, linewidth=2)
        if 'SR-EEG (Proposed)' in reg_grp.columns:
            axes[1].errorbar(reg_grp.index, reg_grp['SR-EEG (Proposed)'], yerr=reg_std['SR-EEG (Proposed)'], fmt='s-', color='#2b6cb0', label='SR-EEG (Proposed Robust)', capsize=4, linewidth=2.5)
        
        n_reg = reg_df['Dataset'].nunique()
        axes[1].set_title(f"Regression Datasets (N={n_reg}): AFR vs Noise ε", fontweight='bold', fontsize=12)
        axes[1].set_xlabel("Noise Perturbation Magnitude ε")
        axes[1].set_ylabel("Allocation Flip Rate (%) [Lower is Better]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[OK] Multi-panel comparative figure saved to {out_fig}")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("CROSS-DATASET STABILITY SUMMARY (GAP 3: ORIGINAL VS SR-EEG)")
    print("=" * 80)
    pivot_afr = master_df.groupby(['Dataset', 'Method', 'Epsilon'])['AFR'].mean().unstack()
    print(pivot_afr.round(2))
    print("=" * 80)

if __name__ == "__main__":
    aggregate_all_stability_results()
