# -*- coding: utf-8 -*-
"""
Generate publication-quality 2-panel comparative figures across both benchmark datasets (Wine and Phoneme).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_multi_dataset_figures():
    csv_path = os.path.join('results', 'csv', 'summary_before_vs_after.csv')
    df = pd.read_csv(csv_path)
    fig_dir = os.path.join('results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    datasets = ['Wine', 'Phoneme']
    methods = [('Original EEG', '#d95f02', 'o'), ('Proposed SR-EEG', '#1b9e77', 's')]
    
    # FIG 1: Allocation Flip Rate (AFR) vs Epsilon (q = 0.50)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for method, color, marker in methods:
            sub = df[(df['Dataset'] == ds) & (df['Method'] == method) & (df['q'] == 0.5)]
            ax.errorbar(sub['Epsilon'], sub['AFR_mean'], yerr=sub['AFR_std'], label=method,
                        color=color, marker=marker, linewidth=2.2, capsize=4)
        ax.set_title(f"({chr(97+idx)}) {ds} (q = 0.50)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Perturbation Strength (Epsilon)", fontsize=11)
        ax.set_ylabel("Allocation Flip Rate (%)", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.suptitle("Allocation Flip Rate vs. Perturbation Magnitude Across Datasets", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_afr_vs_epsilon.png"), bbox_inches='tight')
    plt.close()
    
    # FIG 2: Prediction Consistency vs Epsilon (q = 0.50)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for method, color, marker in methods:
            sub = df[(df['Dataset'] == ds) & (df['Method'] == method) & (df['q'] == 0.5)]
            ax.errorbar(sub['Epsilon'], sub['Prediction_Consistency_mean'], yerr=sub['Prediction_Consistency_std'],
                        label=method, color=color, marker=marker, linewidth=2.2, capsize=4)
        ax.set_title(f"({chr(97+idx)}) {ds} (q = 0.50)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Perturbation Strength (Epsilon)", fontsize=11)
        ax.set_ylabel("Prediction Consistency (%)", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.suptitle("Prediction Consistency vs. Perturbation Magnitude Across Datasets", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_prediction_consistency.png"), bbox_inches='tight')
    plt.close()

    # FIG 3: Task Accuracy on Perturbed Inputs (q = 0.50)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        for method, color, marker in methods:
            sub = df[(df['Dataset'] == ds) & (df['Method'] == method) & (df['q'] == 0.5)]
            ax.errorbar(sub['Epsilon'], sub['Perturbed_Accuracy_mean'], yerr=sub['Perturbed_Accuracy_std'],
                        label=method, color=color, marker=marker, linewidth=2.2, capsize=4)
        ax.set_title(f"({chr(97+idx)}) {ds} (q = 0.50)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Perturbation Strength (Epsilon)", fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.suptitle("Task Accuracy on Perturbed Test Inputs Across Datasets", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_task_accuracy_vs_epsilon.png"), bbox_inches='tight')
    plt.close()

    # FIG 4: Allocation Flip Rate across Explainability Levels q (Epsilon = 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        sub_eps = df[(df['Dataset'] == ds) & (df['Epsilon'] == 0.05)]
        for method, color, marker in methods:
            sub = sub_eps[sub_eps['Method'] == method]
            ax.errorbar(sub['q'], sub['AFR_mean'], yerr=sub['AFR_std'], label=method,
                        color=color, marker=marker, linewidth=2.2, capsize=4)
        ax.set_title(f"({chr(97+idx)}) {ds} (Epsilon = 0.05)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Explainability Level (q)", fontsize=11)
        ax.set_ylabel("Allocation Flip Rate (%)", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.suptitle("Allocation Flip Rate Across Explainability Levels q at Epsilon = 0.05", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_afr_across_q.png"), bbox_inches='tight')
    plt.close()

    # FIG 5: Stability-Performance Trade-off
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        sub_all = df[(df['Dataset'] == ds) & (df['Epsilon'] > 0.0)]
        for method, color, marker in methods:
            sub = sub_all[sub_all['Method'] == method]
            ax.scatter(sub['AFR_mean'], sub['Accuracy_Drop_mean'], label=method,
                       color=color, marker=marker, s=65, alpha=0.85)
        ax.set_title(f"({chr(97+idx)}) {ds} (All Epsilon > 0)", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Allocation Flip Rate (%) [Lower is more stable]", fontsize=11)
        ax.set_ylabel("Accuracy Drop (%) [Lower is more robust]", fontsize=11)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.suptitle("Stability-Performance Trade-off Across Datasets", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_stability_performance_tradeoff.png"), bbox_inches='tight')
    plt.close()
    print("Multi-dataset publication figures successfully rendered to results/figures/")

if __name__ == '__main__':
    generate_multi_dataset_figures()
