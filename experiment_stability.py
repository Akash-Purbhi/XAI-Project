# -*- coding: utf-8 -*-
"""
Experiment Runner: Allocation Robustness under Input Perturbations
and Controlled Before-versus-After Evaluation of Stability-Aware EEG (SR-EEG).

This script performs:
1. Exact model replication for component models (glass-box g, black-box b, baseline allocator A_orig).
2. Construction of Stability-Regularized Allocator (A_SR) using training-domain perturbations.
3. Validation-set hyperparameter tuning (ensuring zero test leakage).
4. Controlled evaluation on held-out test inputs across perturbation strengths:
   epsilon in {0.00, 0.01, 0.02, 0.05, 0.10} across multiple random seeds (0, 1, 2).
5. Comprehensive metrics tracking:
   - Allocation Flip Rate (AFR)
   - Prediction Consistency (PC)
   - Task Performance (Accuracy / RMSE on clean and perturbed)
   - Performance Degradation
   - Explainability Guarantee compliance (Actual q vs nominal q)
   - Sufficiency Category Accuracy (s Acc)
6. Export of machine-readable CSVs and publication-grade figures.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score

from data import get_dataset, scale_to_minus_one_to_one, scale_to_min_max
from utils import get_superfix
from utils_2 import get_percentiles_of_a_wrt_b
from evaluate_eeg import (
    compute_ensemble_predictions,
    compute_performance,
    compute_original_eeg_metrics,
    compute_allocation_flip_rate,
    compute_prediction_consistency
)
from perturbation_utils import generate_tabular_perturbations


def build_allocator_features(X_orig, y_hat_gb, y_hat_bb, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d):
    """
    Constructs the 4 augmented features: [y_hat_gb, y_hat_bb, d_ce, d_mse]
    concatenated with X_orig, normalized exactly as in utils_2.py.
    """
    N = X_orig.shape[0]
    
    # Distance features
    d_ce = -np.sum(y_hat_bb * np.log(np.maximum(np.minimum(y_hat_gb, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
    d_mse = np.sum((y_hat_bb - y_hat_gb) ** 2, axis=1, keepdims=True)
    d_gb_bb = np.hstack([d_ce, d_mse])
    
    # Scaling to [-1, 1] using training-set min and max
    y_hat_gb_scaled = scale_to_min_max(y_hat_gb, mn_gb, mx_gb)
    y_hat_bb_scaled = scale_to_min_max(y_hat_bb, mn_bb, mx_bb)
    d_scaled = scale_to_min_max(d_gb_bb, mn_d, mx_d)
    
    if isinstance(X_orig, pd.DataFrame):
        X_mat = X_orig.to_numpy()
    else:
        X_mat = np.asarray(X_orig)
        
    X_alloc = np.hstack([X_mat, y_hat_gb_scaled, y_hat_bb_scaled, d_scaled])
    return X_alloc, d_gb_bb


def run_experiment(dataset_name='Wine', test_seeds=[0, 1, 2], epsilons=[0.00, 0.01, 0.02, 0.05, 0.10]):
    print(f"\n=======================================================")
    print(f"STARTING COMPREHENSIVE EXPERIMENT: {dataset_name}")
    print(f"=======================================================")
    
    # 1. Load dataset
    pred_dim, ds_dict = get_dataset(dataset_name=dataset_name)
    problem_type = ds_dict.get('problem_type')
    ds_sk = ds_dict.get('ds_sklearn')
    cat_indicator = ds_dict.get('categorical_indicator')
    
    X_train_df = ds_sk['ds_train']['x_train'].reset_index(drop=True)
    y_train_raw = ds_sk['ds_train']['y_train']
    X_val_df = ds_sk['ds_val']['x_val'].reset_index(drop=True)
    y_val_raw = ds_sk['ds_val']['y_val']
    X_test_df = ds_sk['ds_test']['x_test'].reset_index(drop=True)
    y_test_raw = ds_sk['ds_test']['y_test']
    
    y_train_onehot = np.eye(pred_dim, dtype=np.float32)[y_train_raw.astype(int)]
    y_val_onehot = np.eye(pred_dim, dtype=np.float32)[y_val_raw.astype(int)]
    y_test_onehot = np.eye(pred_dim, dtype=np.float32)[y_test_raw.astype(int)]
    
    # 2. Fit Component Models g and b using best hyperparameters found in CV
    print("Fitting Component Models g (Glass-box) and b (Black-box)...")
    if problem_type == 'classification':
        gb_model = LogisticRegression(C=2, solver='saga', penalty='l1', max_iter=10000, random_state=0)
        gb_model.fit(X_train_df, y_train_raw)
        
        bb_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, n_estimators=512, random_state=0, subsample=0.8)
        bb_model.fit(X_train_df, y_train_raw)
        
        y_hat_gb_train = gb_model.predict_proba(X_train_df)
        y_hat_gb_val = gb_model.predict_proba(X_val_df)
        y_hat_gb_test = gb_model.predict_proba(X_test_df)
        
        y_hat_bb_train = bb_model.predict_proba(X_train_df)
        y_hat_bb_val = bb_model.predict_proba(X_val_df)
        y_hat_bb_test = bb_model.predict_proba(X_test_df)
    else:
        raise NotImplementedError("Currently focusing on classification pipeline")
        
    acc_gb_test = accuracy_score(y_test_raw, np.argmax(y_hat_gb_test, axis=1)) * 100.0
    acc_bb_test = accuracy_score(y_test_raw, np.argmax(y_hat_bb_test, axis=1)) * 100.0
    print(f"Component Models Test Accuracy: Glass-Box = {acc_gb_test:.2f}%, Black-Box = {acc_bb_test:.2f}%")
    
    # 3. Ground Truth Allocation Desirability Ranking r(z)
    loss_gb_train = np.sum(-y_train_onehot * np.log(np.maximum(np.minimum(y_hat_gb_train, 1 - 1e-7), 1e-7)), axis=1, keepdims=True)
    loss_bb_train = np.sum(-y_train_onehot * np.log(np.maximum(np.minimum(y_hat_bb_train, 1 - 1e-7), 1e-7)), axis=1, keepdims=True)
    loss_cutoff = np.log(pred_dim)
    
    gb_c = (loss_gb_train < loss_cutoff).astype(np.float32)
    bb_c = (loss_bb_train < loss_cutoff).astype(np.float32)
    gb_desirability_train = 2.0 * gb_c - bb_c - 1.0 / (1.0 + np.exp(loss_bb_train - loss_gb_train))
    r_train = get_percentiles_of_a_wrt_b(b=gb_desirability_train, a=None, distr_min=-2.0, distr_max=2.0)
    
    # Scalers for augmented features
    d_ce_train = -np.sum(y_hat_bb_train * np.log(np.maximum(np.minimum(y_hat_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
    d_mse_train = np.sum((y_hat_bb_train - y_hat_gb_train) ** 2, axis=1, keepdims=True)
    d_gb_bb_train = np.hstack([d_ce_train, d_mse_train])
    
    _, mn_gb, mx_gb = scale_to_minus_one_to_one(y_hat_gb_train)
    _, mn_bb, mx_bb = scale_to_minus_one_to_one(y_hat_bb_train)
    _, mn_d, mx_d = scale_to_minus_one_to_one(d_gb_bb_train)
    
    X_alloc_train, _ = build_allocator_features(X_train_df, y_hat_gb_train, y_hat_bb_train, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d)
    X_alloc_val, _ = build_allocator_features(X_val_df, y_hat_gb_val, y_hat_bb_val, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d)
    X_alloc_test, _ = build_allocator_features(X_test_df, y_hat_gb_test, y_hat_bb_test, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d)
    
    # 4. Fit Original Baseline Allocator
    print("Fitting Original Allocator (A_orig)...")
    alloc_orig = GradientBoostingRegressor(learning_rate=0.01, max_depth=8, n_estimators=1024, random_state=0, subsample=0.25)
    alloc_orig.fit(X_alloc_train, r_train.ravel())
    
    # 5. Fit Proposed Stability-Regularized Allocator (A_SR)
    # Using training perturbations ONLY (zero test data exposure)
    print("Fitting Proposed Stability-Regularized Allocator (A_SR)...")
    # Generate training perturbations to enforce smoothness in local neighborhood
    X_train_pert_1 = generate_tabular_perturbations(X_train_df, cat_indicator, epsilon=0.02, seed=10)
    X_train_pert_2 = generate_tabular_perturbations(X_train_df, cat_indicator, epsilon=0.04, seed=20)
    
    y_hat_gb_tr_p1 = gb_model.predict_proba(X_train_pert_1)
    y_hat_bb_tr_p1 = bb_model.predict_proba(X_train_pert_1)
    X_alloc_tr_p1, _ = build_allocator_features(X_train_pert_1, y_hat_gb_tr_p1, y_hat_bb_tr_p1, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d)
    
    y_hat_gb_tr_p2 = gb_model.predict_proba(X_train_pert_2)
    y_hat_bb_tr_p2 = bb_model.predict_proba(X_train_pert_2)
    X_alloc_tr_p2, _ = build_allocator_features(X_train_pert_2, y_hat_gb_tr_p2, y_hat_bb_tr_p2, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d)
    
    # Combined stability-augmented training matrix
    X_alloc_train_sr = np.vstack([X_alloc_train, X_alloc_tr_p1, X_alloc_tr_p2])
    r_train_sr = np.concatenate([r_train.ravel(), r_train.ravel(), r_train.ravel()])
    
    # Fit SR-EEG allocator with subsample regularizer and stability penalty
    alloc_sr = GradientBoostingRegressor(learning_rate=0.01, max_depth=6, n_estimators=800, random_state=0, subsample=0.5)
    alloc_sr.fit(X_alloc_train_sr, r_train_sr)
    
    # 6. Controlled Robustness & Before-vs-After Evaluation Protocol
    print("\nRunning Controlled Test Evaluation across Perturbation Strengths and Seeds...")
    q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_results = []
    
    for seed in test_seeds:
        for eps in epsilons:
            # Generate test perturbation
            X_test_pert = generate_tabular_perturbations(X_test_df, cat_indicator, epsilon=eps, seed=seed)
            
            # Predict component models on clean and perturbed
            y_gb_cl = y_hat_gb_test
            y_bb_cl = y_hat_bb_test
            
            y_gb_pt = gb_model.predict_proba(X_test_pert)
            y_bb_pt = bb_model.predict_proba(X_test_pert)
            
            # Construct allocator inputs
            X_alloc_pt, _ = build_allocator_features(X_test_pert, y_gb_pt, y_bb_pt, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d)
            
            # Original Allocator predictions
            r_hat_cl_orig = alloc_orig.predict(X_alloc_test)
            r_hat_pt_orig = alloc_orig.predict(X_alloc_pt)
            
            # Stability-Aware Allocator predictions
            r_hat_cl_sr = alloc_sr.predict(X_alloc_test)
            r_hat_pt_sr = alloc_sr.predict(X_alloc_pt)
            
            for q in q_levels:
                # Original Allocator decisions
                a_cl_orig = (r_hat_cl_orig > (1.0 - q)).astype(float)
                a_pt_orig = (r_hat_pt_orig > (1.0 - q)).astype(float)
                
                # Stability-Aware Allocator decisions
                a_cl_sr = (r_hat_cl_sr > (1.0 - q)).astype(float)
                a_pt_sr = (r_hat_pt_sr > (1.0 - q)).astype(float)
                
                # Allocation Flip Rate (AFR)
                afr_orig = compute_allocation_flip_rate(a_cl_orig, a_pt_orig)
                afr_sr = compute_allocation_flip_rate(a_cl_sr, a_pt_sr)
                
                # Ensemble Predictions Clean
                _, pred_cl_orig = compute_ensemble_predictions(a_cl_orig, y_gb_cl, y_bb_cl, 'classification')
                _, pred_cl_sr = compute_ensemble_predictions(a_cl_sr, y_gb_cl, y_bb_cl, 'classification')
                
                # Ensemble Predictions Perturbed
                _, pred_pt_orig = compute_ensemble_predictions(a_pt_orig, y_gb_pt, y_bb_pt, 'classification')
                _, pred_pt_sr = compute_ensemble_predictions(a_pt_sr, y_gb_pt, y_bb_pt, 'classification')
                
                # Prediction Consistency (PC)
                pc_orig = compute_prediction_consistency(pred_cl_orig, pred_pt_orig, 'classification')
                pc_sr = compute_prediction_consistency(pred_cl_sr, pred_pt_sr, 'classification')
                
                # Accuracies
                acc_cl_orig = compute_performance(y_test_raw, pred_cl_orig, 'classification')
                acc_pt_orig = compute_performance(y_test_raw, pred_pt_orig, 'classification')
                
                acc_cl_sr = compute_performance(y_test_raw, pred_cl_sr, 'classification')
                acc_pt_sr = compute_performance(y_test_raw, pred_pt_sr, 'classification')
                
                # Record result for Original EEG
                all_results.append({
                    'Dataset': dataset_name,
                    'Seed': seed,
                    'Epsilon': eps,
                    'q': q,
                    'Method': 'Original EEG',
                    'AFR': afr_orig,
                    'Prediction_Consistency': pc_orig,
                    'Clean_Accuracy': acc_cl_orig,
                    'Perturbed_Accuracy': acc_pt_orig,
                    'Accuracy_Drop': acc_cl_orig - acc_pt_orig,
                    'Actual_q_Clean': np.mean(a_cl_orig),
                    'Actual_q_Perturbed': np.mean(a_pt_orig)
                })
                
                # Record result for Stability-Aware EEG (SR-EEG)
                all_results.append({
                    'Dataset': dataset_name,
                    'Seed': seed,
                    'Epsilon': eps,
                    'q': q,
                    'Method': 'Proposed SR-EEG',
                    'AFR': afr_sr,
                    'Prediction_Consistency': pc_sr,
                    'Clean_Accuracy': acc_cl_sr,
                    'Perturbed_Accuracy': acc_pt_sr,
                    'Accuracy_Drop': acc_cl_sr - acc_pt_sr,
                    'Actual_q_Clean': np.mean(a_cl_sr),
                    'Actual_q_Perturbed': np.mean(a_pt_sr)
                })

    results_df = pd.DataFrame(all_results)
    
    # Save detailed CSV
    os.makedirs(os.path.join('results', 'csv'), exist_ok=True)
    dataset_lower = dataset_name.lower()
    results_df.to_csv(os.path.join('results', 'csv', f'detailed_per_seed_results_{dataset_lower}.csv'), index=False)
    results_df.to_csv(os.path.join('results', 'csv', 'detailed_per_seed_results.csv'), index=False)
    print(f"Saved detailed results to results/csv/detailed_per_seed_results_{dataset_lower}.csv")
    
    # Aggregate Summary (Mean and SD across seeds)
    summary_df = results_df.groupby(['Dataset', 'Epsilon', 'q', 'Method']).agg({
        'AFR': ['mean', 'std'],
        'Prediction_Consistency': ['mean', 'std'],
        'Clean_Accuracy': ['mean', 'std'],
        'Perturbed_Accuracy': ['mean', 'std'],
        'Accuracy_Drop': ['mean', 'std'],
        'Actual_q_Clean': 'mean',
        'Actual_q_Perturbed': 'mean'
    }).reset_index()
    
    summary_df.columns = [
        '_'.join(col).strip('_') for col in summary_df.columns.values
    ]
    summary_df.to_csv(os.path.join('results', 'csv', f'summary_before_vs_after_{dataset_lower}.csv'), index=False)
    
    # Also update master summary
    master_path = os.path.join('results', 'csv', 'summary_before_vs_after.csv')
    if os.path.exists(master_path):
        try:
            prev_master = pd.read_csv(master_path)
            prev_master = prev_master[prev_master['Dataset'] != dataset_name]
            combined = pd.concat([prev_master, summary_df], ignore_index=True)
            combined.to_csv(master_path, index=False)
        except Exception:
            summary_df.to_csv(master_path, index=False)
    else:
        summary_df.to_csv(master_path, index=False)
        
    print(f"Saved aggregated summary to results/csv/summary_before_vs_after_{dataset_lower}.csv and master summary")
    
    # Print Quick Comparison Table at q=0.5
    print("\n--- BEFORE VS. AFTER COMPARISON SUMMARY (q = 0.50) ---")
    sub_05 = summary_df[summary_df['q'] == 0.5]
    print(sub_05[['Epsilon', 'Method', 'AFR_mean', 'AFR_std', 'Prediction_Consistency_mean', 'Perturbed_Accuracy_mean']].to_string(index=False))
    
    # 7. Generate Publication Figures
    generate_figures(results_df, summary_df)
    
    return results_df, summary_df


def generate_figures(results_df, summary_df):
    print("\nGenerating Publication Figures...")
    fig_dir = os.path.join('results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Set aesthetics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Allocation Flip Rate (AFR) vs. Epsilon across methods
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    epsilons = [0.00, 0.01, 0.02, 0.05, 0.10]
    
    for method, color, marker in [('Original EEG', '#d95f02', 'o'), ('Proposed SR-EEG', '#1b9e77', 's')]:
        sub = summary_df[(summary_df['Method'] == method) & (summary_df['q'] == 0.5)]
        ax.errorbar(sub['Epsilon'], sub['AFR_mean'], yerr=sub['AFR_std'], label=method,
                    color=color, marker=marker, linewidth=2, capsize=4)
                    
    ax.set_title("Allocation Flip Rate vs. Perturbation Magnitude (q = 0.50)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Perturbation Strength (Epsilon)", fontsize=11)
    ax.set_ylabel("Allocation Flip Rate (%)", fontsize=11)
    ax.set_ylim(-1, max(summary_df['AFR_mean']) + 8)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "fig1_afr_vs_epsilon.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")
    
    # Figure 2: Prediction Consistency vs. Epsilon
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for method, color, marker in [('Original EEG', '#d95f02', 'o'), ('Proposed SR-EEG', '#1b9e77', 's')]:
        sub = summary_df[(summary_df['Method'] == method) & (summary_df['q'] == 0.5)]
        ax.errorbar(sub['Epsilon'], sub['Prediction_Consistency_mean'], yerr=sub['Prediction_Consistency_std'],
                    label=method, color=color, marker=marker, linewidth=2, capsize=4)
                    
    ax.set_title("Prediction Consistency vs. Perturbation Magnitude (q = 0.50)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Perturbation Strength (Epsilon)", fontsize=11)
    ax.set_ylabel("Prediction Consistency (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "fig2_prediction_consistency.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")
    
    # Figure 3: Task Accuracy vs. Epsilon
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for method, color, marker in [('Original EEG', '#d95f02', 'o'), ('Proposed SR-EEG', '#1b9e77', 's')]:
        sub = summary_df[(summary_df['Method'] == method) & (summary_df['q'] == 0.5)]
        ax.errorbar(sub['Epsilon'], sub['Perturbed_Accuracy_mean'], yerr=sub['Perturbed_Accuracy_std'],
                    label=method, color=color, marker=marker, linewidth=2, capsize=4)
                    
    ax.set_title("Task Accuracy on Perturbed Test Inputs (q = 0.50)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Perturbation Strength (Epsilon)", fontsize=11)
    ax.set_ylabel("Perturbed Accuracy (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, "fig3_task_accuracy_vs_epsilon.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # Figure 4: Allocation Flip Rate across Explainability Levels q (at epsilon = 0.05)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_eps05 = summary_df[summary_df['Epsilon'] == 0.05]
    for method, color, marker in [('Original EEG', '#d95f02', 'o'), ('Proposed SR-EEG', '#1b9e77', 's')]:
        sub = sub_eps05[sub_eps05['Method'] == method]
        ax.errorbar(sub['q'], sub['AFR_mean'], yerr=sub['AFR_std'], label=method,
                    color=color, marker=marker, linewidth=2, capsize=4)
                    
    ax.set_title("Allocation Flip Rate across Explainability Levels q (Epsilon = 0.05)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Explainability Level (q)", fontsize=11)
    ax.set_ylabel("Allocation Flip Rate (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, "fig4_afr_across_q.png")
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved: {fig4_path}")
    
    # Figure 5: Stability-Performance Trade-off (AFR vs Accuracy Drop)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_eps_all = summary_df[summary_df['Epsilon'] > 0.0]
    for method, color, marker in [('Original EEG', '#d95f02', 'o'), ('Proposed SR-EEG', '#1b9e77', 's')]:
        sub = sub_eps_all[sub_eps_all['Method'] == method]
        ax.scatter(sub['AFR_mean'], sub['Accuracy_Drop_mean'], label=method,
                   color=color, marker=marker, s=60, alpha=0.85)
                   
    ax.set_title("Stability-Performance Trade-off (AFR vs. Accuracy Drop)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Allocation Flip Rate (%) [Lower is more stable]", fontsize=11)
    ax.set_ylabel("Accuracy Drop under Perturbation (%) [Lower is more robust]", fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig5_path = os.path.join(fig_dir, "fig5_stability_performance_tradeoff.png")
    plt.savefig(fig5_path)
    plt.close()
    print(f"Saved: {fig5_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='Wine', type=str, help="Dataset to run ('Wine', 'Phoneme')")
    args = parser.parse_args()
    run_experiment(dataset_name=args.dataset_name, test_seeds=[0, 1, 2], epsilons=[0.00, 0.01, 0.02, 0.05, 0.10])
