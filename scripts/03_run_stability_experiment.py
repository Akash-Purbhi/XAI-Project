# -*- coding: utf-8 -*-
"""
Script 03: Run Perturbation Stability Benchmark (Gap 3: Original EEG vs. SR-EEG)
Supports:
- Dynamic dataset selection (Classification & Regression)
- Hardware acceleration (RTX 3050 GPU via XGBoost CUDA or CPU via Scikit-Learn)
- Perturbation stress test across epsilons and multiple seeds
- Detailed metrics export (AFR, PC, Task Performance, Budget Adherence)
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from data import get_dataset, scale_to_minus_one_to_one
from utils_2 import get_percentiles_of_a_wrt_b
from evaluate_eeg import (
    compute_ensemble_predictions,
    compute_performance,
    compute_allocation_flip_rate,
    compute_prediction_consistency
)
from perturbation_utils import generate_tabular_perturbations

# Check GPU support
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def build_allocator_features(X_orig, y_hat_gb, y_hat_bb, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type='classification'):
    """
    Constructs augmented allocator features [X, y_hat_gb, y_hat_bb, d_gb_bb]
    normalized to [-1, 1] per original EEG specification.
    """
    if problem_type == 'classification':
        d_ce = -np.sum(y_hat_bb * np.log(np.maximum(np.minimum(y_hat_gb, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        d_mse = np.sum((y_hat_bb - y_hat_gb) ** 2, axis=1, keepdims=True)
        d_gb_bb = np.hstack([d_ce, d_mse])
    else:
        d_mse = (y_hat_bb.reshape(-1, 1) - y_hat_gb.reshape(-1, 1)) ** 2
        d_abs = np.abs(y_hat_bb.reshape(-1, 1) - y_hat_gb.reshape(-1, 1))
        d_gb_bb = np.hstack([d_mse, d_abs])

    y_hat_gb_scaled = 2.0 * (y_hat_gb - mn_gb) / np.maximum(mx_gb, 1e-7) - 1.0
    y_hat_bb_scaled = 2.0 * (y_hat_bb - mn_bb) / np.maximum(mx_bb, 1e-7) - 1.0
    d_gb_bb_scaled = 2.0 * (d_gb_bb - mn_d) / np.maximum(mx_d, 1e-7) - 1.0

    X_alloc = np.hstack([X_orig, y_hat_gb_scaled, y_hat_bb_scaled, d_gb_bb_scaled])
    return X_alloc, d_gb_bb


def train_tree_model(model_type, problem_type, use_gpu, **kwargs):
    """
    Factory creating tree models, choosing GPU XGBoost or CPU Scikit-Learn.
    """
    gpu_flag = use_gpu and CUDA_AVAILABLE and XGB_AVAILABLE
    if problem_type == 'classification':
        if gpu_flag:
            return xgb.XGBClassifier(
                tree_method='hist',
                device='cuda',
                learning_rate=kwargs.get('lr', 0.01),
                max_depth=kwargs.get('max_depth', 8),
                n_estimators=kwargs.get('n_estimators', 512),
                subsample=kwargs.get('subsample', 0.8),
                random_state=kwargs.get('random_state', 0),
                eval_metric='logloss'
            )
        else:
            return GradientBoostingClassifier(
                learning_rate=kwargs.get('lr', 0.01),
                max_depth=kwargs.get('max_depth', 8),
                n_estimators=kwargs.get('n_estimators', 512),
                subsample=kwargs.get('subsample', 0.8),
                random_state=kwargs.get('random_state', 0)
            )
    else:
        # Regression
        if gpu_flag:
            return xgb.XGBRegressor(
                tree_method='hist',
                device='cuda',
                learning_rate=kwargs.get('lr', 0.01),
                max_depth=kwargs.get('max_depth', 8),
                n_estimators=kwargs.get('n_estimators', 512),
                subsample=kwargs.get('subsample', 0.8),
                random_state=kwargs.get('random_state', 0)
            )
        else:
            return GradientBoostingRegressor(
                learning_rate=kwargs.get('lr', 0.01),
                max_depth=kwargs.get('max_depth', 8),
                n_estimators=kwargs.get('n_estimators', 512),
                subsample=kwargs.get('subsample', 0.8),
                random_state=kwargs.get('random_state', 0)
            )


def run_experiment(dataset_name, epsilons=[0.00, 0.01, 0.02, 0.05, 0.10], test_seeds=[0, 1, 2], use_gpu=True):
    print("=" * 70)
    print(f"RUNNING GAP 3 STABILITY EXPERIMENT: {dataset_name}")
    device_mode = "GPU (CUDA via XGBoost)" if (use_gpu and CUDA_AVAILABLE and XGB_AVAILABLE) else "CPU (Scikit-Learn)"
    print(f"Acceleration Device: {device_mode}")
    print(f"Epsilons: {epsilons} | Seeds: {test_seeds}")
    print("=" * 70)

    # 1. Load Data
    pred_dim, ds_dict = get_dataset(dataset_name=dataset_name)
    problem_type = ds_dict.get('problem_type')
    cat_indicator = ds_dict.get('categorical_features_indicator_dict', {}).get('categorical_features_indicator_list', [])

    ds_sklearn = ds_dict['ds_sklearn']
    X_train_df = ds_sklearn['ds_train']['x_train']
    y_train_raw = ds_sklearn['ds_train']['y_train']
    X_val_df = ds_sklearn['ds_val']['x_val']
    y_val_raw = ds_sklearn['ds_val']['y_val']
    X_test_df = ds_sklearn['ds_test']['x_test']
    y_test_raw = ds_sklearn['ds_test']['y_test']

    print(f"Dataset Loaded: {problem_type.upper()} | Train: {len(X_train_df)}, Val: {len(X_val_df)}, Test: {len(X_test_df)}")

    # 2. Fit Component Models (Glass-Box & Black-Box)
    print("\n[Step 1] Fitting Component Models g (Glass-box) & b (Black-box)...")
    if problem_type == 'classification':
        gb_model = LogisticRegression(C=2, solver='saga', penalty='l1', max_iter=5000, random_state=0)
        gb_model.fit(X_train_df, y_train_raw)

        bb_model = train_tree_model('classifier', 'classification', use_gpu, lr=0.01, max_depth=8, n_estimators=512, subsample=0.8)
        bb_model.fit(X_train_df, y_train_raw)

        y_hat_gb_train = gb_model.predict_proba(X_train_df)
        y_hat_gb_val = gb_model.predict_proba(X_val_df)
        y_hat_gb_test = gb_model.predict_proba(X_test_df)

        y_hat_bb_train = bb_model.predict_proba(X_train_df)
        y_hat_bb_val = bb_model.predict_proba(X_val_df)
        y_hat_bb_test = bb_model.predict_proba(X_test_df)

        acc_gb = accuracy_score(y_test_raw, np.argmax(y_hat_gb_test, axis=1)) * 100.0
        acc_bb = accuracy_score(y_test_raw, np.argmax(y_hat_bb_test, axis=1)) * 100.0
        print(f" -> Component Test Accuracy: Glass-Box = {acc_gb:.2f}%, Black-Box = {acc_bb:.2f}%")

        # Ground truth desirability ranking
        y_train_onehot = np.eye(pred_dim, dtype=np.float32)[y_train_raw.astype(int)]
        loss_gb_tr = np.sum(-y_train_onehot * np.log(np.maximum(np.minimum(y_hat_gb_train, 1 - 1e-7), 1e-7)), axis=1, keepdims=True)
        loss_bb_tr = np.sum(-y_train_onehot * np.log(np.maximum(np.minimum(y_hat_bb_train, 1 - 1e-7), 1e-7)), axis=1, keepdims=True)
        loss_cutoff = np.log(pred_dim)

        gb_c = (loss_gb_tr < loss_cutoff).astype(np.float32)
        bb_c = (loss_bb_tr < loss_cutoff).astype(np.float32)
        gb_desirability = 2.0 * gb_c - bb_c - 1.0 / (1.0 + np.exp(loss_bb_tr - loss_gb_tr))
    else:
        # Regression
        gb_model = Lasso(alpha=0.01, max_iter=5000, random_state=0)
        gb_model.fit(X_train_df, y_train_raw)

        bb_model = train_tree_model('regressor', 'regression', use_gpu, lr=0.01, max_depth=8, n_estimators=512, subsample=0.8)
        bb_model.fit(X_train_df, y_train_raw)

        y_hat_gb_train = gb_model.predict(X_train_df).reshape(-1, 1)
        y_hat_gb_val = gb_model.predict(X_val_df).reshape(-1, 1)
        y_hat_gb_test = gb_model.predict(X_test_df).reshape(-1, 1)

        y_hat_bb_train = bb_model.predict(X_train_df).reshape(-1, 1)
        y_hat_bb_val = bb_model.predict(X_val_df).reshape(-1, 1)
        y_hat_bb_test = bb_model.predict(X_test_df).reshape(-1, 1)

        rmse_gb = np.sqrt(mean_squared_error(y_test_raw, y_hat_gb_test))
        rmse_bb = np.sqrt(mean_squared_error(y_test_raw, y_hat_bb_test))
        print(f" -> Component Test RMSE: Glass-Box = {rmse_gb:.4f}, Black-Box = {rmse_bb:.4f}")

        # Ground truth desirability ranking for regression
        loss_gb_tr = (y_train_raw.reshape(-1, 1) - y_hat_gb_train) ** 2
        loss_bb_tr = (y_train_raw.reshape(-1, 1) - y_hat_bb_train) ** 2
        cutoff_reg = np.median(loss_bb_tr)

        gb_c = (loss_gb_tr < cutoff_reg).astype(np.float32)
        bb_c = (loss_bb_tr < cutoff_reg).astype(np.float32)
        gb_desirability = 2.0 * gb_c - bb_c - (loss_gb_tr - loss_bb_tr) / (np.std(loss_bb_tr) + 1e-7)

    r_train = get_percentiles_of_a_wrt_b(b=gb_desirability, a=None, distr_min=-2.0, distr_max=2.0)

    # Scalers
    _, mn_gb, mx_gb = scale_to_minus_one_to_one(y_hat_gb_train)
    _, mn_bb, mx_bb = scale_to_minus_one_to_one(y_hat_bb_train)
    if problem_type == 'classification':
        d_ce = -np.sum(y_hat_bb_train * np.log(np.maximum(np.minimum(y_hat_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        d_mse = np.sum((y_hat_bb_train - y_hat_gb_train) ** 2, axis=1, keepdims=True)
        d_init = np.hstack([d_ce, d_mse])
    else:
        d_mse = (y_hat_bb_train - y_hat_gb_train) ** 2
        d_abs = np.abs(y_hat_bb_train - y_hat_gb_train)
        d_init = np.hstack([d_mse, d_abs])
    _, mn_d, mx_d = scale_to_minus_one_to_one(d_init)

    X_alloc_train, _ = build_allocator_features(X_train_df, y_hat_gb_train, y_hat_bb_train, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)
    X_alloc_val, _ = build_allocator_features(X_val_df, y_hat_gb_val, y_hat_bb_val, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)
    X_alloc_test, _ = build_allocator_features(X_test_df, y_hat_gb_test, y_hat_bb_test, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)

    # 3. Fit Baseline Allocator (A_orig)
    print("\n[Step 2] Fitting Baseline Allocator (A_orig)...")
    alloc_orig = train_tree_model('regressor', 'regression', use_gpu, lr=0.01, max_depth=8, n_estimators=800, subsample=0.25, random_state=0)
    alloc_orig.fit(X_alloc_train, r_train.ravel())

    # 4. Fit Stability-Robust Allocator (SR-EEG)
    print("\n[Step 3] Fitting Proposed Stability-Robust Allocator (A_SR)...")
    X_tr_pert_1 = generate_tabular_perturbations(X_train_df, cat_indicator, epsilon=0.02, seed=10)
    X_tr_pert_2 = generate_tabular_perturbations(X_train_df, cat_indicator, epsilon=0.04, seed=20)

    if problem_type == 'classification':
        y_gb_p1 = gb_model.predict_proba(X_tr_pert_1)
        y_bb_p1 = bb_model.predict_proba(X_tr_pert_1)
        y_gb_p2 = gb_model.predict_proba(X_tr_pert_2)
        y_bb_p2 = bb_model.predict_proba(X_tr_pert_2)
    else:
        y_gb_p1 = gb_model.predict(X_tr_pert_1).reshape(-1, 1)
        y_bb_p1 = bb_model.predict(X_tr_pert_1).reshape(-1, 1)
        y_gb_p2 = gb_model.predict(X_tr_pert_2).reshape(-1, 1)
        y_bb_p2 = bb_model.predict(X_tr_pert_2).reshape(-1, 1)

    X_alloc_p1, _ = build_allocator_features(X_tr_pert_1, y_gb_p1, y_bb_p1, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)
    X_alloc_p2, _ = build_allocator_features(X_tr_pert_2, y_gb_p2, y_bb_p2, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)

    X_alloc_train_sr = np.vstack([X_alloc_train, X_alloc_p1, X_alloc_p2])
    r_train_sr = np.concatenate([r_train.ravel(), r_train.ravel(), r_train.ravel()])

    alloc_sr = train_tree_model('regressor', 'regression', use_gpu, lr=0.01, max_depth=6, n_estimators=800, subsample=0.50, random_state=0)
    alloc_sr.fit(X_alloc_train_sr, r_train_sr)

    # 5. Controlled Perturbation Evaluation Loop
    print("\n[Step 4] Running Perturbation Stress Tests across Epsilons & Seeds...")
    q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    records = []

    for seed in test_seeds:
        for eps in epsilons:
            X_test_pert = generate_tabular_perturbations(X_test_df, cat_indicator, epsilon=eps, seed=seed)

            if problem_type == 'classification':
                y_gb_cl = y_hat_gb_test
                y_bb_cl = y_hat_bb_test
                y_gb_pt = gb_model.predict_proba(X_test_pert)
                y_bb_pt = bb_model.predict_proba(X_test_pert)
            else:
                y_gb_cl = y_hat_gb_test
                y_bb_cl = y_hat_bb_test
                y_gb_pt = gb_model.predict(X_test_pert).reshape(-1, 1)
                y_bb_pt = bb_model.predict(X_test_pert).reshape(-1, 1)

            X_alloc_pt, _ = build_allocator_features(X_test_pert, y_gb_pt, y_bb_pt, mn_gb, mx_gb, mn_bb, mx_bb, mn_d, mx_d, problem_type)

            r_hat_cl_orig = alloc_orig.predict(X_alloc_test)
            r_hat_pt_orig = alloc_orig.predict(X_alloc_pt)
            r_hat_cl_sr = alloc_sr.predict(X_alloc_test)
            r_hat_pt_sr = alloc_sr.predict(X_alloc_pt)

            for q in q_levels:
                a_cl_orig = (r_hat_cl_orig > (1.0 - q)).astype(float)
                a_pt_orig = (r_hat_pt_orig > (1.0 - q)).astype(float)
                a_cl_sr = (r_hat_cl_sr > (1.0 - q)).astype(float)
                a_pt_sr = (r_hat_pt_sr > (1.0 - q)).astype(float)

                afr_orig = compute_allocation_flip_rate(a_cl_orig, a_pt_orig)
                afr_sr = compute_allocation_flip_rate(a_cl_sr, a_pt_sr)

                _, pred_cl_orig = compute_ensemble_predictions(a_cl_orig, y_gb_cl, y_bb_cl, problem_type)
                _, pred_cl_sr = compute_ensemble_predictions(a_cl_sr, y_gb_cl, y_bb_cl, problem_type)
                _, pred_pt_orig = compute_ensemble_predictions(a_pt_orig, y_gb_pt, y_bb_pt, problem_type)
                _, pred_pt_sr = compute_ensemble_predictions(a_pt_sr, y_gb_pt, y_bb_pt, problem_type)

                pc_orig = compute_prediction_consistency(pred_cl_orig, pred_pt_orig, problem_type)
                pc_sr = compute_prediction_consistency(pred_cl_sr, pred_pt_sr, problem_type)

                perf_cl_orig = compute_performance(y_test_raw, pred_cl_orig, problem_type)
                perf_pt_orig = compute_performance(y_test_raw, pred_pt_orig, problem_type)
                perf_cl_sr = compute_performance(y_test_raw, pred_cl_sr, problem_type)
                perf_pt_sr = compute_performance(y_test_raw, pred_pt_sr, problem_type)

                records.append({
                    'Dataset': dataset_name, 'Problem_Type': problem_type, 'Seed': seed, 'Epsilon': eps, 'q': q,
                    'Method': 'Original EEG', 'AFR': afr_orig, 'Prediction_Consistency': pc_orig,
                    'Clean_Perf': perf_cl_orig, 'Perturbed_Perf': perf_pt_orig,
                    'Actual_q_Clean': np.mean(a_cl_orig), 'Actual_q_Perturbed': np.mean(a_pt_orig)
                })
                records.append({
                    'Dataset': dataset_name, 'Problem_Type': problem_type, 'Seed': seed, 'Epsilon': eps, 'q': q,
                    'Method': 'SR-EEG (Proposed)', 'AFR': afr_sr, 'Prediction_Consistency': pc_sr,
                    'Clean_Perf': perf_cl_sr, 'Perturbed_Perf': perf_pt_sr,
                    'Actual_q_Clean': np.mean(a_cl_sr), 'Actual_q_Perturbed': np.mean(a_pt_sr)
                })

    df_records = pd.DataFrame(records)

    # 6. Save Results
    out_dir = Path(f"results/stability/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_records.to_csv(out_dir / "detailed_per_seed_results.csv", index=False)

    summary = df_records.groupby(['Epsilon', 'Method']).agg({
        'AFR': ['mean', 'std'],
        'Prediction_Consistency': ['mean', 'std'],
        'Clean_Perf': 'mean',
        'Perturbed_Perf': 'mean'
    }).reset_index()
    summary.to_csv(out_dir / "summary_before_vs_after.csv", index=False)

    # 7. Generate Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    orig_sum = df_records[df_records['Method'] == 'Original EEG'].groupby('Epsilon')['AFR'].mean()
    sr_sum = df_records[df_records['Method'] == 'SR-EEG (Proposed)'].groupby('Epsilon')['AFR'].mean()

    axes[0].plot(orig_sum.index, orig_sum.values, 'o--', color='#d9534f', label='Original EEG (Baseline)', linewidth=2)
    axes[0].plot(sr_sum.index, sr_sum.values, 's-', color='#2b6cb0', label='SR-EEG (Proposed Robust)', linewidth=2.5)
    axes[0].set_title(f"Allocation Flip Rate (AFR) vs. Noise ε - {dataset_name}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Perturbation Magnitude ε")
    axes[0].set_ylabel("Allocation Flip Rate (%) [Lower is Better]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    orig_pc = df_records[df_records['Method'] == 'Original EEG'].groupby('Epsilon')['Prediction_Consistency'].mean()
    sr_pc = df_records[df_records['Method'] == 'SR-EEG (Proposed)'].groupby('Epsilon')['Prediction_Consistency'].mean()

    axes[1].plot(orig_pc.index, orig_pc.values, 'o--', color='#d9534f', label='Original EEG', linewidth=2)
    axes[1].plot(sr_pc.index, sr_pc.values, 's-', color='#2b6cb0', label='SR-EEG (Proposed)', linewidth=2.5)
    axes[1].set_title(f"Prediction Consistency vs. Noise ε - {dataset_name}", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Perturbation Magnitude ε")
    metric_label = "Consistency (%) [Higher is Better]" if problem_type == 'classification' else "Mean Abs Shift [Lower is Better]"
    axes[1].set_ylabel(metric_label)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_path = out_dir / "stability_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[DONE] Finished {dataset_name}. Results saved to {out_dir}")
    print(f" -> AFR at eps=0.05: Baseline = {orig_sum.get(0.05, 0):.2f}% | SR-EEG = {sr_sum.get(0.05, 0):.2f}%")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Gap 3 Perturbation Stability Benchmark")
    parser.add_argument('--dataset_name', type=str, default='Wine', help="Dataset to evaluate")
    parser.add_argument('--use_gpu', action='store_true', default=True, help="Use GPU acceleration if available")
    parser.add_argument('--no_gpu', dest='use_gpu', action='store_false', help="Force CPU mode")
    args = parser.parse_args()

    run_experiment(dataset_name=args.dataset_name, use_gpu=args.use_gpu)


if __name__ == "__main__":
    main()
