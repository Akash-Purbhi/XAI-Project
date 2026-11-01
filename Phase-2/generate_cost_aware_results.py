# -*- coding: utf-8 -*-
"""
Generate Quantitative Results, Tables, and Figures for Cost-Aware EEG Extension.

Outputs generated in Phase-2/results/cost_aware_extension/:
1. experiment_config.json
2. baseline_results.csv
3. cost_aware_results.csv
4. comparison_results.csv
5. inference_timing.csv
6. allocation_details.csv
7. Publication figures (300 DPI):
   - fig1_coverage_vs_predictive_performance.png
   - fig2_coverage_vs_inference_latency.png
   - fig3_cost_sensitivity_pareto.png
"""

import os
import sys
import json
import time
import platform
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, log_loss

from cost_aware_allocator import CostAwareAllocator, execute_cost_aware_inference, compute_ground_truth_ranking
from threshold_selection import (
    find_matched_coverage_threshold,
    optimize_cost_aware_threshold,
    ThresholdSelector
)
from benchmark_inference_timing import (
    load_wine_benchmark_data,
    construct_original_allocator_features,
    get_system_environment_info
)


def run_full_pipeline_and_generate_artifacts(output_dir: str = "results/cost_aware_extension"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("GENERATING COMPREHENSIVE EXPERIMENT RESULTS, TABLES, AND FIGURES")
    print(f"Target Output Directory: {output_dir}")
    print("=" * 80)

    # 1. Config Metadata
    env_info = get_system_environment_info()
    seeds = [0, 1, 2]
    q_targets = [0.20, 0.40, 0.60, 0.80]
    lambdas = [0.0, 0.1, 1.0]
    cost_gb = 1.0
    cost_bb = 10.0
    timing_reps = 30

    config = {
        "dataset": "Wine",
        "openml_id": 44091,
        "task_type": "binary_classification",
        "n_classes": 2,
        "replicate_seeds": seeds,
        "matched_q_targets": q_targets,
        "cost_lambdas": lambdas,
        "cost_gb": cost_gb,
        "cost_bb": cost_bb,
        "timing_repetitions": timing_reps,
        "component_models": {
            "glass_box": {
                "model": "LogisticRegression",
                "params": {"C": 2.0, "penalty": "l1", "solver": "saga", "max_iter": 5000}
            },
            "black_box": {
                "model": "GradientBoostingClassifier",
                "params": {"learning_rate": 0.01, "max_depth": 8, "n_estimators": 512, "subsample": 0.8}
            }
        },
        "allocator_models": {
            "original_eeg": {
                "model": "GradientBoostingRegressor",
                "features": "X_alloc [X, y_hat_gb, y_hat_bb, d_ce, d_mse] (15 features)",
                "params": {"learning_rate": 0.01, "max_depth": 6, "n_estimators": 400, "subsample": 0.5}
            },
            "cost_aware_eeg": {
                "model": "CostAwareAllocator (GradientBoostingRegressor)",
                "features": "X strictly (11 features)",
                "params": {"learning_rate": 0.01, "max_depth": 5, "n_estimators": 400, "subsample": 0.8}
            }
        },
        "environment": env_info
    }

    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("  [✓] Generated experiment_config.json")

    # 2. Load Wine Data
    x_train, y_train, x_val, y_val, x_test, y_test = load_wine_benchmark_data(data_home="/tmp/scikit_learn_data", seed=1)
    N_test = len(x_test)
    pred_dim = 2
    y_test_onehot = np.eye(pred_dim, dtype=np.float32)[y_test]

    baseline_rows = []
    cost_aware_rows = []
    comparison_rows = []
    allocation_detail_rows = []

    # Iterate over seeds
    for seed in seeds:
        print(f"\n--- Processing Seed {seed} ---")
        gb = LogisticRegression(C=2.0, solver='saga', penalty='l1', max_iter=5000, random_state=seed)
        gb.fit(x_train, y_train)

        bb = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, n_estimators=512, subsample=0.8, random_state=seed)
        bb.fit(x_train, y_train)

        # Validation & Test Component Probabilities
        p_gb_val = gb.predict_proba(x_val)
        p_bb_val = bb.predict_proba(x_val)
        p_gb_test = gb.predict_proba(x_test)
        p_bb_test = bb.predict_proba(x_test)

        # Training predictions & Scalers
        p_gb_train = gb.predict_proba(x_train)
        p_bb_train = bb.predict_proba(x_train)

        d_ce_tr = -np.sum(p_bb_train * np.log(np.maximum(np.minimum(p_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        d_mse_tr = np.sum((p_bb_train - p_gb_train) ** 2, axis=1, keepdims=True)
        d_gb_bb_tr = np.hstack([d_ce_tr, d_mse_tr])

        scaler_params = {
            'gb': (p_gb_train.min(0), p_gb_train.max(0)),
            'bb': (p_bb_train.min(0), p_bb_train.max(0)),
            'd': (d_gb_bb_tr.min(0), d_gb_bb_tr.max(0))
        }

        # Original Allocator
        y_oh_tr = np.eye(pred_dim, dtype=np.float32)[y_train]
        l_gb_tr = -np.sum(y_oh_tr * np.log(np.maximum(np.minimum(p_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        l_bb_tr = -np.sum(y_oh_tr * np.log(np.maximum(np.minimum(p_bb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        r_tr, _ = compute_ground_truth_ranking(loss_gb=l_gb_tr, loss_bb=l_bb_tr, pred_dim=pred_dim)

        X_alloc_tr = construct_original_allocator_features(x_train, p_gb_train, p_bb_train, scaler_params)
        orig_alloc = GradientBoostingRegressor(learning_rate=0.01, max_depth=6, n_estimators=400, subsample=0.5, random_state=seed)
        orig_alloc.fit(X_alloc_tr, r_tr.ravel())

        X_alloc_val = construct_original_allocator_features(x_val, p_gb_val, p_bb_val, scaler_params)
        X_alloc_test = construct_original_allocator_features(x_test, p_gb_test, p_bb_test, scaler_params)
        r_orig_val = orig_alloc.predict(X_alloc_val)
        r_orig_test = orig_alloc.predict(X_alloc_test)

        # Cost-Aware Allocator
        ca_alloc = CostAwareAllocator(
            estimator=GradientBoostingRegressor(learning_rate=0.01, max_depth=5, n_estimators=400, subsample=0.8, random_state=seed),
            random_state=seed
        )
        ca_alloc.fit(X_train=x_train, y_train=y_train, loss_gb_train=l_gb_tr, loss_bb_train=l_bb_tr, pred_dim=pred_dim)
        r_ca_val = ca_alloc.predict_ranking(x_val)
        r_ca_test = ca_alloc.predict_ranking(x_test)

        # Timing Original Baseline Pipeline
        # Measure latency for original pipeline (flat across q)
        timings_orig = []
        for _ in range(timing_reps):
            t0 = time.perf_counter()
            pg = gb.predict_proba(x_test)
            pb = bb.predict_proba(x_test)
            xa = construct_original_allocator_features(x_test, pg, pb, scaler_params)
            ro = orig_alloc.predict(xa)
            ao = (ro > 0.5).astype(float).reshape(-1, 1)
            _ = np.argmax(ao * pg + (1.0 - ao) * pb, axis=1)
            timings_orig.append((time.perf_counter() - t0) * 1000.0)
        base_lat_mean = float(np.mean(timings_orig))
        base_lat_std = float(np.std(timings_orig))
        base_cost = N_test * (cost_gb + cost_bb)

        # --- A. Evaluate Matched-Coverage Targets ---
        for q in q_targets:
            # 1. Validation Thresholds
            tau_orig = find_matched_coverage_threshold(r_orig_val, target_q=q)
            tau_ca = find_matched_coverage_threshold(r_ca_val, target_q=q)

            # Baseline Test Execution
            a_orig = (r_orig_test > tau_orig).astype(int)
            n_gb_o = int(np.sum(a_orig == 1))
            n_bb_o = int(N_test - n_gb_o)
            cov_o = float(n_gb_o / N_test)
            prob_o = a_orig.reshape(-1, 1) * p_gb_test + (1.0 - a_orig.reshape(-1, 1)) * p_bb_test
            pred_o = np.argmax(prob_o, axis=1)
            acc_o = float(accuracy_score(y_test, pred_o) * 100.0)
            f1_o = float(f1_score(y_test, pred_o, average='macro') * 100.0)
            ce_o = float(-np.mean(np.sum(y_test_onehot * np.log(np.maximum(np.minimum(prob_o, 1.0 - 1e-7), 1e-7)), axis=1)))

            baseline_rows.append({
                "dataset": "Wine",
                "model_config": "LR_GBDT_OrigAlloc",
                "seed": seed,
                "allocation_method": "Original_EEG",
                "target_coverage": q,
                "actual_coverage": cov_o,
                "accuracy": acc_o,
                "macro_f1": f1_o,
                "cross_entropy": ce_o,
                "n_glass_box": n_gb_o,
                "n_black_box": n_bb_o,
                "selected_threshold": tau_orig,
                "cost_sensitivity_lambda": 0.0,
                "mean_latency_ms": base_lat_mean,
                "std_latency_ms": base_lat_std,
                "latency_per_obs_ms": base_lat_mean / N_test,
                "realized_cost": base_cost,
                "cost_savings_pct": 0.0
            })

            # Cost-Aware Test Execution
            a_ca = (r_ca_test > tau_ca).astype(int)
            n_gb_c = int(np.sum(a_ca == 1))
            n_bb_c = int(N_test - n_gb_c)
            cov_c = float(n_gb_c / N_test)

            exec_c = execute_cost_aware_inference(
                X=x_test, a_decisions=a_ca, gb_model=gb, bb_model=bb,
                problem_type='classification', pred_dim=pred_dim, cost_gb=cost_gb, cost_bb=cost_bb
            )
            prob_c = exec_c['y_prob']
            pred_c = exec_c['y_pred']
            acc_c = float(accuracy_score(y_test, pred_c) * 100.0)
            f1_c = float(f1_score(y_test, pred_c, average='macro') * 100.0)
            ce_c = float(-np.mean(np.sum(y_test_onehot * np.log(np.maximum(np.minimum(prob_c, 1.0 - 1e-7), 1e-7)), axis=1)))

            # Timing Cost-Aware complete execution
            timings_ca = []
            for _ in range(timing_reps):
                t0 = time.perf_counter()
                rc = ca_alloc.predict_ranking(x_test)
                ac = (rc > tau_ca).astype(int)
                _ = execute_cost_aware_inference(
                    X=x_test, a_decisions=ac, gb_model=gb, bb_model=bb,
                    problem_type='classification', pred_dim=pred_dim
                )
                timings_ca.append((time.perf_counter() - t0) * 1000.0)
            lat_c_mean = float(np.mean(timings_ca))
            lat_c_std = float(np.std(timings_ca))

            cost_c = float(n_gb_c * cost_gb + n_bb_c * cost_bb)
            savings_c = float((1.0 - cost_c / base_cost) * 100.0)
            lat_reduction = float(((base_lat_mean - lat_c_mean) / base_lat_mean) * 100.0)

            cost_aware_rows.append({
                "dataset": "Wine",
                "model_config": "LR_GBDT_CostAwareAlloc",
                "seed": seed,
                "allocation_method": "CostAware_EEG",
                "mode": "matched_coverage",
                "target_coverage": q,
                "actual_coverage": cov_c,
                "accuracy": acc_c,
                "macro_f1": f1_c,
                "cross_entropy": ce_c,
                "n_glass_box": n_gb_c,
                "n_black_box": n_bb_c,
                "selected_threshold": tau_ca,
                "cost_sensitivity_lambda": 0.0,
                "mean_latency_ms": lat_c_mean,
                "std_latency_ms": lat_c_std,
                "latency_per_obs_ms": lat_c_mean / N_test,
                "realized_cost": cost_c,
                "cost_savings_pct": savings_c,
                "relative_latency_reduction_pct": lat_reduction
            })

            # Side-by-side comparison row
            comparison_rows.append({
                "dataset": "Wine",
                "seed": seed,
                "target_coverage": q,
                "orig_actual_coverage": cov_o,
                "ca_actual_coverage": cov_c,
                "orig_accuracy": acc_o,
                "ca_accuracy": acc_c,
                "accuracy_diff": acc_c - acc_o,
                "orig_macro_f1": f1_o,
                "ca_macro_f1": f1_c,
                "macro_f1_diff": f1_c - f1_o,
                "orig_cross_entropy": ce_o,
                "ca_cross_entropy": ce_c,
                "orig_latency_ms": base_lat_mean,
                "ca_latency_ms": lat_c_mean,
                "latency_reduction_pct": lat_reduction,
                "orig_realized_cost": base_cost,
                "ca_realized_cost": cost_c,
                "cost_savings_pct": savings_c
            })

            # Record per-observation allocation details for seed 0
            if seed == 0:
                for idx_obs in range(min(100, N_test)):
                    allocation_detail_rows.append({
                        "dataset": "Wine",
                        "seed": seed,
                        "observation_idx": idx_obs,
                        "target_q": q,
                        "true_y": int(y_test[idx_obs]),
                        "pred_gb": int(np.argmax(p_gb_test[idx_obs])),
                        "pred_bb": int(np.argmax(p_bb_test[idx_obs])),
                        "r_hat_orig": float(r_orig_test[idx_obs]),
                        "r_hat_ca": float(r_ca_test[idx_obs]),
                        "a_orig": int(a_orig[idx_obs]),
                        "a_ca": int(a_ca[idx_obs]),
                        "allocation_agreement": int(a_orig[idx_obs] == a_ca[idx_obs]),
                        "final_pred_orig": int(pred_o[idx_obs]),
                        "final_pred_ca": int(pred_c[idx_obs])
                    })

        # --- B. Evaluate Cost-Sensitivity Lambda Grid (Mode 2) ---
        for q_min in q_targets:
            for lam in lambdas:
                tau_opt, m_opt, _ = optimize_cost_aware_threshold(
                    r_hat_val=r_ca_val, y_val=y_val, y_hat_gb_val=p_gb_val, y_hat_bb_val=p_bb_val,
                    q_min=q_min, lambda_cost=lam, cost_gb=cost_gb, cost_bb=cost_bb
                )
                a_opt = (r_ca_test > tau_opt).astype(int)
                n_gb_opt = int(np.sum(a_opt == 1))
                n_bb_opt = int(N_test - n_gb_opt)
                cov_opt = float(n_gb_opt / N_test)

                exec_opt = execute_cost_aware_inference(
                    X=x_test, a_decisions=a_opt, gb_model=gb, bb_model=bb,
                    problem_type='classification', pred_dim=pred_dim, cost_gb=cost_gb, cost_bb=cost_bb
                )
                prob_opt = exec_opt['y_prob']
                pred_opt = exec_opt['y_pred']
                acc_opt = float(accuracy_score(y_test, pred_opt) * 100.0)
                f1_opt = float(f1_score(y_test, pred_opt, average='macro') * 100.0)
                ce_opt = float(-np.mean(np.sum(y_test_onehot * np.log(np.maximum(np.minimum(prob_opt, 1.0 - 1e-7), 1e-7)), axis=1)))
                cost_opt = float(n_gb_opt * cost_gb + n_bb_opt * cost_bb)
                savings_opt = float((1.0 - cost_opt / base_cost) * 100.0)

                cost_aware_rows.append({
                    "dataset": "Wine",
                    "model_config": "LR_GBDT_CostAwareAlloc",
                    "seed": seed,
                    "allocation_method": "CostAware_EEG",
                    "mode": "cost_optimized",
                    "target_coverage": q_min,
                    "actual_coverage": cov_opt,
                    "accuracy": acc_opt,
                    "macro_f1": f1_opt,
                    "cross_entropy": ce_opt,
                    "n_glass_box": n_gb_opt,
                    "n_black_box": n_bb_opt,
                    "selected_threshold": tau_opt,
                    "cost_sensitivity_lambda": lam,
                    "mean_latency_ms": np.nan,  # Mode 2 focuses on cost-performance objective
                    "std_latency_ms": np.nan,
                    "latency_per_obs_ms": np.nan,
                    "realized_cost": cost_opt,
                    "cost_savings_pct": savings_opt,
                    "relative_latency_reduction_pct": np.nan
                })

    # Save CSVs
    df_base = pd.DataFrame(baseline_rows)
    df_ca = pd.DataFrame(cost_aware_rows)
    df_comp = pd.DataFrame(comparison_rows)
    df_alloc = pd.DataFrame(allocation_detail_rows)

    df_base.to_csv(os.path.join(output_dir, "baseline_results.csv"), index=False)
    df_ca.to_csv(os.path.join(output_dir, "cost_aware_results.csv"), index=False)
    df_comp.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)
    df_alloc.to_csv(os.path.join(output_dir, "allocation_details.csv"), index=False)

    # Copy / update inference_timing.csv
    timing_src = "results/csv/inference_timing.csv"
    if os.path.exists(timing_src):
        df_timing = pd.read_csv(timing_src)
        df_timing.to_csv(os.path.join(output_dir, "inference_timing.csv"), index=False)
    else:
        # Generate clean timing summary
        df_timing = pd.DataFrame(baseline_rows)
        df_timing.to_csv(os.path.join(output_dir, "inference_timing.csv"), index=False)

    print("  [✓] Generated baseline_results.csv")
    print("  [✓] Generated cost_aware_results.csv")
    print("  [✓] Generated comparison_results.csv")
    print("  [✓] Generated inference_timing.csv")
    print("  [✓] Generated allocation_details.csv")

    # =========================================================================
    # 3. Publication Figures Generation
    # =========================================================================
    print("\n--- Generating Publication Figures (300 DPI) ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Coverage vs. Predictive Performance
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Aggregate matched coverage rows
    matched_ca = df_ca[df_ca["mode"] == "matched_coverage"]
    q_vals = q_targets

    mean_acc_o = [df_base[df_base["target_coverage"] == q]["accuracy"].mean() for q in q_vals]
    std_acc_o = [df_base[df_base["target_coverage"] == q]["accuracy"].std() for q in q_vals]
    mean_acc_c = [matched_ca[matched_ca["target_coverage"] == q]["accuracy"].mean() for q in q_vals]
    std_acc_c = [matched_ca[matched_ca["target_coverage"] == q]["accuracy"].std() for q in q_vals]

    mean_f1_o = [df_base[df_base["target_coverage"] == q]["macro_f1"].mean() for q in q_vals]
    std_f1_o = [df_base[df_base["target_coverage"] == q]["macro_f1"].std() for q in q_vals]
    mean_f1_c = [matched_ca[matched_ca["target_coverage"] == q]["macro_f1"].mean() for q in q_vals]
    std_f1_c = [matched_ca[matched_ca["target_coverage"] == q]["macro_f1"].std() for q in q_vals]

    # Subplot 1: Accuracy
    ax1.errorbar(q_vals, mean_acc_o, yerr=std_acc_o, fmt='-o', color='#1f77b4', label='Original EEG (AAAI-24)', capsize=4, linewidth=2)
    ax1.errorbar(q_vals, mean_acc_c, yerr=std_acc_c, fmt='-s', color='#d62728', label='Proposed Cost-Aware EEG', capsize=4, linewidth=2)
    ax1.axhline(72.07, color='gray', linestyle=':', label='Glass-Box Only (72.07%)')
    ax1.axhline(81.31, color='black', linestyle='--', label='Black-Box Only (81.31%)')
    ax1.set_xlabel('Explainability Target Coverage (q)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Accuracy vs. Explainability Coverage', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True)
    ax1.set_ylim(70, 84)

    # Subplot 2: Macro-F1
    ax2.errorbar(q_vals, mean_f1_o, yerr=std_f1_o, fmt='-o', color='#1f77b4', label='Original EEG (AAAI-24)', capsize=4, linewidth=2)
    ax2.errorbar(q_vals, mean_f1_c, yerr=std_f1_c, fmt='-s', color='#d62728', label='Proposed Cost-Aware EEG', capsize=4, linewidth=2)
    ax2.set_xlabel('Explainability Target Coverage (q)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Test Macro-F1 Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Macro-F1 vs. Explainability Coverage', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.set_ylim(70, 84)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_coverage_vs_predictive_performance.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  [✓] Saved {fig1_path}")

    # Figure 2: Coverage vs. Actual End-to-End Latency
    fig2, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    mean_lat_o = [df_base[df_base["target_coverage"] == q]["mean_latency_ms"].mean() for q in q_vals]
    std_lat_o = [df_base[df_base["target_coverage"] == q]["mean_latency_ms"].std() for q in q_vals]
    mean_lat_c = [matched_ca[matched_ca["target_coverage"] == q]["mean_latency_ms"].mean() for q in q_vals]
    std_lat_c = [matched_ca[matched_ca["target_coverage"] == q]["mean_latency_ms"].std() for q in q_vals]

    ax.plot(q_vals, mean_lat_o, '-o', color='#1f77b4', label='Original EEG (Flat Invariance)', linewidth=2.5, markersize=7)
    ax.fill_between(q_vals, np.array(mean_lat_o) - np.array(std_lat_o), np.array(mean_lat_o) + np.array(std_lat_o), color='#1f77b4', alpha=0.15)

    ax.plot(q_vals, mean_lat_c, '-s', color='#2ca02c', label='Cost-Aware EEG (Dynamic Scaling)', linewidth=2.5, markersize=7)
    ax.fill_between(q_vals, np.array(mean_lat_c) - np.array(std_lat_c), np.array(mean_lat_c) + np.array(std_lat_c), color='#2ca02c', alpha=0.15)

    # Annotate savings
    for i, q in enumerate(q_vals):
        pct = ((mean_lat_o[i] - mean_lat_c[i]) / mean_lat_o[i]) * 100.0
        ax.annotate(
            f"-{pct:.1f}%\nlatency",
            xy=(q, mean_lat_c[i]),
            xytext=(q, mean_lat_c[i] - 2.2),
            ha='center', fontsize=9, fontweight='bold', color='#1b5e20',
            arrowprops=dict(arrowstyle="->", color='#2ca02c', lw=1)
        )

    ax.set_xlabel('Explainability Target Coverage (q)', fontsize=11, fontweight='bold')
    ax.set_ylabel('End-to-End Test Latency (ms for N=537)', fontsize=11, fontweight='bold')
    ax.set_title('Inference Latency Scaling: Original vs. Cost-Aware EEG', fontsize=12, fontweight='bold')
    ax.legend(loc='center left', frameon=True, fontsize=10)
    ax.set_ylim(4, 23)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_coverage_vs_inference_latency.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  [✓] Saved {fig2_path}")

    # Figure 3: Cost-Sensitivity Pareto Trade-off
    fig3, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    df_opt = df_ca[df_ca["mode"] == "cost_optimized"]

    colors = {0.0: '#1f77b4', 0.1: '#ff7f0e', 1.0: '#2ca02c'}
    markers = {0.0: 'o', 0.1: '^', 1.0: 's'}

    for lam in lambdas:
        sub = df_opt[df_opt["cost_sensitivity_lambda"] == lam]
        grouped = sub.groupby("target_coverage")[["accuracy", "cost_savings_pct"]].mean().reset_index()
        ax.plot(
            grouped["cost_savings_pct"],
            grouped["accuracy"],
            marker=markers[lam],
            color=colors[lam],
            linewidth=2,
            label=rf'$\lambda_{{\text{{cost}}}} = {lam}$' + (' (Performance Priority)' if lam == 0.0 else (' (Cost Priority)' if lam == 1.0 else ' (Balanced)'))
        )
        for _, row in grouped.iterrows():
            ax.annotate(
                f"q≥{row['target_coverage']:.2f}",
                xy=(row["cost_savings_pct"], row["accuracy"]),
                xytext=(row["cost_savings_pct"] + 1.2, row["accuracy"] + 0.15),
                fontsize=8, color=colors[lam]
            )

    ax.set_xlabel('Compute Cost Savings over Baseline (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title(r'Cost-Performance Pareto Frontier under Multi-Objective Tuning ($J = L + \lambda C$)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.set_xlim(20, 85)
    ax.set_ylim(76, 83)

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_cost_sensitivity_pareto.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  [✓] Saved {fig3_path}")

    print("\n" + "=" * 80)
    print(f"[SUCCESS] All 6 CSV/JSON deliverables and 3 figures saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline_and_generate_artifacts()
