# -*- coding: utf-8 -*-
"""
Controlled Experimental Evaluation: Original EEG vs. Cost-Aware EEG.

This runner performs an exact, controlled comparison:
1. Exact same dataset, preprocessing, and train/val/test splits (Wine benchmark).
2. Exact same component models (L1-Logistic Regression and GBDT Classifier).
3. Matched-coverage evaluation across target quotas q in [0.20, 0.40, 0.60, 0.80]
   plus full curve q in [0.0, 1.0] for standard EEG Table 2 benchmark metrics (AUC, PPCR, 95TQM).
4. Multi-seed replicates (seeds = [0, 1, 2]).
5. Validation-only threshold selection: thresholds are calibrated strictly on validation data.
6. Test data used ONLY for final unbiased evaluation.
7. Tracks predictive accuracy, cross-entropy loss, actual coverage, model call counts,
   measured inference latency, and computational cost savings.
"""

import os
import argparse
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from cost_aware_allocator import CostAwareAllocator, execute_cost_aware_inference, compute_ground_truth_ranking
from threshold_selection import find_matched_coverage_threshold, evaluate_threshold_on_validation
from benchmark_inference_timing import load_wine_benchmark_data, construct_original_allocator_features
from evaluate_eeg import compute_auc, compute_original_eeg_metrics


# =============================================================================
# 1. Experiment Orchestration
# =============================================================================

def run_controlled_experiment(
    seeds: List[int] = [0, 1, 2],
    q_targets: List[float] = [0.20, 0.40, 0.60, 0.80],
    data_home: str = "/tmp/scikit_learn_data",
    cost_gb: float = 1.0,
    cost_bb: float = 10.0,
    timing_reps: int = 25
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs controlled multi-seed comparison between Original EEG and Cost-Aware EEG.

    Returns:
    --------
    df_detailed : pd.DataFrame with per-seed, per-q, per-method observations.
    df_summary : pd.DataFrame aggregated across seeds (mean ± std).
    """
    print("=" * 80)
    print("STARTING CONTROLLED EXPERIMENT: ORIGINAL EEG VS. COST-AWARE EEG")
    print(f"Seeds: {seeds} | Matched Coverage Targets: {q_targets}")
    print("=" * 80)

    # 1. Load Wine benchmark dataset (exact split)
    x_train, y_train, x_val, y_val, x_test, y_test = load_wine_benchmark_data(data_home=data_home, seed=1)
    N_test = len(x_test)
    pred_dim = 2
    y_test_onehot = np.eye(pred_dim, dtype=np.float32)[y_test]

    detailed_records = []
    eeg_metric_records = []

    for seed in seeds:
        print(f"\n>>> Running Replicate Seed: {seed} <<<")

        # ---------------------------------------------------------------------
        # A. Fit Component Models on Training Data
        # ---------------------------------------------------------------------
        gb_model = LogisticRegression(C=2.0, solver='saga', penalty='l1', max_iter=5000, random_state=seed)
        gb_model.fit(x_train, y_train)

        bb_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, n_estimators=512, subsample=0.8, random_state=seed)
        bb_model.fit(x_train, y_train)

        # Baseline Component Accuracies on Test
        acc_gb_test = float(np.mean(gb_model.predict(x_test) == y_test) * 100.0)
        acc_bb_test = float(np.mean(bb_model.predict(x_test) == y_test) * 100.0)
        print(f"  Component Test Accuracy: Glass-Box = {acc_gb_test:.2f}%, Black-Box = {acc_bb_test:.2f}%")

        # Component Predictions on Validation & Test
        y_hat_gb_val = gb_model.predict_proba(x_val)
        y_hat_bb_val = bb_model.predict_proba(x_val)

        y_hat_gb_test = gb_model.predict_proba(x_test)
        y_hat_bb_test = bb_model.predict_proba(x_test)

        # Training predictions & Ground Truth Ranking for Allocator Fitting
        y_hat_gb_train = gb_model.predict_proba(x_train)
        y_hat_bb_train = bb_model.predict_proba(x_train)

        y_onehot_train = np.eye(pred_dim, dtype=np.float32)[y_train]
        loss_gb_train = -np.sum(y_onehot_train * np.log(np.maximum(np.minimum(y_hat_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        loss_bb_train = -np.sum(y_onehot_train * np.log(np.maximum(np.minimum(y_hat_bb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)

        r_train, _ = compute_ground_truth_ranking(loss_gb=loss_gb_train, loss_bb=loss_bb_train, pred_dim=pred_dim)

        # Scaler parameters for original allocator features
        d_ce_tr = -np.sum(y_hat_bb_train * np.log(np.maximum(np.minimum(y_hat_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        d_mse_tr = np.sum((y_hat_bb_train - y_hat_gb_train) ** 2, axis=1, keepdims=True)
        d_gb_bb_tr = np.hstack([d_ce_tr, d_mse_tr])

        scaler_params = {
            'gb': (y_hat_gb_train.min(0), y_hat_gb_train.max(0)),
            'bb': (y_hat_bb_train.min(0), y_hat_bb_train.max(0)),
            'd': (d_gb_bb_tr.min(0), d_gb_bb_tr.max(0))
        }

        # ---------------------------------------------------------------------
        # B. Fit Original Baseline Allocator (Trained on X_alloc)
        # ---------------------------------------------------------------------
        X_alloc_train = construct_original_allocator_features(x_train, y_hat_gb_train, y_hat_bb_train, scaler_params)
        orig_allocator = GradientBoostingRegressor(learning_rate=0.01, max_depth=6, n_estimators=400, subsample=0.5, random_state=seed)
        orig_allocator.fit(X_alloc_train, r_train.ravel())

        # Validation & Test features for Original Allocator
        X_alloc_val = construct_original_allocator_features(x_val, y_hat_gb_val, y_hat_bb_val, scaler_params)
        X_alloc_test = construct_original_allocator_features(x_test, y_hat_gb_test, y_hat_bb_test, scaler_params)

        r_hat_orig_val = orig_allocator.predict(X_alloc_val)
        r_hat_orig_test = orig_allocator.predict(X_alloc_test)

        # ---------------------------------------------------------------------
        # C. Fit Cost-Aware Allocator (Trained STRICTLY on x_train)
        # ---------------------------------------------------------------------
        ca_allocator = CostAwareAllocator(
            estimator=GradientBoostingRegressor(learning_rate=0.01, max_depth=5, n_estimators=400, subsample=0.8, random_state=seed),
            random_state=seed
        )
        ca_allocator.fit(X_train=x_train, y_train=y_train, loss_gb_train=loss_gb_train, loss_bb_train=loss_bb_train, pred_dim=pred_dim)

        r_hat_ca_val = ca_allocator.predict_ranking(x_val)
        r_hat_ca_test = ca_allocator.predict_ranking(x_test)

        # ---------------------------------------------------------------------
        # D. Matched-Coverage Test Evaluation Across Target Quotas
        # ---------------------------------------------------------------------
        for q in q_targets:
            # 1. Select threshold on VALIDATION DATA ONLY
            tau_orig = find_matched_coverage_threshold(r_hat_orig_val, target_q=q)
            tau_ca = find_matched_coverage_threshold(r_hat_ca_val, target_q=q)

            # --- ORIGINAL EEG EVALUATION ---
            # Decision
            a_orig_test = (r_hat_orig_test > tau_orig).astype(int)
            n_gb_orig = int(np.sum(a_orig_test == 1))
            n_bb_orig = int(N_test - n_gb_orig)
            cov_orig = float(n_gb_orig / N_test)

            a_orig_col = a_orig_test.reshape(-1, 1)
            prob_orig = a_orig_col * y_hat_gb_test + (1.0 - a_orig_col) * y_hat_bb_test
            pred_orig = np.argmax(prob_orig, axis=1)
            acc_orig = float(np.mean(pred_orig == y_test) * 100.0)
            ce_orig = float(-np.mean(np.sum(y_test_onehot * np.log(np.maximum(np.minimum(prob_orig, 1.0 - 1e-7), 1e-7)), axis=1)))

            # Timing original complete pipeline
            timings_orig = []
            for _ in range(timing_reps):
                t0 = time.perf_counter()
                p_g = gb_model.predict_proba(x_test)
                p_b = bb_model.predict_proba(x_test)
                x_al = construct_original_allocator_features(x_test, p_g, p_b, scaler_params)
                r_p = orig_allocator.predict(x_al)
                a_m = (r_p > tau_orig).astype(float).reshape(-1, 1)
                _ = np.argmax(a_m * p_g + (1.0 - a_m) * p_b, axis=1)
                timings_orig.append((time.perf_counter() - t0) * 1000.0)

            lat_mean_orig = float(np.mean(timings_orig))
            lat_std_orig = float(np.std(timings_orig))

            cost_orig = float(N_test * (cost_gb + cost_bb))  # executes both models for all samples

            detailed_records.append({
                "seed": seed,
                "method": "Original_EEG",
                "target_q": q,
                "selected_tau": tau_orig,
                "actual_coverage": cov_orig,
                "n_gb": n_gb_orig,
                "n_bb": n_bb_orig,
                "accuracy": acc_orig,
                "cross_entropy": ce_orig,
                "mean_latency_ms": lat_mean_orig,
                "std_latency_ms": lat_std_orig,
                "latency_per_obs_ms": lat_mean_orig / N_test,
                "realized_cost": cost_orig,
                "cost_savings_pct": 0.0
            })

            # --- COST-AWARE EEG EVALUATION ---
            # Decision
            a_ca_test = (r_hat_ca_test > tau_ca).astype(int)
            n_gb_ca = int(np.sum(a_ca_test == 1))
            n_bb_ca = int(N_test - n_gb_ca)
            cov_ca = float(n_gb_ca / N_test)

            # Conditional Execution (Strictly only assigned model called)
            ca_exec_result = execute_cost_aware_inference(
                X=x_test,
                a_decisions=a_ca_test,
                gb_model=gb_model,
                bb_model=bb_model,
                problem_type='classification',
                pred_dim=pred_dim,
                cost_gb=cost_gb,
                cost_bb=cost_bb
            )
            prob_ca = ca_exec_result['y_prob']
            pred_ca = ca_exec_result['y_pred']
            acc_ca = float(np.mean(pred_ca == y_test) * 100.0)
            ce_ca = float(-np.mean(np.sum(y_test_onehot * np.log(np.maximum(np.minimum(prob_ca, 1.0 - 1e-7), 1e-7)), axis=1)))

            # Timing cost-aware complete pipeline
            timings_ca = []
            for _ in range(timing_reps):
                t0 = time.perf_counter()
                r_c = ca_allocator.predict_ranking(x_test)
                a_c = (r_c > tau_ca).astype(int)
                _ = execute_cost_aware_inference(
                    X=x_test, a_decisions=a_c, gb_model=gb_model, bb_model=bb_model,
                    problem_type='classification', pred_dim=pred_dim
                )
                timings_ca.append((time.perf_counter() - t0) * 1000.0)

            lat_mean_ca = float(np.mean(timings_ca))
            lat_std_ca = float(np.std(timings_ca))

            cost_ca = float(n_gb_ca * cost_gb + n_bb_ca * cost_bb)
            savings_pct = float((1.0 - cost_ca / cost_orig) * 100.0)

            detailed_records.append({
                "seed": seed,
                "method": "CostAware_EEG",
                "target_q": q,
                "selected_tau": tau_ca,
                "actual_coverage": cov_ca,
                "n_gb": n_gb_ca,
                "n_bb": n_bb_ca,
                "accuracy": acc_ca,
                "cross_entropy": ce_ca,
                "mean_latency_ms": lat_mean_ca,
                "std_latency_ms": lat_std_ca,
                "latency_per_obs_ms": lat_mean_ca / N_test,
                "realized_cost": cost_ca,
                "cost_savings_pct": savings_pct
            })

            print(f"  q={q:.2f} | Orig: Acc={acc_orig:.2f}%, Cov={cov_orig:.2f}, Lat={lat_mean_orig:.2f}ms | CA: Acc={acc_ca:.2f}%, Cov={cov_ca:.2f}, Lat={lat_mean_ca:.2f}ms (Saved {savings_pct:.1f}% cost)")

        # ---------------------------------------------------------------------
        # E. Full Curve Evaluation for AAAI-24 EEG Metrics (AUC, PPCR, 95TQM)
        # ---------------------------------------------------------------------
        q_curve = np.linspace(0.0, 1.0, 21)
        perf_orig_curve = []
        perf_ca_curve = []
        perf_rand_curve = []
        perf_oracle_curve = []

        # Oracle ranking using ground-truth test losses
        loss_gb_test = -np.sum(y_test_onehot * np.log(np.maximum(np.minimum(y_hat_gb_test, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        loss_bb_test = -np.sum(y_test_onehot * np.log(np.maximum(np.minimum(y_hat_bb_test, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
        r_oracle_test, _ = compute_ground_truth_ranking(loss_gb=loss_gb_test, loss_bb=loss_bb_test, pred_dim=pred_dim)

        for q_c in q_curve:
            # Original
            tau_o = find_matched_coverage_threshold(r_hat_orig_val, target_q=q_c)
            a_o = (r_hat_orig_test > tau_o).astype(float).reshape(-1, 1)
            p_o = np.argmax(a_o * y_hat_gb_test + (1.0 - a_o) * y_hat_bb_test, axis=1)
            perf_orig_curve.append(float(np.mean(p_o == y_test) * 100.0))

            # Cost-Aware
            tau_c = find_matched_coverage_threshold(r_hat_ca_val, target_q=q_c)
            a_c = (r_hat_ca_test > tau_c).astype(float).reshape(-1, 1)
            p_c = np.argmax(a_c * y_hat_gb_test + (1.0 - a_c) * y_hat_bb_test, axis=1)
            perf_ca_curve.append(float(np.mean(p_c == y_test) * 100.0))

            # Random
            perf_rand_curve.append(q_c * acc_gb_test + (1.0 - q_c) * acc_bb_test)

            # Oracle
            tau_orc = find_matched_coverage_threshold(r_oracle_test, target_q=q_c)
            a_orc = (r_oracle_test > tau_orc).astype(float).reshape(-1, 1)
            p_orc = np.argmax(a_orc * y_hat_gb_test + (1.0 - a_orc) * y_hat_bb_test, axis=1)
            perf_oracle_curve.append(float(np.mean(p_orc == y_test) * 100.0))

        # EEG Metrics
        m_orig = compute_original_eeg_metrics(
            q_grid=q_curve, perf_ensemble=perf_orig_curve, perf_random=perf_rand_curve,
            perf_oracle=perf_oracle_curve, perf_gb=acc_gb_test, perf_bb=acc_bb_test
        )
        m_orig["method"] = "Original_EEG"
        m_orig["seed"] = seed
        eeg_metric_records.append(m_orig)

        m_ca = compute_original_eeg_metrics(
            q_grid=q_curve, perf_ensemble=perf_ca_curve, perf_random=perf_rand_curve,
            perf_oracle=perf_oracle_curve, perf_gb=acc_gb_test, perf_bb=acc_bb_test
        )
        m_ca["method"] = "CostAware_EEG"
        m_ca["seed"] = seed
        eeg_metric_records.append(m_ca)

    df_detailed = pd.DataFrame(detailed_records)
    df_eeg_metrics = pd.DataFrame(eeg_metric_records)

    # -------------------------------------------------------------------------
    # F. Compute Aggregate Summary (Mean ± Std Across Seeds)
    # -------------------------------------------------------------------------
    summary_cols = ["actual_coverage", "accuracy", "cross_entropy", "mean_latency_ms", "latency_per_obs_ms", "realized_cost", "cost_savings_pct"]
    df_summary = df_detailed.groupby(["method", "target_q"])[summary_cols].agg(["mean", "std"]).reset_index()

    return df_detailed, df_summary, df_eeg_metrics


# =============================================================================
# 2. Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run controlled experimental evaluation comparing Original EEG and Cost-Aware EEG.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds for replicate runs.")
    parser.add_argument("--q_targets", nargs="+", type=float, default=[0.20, 0.40, 0.60, 0.80], help="Target explainability quotas.")
    parser.add_argument("--output_dir", default="results/csv", type=str, help="Directory to save experimental results.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df_detailed, df_summary, df_eeg = run_controlled_experiment(
        seeds=args.seeds,
        q_targets=args.q_targets
    )

    # Save to CSV
    path_detailed = os.path.join(args.output_dir, "cost_aware_matched_coverage_wine.csv")
    path_summary = os.path.join(args.output_dir, "cost_aware_summary_wine.csv")
    path_eeg = os.path.join(args.output_dir, "cost_aware_eeg_metrics_wine.csv")

    df_detailed.to_csv(path_detailed, index=False)
    df_summary.to_csv(path_summary, index=False)
    df_eeg.to_csv(path_eeg, index=False)

    print("\n" + "=" * 80)
    print("CONTROLLED EXPERIMENTAL RESULTS SUMMARY (Mean across seeds):")
    print("=" * 80)
    for q in args.q_targets:
        orig_row = df_detailed[(df_detailed["method"] == "Original_EEG") & (df_detailed["target_q"] == q)]
        ca_row = df_detailed[(df_detailed["method"] == "CostAware_EEG") & (df_detailed["target_q"] == q)]

        acc_o = orig_row["accuracy"].mean()
        acc_c = ca_row["accuracy"].mean()
        lat_o = orig_row["mean_latency_ms"].mean()
        lat_c = ca_row["mean_latency_ms"].mean()
        sav_c = ca_row["cost_savings_pct"].mean()

        print(f"Target q={q:.2f}:")
        print(f"  - Original EEG:   Accuracy = {acc_o:.2f}% | Latency = {lat_o:.2f} ms | Cost Savings = 0.0%")
        print(f"  - Cost-Aware EEG: Accuracy = {acc_c:.2f}% | Latency = {lat_c:.2f} ms | Cost Savings = {sav_c:.1f}%")

    print("\n" + "=" * 80)
    print("TABLE 2 BENCHMARK METRICS (Mean across seeds):")
    print("=" * 80)
    for method in ["Original_EEG", "CostAware_EEG"]:
        sub = df_eeg[df_eeg["method"] == method]
        print(f"{method}:")
        print(f"  - AUC:     {sub['AUC'].mean():.2f} ± {sub['AUC'].std():.2f}")
        print(f"  - PPCR:    {sub['PPCR'].mean():.2f}% ± {sub['PPCR'].std():.2f}%")
        print(f"  - 95TQM:   {sub['95TQM'].mean():.2f}% ± {sub['95TQM'].std():.2f}%")
        print(f"  - Max Acc: {sub['Max_Acc'].mean():.2f}% ± {sub['Max_Acc'].std():.2f}%")

    print(f"\n[SUCCESS] All results exported to {args.output_dir}:")
    print(f"  1. {path_detailed}")
    print(f"  2. {path_summary}")
    print(f"  3. {path_eeg}")


if __name__ == "__main__":
    main()
