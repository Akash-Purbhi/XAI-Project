# -*- coding: utf-8 -*-
"""
Inference Latency & Computational Cost Benchmark Utility.

Measures actual wall-clock inference execution latency across:
1. Glass-box model inference g(X)
2. Black-box model inference b(X)
3. Original EEG allocator inference A_orig(X_alloc)
4. Cost-aware allocator inference A_CA(X)
5. Complete Original EEG Ensemble inference path (b(X) + g(X) + distance features + A_orig + routing)
6. Complete Cost-Aware Ensemble inference path (A_CA(X) + conditional execution of ONLY selected model)

Adheres strictly to empirical benchmarking standards:
- time.perf_counter() precision timing
- Explicit warm-up iterations
- Repeated measurement rounds with mean, standard deviation, min, and max
- Milliseconds per observation reporting
- Hardware and environment metadata recording
- Zero synthetic/theoretical FLOP estimation: actual CPU execution profiling
"""

import os
import sys
import platform
import argparse
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from cost_aware_allocator import CostAwareAllocator, execute_cost_aware_inference, compute_ground_truth_ranking


# =============================================================================
# 1. Environment & Hardware Diagnostics
# =============================================================================

def get_system_environment_info() -> Dict[str, str]:
    """Collects hardware and software environment metadata."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "Apple Silicon / ARM64",
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "pandas_version": pd.__version__
    }


# =============================================================================
# 2. Dataset Preparation & Feature Engineering
# =============================================================================

def load_wine_benchmark_data(data_home: str = "/tmp/scikit_learn_data", seed: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and partitions Wine dataset exactly following Grinsztajn / Pisztora benchmark:
    Train: 70%, Val: 9%, Test: 21% (N=2554 total: 1788 train, 229 val, 537 test).
    """
    X, y = fetch_openml(data_id=44091, data_home=data_home, return_X_y=True)
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)

    # Scale to [-1, 1]
    X_scaled = 2.0 * (X - X.min(0)) / X.max(0) - 1.0
    y_flat = y.map({'True': 1, 'False': 0}).to_numpy(dtype=int)
    X_mat = X_scaled.to_numpy(dtype=np.float32)

    # Stratified Train/Val/Test Split
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        X_mat, y_flat, stratify=y_flat, test_size=1.0 - 0.79, shuffle=True, random_state=seed
    )
    n_val = int(len(y_train_val) * 0.09 / 0.79)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, stratify=y_train_val, test_size=n_val, shuffle=True, random_state=seed
    )

    return x_train, y_train, x_val, y_val, x_test, y_test


def scale_to_min_max(arr: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    """Linearly rescales array into [-1, 1] given reference min and max."""
    denom = np.where(np.abs(mx - mn) < 1e-12, 1e-12, mx - mn)
    return 2.0 * (arr - mn) / denom - 1.0


def construct_original_allocator_features(
    X: np.ndarray,
    y_hat_gb: np.ndarray,
    y_hat_bb: np.ndarray,
    scaler_params: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """
    Constructs the 4 augmented features: [y_hat_gb, y_hat_bb, d_ce, d_mse]
    concatenated with X, normalized exactly as in utils_2.py.
    """
    d_ce = -np.sum(y_hat_bb * np.log(np.maximum(np.minimum(y_hat_gb, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
    d_mse = np.sum((y_hat_bb - y_hat_gb) ** 2, axis=1, keepdims=True)
    d_gb_bb = np.hstack([d_ce, d_mse])

    mn_gb, mx_gb = scaler_params['gb']
    mn_bb, mx_bb = scaler_params['bb']
    mn_d, mx_d = scaler_params['d']

    y_hat_gb_scaled = scale_to_min_max(y_hat_gb, mn_gb, mx_gb)
    y_hat_bb_scaled = scale_to_min_max(y_hat_bb, mn_bb, mx_bb)
    d_scaled = scale_to_min_max(d_gb_bb, mn_d, mx_d)

    return np.hstack([X, y_hat_gb_scaled, y_hat_bb_scaled, d_scaled])


# =============================================================================
# 3. Model Training Harness
# =============================================================================

def train_component_and_allocator_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 0
) -> Tuple[LogisticRegression, GradientBoostingClassifier, GradientBoostingRegressor, CostAwareAllocator, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Fits:
    1. Glass-Box: LogisticRegression
    2. Black-Box: GradientBoostingClassifier
    3. Original Allocator: GradientBoostingRegressor on X_alloc
    4. Cost-Aware Allocator: CostAwareAllocator on X
    """
    # 1. Glass-box
    gb_model = LogisticRegression(C=2.0, solver='saga', penalty='l1', max_iter=5000, random_state=random_state)
    gb_model.fit(x_train, y_train)

    # 2. Black-box
    bb_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=8, n_estimators=512, subsample=0.8, random_state=random_state)
    bb_model.fit(x_train, y_train)

    # 3. Predictions for original allocator training
    y_hat_gb_train = gb_model.predict_proba(x_train)
    y_hat_bb_train = bb_model.predict_proba(x_train)

    # Min/max scalers for original allocator features
    d_ce_tr = -np.sum(y_hat_bb_train * np.log(np.maximum(np.minimum(y_hat_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
    d_mse_tr = np.sum((y_hat_bb_train - y_hat_gb_train) ** 2, axis=1, keepdims=True)
    d_gb_bb_tr = np.hstack([d_ce_tr, d_mse_tr])

    scaler_params = {
        'gb': (y_hat_gb_train.min(0), y_hat_gb_train.max(0)),
        'bb': (y_hat_bb_train.min(0), y_hat_bb_train.max(0)),
        'd': (d_gb_bb_tr.min(0), d_gb_bb_tr.max(0))
    }

    # Ground truth ranking
    y_onehot_train = np.eye(2, dtype=np.float32)[y_train]
    loss_gb_train = -np.sum(y_onehot_train * np.log(np.maximum(np.minimum(y_hat_gb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
    loss_bb_train = -np.sum(y_onehot_train * np.log(np.maximum(np.minimum(y_hat_bb_train, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)

    r_train, _ = compute_ground_truth_ranking(loss_gb=loss_gb_train, loss_bb=loss_bb_train, pred_dim=2)

    # Original allocator features
    X_alloc_train = construct_original_allocator_features(x_train, y_hat_gb_train, y_hat_bb_train, scaler_params)
    orig_allocator = GradientBoostingRegressor(learning_rate=0.01, max_depth=6, n_estimators=400, subsample=0.5, random_state=random_state)
    orig_allocator.fit(X_alloc_train, r_train.ravel())

    # Cost-aware allocator (Strictly on x_train)
    ca_allocator = CostAwareAllocator(
        estimator=GradientBoostingRegressor(learning_rate=0.01, max_depth=5, n_estimators=400, subsample=0.8, random_state=random_state),
        random_state=random_state
    )
    ca_allocator.fit(X_train=x_train, y_train=y_train, loss_gb_train=loss_gb_train, loss_bb_train=loss_bb_train, pred_dim=2)

    return gb_model, bb_model, orig_allocator, ca_allocator, scaler_params


# =============================================================================
# 4. Latency Benchmark Timing Core
# =============================================================================

def time_callable(fn, n_warmup: int = 10, n_repetitions: int = 50) -> Tuple[float, float, float, float, List[float]]:
    """
    Executes a callable with warm-up runs and measures wall-clock execution times.

    Returns:
    --------
    mean_ms, std_ms, min_ms, max_ms, all_ms
    """
    # 1. Warm-up
    for _ in range(n_warmup):
        fn()

    # 2. Timing loops
    timings = []
    for _ in range(n_repetitions):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    timings = np.array(timings)
    return float(np.mean(timings)), float(np.std(timings)), float(np.min(timings)), float(np.max(timings)), timings.tolist()


def run_full_inference_benchmark(
    x_test: np.ndarray,
    gb_model: LogisticRegression,
    bb_model: GradientBoostingClassifier,
    orig_allocator: GradientBoostingRegressor,
    ca_allocator: CostAwareAllocator,
    scaler_params: Dict[str, Tuple[np.ndarray, np.ndarray]],
    q_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    n_warmup: int = 10,
    n_repetitions: int = 50,
    batch_size: int = None
) -> pd.DataFrame:
    """
    Runs the complete set of required timing measurements:
    1. Glass-box inference
    2. Black-box inference
    3. Original allocator inference
    4. Cost-aware allocator inference
    5. Original complete ensemble inference (across q)
    6. Cost-aware complete ensemble inference (across q)
    """
    N = len(x_test) if batch_size is None else min(batch_size, len(x_test))
    X_eval = x_test[:N]

    # Pre-computed component predictions for isolated allocator timing
    y_hat_gb_pre = gb_model.predict_proba(X_eval)
    y_hat_bb_pre = bb_model.predict_proba(X_eval)
    X_alloc_eval = construct_original_allocator_features(X_eval, y_hat_gb_pre, y_hat_bb_pre, scaler_params)

    results_records = []

    def record_metric(pipeline_step: str, category: str, q_val: Any, mean_ms: float, std_ms: float, min_ms: float, max_ms: float):
        results_records.append({
            "pipeline_step": pipeline_step,
            "category": category,
            "q_quota": q_val,
            "batch_size": N,
            "mean_latency_ms": round(mean_ms, 4),
            "std_latency_ms": round(std_ms, 4),
            "min_latency_ms": round(min_ms, 4),
            "max_latency_ms": round(max_ms, 4),
            "ms_per_observation": round(mean_ms / N, 6),
            "repetitions": n_repetitions,
            "warmup_runs": n_warmup
        })

    print(f"\n--- Benchmarking Isolated Components (N = {N}) ---")

    # 1. Glass-Box Inference
    mean_ms, std_ms, min_ms, max_ms, _ = time_callable(
        lambda: gb_model.predict_proba(X_eval),
        n_warmup=n_warmup, n_repetitions=n_repetitions
    )
    record_metric("1_GlassBox_Inference", "Component", np.nan, mean_ms, std_ms, min_ms, max_ms)
    print(f"  [1] Glass-box inference:            {mean_ms:.3f} ms ± {std_ms:.3f} ms ({mean_ms/N*1000:.2f} µs/obs)")

    # 2. Black-Box Inference
    mean_ms, std_ms, min_ms, max_ms, _ = time_callable(
        lambda: bb_model.predict_proba(X_eval),
        n_warmup=n_warmup, n_repetitions=n_repetitions
    )
    record_metric("2_BlackBox_Inference", "Component", np.nan, mean_ms, std_ms, min_ms, max_ms)
    print(f"  [2] Black-box inference:            {mean_ms:.3f} ms ± {std_ms:.3f} ms ({mean_ms/N*1000:.2f} µs/obs)")

    # 3. Original Allocator Inference (isolated regressor call on X_alloc)
    mean_ms, std_ms, min_ms, max_ms, _ = time_callable(
        lambda: orig_allocator.predict(X_alloc_eval),
        n_warmup=n_warmup, n_repetitions=n_repetitions
    )
    record_metric("3_Original_Allocator_Inference", "Allocator", np.nan, mean_ms, std_ms, min_ms, max_ms)
    print(f"  [3] Original allocator inference:   {mean_ms:.3f} ms ± {std_ms:.3f} ms ({mean_ms/N*1000:.2f} µs/obs)")

    # 4. Cost-Aware Allocator Inference (isolated regressor call on X)
    mean_ms, std_ms, min_ms, max_ms, _ = time_callable(
        lambda: ca_allocator.predict_ranking(X_eval),
        n_warmup=n_warmup, n_repetitions=n_repetitions
    )
    record_metric("4_CostAware_Allocator_Inference", "Allocator", np.nan, mean_ms, std_ms, min_ms, max_ms)
    print(f"  [4] Cost-aware allocator inference: {mean_ms:.3f} ms ± {std_ms:.3f} ms ({mean_ms/N*1000:.2f} µs/obs)")

    # 5. Original Complete Ensemble Inference (Across q levels)
    # MUST execute: g(X) + b(X) + distance features + feature scaling + orig_allocator + routing + final pred
    print(f"\n--- Benchmarking Complete Ensemble Pipelines Across q ---")

    for q in q_levels:
        def run_original_complete_ensemble():
            p_gb = gb_model.predict_proba(X_eval)
            p_bb = bb_model.predict_proba(X_eval)
            x_alloc = construct_original_allocator_features(X_eval, p_gb, p_bb, scaler_params)
            r_pred = orig_allocator.predict(x_alloc)
            a_mask = (r_pred > (1.0 - q)).astype(float).reshape(-1, 1)
            ens_prob = a_mask * p_gb + (1.0 - a_mask) * p_bb
            return np.argmax(ens_prob, axis=1)

        mean_ms, std_ms, min_ms, max_ms, _ = time_callable(
            run_original_complete_ensemble,
            n_warmup=n_warmup, n_repetitions=n_repetitions
        )
        record_metric(f"5_Original_Ensemble_Complete", "Original_Pipeline", q, mean_ms, std_ms, min_ms, max_ms)
        print(f"  [5] Original complete pipeline (q={q:.1f}):   {mean_ms:.3f} ms ± {std_ms:.3f} ms ({mean_ms/N:.4f} ms/obs)")

    # 6. Cost-Aware Complete Ensemble Inference (Across q levels)
    # MUST execute: ca_allocator(X) + routing decision + execute ONLY selected model + final pred
    for q in q_levels:
        def run_cost_aware_complete_ensemble():
            r_hat_ca = ca_allocator.predict_ranking(X_eval)
            a_dec = (r_hat_ca > (1.0 - q)).astype(int)
            return execute_cost_aware_inference(
                X=X_eval,
                a_decisions=a_dec,
                gb_model=gb_model,
                bb_model=bb_model,
                problem_type='classification',
                pred_dim=2
            )

        mean_ms, std_ms, min_ms, max_ms, _ = time_callable(
            run_cost_aware_complete_ensemble,
            n_warmup=n_warmup, n_repetitions=n_repetitions
        )
        record_metric(f"6_CostAware_Ensemble_Complete", "CostAware_Pipeline", q, mean_ms, std_ms, min_ms, max_ms)
        print(f"  [6] Cost-aware complete pipeline (q={q:.1f}): {mean_ms:.3f} ms ± {std_ms:.3f} ms ({mean_ms/N:.4f} ms/obs)")

    df_results = pd.DataFrame(results_records)

    # Attach Environment Metadata as DataFrame attributes / columns
    env_info = get_system_environment_info()
    for k, v in env_info.items():
        df_results[f"env_{k}"] = v

    return df_results


# =============================================================================
# 5. CLI Execution & CSV Export
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark actual wall-clock inference latency for EEG vs Cost-Aware EEG.")
    parser.add_argument("--data_home", default="/tmp/scikit_learn_data", type=str, help="Cache directory for OpenML datasets.")
    parser.add_argument("--n_warmup", default=10, type=int, help="Number of warm-up iterations.")
    parser.add_argument("--n_repetitions", default=50, type=int, help="Number of repeated timing loops.")
    parser.add_argument("--output_csv", default="results/csv/inference_timing.csv", type=str, help="Output path for timing results.")
    args = parser.parse_args()

    print("=" * 70)
    print("INFERENCE LATENCY BENCHMARK: ORIGINAL EEG VS. COST-AWARE EEG")
    print("=" * 70)

    env = get_system_environment_info()
    print("Environment Specifications:")
    for k, v in env.items():
        print(f"  - {k}: {v}")

    # 1. Load data
    print("\nLoading Wine benchmark dataset...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_wine_benchmark_data(data_home=args.data_home)
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # 2. Train models
    print("\nTraining component and allocator models...")
    gb_model, bb_model, orig_allocator, ca_allocator, scaler_params = train_component_and_allocator_models(
        x_train, y_train, random_state=0
    )
    acc_gb = gb_model.score(x_test, y_test) * 100.0
    acc_bb = bb_model.score(x_test, y_test) * 100.0
    print(f"  Fitted Component Models: Glass-Box Acc = {acc_gb:.2f}%, Black-Box Acc = {acc_bb:.2f}%")

    # 3. Benchmark across full batch (N=537)
    print("\n" + "=" * 70)
    print(f"RUNNING BENCHMARK: FULL TEST BATCH (N = {len(x_test)})")
    print("=" * 70)
    df_batch = run_full_inference_benchmark(
        x_test=x_test,
        gb_model=gb_model,
        bb_model=bb_model,
        orig_allocator=orig_allocator,
        ca_allocator=ca_allocator,
        scaler_params=scaler_params,
        q_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        n_warmup=args.n_warmup,
        n_repetitions=args.n_repetitions,
        batch_size=len(x_test)
    )

    # 4. Benchmark single-query latency (N=1)
    print("\n" + "=" * 70)
    print(f"RUNNING BENCHMARK: SINGLE-QUERY ONLINE INFERENCE (N = 1)")
    print("=" * 70)
    df_single = run_full_inference_benchmark(
        x_test=x_test,
        gb_model=gb_model,
        bb_model=bb_model,
        orig_allocator=orig_allocator,
        ca_allocator=ca_allocator,
        scaler_params=scaler_params,
        q_levels=[0.1, 0.5, 0.9],
        n_warmup=args.n_warmup,
        n_repetitions=args.n_repetitions,
        batch_size=1
    )

    # Combine results
    df_all = pd.concat([df_batch, df_single], ignore_index=True)

    # Export to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_all.to_csv(args.output_csv, index=False)
    print(f"\n[SUCCESS] Benchmark results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
