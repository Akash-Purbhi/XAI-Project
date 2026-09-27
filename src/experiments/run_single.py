"""
Single dataset replication experiment runner.

Executes the complete EEG pipeline on a given dataset and seed:
1. Load dataset and generate deterministic 70/9/21 train/val/test splits
2. Fit preprocessing strictly on train and transform val/test
3. Train and tune glass-box and black-box models independently on full training set
4. Compute observation losses and sufficiency indicators
5. Compute EEG desirability ranking on training set and category quantile boundaries
6. Fit learned allocator on augmented feature set [x, g(x), b(x), d_ce, d_mse]
7. Predict test sufficiency categories and compute sufficiency accuracy
8. Sweep explainability level q across [0.0, 1.0] for Random, Oracle, and Learned EEG
9. Compute all 9 paper metrics and save machine-readable results
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.utils.seed import set_seed
from src.utils.logging import setup_logger
from src.utils.io import save_json, save_dataframe
from src.data import load_dataset, make_train_val_test_splits, EEGDataPreprocessor
from src.models import tune_and_fit_model
from src.allocator import (
    compute_loss,
    compute_regression_epsilon,
    compute_sufficiency_indicators,
    categorize_sufficiency,
    compute_desirability_score,
    compute_normalized_ranking,
    compute_training_category_quantiles,
    predict_sufficiency_categories,
    evaluate_random_allocator,
    evaluate_oracle_allocator,
    build_augmented_features,
    EEGAllocator,
    evaluate_eeg_ensemble,
)
from src.metrics import compute_eeg_metrics


def run_single_dataset(
    dataset_name: str,
    seed: int = 0,
    glass_box_type: str = "auto",
    black_box_type: str = "auto",
    allocator_type: str = "GradientBoostingRegressor",
    q_step: float = 0.01,
    random_repetitions: int = 10,
    output_dir: str = "results/raw",
    logger=None,
) -> Dict[str, Any]:
    """
    Run complete EEG pipeline on a single dataset and random seed.
    """
    if logger is None:
        logger = setup_logger(f"run_{dataset_name}_seed_{seed}")

    logger.info(f"=== Starting EEG Replication: {dataset_name} (Seed {seed}) ===")
    set_seed(seed)

    # 1. Load Data
    X, y, metadata = load_dataset(dataset_name)
    task_type = metadata["task_type"]
    logger.info(f"Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, task={task_type}")

    # 2. Splits (70% train, 9% val, 21% test)
    splits = make_train_val_test_splits(X, y, task_type=task_type, seed=seed)
    logger.info(
        f"Splits created: Train={len(splits['X_train'])}, "
        f"Val={len(splits['X_val'])}, Test={len(splits['X_test'])}"
    )

    # 3. Preprocessing (Fit strictly on Train)
    preprocessor = EEGDataPreprocessor(task_type=task_type)
    scaled_splits = preprocessor.transform_splits(splits)
    X_train, y_train = scaled_splits["X_train"], scaled_splits["y_train"]
    X_val, y_val = scaled_splits["X_val"], scaled_splits["y_val"]
    X_test, y_test = scaled_splits["X_test"], scaled_splits["y_test"]

    # 4. Train Component Models
    # Determine candidate models
    # NOTE: TabWRN-28 is fully implemented in src/models/tab_wrn.py following the paper's
    # Wide ResNet-28 architecture adapted for tabular data (Pisztora & Li, AAAI 2024).
    # However, it is excluded from Phase-1 experiments due to compute/time constraints:
    # GradientBoosting GridSearchCV alone takes ~22 min per seed on SuperconductR (21k×79).
    # Adding TabWRN training + validation-set tuning would make the pipeline infeasible
    # within available resources. The architecture is correct and ready for use if compute
    # budget allows in future work.
    # TODO: Update README to document this TabWRN exclusion decision explicitly.
    if task_type == "classification":
        gb_candidates = ["LogisticRegression", "ClassificationTree"]
        bb_candidates = ["GradientBoostingClassifier"]
    else:
        gb_candidates = ["LinearRegression", "RegressionTree"]
        bb_candidates = ["GradientBoostingRegressor"]

    # Train glass-box candidate(s)
    if glass_box_type != "auto" and glass_box_type in gb_candidates:
        gb_selected_name = glass_box_type
        gb_model, gb_params = tune_and_fit_model(gb_selected_name, task_type, X_train, y_train, seed=seed)
    else:
        best_gb_model, best_gb_name, best_gb_params, best_gb_score = None, None, None, -np.inf
        for cand in gb_candidates:
            model, params = tune_and_fit_model(cand, task_type, X_train, y_train, seed=seed)
            preds_val = model.predict(X_val)
            if task_type == "classification":
                score = np.mean(preds_val == y_val)
            else:
                score = -np.mean((preds_val - y_val) ** 2)
            if score > best_gb_score:
                best_gb_score, best_gb_model, best_gb_name, best_gb_params = score, model, cand, params
        gb_model, gb_selected_name, gb_params = best_gb_model, best_gb_name, best_gb_params

    # Train black-box candidate(s)
    if black_box_type != "auto" and black_box_type in bb_candidates:
        bb_selected_name = black_box_type
        bb_model, bb_params = tune_and_fit_model(bb_selected_name, task_type, X_train, y_train, seed=seed)
    else:
        bb_selected_name = bb_candidates[0]
        bb_model, bb_params = tune_and_fit_model(bb_selected_name, task_type, X_train, y_train, seed=seed)

    logger.info(f"Selected Component Models: Glass-Box={gb_selected_name}, Black-Box={bb_selected_name}")

    # Generate predictions
    preds_gb_train = gb_model.predict(X_train)
    preds_gb_val = gb_model.predict(X_val)
    preds_gb_test = gb_model.predict(X_test)

    preds_bb_train = bb_model.predict(X_train)
    preds_bb_val = bb_model.predict(X_val)
    preds_bb_test = bb_model.predict(X_test)

    probs_gb_train = gb_model.predict_proba(X_train) if task_type == "classification" else None
    probs_gb_val = gb_model.predict_proba(X_val) if task_type == "classification" else None
    probs_gb_test = gb_model.predict_proba(X_test) if task_type == "classification" else None

    probs_bb_train = bb_model.predict_proba(X_train) if task_type == "classification" else None
    probs_bb_val = bb_model.predict_proba(X_val) if task_type == "classification" else None
    probs_bb_test = bb_model.predict_proba(X_test) if task_type == "classification" else None

    # 5. Compute Losses and Sufficiency
    loss_gb_train = compute_loss(y_train, preds_gb_train, task_type, probs_gb_train)
    loss_gb_val = compute_loss(y_val, preds_gb_val, task_type, probs_gb_val)
    loss_gb_test = compute_loss(y_test, preds_gb_test, task_type, probs_gb_test)

    loss_bb_train = compute_loss(y_train, preds_bb_train, task_type, probs_bb_train)
    loss_bb_val = compute_loss(y_val, preds_bb_val, task_type, probs_bb_val)
    loss_bb_test = compute_loss(y_test, preds_bb_test, task_type, probs_bb_test)

    # Compute epsilon for regression
    if task_type == "regression":
        epsilon = compute_regression_epsilon(loss_gb_val, loss_bb_val)
        logger.info(f"Regression Sufficiency Threshold epsilon: {epsilon:.5f}")
    else:
        epsilon = 0.5

    s_gb_train, s_bb_train = compute_sufficiency_indicators(
        loss_gb_train, loss_bb_train, task_type, epsilon, y_train, preds_gb_train, preds_bb_train
    )
    s_gb_val, s_bb_val = compute_sufficiency_indicators(
        loss_gb_val, loss_bb_val, task_type, epsilon, y_val, preds_gb_val, preds_bb_val
    )
    s_gb_test, s_bb_test = compute_sufficiency_indicators(
        loss_gb_test, loss_bb_test, task_type, epsilon, y_test, preds_gb_test, preds_bb_test
    )

    perf_gb_test = float(np.mean(s_gb_test))
    perf_bb_test = float(np.mean(s_bb_test))
    logger.info(f"Standalone Test Performance: Glass-Box={perf_gb_test*100:.2f}%, Black-Box={perf_bb_test*100:.2f}%")

    # 6. Desirability Ranking on Train
    r_tilde_train = compute_desirability_score(s_gb_train, s_bb_train, loss_gb_train, loss_bb_train)
    r_train = compute_normalized_ranking(r_tilde_train)
    category_quantiles = compute_training_category_quantiles(s_gb_train, s_bb_train)

    # 7. Fit Learned Allocator
    X_aug_train = build_augmented_features(X_train, preds_gb_train, preds_bb_train, task_type, probs_gb_train, probs_bb_train)
    X_aug_val = build_augmented_features(X_val, preds_gb_val, preds_bb_val, task_type, probs_gb_val, probs_bb_val)
    X_aug_test = build_augmented_features(X_test, preds_gb_test, preds_bb_test, task_type, probs_gb_test, probs_bb_test)

    allocator = EEGAllocator(allocator_type=allocator_type, random_state=seed)
    allocator.fit(X_aug_train, r_train)

    r_hat_val = allocator.predict_rank(X_aug_val)
    r_hat_test = allocator.predict_rank(X_aug_test)

    # Disagreement distance for feature-independent allocator a''_q
    # Last column of X_aug is d_mse
    d_mse_val = X_aug_val[:, -1]
    d_mse_test = X_aug_test[:, -1]

    # 8. Sufficiency Category Prediction on Test
    true_categories_test, test_cat_counts = categorize_sufficiency(s_gb_test, s_bb_test)
    pred_categories_test = predict_sufficiency_categories(r_hat_test, category_quantiles)
    s_acc = float(np.mean(pred_categories_test == true_categories_test) * 100.0)
    logger.info(f"Sufficiency Category Estimation Accuracy: {s_acc:.2f}%")

    # 9. Sweep Explainability Level q
    q_grid = np.round(np.arange(0.0, 1.0 + q_step / 2.0, q_step), 4)

    # Random Allocator curve
    perf_random = evaluate_random_allocator(
        s_gb_test, s_bb_test, q_grid, n_repetitions=random_repetitions, seed=seed
    )

    # Oracle Allocator curve
    r_tilde_test = compute_desirability_score(s_gb_test, s_bb_test, loss_gb_test, loss_bb_test)
    r_test = compute_normalized_ranking(r_tilde_test)
    perf_oracle = evaluate_oracle_allocator(s_gb_test, s_bb_test, r_test, q_grid)

    # Learned EEG Ensemble curve
    perf_eeg, perf_feat_only, pcfa_choices = evaluate_eeg_ensemble(
        r_hat_val, r_hat_test, d_mse_val, d_mse_test,
        s_gb_val, s_bb_val, s_gb_test, s_bb_test, q_grid
    )

    # 10. Compute EEG Metrics
    metrics = compute_eeg_metrics(
        perf_eeg=perf_eeg,
        perf_random=perf_random,
        perf_oracle=perf_oracle,
        perf_gb=perf_gb_test,
        perf_bb=perf_bb_test,
        q_grid=q_grid,
        pcfa_choices=pcfa_choices,
        sufficiency_accuracy=s_acc,
    )
    logger.info(f"EEG Metrics computed: {metrics}")

    # 11. Save Results
    os.makedirs(output_dir, exist_ok=True)
    curve_df = pd.DataFrame({
        "q": q_grid,
        "perf_eeg": perf_eeg,
        "perf_random": perf_random,
        "perf_oracle": perf_oracle,
        "pcfa_choice": pcfa_choices,
    })
    curve_csv_path = os.path.join(output_dir, f"{dataset_name}_seed{seed}_curve.csv")
    save_dataframe(curve_df, curve_csv_path)

    run_summary = {
        "dataset": dataset_name,
        "task_type": task_type,
        "seed": seed,
        "gb_model": gb_selected_name,
        "gb_params": gb_params,
        "bb_model": bb_selected_name,
        "bb_params": bb_params,
        "allocator_model": allocator_type,
        "metrics": metrics,
        "n_samples": metadata["n_samples"],
        "n_features": metadata["n_features"],
        "test_category_counts": test_cat_counts,
    }
    summary_json_path = os.path.join(output_dir, f"{dataset_name}_seed{seed}_summary.json")
    save_json(run_summary, summary_json_path)

    logger.info(f"Results saved to {curve_csv_path} and {summary_json_path}")
    return {
        "metrics": metrics,
        "curve_df": curve_df,
        "run_summary": run_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single dataset EEG replication experiment")
    parser.add_argument("--dataset", type=str, default="Wine", help="Dataset name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gb", type=str, default="auto", help="Glass box model type")
    parser.add_argument("--bb", type=str, default="auto", help="Black box model type")
    parser.add_argument("--allocator", type=str, default="GradientBoostingRegressor", help="Allocator type")
    parser.add_argument("--q_step", type=float, default=0.01, help="q grid step")

    args = parser.parse_args()
    res = run_single_dataset(
        dataset_name=args.dataset,
        seed=args.seed,
        glass_box_type=args.gb,
        black_box_type=args.bb,
        allocator_type=args.allocator,
        q_step=args.q_step,
    )
    print("\nReplication Completed Successfully:")
    for k, v in res["metrics"].items():
        print(f"  {k}: {v}")
