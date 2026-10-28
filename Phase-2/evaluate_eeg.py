# -*- coding: utf-8 -*-
"""
Evaluation and Metric Computation for Ensembles with Explainability Guarantees (EEG)
and Allocation Stability Extension under Input Perturbations.

This module computes:
1. Original Paper EEG Benchmark Metrics (AAAI 2024 Table 2):
   - AUC: Area under the explainability-performance curve across q in [0, 1]
   - PPCR: Percentage Performance Captured over Random
   - PQEOM: Percent Q where ensemble performance >= max(perf(g), perf(b))
   - PQOM: Percent Q where ensemble performance > max(perf(g), perf(b))
   - PCFA: Percent Contribution of Feature-dependent Allocator
   - 95TQM: 95% Threshold Q Max
   - Max Acc: Maximum accuracy achieved across all q
   - Argmax q: Explainability level at which Max Acc is attained
   - s Acc: Sufficiency category accuracy

2. Allocation Stability Metrics (Phase 2 Research Extension):
   - Allocation Flip Rate (AFR) across q and mean AFR
   - Prediction Consistency (PC) across q
   - Performance Degradation under perturbation
   - Glass-Box Allocation Proportion compliance (Nominal vs. Actual q)
"""

import numpy as np
import pandas as pd


def compute_ensemble_predictions(a_decisions, y_hat_gb, y_hat_bb, problem_type='classification'):
    """
    Computes ensemble predictions given binary allocation decisions a_q in {0, 1}^N.
    a_q = 1 => allocated to glass-box (gb)
    a_q = 0 => allocated to black-box (bb)
    """
    a = np.asarray(a_decisions).reshape(-1, 1)
    if problem_type == 'classification':
        y_prob = a * y_hat_gb + (1.0 - a) * y_hat_bb
        y_pred = np.argmax(y_prob, axis=-1)
        return y_prob, y_pred
    elif problem_type == 'regression':
        y_pred = a.ravel() * y_hat_gb.ravel() + (1.0 - a.ravel()) * y_hat_bb.ravel()
        return y_pred, y_pred
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def compute_performance(y_true, y_pred, problem_type='classification'):
    """
    Computes task performance metric:
    - Classification: accuracy (percentage 0-100)
    - Regression: negative RMSE or R^2 / accuracy threshold (scaled to percentage 0-100 or 1 - normalized RMSE)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if problem_type == 'classification':
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true.ravel().astype(int)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = y_pred.ravel().astype(int)
        return float(np.mean(y_true_labels == y_pred_labels) * 100.0)
    elif problem_type == 'regression':
        # Normalized negative RMSE or 100 - RMSE percentage
        rmse = np.sqrt(np.mean((y_true.ravel() - y_pred.ravel()) ** 2))
        return float(rmse)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")


def compute_auc(q_grid, perf_values):
    """
    Trapezoidal AUC of performance over q in [0, 1].
    """
    q_grid = np.asarray(q_grid)
    perf_values = np.asarray(perf_values)
    return float(np.trapz(perf_values, q_grid))


def compute_original_eeg_metrics(q_grid, perf_ensemble, perf_random, perf_oracle,
                                perf_gb, perf_bb, pcfa_values=None,
                                true_sufficiency_cat=None, pred_sufficiency_cat=None,
                                problem_type='classification'):
    """
    Computes the standard EEG metrics reported in Table 2 of Pisztora & Li (AAAI 2024).
    """
    q_grid = np.asarray(q_grid)
    perf_ensemble = np.asarray(perf_ensemble)
    perf_random = np.asarray(perf_random)
    perf_oracle = np.asarray(perf_oracle)

    auc_ens = compute_auc(q_grid, perf_ensemble)
    auc_rand = compute_auc(q_grid, perf_random)
    auc_oracle = compute_auc(q_grid, perf_oracle)

    # PPCR: Percentage Performance Captured over Random
    denom = (auc_oracle - auc_rand)
    if abs(denom) < 1e-9:
        ppcr = 0.0
    else:
        ppcr = float((auc_ens - auc_rand) / denom * 100.0)

    best_component_perf = max(perf_gb, perf_bb)

    # PQEOM: % Q Equal or Over Max component model
    pqeom = float(np.mean(perf_ensemble >= (best_component_perf - 1e-5)) * 100.0)

    # PQOM: % Q strictly Over Max component model
    pqom = float(np.mean(perf_ensemble > (best_component_perf + 1e-5)) * 100.0)

    # PCFA: % Q feature-dependent allocator is used
    if pcfa_values is not None:
        pcfa = float(np.mean(np.asarray(pcfa_values) > 0.5) * 100.0)
    else:
        pcfa = 100.0  # purely feature dependent if no ensemble rule provided

    # 95TQM: Highest q where ensemble >= 95% of best component performance
    threshold_95 = 0.95 * best_component_perf if problem_type == 'classification' else 1.05 * best_component_perf
    qualifying_q = [q for q, p in zip(q_grid, perf_ensemble) if (p >= threshold_95 if problem_type == 'classification' else p <= threshold_95)]
    q_95tqm = float(max(qualifying_q) * 100.0) if len(qualifying_q) > 0 else 0.0

    # Max Acc and Argmax q
    if problem_type == 'classification':
        max_idx = int(np.argmax(perf_ensemble))
        max_acc = float(perf_ensemble[max_idx])
        argmax_q = float(q_grid[max_idx] * 100.0)
    else:
        min_idx = int(np.argmin(perf_ensemble))
        max_acc = float(perf_ensemble[min_idx])  # best RMSE
        argmax_q = float(q_grid[min_idx] * 100.0)

    # Sufficiency Accuracy (s Acc)
    if true_sufficiency_cat is not None and pred_sufficiency_cat is not None:
        s_acc = float(np.mean(np.asarray(true_sufficiency_cat) == np.asarray(pred_sufficiency_cat)) * 100.0)
    else:
        s_acc = np.nan

    return {
        'AUC': auc_ens,
        'AUC_Random': auc_rand,
        'AUC_Oracle': auc_oracle,
        'PPCR': ppcr,
        'PQEOM': pqeom,
        'PQOM': pqom,
        'PCFA': pcfa,
        '95TQM': q_95tqm,
        'Max_Acc': max_acc,
        'Argmax_q': argmax_q,
        's_Acc': s_acc
    }


def compute_allocation_flip_rate(alloc_clean, alloc_perturbed):
    """
    Computes Allocation Flip Rate (AFR) between clean and perturbed allocation decisions:
    AFR = (Number of changed allocations / Total evaluated observations) * 100%
    """
    alloc_clean = np.asarray(alloc_clean).ravel()
    alloc_perturbed = np.asarray(alloc_perturbed).ravel()
    assert len(alloc_clean) == len(alloc_perturbed), "Shape mismatch in flip rate calculation"
    flips = np.sum(alloc_clean != alloc_perturbed)
    return float(flips / len(alloc_clean) * 100.0)


def compute_prediction_consistency(y_pred_clean, y_pred_perturbed, problem_type='classification'):
    """
    Computes consistency between clean and perturbed predictions:
    - Classification: % where predicted class is identical
    - Regression: Mean Absolute Prediction Shift |y_clean - y_perturbed|
    """
    y_pred_clean = np.asarray(y_pred_clean).ravel()
    y_pred_perturbed = np.asarray(y_pred_perturbed).ravel()
    if problem_type == 'classification':
        return float(np.mean(y_pred_clean == y_pred_perturbed) * 100.0)
    elif problem_type == 'regression':
        return float(np.mean(np.abs(y_pred_clean - y_pred_perturbed)))
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")
