# -*- coding: utf-8 -*-
"""
Validation-Based Threshold Selection and Multi-Objective Optimization for Cost-Aware EEG.

This module provides rigorous validation-only threshold selection:
1. Strict isolation: Validation data ONLY is used for threshold selection and lambda tuning.
2. Mode 1 (Matched Coverage): Selects thresholds achieving exact or nearest empirical coverage target q.
3. Mode 2 (Coverage-Constrained Cost-Performance Optimization):
   Minimizes combined objective J = L_norm + lambda_cost * C_norm subject to actual coverage >= q_min.
4. Robust edge-case handling:
   - Empty/infeasible candidate sets
   - All-glass-box (q=1) and all-black-box (q=0) allocations
   - Tied scores (deterministic stable tie-breaking)
   - NaNs and non-finite inputs
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


# =============================================================================
# 1. Metric Evaluation for a Single Candidate Threshold
# =============================================================================

def evaluate_threshold_on_validation(
    r_hat: np.ndarray,
    y_true: np.ndarray,
    y_hat_gb: np.ndarray,
    y_hat_bb: np.ndarray,
    threshold: float,
    problem_type: str = 'classification',
    pred_dim: int = 2,
    cost_gb: float = 1.0,
    cost_bb: float = 10.0,
    lambda_cost: float = 0.1
) -> Dict[str, Any]:
    """
    Evaluates predictive loss, compute cost, and combined objective J for a single candidate threshold.

    Parameters:
    -----------
    r_hat : np.ndarray of shape (N,)
        Predicted desirability ranking scores on validation data.
    y_true : np.ndarray of shape (N,) or (N, K)
        Ground truth validation labels.
    y_hat_gb : np.ndarray of shape (N, K) or (N,)
        Glass-box validation predictions (probabilities for classification).
    y_hat_bb : np.ndarray of shape (N, K) or (N,)
        Black-box validation predictions (probabilities for classification).
    threshold : float
        Candidate cutoff: a = 1 if r_hat > threshold else 0.
    problem_type : str, default='classification'
    pred_dim : int, default=2
    cost_gb : float, default=1.0
    cost_bb : float, default=10.0
    lambda_cost : float, default=0.1
        Weight on normalized cost in J = L_norm + lambda_cost * C_norm.

    Returns:
    --------
    metrics : dict containing loss, coverage, counts, cost, and objective J.
    """
    N = len(r_hat)
    if N == 0:
        raise ValueError("Cannot evaluate threshold on empty validation set.")

    # Sanitize r_hat
    r_clean = np.asarray(r_hat, dtype=np.float32).ravel()
    if not np.all(np.isfinite(r_clean)):
        r_clean = np.nan_to_num(r_clean, nan=0.5, posinf=1.0, neginf=0.0)

    # 1. Routing decisions: high scores (> threshold) select Glass-Box
    a_decisions = (r_clean > threshold).astype(int)
    n_gb = int(np.sum(a_decisions == 1))
    n_bb = int(N - n_gb)
    coverage_gb = float(n_gb / N)

    # 2. Compute Ensemble Predictions
    y_true_arr = np.asarray(y_true)
    if problem_type == 'classification':
        if y_true_arr.ndim == 1 or (y_true_arr.ndim == 2 and y_true_arr.shape[1] == 1):
            y_onehot = np.eye(pred_dim, dtype=np.float32)[y_true_arr.ravel().astype(int)]
            y_labels = y_true_arr.ravel().astype(int)
        else:
            y_onehot = y_true_arr.astype(np.float32)
            y_labels = np.argmax(y_onehot, axis=1)

        a_col = a_decisions.reshape(-1, 1)
        prob_ens = a_col * y_hat_gb + (1.0 - a_col) * y_hat_bb
        prob_clipped = np.maximum(np.minimum(prob_ens, 1.0 - 1e-7), 1e-7)

        # Cross-entropy loss
        ce_loss = float(-np.mean(np.sum(y_onehot * np.log(prob_clipped), axis=1)))
        # Accuracy
        preds_ens = np.argmax(prob_ens, axis=1)
        accuracy = float(np.mean(preds_ens == y_labels) * 100.0)

        # Normalized loss L_norm in [0, 1] relative to uniform chance log(K)
        max_loss = max(1e-6, float(np.log(pred_dim)))
        loss_norm = float(np.clip(ce_loss / max_loss, 0.0, 2.0))
        primary_loss = ce_loss

    elif problem_type == 'regression':
        y_flat = y_true_arr.ravel().astype(np.float32)
        y_pred = a_decisions * y_hat_gb.ravel() + (1.0 - a_decisions) * y_hat_bb.ravel()

        mse_loss = float(np.mean((y_flat - y_pred) ** 2))
        rmse = float(np.sqrt(mse_loss))
        accuracy = rmse

        # Normalized loss L_norm by target variance
        y_var = float(np.var(y_flat))
        loss_norm = float(mse_loss / max(1e-6, y_var))
        primary_loss = mse_loss

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    # 3. Compute Normalized Cost C_norm in [0, 1]
    # Cost = n_gb * C_G + n_bb * C_B
    # Min possible cost = N * C_G (100% GB)
    # Max possible cost = N * C_B (100% BB)
    realized_cost = float(n_gb * cost_gb + n_bb * cost_bb)
    min_cost = N * cost_gb
    max_cost = N * cost_bb
    cost_range = max(1e-12, max_cost - min_cost)
    cost_norm = float(np.clip((realized_cost - min_cost) / cost_range, 0.0, 1.0))

    # 4. Combined Objective: J = L_norm + lambda_cost * C_norm
    j_objective = float(loss_norm + lambda_cost * cost_norm)

    return {
        "threshold": float(threshold),
        "primary_loss": primary_loss,
        "accuracy": accuracy,
        "loss_norm": loss_norm,
        "cost_norm": cost_norm,
        "j_objective": j_objective,
        "coverage_gb": coverage_gb,
        "n_gb": n_gb,
        "n_bb": n_bb,
        "realized_cost": realized_cost,
        "lambda_cost": float(lambda_cost)
    }


# =============================================================================
# 2. Mode 1: Matched-Coverage Threshold Selection
# =============================================================================

def find_matched_coverage_threshold(r_hat_val: np.ndarray, target_q: float) -> float:
    """
    Finds validation threshold tau such that actual coverage on validation closely matches target_q.

    Convention:
    a = 1 if r_hat > tau
    Target coverage: fraction target_q should have a = 1 (i.e. top target_q fraction).
    Theoretical quantile: tau = quantile(r_hat, 1.0 - target_q).

    Handles edge cases:
    - target_q = 0.0 -> tau > max(r_hat) (all BB)
    - target_q = 1.0 -> tau < min(r_hat) (all GB)
    - Identical/tied scores: deterministic selection
    """
    r_clean = np.asarray(r_hat_val, dtype=np.float32).ravel()
    if len(r_clean) == 0:
        return 0.5

    # Sanitize NaNs
    if not np.all(np.isfinite(r_clean)):
        r_clean = np.nan_to_num(r_clean, nan=0.5, posinf=1.0, neginf=0.0)

    target_q = float(np.clip(target_q, 0.0, 1.0))

    if target_q <= 0.0:
        return float(np.max(r_clean) + 1.0)  # All a = 0 (100% BB)
    if target_q >= 1.0:
        return float(np.min(r_clean) - 1.0)  # All a = 1 (100% GB)

    # Cutoff at the (1 - target_q) quantile
    percentile_val = (1.0 - target_q) * 100.0
    threshold = float(np.percentile(r_clean, percentile_val))
    return threshold


# =============================================================================
# 3. Mode 2: Coverage-Constrained Cost-Performance Optimization
# =============================================================================

def optimize_cost_aware_threshold(
    r_hat_val: np.ndarray,
    y_val: np.ndarray,
    y_hat_gb_val: np.ndarray,
    y_hat_bb_val: np.ndarray,
    q_min: float = 0.20,
    lambda_cost: float = 0.1,
    problem_type: str = 'classification',
    pred_dim: int = 2,
    cost_gb: float = 1.0,
    cost_bb: float = 10.0,
    num_candidates: int = 101
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """
    Searches candidate thresholds on validation data to minimize:
        J = L_norm + lambda_cost * C_norm
    subject to:
        actual_coverage_gb >= q_min.

    Parameters:
    -----------
    r_hat_val : np.ndarray
        Validation ranking predictions from allocator.
    y_val : np.ndarray
        Ground-truth validation targets.
    y_hat_gb_val : np.ndarray
        Glass-box validation predictions.
    y_hat_bb_val : np.ndarray
        Black-box validation predictions.
    q_min : float, default=0.20
        Minimum required explainability coverage.
    lambda_cost : float, default=0.1
        Weight of normalized compute cost in objective.
    problem_type : str, default='classification'
    pred_dim : int, default=2
    cost_gb : float, default=1.0
    cost_bb : float, default=10.0
    num_candidates : int, default=101
        Number of candidate thresholds to evaluate.

    Returns:
    --------
    best_threshold : float
    best_metrics : dict
    candidates_df : pd.DataFrame with all evaluated candidates
    """
    r_clean = np.asarray(r_hat_val, dtype=np.float32).ravel()
    if not np.all(np.isfinite(r_clean)):
        r_clean = np.nan_to_num(r_clean, nan=0.5, posinf=1.0, neginf=0.0)

    # 1. Generate candidate thresholds across score range with margin
    r_min = float(np.min(r_clean))
    r_max = float(np.max(r_clean))

    if abs(r_max - r_min) < 1e-7:
        # All equal scores: only all-GB or all-BB possible
        candidates = [r_min - 0.1, r_min + 0.1]
    else:
        # Include boundary conditions
        interior_candidates = np.linspace(r_min - 1e-4, r_max + 1e-4, num_candidates)
        # Also include exact sample percentiles
        percentile_candidates = np.percentile(r_clean, np.linspace(0, 100, 21))
        candidates = np.unique(np.concatenate([
            [r_min - 1.0, r_max + 1.0],  # All-GB, All-BB
            interior_candidates,
            percentile_candidates
        ]))

    candidates = np.sort(candidates)

    eval_records = []
    feasible_records = []

    for tau in candidates:
        m = evaluate_threshold_on_validation(
            r_hat=r_clean,
            y_true=y_val,
            y_hat_gb=y_hat_gb_val,
            y_hat_bb=y_hat_bb_val,
            threshold=tau,
            problem_type=problem_type,
            pred_dim=pred_dim,
            cost_gb=cost_gb,
            cost_bb=cost_bb,
            lambda_cost=lambda_cost
        )
        eval_records.append(m)

        # Feasibility check: actual coverage >= q_min (with 1e-6 float tolerance)
        if m["coverage_gb"] >= (q_min - 1e-6):
            feasible_records.append(m)

    candidates_df = pd.DataFrame(eval_records)

    # Handle no feasible threshold edge case:
    # If no candidate satisfies coverage >= q_min, choose threshold that maximizes coverage
    if len(feasible_records) == 0:
        max_cov_idx = candidates_df["coverage_gb"].idxmax()
        best_m = eval_records[max_cov_idx]
        best_tau = float(best_m["threshold"])
    else:
        # Sort by J ascending, with deterministic tie-breaking (prefer higher coverage, then lower threshold)
        feasible_records.sort(key=lambda x: (x["j_objective"], -x["coverage_gb"], x["threshold"]))
        best_m = feasible_records[0]
        best_tau = float(best_m["threshold"])

    return best_tau, best_m, candidates_df


# =============================================================================
# 4. High-Level ThresholdSelector Class
# =============================================================================

class ThresholdSelector:
    """
    Manages validation-only threshold tuning for both Matched-Coverage and Cost-Optimized modes.
    Ensures zero test-label leakage.
    """

    def __init__(
        self,
        q_targets: List[float] = [0.20, 0.40, 0.60, 0.80],
        cost_lambdas: List[float] = [0.0, 0.1, 1.0],
        cost_gb: float = 1.0,
        cost_bb: float = 10.0,
        problem_type: str = 'classification',
        pred_dim: int = 2
    ):
        self.q_targets = q_targets
        self.cost_lambdas = cost_lambdas
        self.cost_gb = cost_gb
        self.cost_bb = cost_bb
        self.problem_type = problem_type
        self.pred_dim = pred_dim

        # Fitted calibration lookup tables (validation only)
        self.matched_thresholds_: Dict[float, float] = {}
        self.cost_optimized_thresholds_: Dict[Tuple[float, float], float] = {}
        self.validation_summary_: Optional[pd.DataFrame] = None
        self.is_calibrated = False

    def calibrate(
        self,
        r_hat_val: np.ndarray,
        y_val: np.ndarray,
        y_hat_gb_val: np.ndarray,
        y_hat_bb_val: np.ndarray
    ) -> "ThresholdSelector":
        """
        Calibrates thresholds strictly on validation data.
        Test data must NEVER be passed to this method.
        """
        summary_rows = []

        # 1. Mode 1: Matched Coverage Thresholds
        for q in self.q_targets:
            tau_matched = find_matched_coverage_threshold(r_hat_val, target_q=q)
            self.matched_thresholds_[q] = tau_matched

            m = evaluate_threshold_on_validation(
                r_hat=r_hat_val,
                y_true=y_val,
                y_hat_gb=y_hat_gb_val,
                y_hat_bb=y_hat_bb_val,
                threshold=tau_matched,
                problem_type=self.problem_type,
                pred_dim=self.pred_dim,
                cost_gb=self.cost_gb,
                cost_bb=self.cost_bb,
                lambda_cost=0.0
            )
            m["mode"] = "matched_coverage"
            m["target_q"] = q
            summary_rows.append(m)

        # 2. Mode 2: Coverage-Constrained Cost-Performance Optimization
        for q_min in self.q_targets:
            for lam in self.cost_lambdas:
                tau_opt, m_opt, _ = optimize_cost_aware_threshold(
                    r_hat_val=r_hat_val,
                    y_val=y_val,
                    y_hat_gb_val=y_hat_gb_val,
                    y_hat_bb_val=y_hat_bb_val,
                    q_min=q_min,
                    lambda_cost=lam,
                    problem_type=self.problem_type,
                    pred_dim=self.pred_dim,
                    cost_gb=self.cost_gb,
                    cost_bb=self.cost_bb
                )
                self.cost_optimized_thresholds_[(q_min, lam)] = tau_opt

                m_opt["mode"] = "cost_optimized"
                m_opt["target_q"] = q_min
                m_opt["lambda_cost"] = lam
                summary_rows.append(m_opt)

        self.validation_summary_ = pd.DataFrame(summary_rows)
        self.is_calibrated = True
        return self

    def get_threshold(self, mode: str = "matched_coverage", q: float = 0.5, lambda_cost: float = 0.1) -> float:
        """Retrieves calibrated threshold for specified mode and parameters."""
        if not self.is_calibrated:
            raise RuntimeError("ThresholdSelector must be calibrated on validation data before retrieving thresholds.")

        if mode == "matched_coverage":
            if q in self.matched_thresholds_:
                return self.matched_thresholds_[q]
            return find_matched_coverage_threshold(r_hat_val=None, target_q=q)

        elif mode == "cost_optimized":
            key = (q, lambda_cost)
            if key in self.cost_optimized_thresholds_:
                return self.cost_optimized_thresholds_[key]
            raise KeyError(f"No calibrated threshold for (q_min={q}, lambda={lambda_cost}). Available: {list(self.cost_optimized_thresholds_.keys())}")

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'matched_coverage' or 'cost_optimized'.")

    def route_test(
        self,
        r_hat_test: np.ndarray,
        mode: str = "matched_coverage",
        q: float = 0.5,
        lambda_cost: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """
        Applies pre-calibrated validation threshold to test scores r_hat_test.
        Ensures zero test-label exposure.
        """
        threshold = self.get_threshold(mode=mode, q=q, lambda_cost=lambda_cost)
        r_clean = np.asarray(r_hat_test, dtype=np.float32).ravel()
        a_decisions = (r_clean > threshold).astype(int)
        return a_decisions, threshold
