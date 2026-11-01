# -*- coding: utf-8 -*-
"""
Cost-Aware Allocation Mechanism for Ensembles with Explainability Guarantees (EEG).

This module implements a decoupled, Stage-0 (X-only) cost-aware allocation mechanism:
1. Calculates ground-truth allocation desirability ranking r exactly as in the AAAI 2024 paper.
2. Trains an allocator strictly on input features X (zero dependency on black-box predictions at test time).
3. Routes high glass-box desirability scores (r_hat > 1 - q) to the glass-box model (a = 1).
4. Routes remaining observations (r_hat <= 1 - q) to the black-box model (a = 0).
5. Executes strictly the selected model per observation to eliminate wasted computation.

Allocation Convention (Preserved from AAAI 2024):
- a = 1: Glass-box model selected (interpretable, low compute cost)
- a = 0: Black-box model selected (high-capacity, high compute cost)
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


# =============================================================================
# 1. Ground-Truth Desirability & Percentile Ranking Target
# =============================================================================

def get_percentiles_of_a_wrt_b(b, a=None, distr_min=-2.0, distr_max=2.0):
    """
    Computes empirical percentiles of array 'a' relative to the distribution of 'b'.
    Exact mathematical reproduction of utils_2.py (NumPy-only, zero TF dependency).

    Parameters:
    -----------
    b : np.ndarray of shape (N_b, 1) or (N_b,)
        Reference distribution (typically training desirability scores).
    a : np.ndarray of shape (N_a, 1) or (N_a,), optional
        Target array to map into percentiles of b. If None, computes percentiles of b.
    distr_min : float, default=-2.0
        Lower theoretical bound of desirability distribution.
    distr_max : float, default=2.0
        Upper theoretical bound of desirability distribution.

    Returns:
    --------
    p_a : np.ndarray of shape (N_a, 1)
        Empirical percentiles in [0, 1].
    """
    b = np.asarray(b).reshape(-1, 1)
    if distr_min is None or distr_max is None:
        raise ValueError("get_percentiles_of_a_wrt_b: Unknown distr min and/or max is not implemented")

    mm = np.expand_dims(np.array([distr_min, distr_max]), -1)
    b_aug = np.concatenate([b, mm], axis=0)
    sort_idxs_b = b_aug.argsort(axis=0)
    r_b = sort_idxs_b.argsort(axis=0)
    p_b = r_b / (r_b.shape[0] - 1)
    brp = np.concatenate([b_aug, r_b, p_b], axis=1)
    brp_sorted = brp[brp[:, -1].argsort()]

    if a is None:
        p_a = p_b[:-2, :]
    else:
        a_arr = np.asarray(a).flatten()
        U_idx = np.digitize(a_arr, bins=brp_sorted[:, 0])
        U_idx = np.clip(U_idx, 1, len(brp_sorted) - 1)
        L_idx = U_idx - 1
        denom = brp_sorted[U_idx, 0] - brp_sorted[L_idx, 0]
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        q = (brp_sorted[U_idx, 0] - a_arr) / denom
        q = np.nan_to_num(q, nan=0.0)
        p_a = np.expand_dims(brp_sorted[L_idx, 2] * q + brp_sorted[U_idx, 2] * (1.0 - q), -1)

    return p_a.astype(np.float32)


def compute_ground_truth_ranking(loss_gb, loss_bb, problem_type='classification', pred_dim=2, rank_type='bb_minus_gb'):
    """
    Computes continuous allocation desirability and maps it to percentile ranking r in [0, 1].

    Parameters:
    -----------
    loss_gb : np.ndarray of shape (N, 1) or (N,)
        Observation-level losses of the glass-box model.
    loss_bb : np.ndarray of shape (N, 1) or (N,)
        Observation-level losses of the black-box model.
    problem_type : str, default='classification'
        'classification' or 'regression'.
    pred_dim : int, default=2
        Number of classes for classification (used for sufficiency cutoff log(K)).
    rank_type : str, default='bb_minus_gb'
        'bb_minus_gb' (AAAI 2024 paper formulation) or 'gb'.

    Returns:
    --------
    r : np.ndarray of shape (N, 1)
        True continuous percentile rank r in [0, 1].
        Values near 1.0 represent observations most suited for the glass-box model.
    desirability : np.ndarray of shape (N, 1)
        Raw continuous desirability score in [-2, +2].
    """
    loss_gb = np.asarray(loss_gb, dtype=np.float32).reshape(-1, 1)
    loss_bb = np.asarray(loss_bb, dtype=np.float32).reshape(-1, 1)

    if problem_type == 'classification':
        loss_cutoff = float(np.log(pred_dim))
    elif problem_type == 'regression':
        loss_cutoff = float(np.minimum(np.mean(loss_gb), np.mean(loss_bb)))
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    gb_c = (loss_gb < loss_cutoff).astype(np.float32)
    bb_c = (loss_bb < loss_cutoff).astype(np.float32)

    if rank_type == 'bb_minus_gb':
        desirability = (2.0 * gb_c - bb_c - 1.0 / (1.0 + np.exp(loss_bb - loss_gb))).astype(np.float32)
    elif rank_type == 'gb':
        desirability = (2.0 * gb_c - bb_c - 1.0 / (1.0 + np.exp(-loss_gb))).astype(np.float32)
    else:
        raise ValueError(f"Unknown rank_type: {rank_type}")

    r = get_percentiles_of_a_wrt_b(b=desirability, a=None, distr_min=-2.0, distr_max=2.0)
    return r, desirability


# =============================================================================
# 2. Cost-Aware Allocator Class (X-Only Routing Gate)
# =============================================================================

class CostAwareAllocator:
    """
    Decoupled Cost-Aware Allocator (Stage-0 Gating).

    Unlike the original EEG allocator which requires black-box predictions as inputs,
    this allocator is trained and evaluated STRICTLY on input features X.
    This eliminates pre-routing execution of the black-box model.
    """

    def __init__(self, estimator=None, rank_type='bb_minus_gb', random_state=0):
        """
        Parameters:
        -----------
        estimator : sklearn-compatible regressor, optional
            Regression model used to predict continuous rank r_hat in [0, 1].
            Default: GradientBoostingRegressor with tuned parameters from Phase-2.
        rank_type : str, default='bb_minus_gb'
            Tie-breaking rule for desirability ranking ('bb_minus_gb' or 'gb').
        random_state : int, default=0
            Seed for reproducibility.
        """
        self.rank_type = rank_type
        self.random_state = random_state
        if estimator is None:
            self.estimator = GradientBoostingRegressor(
                learning_rate=0.01,
                max_depth=5,
                n_estimators=400,
                subsample=0.8,
                random_state=random_state
            )
        else:
            self.estimator = estimator

        self.is_fitted = False
        self.desirability_train_ = None

    def fit(self, X_train, y_train, gb_model=None, bb_model=None, loss_gb_train=None, loss_bb_train=None, problem_type='classification', pred_dim=2):
        """
        Fits the cost-aware allocator on training features X_train and ground-truth rank r_train.

        Parameters:
        -----------
        X_train : array-like or pd.DataFrame of shape (N, D)
            Raw input features (normalized tabular domain).
        y_train : array-like of shape (N,) or (N, K)
            Ground truth labels (one-hot or class indices for classification).
        gb_model : fitted glass-box estimator, optional
            Required if loss_gb_train is not directly provided.
        bb_model : fitted black-box estimator, optional
            Required if loss_bb_train is not directly provided.
        loss_gb_train : array-like of shape (N, 1), optional
            Pre-computed observation losses for glass-box.
        loss_bb_train : array-like of shape (N, 1), optional
            Pre-computed observation losses for black-box.
        problem_type : str, default='classification'
        pred_dim : int, default=2
        """
        X_mat = np.asarray(X_train)

        # 1. Compute observation-level losses if not provided
        if loss_gb_train is None or loss_bb_train is None:
            if gb_model is None or bb_model is None:
                raise ValueError("Must provide either (loss_gb_train, loss_bb_train) or (gb_model, bb_model).")

            y_arr = np.asarray(y_train)
            if problem_type == 'classification':
                if y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1):
                    y_onehot = np.eye(pred_dim, dtype=np.float32)[y_arr.ravel().astype(int)]
                else:
                    y_onehot = y_arr.astype(np.float32)

                y_hat_gb = gb_model.predict_proba(X_train)
                y_hat_bb = bb_model.predict_proba(X_train)

                loss_gb_train = -np.sum(y_onehot * np.log(np.maximum(np.minimum(y_hat_gb, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
                loss_bb_train = -np.sum(y_onehot * np.log(np.maximum(np.minimum(y_hat_bb, 1.0 - 1e-7), 1e-7)), axis=1, keepdims=True)
            elif problem_type == 'regression':
                y_flat = y_arr.ravel().astype(np.float32)
                y_hat_gb = gb_model.predict(X_train).ravel().astype(np.float32)
                y_hat_bb = bb_model.predict(X_train).ravel().astype(np.float32)

                loss_gb_train = ((y_flat - y_hat_gb) ** 2).reshape(-1, 1)
                loss_bb_train = ((y_flat - y_hat_bb) ** 2).reshape(-1, 1)
            else:
                raise ValueError(f"Unknown problem_type: {problem_type}")

        # 2. Ground-truth ranking target r
        r_train, desirability_train = compute_ground_truth_ranking(
            loss_gb=loss_gb_train,
            loss_bb=loss_bb_train,
            problem_type=problem_type,
            pred_dim=pred_dim,
            rank_type=self.rank_type
        )
        self.desirability_train_ = desirability_train

        # 3. Fit allocator STRICTLY on X_mat (Zero dependency on black-box predictions)
        self.estimator.fit(X_mat, r_train.ravel())
        self.is_fitted = True
        return self

    def predict_ranking(self, X):
        """
        Predicts continuous allocation ranking r_hat_CA using ONLY input features X.

        Parameters:
        -----------
        X : array-like or pd.DataFrame of shape (N, D)

        Returns:
        --------
        r_hat_ca : np.ndarray of shape (N,)
            Predicted glass-box desirability percentile ranking in [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("CostAwareAllocator must be fitted before predict_ranking.")
        X_mat = np.asarray(X)
        return self.estimator.predict(X_mat)

    def route(self, X=None, r_hat=None, q=0.5):
        """
        Determines binary allocation decisions for explainability quota q.

        Decision Rule:
        a = 1 if r_hat_CA > (1.0 - q) else 0

        Parameters:
        -----------
        X : array-like of shape (N, D), optional
            Input features (used if r_hat is None).
        r_hat : array-like of shape (N,), optional
            Precomputed ranking scores.
        q : float in [0, 1]
            Explainability quota constraint.

        Returns:
        --------
        a_decisions : np.ndarray of shape (N,)
            Binary allocation decisions: 1 = Glass-box, 0 = Black-box.
        r_hat : np.ndarray of shape (N,)
            Ranking scores used for allocation.
        """
        if r_hat is None:
            if X is None:
                raise ValueError("Must provide either X or r_hat.")
            r_hat = self.predict_ranking(X)
        else:
            r_hat = np.asarray(r_hat).ravel()

        threshold = 1.0 - float(q)
        a_decisions = (r_hat > threshold).astype(int)
        return a_decisions, r_hat


# =============================================================================
# 3. Cost-Aware Inference Execution Function
# =============================================================================

def execute_cost_aware_inference(X, a_decisions, gb_model, bb_model, problem_type='classification', pred_dim=2, cost_gb=1.0, cost_bb=10.0):
    """
    Executes conditional inference strictly evaluating only the assigned model per observation.

    Guarantees:
    - Observations with a = 1 trigger ONLY gb_model (bb_model is never called).
    - Observations with a = 0 trigger ONLY bb_model (gb_model is never called).

    Parameters:
    -----------
    X : array-like or pd.DataFrame of shape (N, D)
        Observations to evaluate.
    a_decisions : array-like of shape (N,)
        Binary routing decisions in {0, 1}. 1 = Glass-box, 0 = Black-box.
    gb_model : fitted glass-box estimator
    bb_model : fitted black-box estimator
    problem_type : str, default='classification'
    pred_dim : int, default=2
    cost_gb : float, default=1.0
        Arbitrary/relative unit compute cost of glass-box query.
    cost_bb : float, default=10.0
        Arbitrary/relative unit compute cost of black-box query (e.g. 10x higher).

    Returns:
    --------
    results : dict containing:
        - 'y_pred': np.ndarray of shape (N,) with final ensemble predictions.
        - 'y_prob': np.ndarray of shape (N, K) with class probabilities (classification only).
        - 'a_decisions': np.ndarray of shape (N,) with allocation mask.
        - 'metrics': dict of execution statistics (GB/BB calls, coverage, compute cost, savings).
    """
    N = len(a_decisions)
    a_arr = np.asarray(a_decisions).ravel().astype(int)
    X_mat = np.asarray(X)

    idx_gb = np.where(a_arr == 1)[0]
    idx_bb = np.where(a_arr == 0)[0]

    num_gb = len(idx_gb)
    num_bb = len(idx_bb)

    # 1. Classification Output Container
    if problem_type == 'classification':
        y_prob = np.zeros((N, pred_dim), dtype=np.float32)
        y_pred = np.zeros(N, dtype=int)

        t_start_gb = time.perf_counter()
        if num_gb > 0:
            X_gb_sub = X_mat[idx_gb] if not isinstance(X, pd.DataFrame) else X.iloc[idx_gb]
            prob_gb = gb_model.predict_proba(X_gb_sub)
            y_prob[idx_gb] = prob_gb
        t_gb = time.perf_counter() - t_start_gb

        t_start_bb = time.perf_counter()
        if num_bb > 0:
            X_bb_sub = X_mat[idx_bb] if not isinstance(X, pd.DataFrame) else X.iloc[idx_bb]
            prob_bb = bb_model.predict_proba(X_bb_sub)
            y_prob[idx_bb] = prob_bb
        t_bb = time.perf_counter() - t_start_bb

        y_pred = np.argmax(y_prob, axis=1)

    # 2. Regression Output Container
    elif problem_type == 'regression':
        y_prob = None
        y_pred = np.zeros(N, dtype=np.float32)

        t_start_gb = time.perf_counter()
        if num_gb > 0:
            X_gb_sub = X_mat[idx_gb] if not isinstance(X, pd.DataFrame) else X.iloc[idx_gb]
            y_pred[idx_gb] = gb_model.predict(X_gb_sub).ravel()
        t_gb = time.perf_counter() - t_start_gb

        t_start_bb = time.perf_counter()
        if num_bb > 0:
            X_bb_sub = X_mat[idx_bb] if not isinstance(X, pd.DataFrame) else X.iloc[idx_bb]
            y_pred[idx_bb] = bb_model.predict(X_bb_sub).ravel()
        t_bb = time.perf_counter() - t_start_bb

    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    # 3. Compute Realized & Baseline Compute Costs
    # Baseline EEG: executes both models on all observations (N * (C_G + C_B))
    baseline_compute_cost = N * (cost_gb + cost_bb)
    # Cost-Aware EEG: executes only the selected model per observation
    realized_compute_cost = num_gb * cost_gb + num_bb * cost_bb
    compute_savings_pct = (1.0 - (realized_compute_cost / max(1e-12, baseline_compute_cost))) * 100.0

    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'a_decisions': a_arr,
        'metrics': {
            'n_total': N,
            'num_gb_calls': num_gb,
            'num_bb_calls': num_bb,
            'actual_gb_coverage': float(num_gb / N),
            'actual_bb_coverage': float(num_bb / N),
            'time_gb_seconds': float(t_gb),
            'time_bb_seconds': float(t_bb),
            'realized_compute_cost': float(realized_compute_cost),
            'baseline_compute_cost': float(baseline_compute_cost),
            'compute_savings_pct': float(compute_savings_pct)
        }
    }
