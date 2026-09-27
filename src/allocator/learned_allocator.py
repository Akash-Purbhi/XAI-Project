"""
Learned EEG Allocator and Ensembling Mechanism.

Implements:
1. Feature augmentation: [x, g(x), b(x), d_ce(g(x), b(x)), d_mse(g(x), b(x))]
2. Feature-dependent allocator a'_q (regressor predicting normalized desirability rank r)
3. Feature-independent allocator a''_q (distance-based "assume black-box is correct" rule)
4. Model-selection / ensembling between a'_q and a''_q on validation performance per q
"""

from typing import Tuple, Optional, List
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from src.allocator.sufficiency import compute_sufficient_performance


def build_augmented_features(
    X: np.ndarray,
    preds_gb: np.ndarray,
    preds_bb: np.ndarray,
    task_type: str,
    probs_gb: Optional[np.ndarray] = None,
    probs_bb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct augmented feature matrix for allocator:
    [x, g(x), b(x), d_ce, d_mse]
    
    Args:
        X: Original feature matrix (n, p)
        preds_gb: Predictions of glass-box model (n,)
        preds_bb: Predictions of black-box model (n,)
        task_type: 'classification' or 'regression'
        probs_gb: Optional glass-box class probabilities (n, 2)
        probs_bb: Optional black-box class probabilities (n, 2)
        
    Returns:
        X_aug: Augmented feature matrix (n, p + 4)
    """
    n = len(X)
    X = np.asarray(X, dtype=np.float32)

    if task_type == "classification" and probs_gb is not None and probs_bb is not None:
        p_gb = probs_gb[:, 1] if probs_gb.ndim == 2 else probs_gb
        p_bb = probs_bb[:, 1] if probs_bb.ndim == 2 else probs_bb

        # Mean squared distance
        d_mse = (p_gb - p_bb) ** 2

        # Cross-entropy distance between predicted distributions
        eps = 1e-12
        p_gb_c = np.clip(p_gb, eps, 1.0 - eps)
        p_bb_c = np.clip(p_bb, eps, 1.0 - eps)
        d_ce = -(p_gb_c * np.log(p_bb_c) + (1.0 - p_gb_c) * np.log(1.0 - p_bb_c))

        g_col = p_gb.reshape(-1, 1)
        b_col = p_bb.reshape(-1, 1)
    else:
        g_val = np.asarray(preds_gb, dtype=np.float32).reshape(-1, 1)
        b_val = np.asarray(preds_bb, dtype=np.float32).reshape(-1, 1)

        d_mse = (g_val - b_val) ** 2
        d_ce = np.abs(g_val - b_val)  # L1 distance for regression

        g_col = g_val
        b_col = b_val

    d_mse_col = np.asarray(d_mse, dtype=np.float32).reshape(-1, 1)
    d_ce_col = np.asarray(d_ce, dtype=np.float32).reshape(-1, 1)

    X_aug = np.hstack([X, g_col, b_col, d_ce_col, d_mse_col])
    return X_aug.astype(np.float32)


class EEGAllocator:
    """
    EEG Learned Allocator combining feature-dependent a'_q and feature-independent a''_q.
    """

    def __init__(
        self,
        allocator_type: str = "GradientBoostingRegressor",
        learning_rate: float = 0.05,
        max_depth: int = 5,
        n_estimators: int = 100,
        subsample: float = 0.8,
        random_state: int = 0,
    ):
        self.allocator_type = allocator_type
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.random_state = random_state

        self.regressor = GradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self.fitted = False

    def fit(self, X_aug_train: np.ndarray, r_train: np.ndarray) -> "EEGAllocator":
        """
        Fit allocator to predict normalized desirability ranking r in [0, 1].
        """
        self.regressor.fit(X_aug_train, r_train)
        self.fitted = True
        return self

    def predict_rank(self, X_aug: np.ndarray) -> np.ndarray:
        """Predict desirability ranking score for observations."""
        if not self.fitted:
            raise RuntimeError("Allocator must be fitted before predicting.")
        return self.regressor.predict(X_aug)


def evaluate_eeg_ensemble(
    r_hat_val: np.ndarray,
    r_hat_test: np.ndarray,
    d_val: np.ndarray,
    d_test: np.ndarray,
    s_gb_val: np.ndarray,
    s_bb_val: np.ndarray,
    s_gb_test: np.ndarray,
    s_bb_test: np.ndarray,
    q_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Evaluate EEG ensemble across the explainability q grid.
    
    At each q:
    1. Evaluates feature-dependent allocator a'_q on validation set.
    2. Evaluates feature-independent allocator a''_q on validation set.
    3. Selects whichever allocator achieves higher validation sufficient performance.
    4. Applies chosen allocator to test set.
    5. Returns test performance curve and PCFA choices.
    
    Args:
        r_hat_val, r_hat_test: Predicted desirability rankings for val and test
        d_val, d_test: Disagreement distance for val and test
        s_gb_val, s_bb_val: Validation sufficiency indicators
        s_gb_test, s_bb_test: Test sufficiency indicators
        q_grid: Array of explainability levels q in [0, 1]
        
    Returns:
        perf_curve_test: Array of sufficient performance on test set across q_grid
        perf_curve_feat_test: Array of purely feature-dependent a'_q performance
        pcfa_choices: List of binary choices (1 if a'_q selected, 0 if a''_q selected)
    """
    n_val = len(s_gb_val)
    n_test = len(s_gb_test)

    # Pre-sort indices for feature-dependent allocator (highest r_hat first)
    sort_feat_val = np.argsort(-r_hat_val)
    sort_feat_test = np.argsort(-r_hat_test)

    # Pre-sort indices for feature-independent allocator (lowest distance first)
    sort_dist_val = np.argsort(d_val)
    sort_dist_test = np.argsort(d_test)

    perf_curve_test = np.zeros(len(q_grid), dtype=np.float32)
    perf_curve_feat_test = np.zeros(len(q_grid), dtype=np.float32)
    pcfa_choices = []

    for q_idx, q in enumerate(q_grid):
        n_g_val = int(round(q * n_val))
        n_g_test = int(round(q * n_test))

        # a'_q: Feature-dependent allocation
        alloc_feat_val = np.zeros(n_val, dtype=np.float32)
        if n_g_val > 0:
            alloc_feat_val[sort_feat_val[:n_g_val]] = 1.0
        perf_feat_val = compute_sufficient_performance(s_gb_val, s_bb_val, alloc_feat_val)

        # a''_q: Feature-independent allocation
        alloc_dist_val = np.zeros(n_val, dtype=np.float32)
        if n_g_val > 0:
            alloc_dist_val[sort_dist_val[:n_g_val]] = 1.0
        perf_dist_val = compute_sufficient_performance(s_gb_val, s_bb_val, alloc_dist_val)

        # Select allocator based on validation performance
        # Default ties to feature-independent a''_q unless a'_q strictly outperforms
        use_feat = bool(perf_feat_val > perf_dist_val)
        pcfa_choices.append(1.0 if use_feat else 0.0)

        # Test allocations
        alloc_feat_test = np.zeros(n_test, dtype=np.float32)
        if n_g_test > 0:
            alloc_feat_test[sort_feat_test[:n_g_test]] = 1.0
        perf_curve_feat_test[q_idx] = compute_sufficient_performance(s_gb_test, s_bb_test, alloc_feat_test)

        if use_feat:
            perf_curve_test[q_idx] = perf_curve_feat_test[q_idx]
        else:
            alloc_dist_test = np.zeros(n_test, dtype=np.float32)
            if n_g_test > 0:
                alloc_dist_test[sort_dist_test[:n_g_test]] = 1.0
            perf_curve_test[q_idx] = compute_sufficient_performance(s_gb_test, s_bb_test, alloc_dist_test)

    return perf_curve_test, perf_curve_feat_test, pcfa_choices
