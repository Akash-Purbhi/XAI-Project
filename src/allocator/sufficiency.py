"""
Sufficiency indicators and categories for EEG replication.

Implements the paper's definitions:
- Classification: s_f(z) = I{f(x) == y}
- Regression: s_f(z) = I{l_U(f(x), y) < epsilon}
  where epsilon is the lower of the average validation losses of g and b.
- 4 Sufficiency categories:
  Zg: g sufficient, b insufficient
  Zb: g insufficient, b sufficient
  Z2: both sufficient
  Z0: neither sufficient
"""

from typing import Tuple, Dict
import numpy as np


def compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_prob: np.ndarray = None,
) -> np.ndarray:
    """
    Compute observation-level underlying task loss l_U(f(x), y).
    
    Args:
        y_true: Ground truth targets (n,)
        y_pred: Model predictions (n,)
        task_type: 'classification' or 'regression'
        y_prob: Optional predicted class probabilities (n, 2)
        
    Returns:
        losses: Array of shape (n,) containing per-sample loss
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if task_type == "classification":
        if y_prob is not None and y_prob.ndim == 2:
            eps = 1e-12
            prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
            y_int = y_true.astype(int)
            # Cross-entropy loss per sample
            loss = -np.log(prob_clipped[np.arange(len(y_int)), y_int])
            return loss
        else:
            # 0/1 misclassification loss
            return (y_pred != y_true).astype(np.float32)
    else:
        # Squared error loss
        return (y_pred - y_true) ** 2


def compute_regression_epsilon(
    loss_gb_val: np.ndarray,
    loss_bb_val: np.ndarray,
) -> float:
    """
    Compute epsilon cutoff for regression sufficiency:
    lower of average validation losses of g and b.
    """
    mean_val_gb = float(np.mean(loss_gb_val))
    mean_val_bb = float(np.mean(loss_bb_val))
    epsilon = min(mean_val_gb, mean_val_bb)
    return epsilon


def compute_sufficiency_indicators(
    loss_gb: np.ndarray,
    loss_bb: np.ndarray,
    task_type: str,
    epsilon: float = 0.5,
    y_true: np.ndarray = None,
    y_pred_gb: np.ndarray = None,
    y_pred_bb: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute binary sufficiency indicators s_g and s_b for each sample.
    
    Args:
        loss_gb: Losses for glass-box model (n,)
        loss_bb: Losses for black-box model (n,)
        task_type: 'classification' or 'regression'
        epsilon: Sufficiency threshold for regression
        y_true, y_pred_gb, y_pred_bb: Optional discrete predictions for classification
        
    Returns:
        s_g: Binary array (n,) where 1 indicates g is sufficient
        s_b: Binary array (n,) where 1 indicates b is sufficient
    """
    if task_type == "classification":
        if y_true is not None and y_pred_gb is not None and y_pred_bb is not None:
            s_g = (np.asarray(y_pred_gb) == np.asarray(y_true)).astype(np.float32)
            s_b = (np.asarray(y_pred_bb) == np.asarray(y_true)).astype(np.float32)
        else:
            s_g = (loss_gb < 0.5).astype(np.float32)
            s_b = (loss_bb < 0.5).astype(np.float32)
    else:
        s_g = (loss_gb < epsilon).astype(np.float32)
        s_b = (loss_bb < epsilon).astype(np.float32)

    return s_g, s_b


def categorize_sufficiency(
    s_g: np.ndarray,
    s_b: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Categorize observations into the 4 sufficiency categories:
    'Zg', 'Zb', 'Z2', 'Z0'.
    
    Returns:
        categories: Array of strings of shape (n,)
        counts: Dictionary with counts of each category
    """
    s_g = np.asarray(s_g, dtype=int)
    s_b = np.asarray(s_b, dtype=int)

    categories = np.empty(len(s_g), dtype=object)

    mask_Zg = (s_g == 1) & (s_b == 0)
    mask_Zb = (s_g == 0) & (s_b == 1)
    mask_Z2 = (s_g == 1) & (s_b == 1)
    mask_Z0 = (s_g == 0) & (s_b == 0)

    categories[mask_Zg] = "Zg"
    categories[mask_Zb] = "Zb"
    categories[mask_Z2] = "Z2"
    categories[mask_Z0] = "Z0"

    counts = {
        "Zg": int(np.sum(mask_Zg)),
        "Zb": int(np.sum(mask_Zb)),
        "Z2": int(np.sum(mask_Z2)),
        "Z0": int(np.sum(mask_Z0)),
    }
    return categories, counts


def compute_sufficient_performance(
    s_g: np.ndarray,
    s_b: np.ndarray,
    allocation: np.ndarray,
) -> float:
    """
    Compute sufficient ensemble performance:
    t_bar(a) = (1/n) * sum(s_g * a + s_b * (1 - a))
    
    Args:
        s_g: Sufficiency indicator for glass box (n,)
        s_b: Sufficiency indicator for black box (n,)
        allocation: Binary allocation vector (n,) where 1 is allocated to g, 0 to b
        
    Returns:
        Scalar sufficient ensemble performance in [0, 1]
    """
    s_g = np.asarray(s_g, dtype=np.float32)
    s_b = np.asarray(s_b, dtype=np.float32)
    alloc = np.asarray(allocation, dtype=np.float32)

    return float(np.mean(s_g * alloc + s_b * (1.0 - alloc)))
