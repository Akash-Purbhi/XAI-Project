"""
EEG allocation desirability ranking and sufficiency quantile estimation.

Implements the core EEG ranking formula from AAAI 2024:
r_tilde(z) = 2 * s_g(z) - s_b(z) - sigma(l_U(g(x), y) - l_U(b(x), y))
where sigma(x) = 1 / (1 + exp(-x))

r(z) = rank_Dn(r_tilde(z)) / n
"""

from typing import Dict, Tuple
import numpy as np
from scipy.stats import rankdata


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def compute_desirability_score(
    s_g: np.ndarray,
    s_b: np.ndarray,
    loss_gb: np.ndarray,
    loss_bb: np.ndarray,
) -> np.ndarray:
    """
    Compute raw allocation desirability score r_tilde(z):
    r_tilde(z) = 2 * s_g - s_b - sigma(l_U(g) - l_U(b))
    
    Category value ranges:
    Zg (s_g=1, s_b=0) -> (1, 2)
    Z2 (s_g=1, s_b=1) -> (0, 1)
    Z0 (s_g=0, s_b=0) -> (-1, 0)
    Zb (s_g=0, s_b=1) -> (-2, -1)
    """
    s_g = np.asarray(s_g, dtype=np.float32)
    s_b = np.asarray(s_b, dtype=np.float32)
    delta_loss = np.asarray(loss_gb, dtype=np.float32) - np.asarray(loss_bb, dtype=np.float32)

    r_tilde = 2.0 * s_g - s_b - sigmoid(delta_loss)
    return r_tilde


def compute_normalized_ranking(r_tilde: np.ndarray) -> np.ndarray:
    """
    Compute normalized percentile ranking r(z) in (0, 1]:
    r(z) = rank_Dn(r_tilde(z)) / n
    Higher rank means higher desirability for allocation to glass-box model g.
    """
    n = len(r_tilde)
    ranks = rankdata(r_tilde, method="average")
    return ranks / float(n)


def compute_training_category_quantiles(
    s_g_train: np.ndarray,
    s_b_train: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute category boundary quantiles on training set:
    Sorted order: Zb, Z0, Z2, Zg
    c_Zb = n_b / n
    c_Z0 = (n_b + n_0) / n
    c_Z2 = (n_b + n_0 + n_2) / n
    """
    n = float(len(s_g_train))
    s_g = np.asarray(s_g_train, dtype=int)
    s_b = np.asarray(s_b_train, dtype=int)

    n_b = np.sum((s_g == 0) & (s_b == 1))
    n_0 = np.sum((s_g == 0) & (s_b == 0))
    n_2 = np.sum((s_g == 1) & (s_b == 1))

    c_Zb = float(n_b / n)
    c_Z0 = float((n_b + n_0) / n)
    c_Z2 = float((n_b + n_0 + n_2) / n)

    return c_Zb, c_Z0, c_Z2


def predict_sufficiency_categories(
    r_hat: np.ndarray,
    quantiles: Tuple[float, float, float],
) -> np.ndarray:
    """
    Predict sufficiency categories from predicted rank r_hat using training quantile boundaries.
    
    Args:
        r_hat: Predicted ranking in [0, 1]
        quantiles: (c_Zb, c_Z0, c_Z2)
        
    Returns:
        pred_cat: Array of predicted categories ('Zb', 'Z0', 'Z2', 'Zg')
    """
    c_Zb, c_Z0, c_Z2 = quantiles
    r = np.asarray(r_hat, dtype=np.float32)

    pred_cat = np.where(
        r <= c_Zb,
        "Zb",
        np.where(
            r <= c_Z0,
            "Z0",
            np.where(r <= c_Z2, "Z2", "Zg"),
        ),
    )
    return pred_cat
