"""
Oracle allocator for EEG replication.

Implements the theoretical upper-bound allocator defined in AAAI 2024:
The oracle allocator has access to the true sufficiency information and represents
the optimal allocation under the experimental definition:
a*_q(z) = I{r(z) > 1 - q}
where r(z) is the true normalized allocation desirability rank.
"""

from typing import Tuple
import numpy as np
from src.allocator.sufficiency import compute_sufficient_performance


def evaluate_oracle_allocator(
    s_g: np.ndarray,
    s_b: np.ndarray,
    true_ranking: np.ndarray,
    q_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute oracle allocator performance across the q grid.
    
    Args:
        s_g: Sufficiency indicators for glass box (n,)
        s_b: Sufficiency indicators for black box (n,)
        true_ranking: Normalized ranking r(z) in (0, 1] computed from ground truth
        q_grid: Array of explainability levels q in [0, 1]
        
    Returns:
        perf_curve: Array of shape (len(q_grid),) with oracle sufficient performance
    """
    n = len(s_g)
    perf_curve = np.zeros(len(q_grid), dtype=np.float32)

    # Sort indices by ranking descending (highest desirability for g first)
    sorted_indices = np.argsort(-true_ranking)

    for q_idx, q in enumerate(q_grid):
        n_g = int(round(q * n))
        alloc = np.zeros(n, dtype=np.float32)
        if n_g > 0:
            alloc[sorted_indices[:n_g]] = 1.0
        perf_curve[q_idx] = compute_sufficient_performance(s_g, s_b, alloc)

    return perf_curve
