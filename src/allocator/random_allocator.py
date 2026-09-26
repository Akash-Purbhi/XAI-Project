"""
Random allocator baseline for EEG replication.

Implements random allocation baseline over multiple repetitions:
For any explainability level q, randomly allocates approximately q proportion
of observations to the glass-box model g, with the remaining assigned to b.
"""

from typing import List, Tuple
import numpy as np
from src.allocator.sufficiency import compute_sufficient_performance


def evaluate_random_allocator(
    s_g: np.ndarray,
    s_b: np.ndarray,
    q_grid: np.ndarray,
    n_repetitions: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute average performance curve for random allocation over multiple repetitions.
    
    To maintain valid comparison and monotonicity within each random trial,
    a random permutation of sample indices is drawn for each trial, and at level q,
    the first int(round(q * n)) observations are allocated to g.
    
    Args:
        s_g: Sufficiency indicators for glass box (n,)
        s_b: Sufficiency indicators for black box (n,)
        q_grid: Array of explainability levels q in [0, 1]
        n_repetitions: Number of random runs to average over
        seed: Random seed
        
    Returns:
        mean_perf_curve: Array of shape (len(q_grid),) with average sufficient performance
    """
    n = len(s_g)
    perf_matrix = np.zeros((n_repetitions, len(q_grid)), dtype=np.float32)
    rng = np.random.RandomState(seed)

    for rep in range(n_repetitions):
        perm = rng.permutation(n)
        for q_idx, q in enumerate(q_grid):
            n_g = int(round(q * n))
            alloc = np.zeros(n, dtype=np.float32)
            if n_g > 0:
                alloc[perm[:n_g]] = 1.0
            perf_matrix[rep, q_idx] = compute_sufficient_performance(s_g, s_b, alloc)

    mean_perf_curve = np.mean(perf_matrix, axis=0)
    return mean_perf_curve
