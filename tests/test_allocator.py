"""
Unit tests for EEG desirability ranking, monotonicity, and allocation mechanisms.
"""

import unittest
import numpy as np
from src.allocator.desirability import (
    compute_desirability_score,
    compute_normalized_ranking,
    compute_training_category_quantiles,
    predict_sufficiency_categories,
)
from src.allocator.random_allocator import evaluate_random_allocator
from src.allocator.oracle_allocator import evaluate_oracle_allocator


class TestAllocator(unittest.TestCase):
    def test_desirability_category_ordering(self):
        """
        Verify the mathematical ranking property established in AAAI 2024:
        r_tilde(Zg) in (1, 2) > r_tilde(Z2) in (0, 1) > r_tilde(Z0) in (-1, 0) > r_tilde(Zb) in (-2, -1)
        """
        # Represent one sample from each category: [Zg, Z2, Z0, Zb]
        s_g = np.array([1, 1, 0, 0])
        s_b = np.array([0, 1, 0, 1])
        loss_g = np.array([0.1, 0.2, 0.3, 0.4])
        loss_b = np.array([0.4, 0.2, 0.5, 0.1])

        r_tilde = compute_desirability_score(s_g, s_b, loss_g, loss_b)

        # Check category score bounds
        self.assertTrue(1.0 < r_tilde[0] < 2.0, f"Zg out of bounds: {r_tilde[0]}")
        self.assertTrue(0.0 < r_tilde[1] < 1.0, f"Z2 out of bounds: {r_tilde[1]}")
        self.assertTrue(-1.0 < r_tilde[2] < 0.0, f"Z0 out of bounds: {r_tilde[2]}")
        self.assertTrue(-2.0 < r_tilde[3] < -1.0, f"Zb out of bounds: {r_tilde[3]}")

        # Strict ordering
        self.assertTrue(r_tilde[0] > r_tilde[1] > r_tilde[2] > r_tilde[3])

    def test_monotonic_allocation(self):
        """
        Proposition 3 (Monotone Allocation):
        For any q_i < q_j, observations allocated to g at q_i must remain allocated to g at q_j.
        """
        n = 100
        rng = np.random.RandomState(42)
        r_hat = rng.rand(n)
        sorted_indices = np.argsort(-r_hat)

        q_grid = [0.1, 0.3, 0.5, 0.8, 1.0]
        allocations = []
        for q in q_grid:
            n_g = int(round(q * n))
            alloc = set(sorted_indices[:n_g])
            allocations.append(alloc)

        for i in range(len(allocations) - 1):
            # allocation at q_i must be a subset of allocation at q_j
            self.assertTrue(allocations[i].issubset(allocations[i + 1]))

    def test_oracle_greater_than_or_equal_to_random(self):
        """Oracle allocator must outperform or match random allocator across q."""
        np.random.seed(42)
        n = 200
        s_g = (np.random.rand(n) > 0.4).astype(np.float32)
        s_b = (np.random.rand(n) > 0.3).astype(np.float32)
        loss_g = np.random.rand(n)
        loss_b = np.random.rand(n)

        r_tilde = compute_desirability_score(s_g, s_b, loss_g, loss_b)
        r = compute_normalized_ranking(r_tilde)

        q_grid = np.linspace(0.0, 1.0, 11)
        perf_oracle = evaluate_oracle_allocator(s_g, s_b, r, q_grid)
        perf_random = evaluate_random_allocator(s_g, s_b, q_grid, n_repetitions=20, seed=42)

        # AUC(oracle) >= AUC(random)
        auc_oracle = np.trapz(perf_oracle, q_grid)
        auc_random = np.trapz(perf_random, q_grid)
        self.assertGreaterEqual(auc_oracle, auc_random - 1e-4)


if __name__ == "__main__":
    unittest.main()
