"""
Unit tests for EEG sufficiency indicator and categorization logic.
"""

import unittest
import numpy as np
from src.allocator.sufficiency import (
    compute_loss,
    compute_regression_epsilon,
    compute_sufficiency_indicators,
    categorize_sufficiency,
    compute_sufficient_performance,
)


class TestSufficiency(unittest.TestCase):
    def test_classification_sufficiency(self):
        """Test classification sufficiency: s_f(z) = I{f(x) == y}."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])

        s_g, _ = compute_sufficiency_indicators(
            loss_gb=np.zeros(4),
            loss_bb=np.zeros(4),
            task_type="classification",
            y_true=y_true,
            y_pred_gb=y_pred,
            y_pred_bb=y_pred,
        )
        np.testing.assert_array_equal(s_g, [1, 0, 1, 0])

    def test_regression_sufficiency_and_epsilon(self):
        """Test regression sufficiency: s_f(z) = I{loss < epsilon}."""
        loss_val_gb = np.array([0.1, 0.3, 0.2])  # mean = 0.2
        loss_val_bb = np.array([0.05, 0.15, 0.1])  # mean = 0.1
        epsilon = compute_regression_epsilon(loss_val_gb, loss_val_bb)
        self.assertAlmostEqual(epsilon, 0.1, places=5)

        loss_test = np.array([0.05, 0.15, 0.08, 0.2])
        s_g, _ = compute_sufficiency_indicators(
            loss_gb=loss_test,
            loss_bb=loss_test,
            task_type="regression",
            epsilon=epsilon,
        )
        np.testing.assert_array_equal(s_g, [1, 0, 1, 0])

    def test_four_sufficiency_categories(self):
        """Verify the 4 disjoint sufficiency categories Zg, Zb, Z2, Z0."""
        s_g = np.array([1, 0, 1, 0])
        s_b = np.array([0, 1, 1, 0])

        categories, counts = categorize_sufficiency(s_g, s_b)
        np.testing.assert_array_equal(categories, ["Zg", "Zb", "Z2", "Z0"])
        self.assertEqual(counts["Zg"], 1)
        self.assertEqual(counts["Zb"], 1)
        self.assertEqual(counts["Z2"], 1)
        self.assertEqual(counts["Z0"], 1)

    def test_sufficient_ensemble_performance(self):
        """Test t_bar(a) = (1/n) * sum(s_g * a + s_b * (1 - a))."""
        s_g = np.array([1, 0, 1, 0])
        s_b = np.array([0, 1, 1, 0])

        # Allocate [1, 0, 1, 0] -> selects s_g for index 0 and 2, s_b for 1 and 3
        # s_g[0]=1, s_b[1]=1, s_g[2]=1, s_b[3]=0 -> 3/4 = 0.75
        alloc = np.array([1, 0, 1, 0])
        perf = compute_sufficient_performance(s_g, s_b, alloc)
        self.assertAlmostEqual(perf, 0.75)


if __name__ == "__main__":
    unittest.main()
