"""
Unit tests for data splitting and preprocessing.
"""

import unittest
import numpy as np
from src.data.splits import make_train_val_test_splits
from src.data.preprocessing import EEGDataPreprocessor


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 10
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y_cls = np.random.randint(0, 2, self.n_samples)
        self.y_reg = (np.random.rand(self.n_samples) * 50.0 + 10.0).astype(np.float32)

    def test_split_proportions(self):
        """Test exact 70/9/21 train/val/test split proportions."""
        splits = make_train_val_test_splits(self.X, self.y_cls, task_type="classification", seed=42)
        n_train = len(splits["X_train"])
        n_val = len(splits["X_val"])
        n_test = len(splits["X_test"])

        self.assertEqual(n_train + n_val + n_test, self.n_samples)
        # Check proportions within ±1 sample
        self.assertAlmostEqual(n_train / self.n_samples, 0.70, delta=0.01)
        self.assertAlmostEqual(n_val / self.n_samples, 0.09, delta=0.01)
        self.assertAlmostEqual(n_test / self.n_samples, 0.21, delta=0.01)

    def test_scaling_train_only_fit(self):
        """Ensure scaling statistics are fitted strictly on train data to prevent leakage."""
        splits = make_train_val_test_splits(self.X, self.y_reg, task_type="regression", seed=42)
        preprocessor = EEGDataPreprocessor(task_type="regression")
        scaled = preprocessor.transform_splits(splits)

        # Train data must be strictly bounded in [-1.0, 1.0]
        self.assertAlmostEqual(float(np.min(scaled["X_train"])), -1.0, places=4)
        self.assertAlmostEqual(float(np.max(scaled["X_train"])), 1.0, places=4)
        self.assertAlmostEqual(float(np.min(scaled["y_train"])), -1.0, places=4)
        self.assertAlmostEqual(float(np.max(scaled["y_train"])), 1.0, places=4)

        # Inverse transform should reconstruct original target values
        recon_y_train = preprocessor.inverse_transform_target(scaled["y_train"])
        np.testing.assert_allclose(recon_y_train, splits["y_train"], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
