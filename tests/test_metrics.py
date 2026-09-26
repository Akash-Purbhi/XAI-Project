"""
Unit tests for all 9 EEG evaluation metrics from AAAI 2024.
"""

import unittest
import numpy as np
from src.metrics.eeg_metrics import compute_auc, compute_eeg_metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.q_grid = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0]

    def test_auc_calculation(self):
        """Test trapezoidal AUC integration."""
        # Constant line y = 0.8 over [0, 1] has AUC = 0.80
        perf = np.full(len(self.q_grid), 0.8)
        auc = compute_auc(perf, self.q_grid)
        self.assertAlmostEqual(auc, 0.8, places=4)

    def test_ppcr_known_cases(self):
        """Test PPCR formula: (AUC(a) - AUC(r)) / (AUC(o) - AUC(r))."""
        perf_rand = np.full(len(self.q_grid), 0.70)
        perf_orac = np.full(len(self.q_grid), 0.90)

        # Case 1: Allocator matches random -> PPCR = 0%
        m_zero = compute_eeg_metrics(
            perf_eeg=perf_rand,
            perf_random=perf_rand,
            perf_oracle=perf_orac,
            perf_gb=0.70,
            perf_bb=0.70,
            q_grid=self.q_grid,
            pcfa_choices=[1.0] * len(self.q_grid),
            sufficiency_accuracy=75.0,
        )
        self.assertAlmostEqual(m_zero["PPCR"], 0.0, delta=0.5)

        # Case 2: Allocator matches oracle -> PPCR = 100%
        m_full = compute_eeg_metrics(
            perf_eeg=perf_orac,
            perf_random=perf_rand,
            perf_oracle=perf_orac,
            perf_gb=0.70,
            perf_bb=0.70,
            q_grid=self.q_grid,
            pcfa_choices=[1.0] * len(self.q_grid),
            sufficiency_accuracy=75.0,
        )
        self.assertAlmostEqual(m_full["PPCR"], 100.0, delta=0.5)

    def test_pqeom_and_pqom(self):
        """Test PQEOM (% >= max individual) and PQOM (% > max individual)."""
        gb_perf = 0.80
        bb_perf = 0.80
        # 11 points: 3 points at 0.75 (< max), 5 points at 0.80 (== max), 3 points at 0.85 (> max)
        perf_eeg = np.array([0.75, 0.75, 0.75, 0.80, 0.80, 0.80, 0.80, 0.80, 0.85, 0.85, 0.85])
        perf_dummy = np.full(11, 0.75)

        m = compute_eeg_metrics(
            perf_eeg=perf_eeg,
            perf_random=perf_dummy,
            perf_oracle=perf_dummy,
            perf_gb=gb_perf,
            perf_bb=bb_perf,
            q_grid=self.q_grid,
            pcfa_choices=[1.0] * 11,
            sufficiency_accuracy=80.0,
        )

        # >= 0.80 occurs 8 times out of 11: 8/11 * 100 = 72.73%
        self.assertAlmostEqual(m["PQEOM"], (8 / 11) * 100.0, delta=0.5)
        # > 0.80 occurs 3 times out of 11: 3/11 * 100 = 27.27%
        self.assertAlmostEqual(m["PQOM"], (3 / 11) * 100.0, delta=0.5)

    def test_95tqm(self):
        """Test 95TQM: highest q where performance >= 0.95 * max(g, b)."""
        gb_perf = 1.00
        bb_perf = 1.00
        # 0.95 * 1.0 = 0.95
        # perf stays >= 0.95 up to index 7 (q = 0.70)
        perf_eeg = np.array([0.98, 0.97, 0.96, 0.96, 0.95, 0.95, 0.95, 0.95, 0.90, 0.85, 0.80])
        perf_dummy = np.full(11, 0.80)

        m = compute_eeg_metrics(
            perf_eeg=perf_eeg,
            perf_random=perf_dummy,
            perf_oracle=perf_dummy,
            perf_gb=gb_perf,
            perf_bb=bb_perf,
            q_grid=self.q_grid,
            pcfa_choices=[1.0] * 11,
            sufficiency_accuracy=80.0,
        )
        self.assertAlmostEqual(m["95TQM"], 70.0, delta=0.5)

    def test_max_acc_and_argmax_q(self):
        """Test Max Acc and highest q that maintains maximum performance."""
        # Max is 0.90 at q=0.4 and q=0.7
        perf_eeg = np.array([0.80, 0.82, 0.85, 0.88, 0.90, 0.89, 0.88, 0.90, 0.85, 0.80, 0.75])
        perf_dummy = np.full(11, 0.75)

        m = compute_eeg_metrics(
            perf_eeg=perf_eeg,
            perf_random=perf_dummy,
            perf_oracle=perf_dummy,
            perf_gb=0.80,
            perf_bb=0.75,
            q_grid=self.q_grid,
            pcfa_choices=[1.0] * 11,
            sufficiency_accuracy=80.0,
        )
        self.assertAlmostEqual(m["Max_Acc"], 90.0, delta=0.1)
        # Argmax q should be 0.70 (i.e. 70.0%), the highest q with max performance
        self.assertAlmostEqual(m["Argmax_q"], 70.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
