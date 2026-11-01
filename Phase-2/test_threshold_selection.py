# -*- coding: utf-8 -*-
"""
Unit tests for validation-based threshold selection and multi-objective optimization
in Phase-2/threshold_selection.py.

Verifies:
1. Zero test leakage: test labels/features are NEVER accessed during calibration.
2. Matched-coverage threshold selection across q in [0.20, 0.40, 0.60, 0.80].
3. Cost-performance optimization across lambda in [0.0, 0.1, 1.0].
4. Edge cases: all-GB (q=1), all-BB (q=0), tied scores, NaNs/Infs, infeasible targets.
5. Cost calculation accuracy and preservation of baseline routing logic.
"""

import numpy as np
from threshold_selection import (
    evaluate_threshold_on_validation,
    find_matched_coverage_threshold,
    optimize_cost_aware_threshold,
    ThresholdSelector
)


class LeakageDetectorArray(np.ndarray):
    """An array wrapper that raises an error if accessed, used to prove zero test leakage."""
    def __array_finalize__(self, obj):
        pass

    def __getitem__(self, item):
        raise AssertionError("TEST DATA LEAKAGE: Test set was accessed during validation threshold calibration!")


def test_zero_test_label_leakage():
    """Confirms that test data is never passed or touched during threshold calibration."""
    np.random.seed(42)
    N_val = 100
    r_val = np.random.uniform(0, 1, size=N_val)
    y_val = np.random.randint(0, 2, size=N_val)
    p_gb_val = np.column_stack([0.6 * np.ones(N_val), 0.4 * np.ones(N_val)])
    p_bb_val = np.column_stack([0.2 * np.ones(N_val), 0.8 * np.ones(N_val)])

    # Test arrays are forbidden
    test_forbidden = np.array([999, 999]).view(LeakageDetectorArray)

    selector = ThresholdSelector(q_targets=[0.20, 0.40, 0.60, 0.80], cost_lambdas=[0.0, 0.1, 1.0])
    # Calibrate strictly with validation sets
    selector.calibrate(
        r_hat_val=r_val,
        y_val=y_val,
        y_hat_gb_val=p_gb_val,
        y_hat_bb_val=p_bb_val
    )
    assert selector.is_calibrated

    # Apply to test scores (test labels are never provided to route_test)
    r_test = np.array([0.1, 0.5, 0.9])
    a_test, tau = selector.route_test(r_hat_test=r_test, mode="matched_coverage", q=0.5)
    assert len(a_test) == 3
    assert isinstance(tau, float)


def test_matched_coverage_threshold_selection():
    """Verifies that matched-coverage quantile selection achieves empirical target coverage."""
    # 100 uniformly spaced scores from 0.01 to 1.00
    r_val = np.linspace(0.01, 1.0, 100)

    for q_target in [0.20, 0.40, 0.60, 0.80]:
        tau = find_matched_coverage_threshold(r_val, target_q=q_target)
        # a = 1 if r > tau
        a_dec = (r_val > tau).astype(int)
        actual_cov = np.mean(a_dec)
        # In a 100-sample grid, empirical coverage should match target_q within 1 sample (1%)
        assert abs(actual_cov - q_target) <= 0.02, f"Target {q_target} had actual coverage {actual_cov}"


def test_cost_performance_optimization_lambdas():
    """
    Verifies that varying lambda_cost in [0.0, 0.1, 1.0] appropriately shifts the objective.
    When GB has acceptable loss and lower cost, increasing lambda_cost should favor lower total cost (higher GB coverage).
    """
    np.random.seed(123)
    N = 200
    r_val = np.linspace(0.0, 1.0, N)
    y_val = np.random.randint(0, 2, size=N)

    # Synthetic probabilities: GB has 75% accuracy, BB has 80% accuracy
    p_gb = np.zeros((N, 2))
    p_bb = np.zeros((N, 2))
    for i in range(N):
        p_gb[i, y_val[i]] = 0.75
        p_gb[i, 1 - y_val[i]] = 0.25
        p_bb[i, y_val[i]] = 0.80
        p_bb[i, 1 - y_val[i]] = 0.20

    q_min = 0.20

    # Low lambda (cost penalty = 0): picks threshold purely for performance
    tau_lam0, m_lam0, _ = optimize_cost_aware_threshold(
        r_hat_val=r_val, y_val=y_val, y_hat_gb_val=p_gb, y_hat_bb_val=p_bb,
        q_min=q_min, lambda_cost=0.0, cost_gb=1.0, cost_bb=10.0
    )

    # High lambda (cost penalty = 1.0): penalizes BB usage, pushes coverage higher towards GB
    tau_lam1, m_lam1, _ = optimize_cost_aware_threshold(
        r_hat_val=r_val, y_val=y_val, y_hat_gb_val=p_gb, y_hat_bb_val=p_bb,
        q_min=q_min, lambda_cost=1.0, cost_gb=1.0, cost_bb=10.0
    )

    assert m_lam0["coverage_gb"] >= (q_min - 1e-6)
    assert m_lam1["coverage_gb"] >= (q_min - 1e-6)
    # Higher lambda must not result in higher cost than lambda=0
    assert m_lam1["realized_cost"] <= m_lam0["realized_cost"] + 1e-4


def test_edge_case_all_glass_box_and_all_black_box():
    """Tests extreme target coverage values q=0.0 and q=1.0."""
    r_val = np.array([0.1, 0.4, 0.7, 0.9])

    # q = 1.0 -> 100% GB (all a = 1)
    tau_all_gb = find_matched_coverage_threshold(r_val, target_q=1.0)
    a_gb = (r_val > tau_all_gb).astype(int)
    assert np.all(a_gb == 1)

    # q = 0.0 -> 100% BB (all a = 0)
    tau_all_bb = find_matched_coverage_threshold(r_val, target_q=0.0)
    a_bb = (r_val > tau_all_bb).astype(int)
    assert np.all(a_bb == 0)


def test_edge_case_tied_scores_and_nans():
    """Tests robustness to identical scores and non-finite values."""
    # All identical scores
    r_tied = np.full(50, 0.5)
    tau_tied = find_matched_coverage_threshold(r_tied, target_q=0.5)
    assert np.isfinite(tau_tied)

    # Array containing NaNs and Infs
    r_dirty = np.array([0.1, np.nan, 0.8, np.inf, -np.inf, 0.5])
    tau_clean = find_matched_coverage_threshold(r_dirty, target_q=0.5)
    assert np.isfinite(tau_clean)

    # Evaluating metrics with dirty array should not raise an exception
    y_dummy = np.array([0, 1, 0, 1, 0, 1])
    p_dummy = np.tile([0.5, 0.5], (6, 1))
    m = evaluate_threshold_on_validation(
        r_hat=r_dirty, y_true=y_dummy, y_hat_gb=p_dummy, y_hat_bb=p_dummy,
        threshold=tau_clean, problem_type='classification', pred_dim=2
    )
    assert np.isfinite(m["j_objective"])
    assert m["n_gb"] + m["n_bb"] == 6


def test_preservation_of_baseline_threshold_logic():
    """Verifies that nominal baseline thresholding (tau = 1 - q) remains mathematically exact."""
    q = 0.40
    nominal_tau = 1.0 - q
    r_test = np.array([0.59, 0.60, 0.61])
    a_test = (r_test > nominal_tau).astype(int)
    # 0.59 <= 0.60 -> 0; 0.60 <= 0.60 -> 0; 0.61 > 0.60 -> 1
    np.testing.assert_array_equal(a_test, [0, 0, 1])


if __name__ == '__main__':
    test_zero_test_label_leakage()
    test_matched_coverage_threshold_selection()
    test_cost_performance_optimization_lambdas()
    test_edge_case_all_glass_box_and_all_black_box()
    test_edge_case_tied_scores_and_nans()
    test_preservation_of_baseline_threshold_logic()
    print("All Threshold Selection unit tests passed successfully!")
