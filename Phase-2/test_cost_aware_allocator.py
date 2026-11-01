# -*- coding: utf-8 -*-
"""
Unit tests for the Cost-Aware Allocation Mechanism (Phase-2/cost_aware_allocator.py).

Verifies:
1. Ground truth ranking formulation and direction.
2. X-only allocator training and inference.
3. Strict execution isolation: models are ONLY called for their allocated samples.
4. Correctness of class probabilities, predictions, and computational savings metrics.
"""

import numpy as np

from cost_aware_allocator import (
    get_percentiles_of_a_wrt_b,
    compute_ground_truth_ranking,
    CostAwareAllocator,
    execute_cost_aware_inference
)


class MockComponentClassifier:
    """Mock classifier that logs call counts and sample sizes."""
    def __init__(self, fixed_probs):
        self.fixed_probs = np.asarray(fixed_probs)
        self.call_count = 0
        self.last_evaluated_samples = 0

    def predict_proba(self, X):
        self.call_count += 1
        n = len(X)
        self.last_evaluated_samples = n
        # Tile fixed probabilities to match n
        return np.tile(self.fixed_probs, (n, 1))

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class MockComponentRegressor:
    """Mock regressor that logs call counts and sample sizes."""
    def __init__(self, fixed_val):
        self.fixed_val = fixed_val
        self.call_count = 0
        self.last_evaluated_samples = 0

    def predict(self, X):
        self.call_count += 1
        n = len(X)
        self.last_evaluated_samples = n
        return np.full(n, self.fixed_val, dtype=np.float32)


def test_percentiles_calculation():
    # b is 10 sorted numbers from 1 to 10
    b = np.linspace(-2.0, 2.0, 100)
    p = get_percentiles_of_a_wrt_b(b, a=None, distr_min=-2.0, distr_max=2.0)
    assert p.shape == (100, 1)
    assert p.min() >= 0.0
    assert p.max() <= 1.0
    # Values should be monotonically increasing
    assert np.all(np.diff(p.ravel()) >= 0.0)


def test_ground_truth_ranking_direction():
    # Observation 0: GB loss 0.1, BB loss 2.0 (GB strongly superior -> high rank r)
    # Observation 1: GB loss 2.0, BB loss 0.1 (BB strongly superior -> low rank r)
    loss_gb = np.array([0.1, 2.0])
    loss_bb = np.array([2.0, 0.1])
    r, desirability = compute_ground_truth_ranking(loss_gb, loss_bb, problem_type='classification', pred_dim=2)

    assert r[0] > r[1], "Sample with superior Glass-Box loss must receive higher desirability percentile r."
    assert desirability[0] > desirability[1], "Sample with superior Glass-Box must receive higher raw desirability."


def test_cost_aware_allocator_x_only_pipeline():
    np.random.seed(42)
    N = 100
    D = 4
    X = np.random.uniform(-1.0, 1.0, size=(N, D))
    y = np.random.randint(0, 2, size=N)

    # Synthetic losses
    loss_gb = np.random.exponential(scale=0.5, size=(N, 1))
    loss_bb = np.random.exponential(scale=0.5, size=(N, 1))

    allocator = CostAwareAllocator(random_state=42)
    allocator.fit(X_train=X, y_train=y, loss_gb_train=loss_gb, loss_bb_train=loss_bb, pred_dim=2)
    assert allocator.is_fitted

    # Test-time ranking: must ONLY accept X, zero black-box dependencies
    X_test = np.random.uniform(-1.0, 1.0, size=(20, D))
    r_hat = allocator.predict_ranking(X_test)
    assert r_hat.shape == (20,)

    # Routing decisions
    q = 0.6
    a_decisions, _ = allocator.route(r_hat=r_hat, q=q)
    assert a_decisions.shape == (20,)
    assert set(np.unique(a_decisions)).issubset({0, 1})

    # High scores (> 1 - q = 0.4) must be allocated to GB (a = 1)
    for i in range(len(r_hat)):
        if r_hat[i] > (1.0 - q):
            assert a_decisions[i] == 1
        else:
            assert a_decisions[i] == 0


def test_strict_execution_isolation_classification():
    # 5 test samples: 3 routed to GB (a=1), 2 routed to BB (a=0)
    N = 5
    X_test = np.random.randn(N, 3)
    a_decisions = np.array([1, 0, 1, 1, 0])

    mock_gb = MockComponentClassifier(fixed_probs=[0.8, 0.2])
    mock_bb = MockComponentClassifier(fixed_probs=[0.1, 0.9])

    results = execute_cost_aware_inference(
        X=X_test,
        a_decisions=a_decisions,
        gb_model=mock_gb,
        bb_model=mock_bb,
        problem_type='classification',
        pred_dim=2,
        cost_gb=1.0,
        cost_bb=10.0
    )

    # Verification of execution isolation:
    # GB model should be called ONCE with exactly 3 samples (idx 0, 2, 3)
    assert mock_gb.call_count == 1
    assert mock_gb.last_evaluated_samples == 3

    # BB model should be called ONCE with exactly 2 samples (idx 1, 4)
    assert mock_bb.call_count == 1
    assert mock_bb.last_evaluated_samples == 2

    # Verify probability assembly
    y_prob = results['y_prob']
    np.testing.assert_allclose(y_prob[0], [0.8, 0.2])
    np.testing.assert_allclose(y_prob[1], [0.1, 0.9])
    np.testing.assert_allclose(y_prob[2], [0.8, 0.2])
    np.testing.assert_allclose(y_prob[3], [0.8, 0.2])
    np.testing.assert_allclose(y_prob[4], [0.1, 0.9])

    # Verify final predictions
    expected_preds = np.array([0, 1, 0, 0, 1])
    np.testing.assert_array_equal(results['y_pred'], expected_preds)

    # Verify cost savings
    # Baseline cost: 5 * (1 + 10) = 55
    # Realized cost: 3 * 1 + 2 * 10 = 23
    # Savings: (55 - 23) / 55 = 32 / 55 = 58.18%
    metrics = results['metrics']
    assert metrics['num_gb_calls'] == 3
    assert metrics['num_bb_calls'] == 2
    assert metrics['baseline_compute_cost'] == 55.0
    assert metrics['realized_compute_cost'] == 23.0
    assert abs(metrics['compute_savings_pct'] - (32.0 / 55.0 * 100.0)) < 1e-4


def test_zero_call_edge_cases():
    # Case 1: 100% GB (a = all 1s). BB must NEVER be called.
    X_test = np.random.randn(4, 2)
    mock_gb = MockComponentClassifier(fixed_probs=[0.7, 0.3])
    mock_bb = MockComponentClassifier(fixed_probs=[0.2, 0.8])

    results = execute_cost_aware_inference(
        X=X_test,
        a_decisions=np.ones(4),
        gb_model=mock_gb,
        bb_model=mock_bb,
        problem_type='classification',
        pred_dim=2
    )
    assert mock_gb.call_count == 1
    assert mock_bb.call_count == 0, "BB model must NOT be called when a = all 1s."
    assert results['metrics']['num_gb_calls'] == 4
    assert results['metrics']['num_bb_calls'] == 0

    # Case 2: 100% BB (a = all 0s). GB must NEVER be called.
    mock_gb_2 = MockComponentClassifier(fixed_probs=[0.7, 0.3])
    mock_bb_2 = MockComponentClassifier(fixed_probs=[0.2, 0.8])

    results_2 = execute_cost_aware_inference(
        X=X_test,
        a_decisions=np.zeros(4),
        gb_model=mock_gb_2,
        bb_model=mock_bb_2,
        problem_type='classification',
        pred_dim=2
    )
    assert mock_gb_2.call_count == 0, "GB model must NOT be called when a = all 0s."
    assert mock_bb_2.call_count == 1
    assert results_2['metrics']['num_gb_calls'] == 0
    assert results_2['metrics']['num_bb_calls'] == 4


def test_regression_inference_execution():
    X_test = np.random.randn(4, 2)
    mock_gb = MockComponentRegressor(fixed_val=3.14)
    mock_bb = MockComponentRegressor(fixed_val=42.0)

    a_decisions = np.array([1, 0, 1, 0])
    results = execute_cost_aware_inference(
        X=X_test,
        a_decisions=a_decisions,
        gb_model=mock_gb,
        bb_model=mock_bb,
        problem_type='regression'
    )
    assert mock_gb.call_count == 1
    assert mock_bb.call_count == 1
    np.testing.assert_allclose(results['y_pred'], [3.14, 42.0, 3.14, 42.0])


if __name__ == '__main__':
    test_percentiles_calculation()
    test_ground_truth_ranking_direction()
    test_cost_aware_allocator_x_only_pipeline()
    test_strict_execution_isolation_classification()
    test_zero_call_edge_cases()
    test_regression_inference_execution()
    print("All Cost-Aware Allocator unit tests passed successfully!")
