# -*- coding: utf-8 -*-
"""
Unit tests for evaluate_eeg.py functions.
Tests standard EEG metrics, allocation flip rate, and prediction consistency on known cases.
"""

import numpy as np
from evaluate_eeg import (
    compute_ensemble_predictions,
    compute_performance,
    compute_auc,
    compute_original_eeg_metrics,
    compute_allocation_flip_rate,
    compute_prediction_consistency
)


def test_compute_ensemble_predictions():
    # 3 samples, 2 classes
    a_decisions = np.array([1, 0, 1])  # sample 0 and 2 from gb, sample 1 from bb
    y_hat_gb = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    y_hat_bb = np.array([[0.3, 0.7], [0.1, 0.9], [0.4, 0.6]])

    prob, preds = compute_ensemble_predictions(a_decisions, y_hat_gb, y_hat_bb, 'classification')
    # sample 0: from gb -> [0.9, 0.1] -> pred 0
    # sample 1: from bb -> [0.1, 0.9] -> pred 1
    # sample 2: from gb -> [0.7, 0.3] -> pred 0
    np.testing.assert_allclose(prob[0], [0.9, 0.1])
    np.testing.assert_allclose(prob[1], [0.1, 0.9])
    np.testing.assert_allclose(prob[2], [0.7, 0.3])
    np.testing.assert_array_equal(preds, [0, 1, 0])


def test_compute_performance_classification():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])  # 3 out of 4 correct -> 75%
    acc = compute_performance(y_true, y_pred, 'classification')
    assert acc == 75.0


def test_compute_auc():
    # Straight line y = x over [0, 1] -> integral = 0.5
    q_grid = np.linspace(0, 1, 101)
    perf = q_grid
    auc = compute_auc(q_grid, perf)
    assert abs(auc - 0.5) < 1e-4


def test_compute_allocation_flip_rate():
    alloc_clean = [1, 1, 0, 0, 1]
    alloc_pert =  [1, 0, 0, 1, 1]  # 2 flips out of 5 -> 40%
    afr = compute_allocation_flip_rate(alloc_clean, alloc_pert)
    assert afr == 40.0


def test_compute_prediction_consistency():
    y_clean = [0, 1, 1, 0, 1]
    y_pert =  [0, 1, 0, 0, 1]  # 4 out of 5 identical -> 80%
    pc = compute_prediction_consistency(y_clean, y_pert, 'classification')
    assert pc == 80.0


def test_compute_original_eeg_metrics():
    q_grid = np.linspace(0, 1, 11)
    # perfect oracle: always 90%, random: 50%, ensemble: 70%
    perf_oracle = np.ones(11) * 90.0
    perf_random = np.ones(11) * 50.0
    perf_ens = np.ones(11) * 70.0
    perf_gb = 60.0
    perf_bb = 65.0

    metrics = compute_original_eeg_metrics(
        q_grid, perf_ens, perf_random, perf_oracle,
        perf_gb=perf_gb, perf_bb=perf_bb, problem_type='classification'
    )
    # PPCR = (70 - 50) / (90 - 50) * 100% = 50%
    assert abs(metrics['PPCR'] - 50.0) < 1e-4
    # PQEOM: 70 >= 65 is always true -> 100%
    assert metrics['PQEOM'] == 100.0
    # PQOM: 70 > 65 is always true -> 100%
    assert metrics['PQOM'] == 100.0
    # Max Acc: 70.0
    assert metrics['Max_Acc'] == 70.0


if __name__ == '__main__':
    test_compute_ensemble_predictions()
    test_compute_performance_classification()
    test_compute_auc()
    test_compute_allocation_flip_rate()
    test_compute_prediction_consistency()
    test_compute_original_eeg_metrics()
    print("All unit tests passed successfully!")
