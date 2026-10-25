# -*- coding: utf-8 -*-
"""
Evaluate baseline EEG metrics for Wine dataset.
"""

import os
import pandas as pd
import numpy as np
from evaluate_eeg import compute_ensemble_predictions, compute_performance, compute_original_eeg_metrics

out_csv = os.path.join(
    'results', 'Exp_Wine',
    'allocator_gradient_boosting_trees_regressor_gb_logistic_regression_bb_gradient_boosting_trees_classifier_rep_0_f_all_d_cemse',
    'out_test.csv'
)

df = pd.read_csv(out_csv, index_col=0)
r_hat = df['r_hat_test'].to_numpy()
r_true = df['r_test'].to_numpy()
y_true = df[['y_test_0', 'y_test_1']].to_numpy()
y_gb = df[['y_hat_wb_test_0', 'y_hat_wb_test_1']].to_numpy()
y_bb = df[['y_hat_bb_test_0', 'y_hat_bb_test_1']].to_numpy()

q_grid = np.linspace(0, 1, 101)
perf_ens = []
perf_oracle = []
perf_rand = []

perf_gb = compute_performance(y_true, y_gb, 'classification')
perf_bb = compute_performance(y_true, y_bb, 'classification')

for q in q_grid:
    # a'_q = (r_hat > 1 - q)
    a_ens = (r_hat > (1.0 - q)).astype(float)
    _, p_ens = compute_ensemble_predictions(a_ens, y_gb, y_bb, 'classification')
    perf_ens.append(compute_performance(y_true, p_ens, 'classification'))
    
    # oracle
    a_orc = (r_true > (1.0 - q)).astype(float)
    _, p_orc = compute_ensemble_predictions(a_orc, y_gb, y_bb, 'classification')
    perf_oracle.append(compute_performance(y_true, p_orc, 'classification'))
    
    # random
    perf_rand.append(q * perf_gb + (1.0 - q) * perf_bb)

metrics = compute_original_eeg_metrics(
    q_grid, perf_ens, perf_rand, perf_oracle,
    perf_gb=perf_gb, perf_bb=perf_bb, problem_type='classification'
)

print("--- LOCALLY REPRODUCED ORIGINAL EEG BASELINE (WINE) ---")
for k, v in metrics.items():
    print(f"  {k}: {v:.2f}")
print(f"  Glass-box accuracy: {perf_gb:.2f}%")
print(f"  Black-box accuracy: {perf_bb:.2f}%")

# Save to CSV
res_df = pd.DataFrame([metrics])
res_df['Dataset'] = 'Wine'
res_df['GlassBox'] = 'logistic_regression'
res_df['BlackBox'] = 'gradient_boosting_trees_classifier'
res_df['GB_Acc'] = perf_gb
res_df['BB_Acc'] = perf_bb
os.makedirs(os.path.join('results', 'csv'), exist_ok=True)
res_df.to_csv(os.path.join('results', 'csv', 'eeg_baseline_wine.csv'), index=False)
print("Saved baseline metrics to results/csv/eeg_baseline_wine.csv")
