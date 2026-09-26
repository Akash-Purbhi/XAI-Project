"""
Script to run all remaining replicate seeds for the completed datasets.
Run after seed 0 is complete for all datasets.
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from src.experiments.run_single import run_single_dataset

DATASETS = ['Wine', 'Bank', 'PolR', 'BrazilianHousesR']
SEEDS = [1, 2, 3, 4]

results = []
for ds in DATASETS:
    for seed in SEEDS:
        print(f'\n>>> Running {ds} seed {seed}...', flush=True)
        try:
            res = run_single_dataset(
                dataset_name=ds,
                seed=seed,
                q_step=0.01,
                random_repetitions=10,
                output_dir='results/raw',
            )
            m = res['metrics']
            print(f'    Done: AUC={m["AUC"]:.1f}, PPCR={m["PPCR"]:.1f}, '
                  f'Max={m["Max_Acc"]:.1f}, s_Acc={m["s_Acc"]:.1f}', flush=True)
            results.append((ds, seed, m))
        except Exception as e:
            print(f'    ERROR: {e}', flush=True)

print('\n\nSummary of seeds 1-4:')
for ds, seed, m in results:
    print(f'  {ds} seed{seed}: AUC={m["AUC"]:.1f}')

print('COMPLETE')
