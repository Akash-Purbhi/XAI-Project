# -*- coding: utf-8 -*-
"""
Script 02: Prepare and Partition Datasets for Gap 3 Experiments
Downloads (via OpenML/data.py) and caches 4-fold data splits for selected datasets.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from time import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from utils import get_superfix_prefix_and_make_dir
from data import get_dataset

DEFAULT_DATASETS = [
    # Classification
    'Wine',
    'Phoneme',
    'KDDIPUMS',
    'EyeMovements',
    'Pol',
    'Bank',
    'MagicTelescope',
    'Credit',
    # Regression
    'Wine_R',
    'CPU_R',
    'Isolet_R',
    'BrazilianHouses_R'
]

def prepare_dataset(dataset_name, run_machine='local', n_splits=4, force=False):
    experiment_name = f'Exp_{dataset_name}'
    _, dir_prefix = get_superfix_prefix_and_make_dir(run_machine=run_machine, experiment_name=experiment_name, overwrite_check=False)
    dir_splits = os.path.join(dir_prefix, 'data_splits')
    splits_file = os.path.join(dir_splits, 'splits_kfold')
    
    if os.path.exists(splits_file) and not force:
        print(f"[SKIP] {dataset_name}: Splits already exist at {splits_file}")
        return True

    print(f"[PREPARE] {dataset_name}: Loading dataset from OpenML/local cache...")
    t0 = time()
    Path(dir_splits).mkdir(parents=True, exist_ok=True)

    try:
        pred_dim, ds_dict = get_dataset(dataset_name=dataset_name)
        problem_type = ds_dict.get('problem_type')
        ds_sklearn = ds_dict.get('ds_sklearn')
        ds_train = ds_sklearn.get('ds_train')
        x = ds_train.get('x_train')
        y = ds_train.get('y_train')

        if problem_type == 'classification':
            splits = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=0).split(X=x, y=y)
        elif problem_type == 'regression':
            splits = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=0).split(X=x, y=y)
        else:
            raise ValueError(f"Unknown problem_type: {problem_type}")

        splits = list(splits)
        with open(splits_file, 'wb') as f:
            pickle.dump(splits, f)

        total_samples = len(y) + len(ds_sklearn['ds_val']['y_val']) + len(ds_sklearn['ds_test']['y_test'])
        n_features = x.shape[1]
        elapsed = time() - t0
        print(f"[DONE] {dataset_name} ({problem_type}): {total_samples} samples, {n_features} features | {elapsed:.2f}s")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to prepare {dataset_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare data splits for Gap 3 datasets")
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS, help="List of dataset names or 'all'")
    parser.add_argument('--force', action='store_true', help="Re-generate splits even if already existing")
    args = parser.parse_args()

    targets = DEFAULT_DATASETS if args.datasets == ['all'] or args.datasets == DEFAULT_DATASETS else args.datasets
    print(f"Preparing {len(targets)} datasets: {', '.join(targets)}")
    
    success = 0
    for idx, ds in enumerate(targets, 1):
        print(f"\n--- ({idx}/{len(targets)}) Processing {ds} ---")
        if prepare_dataset(ds, force=args.force):
            success += 1

    print("\n" + "=" * 60)
    print(f"Preparation Complete: {success}/{len(targets)} datasets ready.")
    print("=" * 60)

if __name__ == "__main__":
    main()
