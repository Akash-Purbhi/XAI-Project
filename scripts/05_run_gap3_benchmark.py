# -*- coding: utf-8 -*-
"""
Script 05: Master Gap 3 Benchmark Runner
Executes the stability experiment sequentially across all selected datasets
and automatically produces the aggregated cross-dataset analysis and plots.
"""

import os
import sys
from pathlib import Path
import argparse
from time import time
import importlib.util

# Add repository root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Dynamically import script 03 and script 04
spec3 = importlib.util.spec_from_file_location("stab_exp", str(PROJECT_ROOT / "scripts" / "03_run_stability_experiment.py"))
stab_exp = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(stab_exp)

spec4 = importlib.util.spec_from_file_location("agg_exp", str(PROJECT_ROOT / "scripts" / "04_aggregate_and_plot_gap3.py"))
agg_exp = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(agg_exp)

DEFAULT_DATASETS = [
    # 12 Classification Datasets
    'Wine',
    'Phoneme',
    'KDDIPUMS',
    'EyeMovements',
    'Pol',
    'Bank',
    'MagicTelescope',
    'Credit',
    'House16H',
    'California',
    'Electricity',
    'Jannis',
    # 12 Regression Datasets
    'Wine_R',
    'CPU_R',
    'Isolet_R',
    'BrazilianHouses_R',
    'Sulfur_R',
    'Ailerons_R',
    'MiamiHousing_R',
    'Pol_R',
    'Elevators_R',
    'BikeSharingDemand_R',
    'Fifa_R',
    'California_R'
]

def main():
    parser = argparse.ArgumentParser(description="Master Gap 3 Benchmark Runner")
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS, help="Datasets to run")
    parser.add_argument('--use_gpu', action='store_true', default=True, help="Use GPU acceleration")
    parser.add_argument('--no_gpu', dest='use_gpu', action='store_false', help="Force CPU mode")
    parser.add_argument('--quick_test', action='store_true', help="Run only on Wine and Phoneme for quick testing")
    parser.add_argument('--force', action='store_true', help="Re-run datasets even if results exist")
    args = parser.parse_args()

    targets = ['Wine', 'Phoneme'] if args.quick_test else args.datasets
    print("=" * 75)
    print(f"MASTER GAP 3 BENCHMARK: {len(targets)} DATASETS")
    print(f"Targets: {', '.join(targets)}")
    print(f"GPU Acceleration: {'ENABLED (CUDA / RTX 3050)' if args.use_gpu else 'DISABLED (CPU)'}")
    print("=" * 75)

    start_total = time()
    completed = []
    failed = []

    for idx, ds in enumerate(targets, 1):
        print(f"\n[{idx}/{len(targets)}] Starting Dataset: {ds}")
        t0 = time()
        try:
            stab_exp.run_experiment(dataset_name=ds, use_gpu=args.use_gpu, force=args.force)
            elapsed = time() - t0
            completed.append((ds, elapsed))
            print(f"[OK] Completed {ds} in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time() - t0
            failed.append((ds, str(e)))
            print(f"[ERROR] Failed {ds} after {elapsed:.1f}s: {e}")

    # Aggregate & Plot across all evaluated datasets
    print("\n" + "=" * 75)
    print("Aggregating Results Across All Evaluated Datasets...")
    agg_exp.aggregate_all_stability_results()

    total_time = (time() - start_total) / 60.0
    print("\n" + "=" * 75)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Total Time: {total_time:.2f} minutes")
    print(f"Successfully Completed: {len(completed)}/{len(targets)}")
    for ds, el in completed:
        print(f"  - {ds:<20}: {el:.1f}s")
    if failed:
        print(f"Failed Datasets ({len(failed)}):")
        for ds, err in failed:
            print(f"  - {ds:<20}: {err}")
    print("=" * 75)

if __name__ == "__main__":
    main()
