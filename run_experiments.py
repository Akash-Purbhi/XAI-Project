"""
Phase 1 EEG Replication - Consolidated Experiment Runner.

Provides a unified command-line interface to execute replication experiments
for any dataset and seed, batch run multiple seeds, or run the entire suite.

Usage Examples:
    # Run a single dataset and seed:
    python run_experiments.py --dataset Wine --seed 0
    python run_experiments.py --dataset SuperconductR --seed 2

    # Run specific seeds for a dataset:
    python run_experiments.py --dataset SuperconductR --seeds 2 3 4

    # Run all 5 datasets for specific seeds:
    python run_experiments.py --datasets Wine Bank PolR --seeds 0 1 2

    # Run the full 5-dataset x 5-seed study (25 runs):
    python run_experiments.py --all
"""

import sys
import time
import argparse
from typing import List

# Ensure utf-8 output encoding for Windows terminals
sys.stdout.reconfigure(encoding="utf-8")

from src.experiments.run_single import run_single_dataset
from src.utils.logging import setup_logger

ALL_DATASETS = ["Wine", "Bank", "PolR", "SuperconductR", "BrazilianHousesR"]
ALL_SEEDS = [0, 1, 2, 3, 4]


def run_experiments(
    datasets: List[str],
    seeds: List[int],
    q_step: float = 0.01,
    random_repetitions: int = 10,
    output_dir: str = "results/raw",
):
    """
    Execute replication experiments across specified datasets and seeds.
    """
    logger = setup_logger("run_experiments")
    total_runs = len(datasets) * len(seeds)
    run_idx = 0
    start_time = time.time()

    print(f"\n{'='*65}")
    print(f"  EEG Replication Runner — Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"  Replicate Seeds ({len(seeds)}): {seeds}")
    print(f"  Total Runs Scheduled: {total_runs}")
    print(f"{'='*65}\n", flush=True)

    completed = []
    failed = []

    for d_name in datasets:
        for seed in seeds:
            run_idx += 1
            t0 = time.time()
            print(f"\n[{run_idx}/{total_runs}] Running {d_name} (Seed {seed})...", flush=True)
            try:
                res = run_single_dataset(
                    dataset_name=d_name,
                    seed=seed,
                    q_step=q_step,
                    random_repetitions=random_repetitions,
                    output_dir=output_dir,
                    logger=logger,
                )
                m = res["metrics"]
                elapsed = time.time() - t0
                print(
                    f"  ✓ Done {d_name} (Seed {seed}) in {elapsed:.1f}s | "
                    f"AUC={m['AUC']:.1f}, PPCR={m['PPCR']:.1f}, Max={m['Max_Acc']:.1f}, s_Acc={m['s_Acc']:.1f}",
                    flush=True,
                )
                completed.append((d_name, seed, m, elapsed))
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  ✗ FAILED {d_name} (Seed {seed}) after {elapsed:.1f}s: {e}", flush=True)
                failed.append((d_name, seed, str(e)))

    total_elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"  Execution Finished in {total_elapsed / 60:.1f} minutes")
    print(f"  Successful: {len(completed)}/{total_runs} | Failed: {len(failed)}/{total_runs}")
    print(f"{'='*65}\n", flush=True)

    if failed:
        print("Failed runs:")
        for d, s, err in failed:
            print(f"  - {d} (seed {s}): {err}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner for AAAI 2024 EEG Replication"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=ALL_DATASETS,
        help="Single dataset to run",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of datasets to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single replicate seed (0-4)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="List of replicate seeds (e.g. 0 1 2 3 4)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 datasets across all 5 seeds (25 runs total)",
    )
    parser.add_argument(
        "--q_step",
        type=float,
        default=0.01,
        help="Grid resolution for explainability parameter q (default: 0.01)",
    )
    parser.add_argument(
        "--random_repetitions",
        type=int,
        default=10,
        help="Monte Carlo repetitions for random baseline (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/raw",
        help="Output directory for raw CSV/JSON results (default: results/raw)",
    )

    args = parser.parse_args()

    # Resolve datasets
    if args.all:
        datasets = ALL_DATASETS
        seeds = ALL_SEEDS
    else:
        if args.dataset:
            datasets = [args.dataset]
        elif args.datasets:
            datasets = args.datasets
        else:
            datasets = ALL_DATASETS

        if args.seed is not None:
            seeds = [args.seed]
        elif args.seeds:
            seeds = args.seeds
        else:
            seeds = [0]

    run_experiments(
        datasets=datasets,
        seeds=seeds,
        q_step=args.q_step,
        random_repetitions=args.random_repetitions,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
