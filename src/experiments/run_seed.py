"""Phase 1 EEG Replication - Seeded run script.

Runs all 5 datasets for a single seed. Used as a worker task in the
5-replicate study when running datasets sequentially.
"""

import argparse
import sys
import os

from src.experiments.run_single import run_single_dataset
from src.utils.logging import setup_logger


def run_all_datasets_single_seed(
    seed: int,
    datasets=None,
    q_step: float = 0.01,
    random_repetitions: int = 10,
    output_dir: str = "results/raw",
) -> list:
    if datasets is None:
        datasets = ["Wine", "Bank", "PolR", "SuperconductR", "BrazilianHousesR"]

    logger = setup_logger(f"phase1_seed_{seed}")
    logger.info(f"=== Running all datasets for seed {seed} ===")

    all_runs = []
    for dataset in datasets:
        logger.info(f"\n--- Dataset: {dataset} | Seed: {seed} ---")
        res = run_single_dataset(
            dataset_name=dataset,
            seed=seed,
            q_step=q_step,
            random_repetitions=random_repetitions,
            output_dir=output_dir,
            logger=logger,
        )
        all_runs.append(res["run_summary"])
        print(
            f"[Seed {seed}] {dataset} complete. "
            f"AUC={res['metrics']['AUC']:.1f}, "
            f"s_Acc={res['metrics']['s_Acc']:.1f}"
        )
    return all_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all datasets for a single replicate seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--q_step", type=float, default=0.01)
    args = parser.parse_args()

    run_all_datasets_single_seed(
        seed=args.seed,
        datasets=args.datasets,
        q_step=args.q_step,
    )
