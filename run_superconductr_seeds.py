"""
Fast runner for SuperconductR replicate seeds 2, 3, 4 with streamlined grid.
"""

import sys
import time
sys.stdout.reconfigure(encoding='utf-8')

from src.experiments.run_single import run_single_dataset

SEEDS = [2, 3, 4]

for seed in SEEDS:
    t0 = time.time()
    print(f"\n>>> Running SuperconductR (Seed {seed}) <<<", flush=True)
    res = run_single_dataset(
        dataset_name="SuperconductR",
        seed=seed,
        q_step=0.01,
        random_repetitions=10,
        output_dir="results/raw",
    )
    m = res["metrics"]
    elapsed = time.time() - t0
    print(f"Done SuperconductR seed {seed} in {elapsed:.1f}s: AUC={m['AUC']:.2f}, PPCR={m['PPCR']:.2f}, Max={m['Max_Acc']:.2f}, s_Acc={m['s_Acc']:.2f}", flush=True)

print("ALL SuperconductR seeds finished successfully!", flush=True)
