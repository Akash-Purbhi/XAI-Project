"""
Robust single-seed SuperconductR runner with progress logging.

Runs one seed at a time (pass via --seed) to avoid the timeout issue where
the previous run_superconductr_seeds.py tried all 3 seeds sequentially and
stalled during the GridSearchCV model selection phase (which takes ~22 min
per seed on 21k samples × 79 features).

Usage:
    python run_superconductr_single_seed.py --seed 2
    python run_superconductr_single_seed.py --seed 3
    python run_superconductr_single_seed.py --seed 4
"""

import sys
import os
import time
import argparse
import traceback

sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Run a single SuperconductR seed")
    parser.add_argument("--seed", type=int, required=True, help="Random seed (2, 3, or 4)")
    args = parser.parse_args()

    seed = args.seed
    print(f"\n{'='*60}", flush=True)
    print(f"  SuperconductR Seed {seed} — Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()

    try:
        # Import after arg parsing so startup is instant and any import errors
        # are caught and logged clearly
        print("[STEP 1/3] Importing pipeline modules...", flush=True)
        from src.experiments.run_single import run_single_dataset
        t_import = time.time() - t0
        print(f"  Imports done in {t_import:.1f}s", flush=True)

        # Run the full EEG pipeline for this seed
        print(f"[STEP 2/3] Running EEG pipeline (dataset=SuperconductR, seed={seed})...", flush=True)
        print(f"  NOTE: Model selection (GridSearchCV on 21k×79 data) takes ~20-25 min.", flush=True)
        print(f"  If this step takes longer than 40 min, there may be a resource issue.", flush=True)

        res = run_single_dataset(
            dataset_name="SuperconductR",
            seed=seed,
            q_step=0.01,
            random_repetitions=10,
            output_dir="results/raw",
        )

        t_pipeline = time.time() - t0
        print(f"  Pipeline completed in {t_pipeline:.1f}s", flush=True)

        # Report results
        m = res["metrics"]
        print(f"\n[STEP 3/3] Results for SuperconductR seed {seed}:", flush=True)
        for k, v in m.items():
            print(f"  {k}: {v}", flush=True)

        elapsed = time.time() - t0
        print(f"\n{'='*60}", flush=True)
        print(f"  SUCCESS: SuperconductR seed {seed} finished in {elapsed:.1f}s", flush=True)
        print(f"  AUC={m['AUC']:.2f}, PPCR={m['PPCR']:.2f}, Max={m['Max_Acc']:.2f}, s_Acc={m['s_Acc']:.2f}", flush=True)
        print(f"{'='*60}\n", flush=True)

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n{'='*60}", flush=True)
        print(f"  FAILED: SuperconductR seed {seed} after {elapsed:.1f}s", flush=True)
        print(f"  Error: {e}", flush=True)
        print(f"{'='*60}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
