"""
Results aggregation and table generation for Phase 1 EEG Replication.

Produces:
- Table 1: Dataset Characteristics
- Table 2: Component Model Performance (GB vs BB vs EEG)
- Table 3: Consolidated EEG Metrics (Mean ± Std over 5 replicates)
- Table 4: Original Paper vs Our Replication Comparison
"""

import os
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from src.data import DATASET_REGISTRY
from src.utils.io import save_dataframe


PAPER_REPORTED_TABLE2 = {
    "Wine": {
        "AUC": 79.0, "PPCR": 21.0, "PQEOM": 71.0, "PQOM": 0.0,
        "PCFA": 7.0, "95TQM": 98.0, "Max_Acc": 80.0, "Argmax_q": 70.0, "s_Acc": 78.0
    },
    "Bank": {
        "AUC": 76.0, "PPCR": -19.0, "PQEOM": 4.0, "PQOM": 0.0,
        "PCFA": 0.0, "95TQM": 100.0, "Max_Acc": 79.0, "Argmax_q": 100.0, "s_Acc": 71.0
    },
    "PolR": {
        "AUC": 98.0, "PPCR": 44.0, "PQEOM": 96.0, "PQOM": 93.0,
        "PCFA": 88.0, "95TQM": 100.0, "Max_Acc": 88.0, "Argmax_q": 81.0, "s_Acc": 84.0
    },
    "SuperconductR": {
        "AUC": 83.0, "PPCR": 42.0, "PQEOM": 60.0, "PQOM": 24.0,
        "PCFA": 95.0, "95TQM": 95.0, "Max_Acc": 83.0, "Argmax_q": 57.0, "s_Acc": 76.0
    },
    "BrazilianHousesR": {
        "AUC": 96.0, "PPCR": 71.0, "PQEOM": 83.0, "PQOM": 64.0,
        "PCFA": 1.0, "95TQM": 93.0, "Max_Acc": 98.0, "Argmax_q": 8.0, "s_Acc": 88.0
    },
}


def build_table1_dataset_characteristics(
    dataset_metadata: Dict[str, Dict[str, Any]],
    output_dir: str = "tables",
) -> pd.DataFrame:
    """
    Generate Table 1: Dataset Characteristics (Samples, Features, Split counts).
    """
    rows = []
    for d_name, meta in dataset_metadata.items():
        n = meta["n_samples"]
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.09))
        n_test = n - n_train - n_val

        rows.append({
            "Dataset": d_name,
            "Task": meta["task_type"].capitalize(),
            "Samples": n,
            "Features": meta["n_features"],
            "Train": n_train,
            "Validation": n_val,
            "Test": n_test,
            "OpenML_ID": meta["openml_id"],
        })

    df = pd.DataFrame(rows)
    save_dataframe(df, os.path.join(output_dir, "table1_dataset_characteristics.csv"))
    return df


def aggregate_replicate_results(
    all_runs: List[Dict[str, Any]],
    output_dir: str = "results/processed",
    tables_dir: str = "tables",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate results across replicates and generate Tables 2, 3, and 4.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Flatten run metrics
    flat_rows = []
    for run in all_runs:
        row = {
            "dataset": run["dataset"],
            "seed": run["seed"],
            "gb_model": run["gb_model"],
            "bb_model": run["bb_model"],
        }
        row.update(run["metrics"])
        flat_rows.append(row)

    runs_df = pd.DataFrame(flat_rows)
    save_dataframe(runs_df, os.path.join(output_dir, "phase1_all_runs.csv"))

    # Table 2: Component Model Performance Comparison
    comp_rows = []
    for dataset in runs_df["dataset"].unique():
        sub = runs_df[runs_df["dataset"] == dataset]
        comp_rows.append({
            "Dataset": dataset,
            "Glass_Box_Model": sub["gb_model"].iloc[0],
            "Glass_Box_Perf": f"{sub['GB_Perf'].mean():.1f} ± {sub['GB_Perf'].std():.1f}%",
            "Black_Box_Model": sub["bb_model"].iloc[0],
            "Black_Box_Perf": f"{sub['BB_Perf'].mean():.1f} ± {sub['BB_Perf'].std():.1f}%",
            "Learned_EEG_Max": f"{sub['Max_Acc'].mean():.1f} ± {sub['Max_Acc'].std():.1f}%",
            "Delta_Over_Best": f"{sub['Max_Acc'].mean() - max(sub['GB_Perf'].mean(), sub['BB_Perf'].mean()):+.1f}%",
        })
    table2_df = pd.DataFrame(comp_rows)
    save_dataframe(table2_df, os.path.join(tables_dir, "table2_component_model_performance.csv"))

    # Table 3: Consolidated EEG Metrics (Mean ± Std)
    metric_cols = ["AUC", "PPCR", "PQEOM", "PQOM", "PCFA", "95TQM", "Max_Acc", "Argmax_q", "s_Acc"]
    t3_rows = []
    numeric_summary = {}

    for dataset in runs_df["dataset"].unique():
        sub = runs_df[runs_df["dataset"] == dataset]
        row = {"Dataset": dataset}
        numeric_summary[dataset] = {}
        for m in metric_cols:
            mean_val = sub[m].mean()
            std_val = sub[m].std()
            row[m] = f"{mean_val:.0f} ± {std_val:.0f}"
            numeric_summary[dataset][m] = (mean_val, std_val)
        t3_rows.append(row)

    # Average row across all datasets
    avg_row = {"Dataset": "Average"}
    for m in metric_cols:
        means = [runs_df[runs_df["dataset"] == d][m].mean() for d in runs_df["dataset"].unique()]
        avg_mean = np.mean(means)
        avg_std = np.std(means)
        avg_row[m] = f"{avg_mean:.0f} ± {avg_std:.0f}"
    t3_rows.append(avg_row)

    table3_df = pd.DataFrame(t3_rows)
    save_dataframe(table3_df, os.path.join(tables_dir, "table3_eeg_metrics_summary.csv"))
    save_dataframe(table3_df, os.path.join(output_dir, "phase1_summary.csv"))

    # Table 4: Paper vs Replication Comparison
    t4_rows = []
    for dataset, paper_vals in PAPER_REPORTED_TABLE2.items():
        if dataset in numeric_summary:
            for m in metric_cols:
                paper_v = paper_vals.get(m, np.nan)
                our_mean, our_std = numeric_summary[dataset].get(m, (np.nan, np.nan))
                abs_diff = our_mean - paper_v if not np.isnan(our_mean) else np.nan
                rel_diff = (abs_diff / paper_v * 100.0) if paper_v != 0 and not np.isnan(abs_diff) else np.nan

                t4_rows.append({
                    "Dataset": dataset,
                    "Metric": m,
                    "Paper": paper_v,
                    "Ours": round(our_mean, 1),
                    "Ours_Std": round(our_std, 1),
                    "Abs_Difference": round(abs_diff, 1),
                    "Rel_Difference_%": round(rel_diff, 1) if not np.isnan(rel_diff) else "-",
                })

    table4_df = pd.DataFrame(t4_rows)
    save_dataframe(table4_df, os.path.join(tables_dir, "table4_paper_vs_replication.csv"))
    save_dataframe(table4_df, "results/paper_comparison/paper_vs_replication.csv")

    return runs_df, table2_df, table3_df, table4_df
