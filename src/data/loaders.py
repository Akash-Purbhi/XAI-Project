"""
Dataset loaders for EEG replication.

Explicitly loads the 5 specified benchmark datasets from OpenML:
1. Wine (OpenML ID: 44091, Classification)
2. Bank (OpenML ID: 44126, Classification)
3. PolR (OpenML ID: 44133, Regression)
4. SuperconductR (OpenML ID: 44148, Regression)
5. BrazilianHousesR (OpenML ID: 44141, Regression)
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


DATASET_REGISTRY = {
    "Wine": {
        "openml_id": 44091,
        "task_type": "classification",
        "target_col": "class",
        "description": "Wine quality binary classification (Grinsztajn benchmark)",
        "label_mapping": {"False": 0, "True": 1, False: 0, True: 1, "0": 0, "1": 1, 0: 0, 1: 1},
        "url": "https://www.openml.org/d/44091",
    },
    "Bank": {
        "openml_id": 44126,
        "task_type": "classification",
        "target_col": "Class",
        "description": "Bank marketing binary classification (Grinsztajn benchmark)",
        "label_mapping": {"1": 0, "2": 1, 1: 0, 2: 1, "False": 0, "True": 1},
        "url": "https://www.openml.org/d/44126",
    },
    "PolR": {
        "openml_id": 44133,
        "task_type": "regression",
        "target_col": "target",
        "description": "Pol commercial dataset regression (Grinsztajn benchmark)",
        "label_mapping": None,
        "url": "https://www.openml.org/d/44133",
    },
    "SuperconductR": {
        "openml_id": 44148,
        "task_type": "regression",
        "target_col": "critical_temp",
        "description": "Superconductivity critical temperature regression (Grinsztajn benchmark)",
        "label_mapping": None,
        "url": "https://www.openml.org/d/44148",
    },
    "BrazilianHousesR": {
        "openml_id": 44141,
        "task_type": "regression",
        "target_col": "total",
        "description": "Brazilian houses rent pricing regression (Grinsztajn benchmark)",
        "label_mapping": None,
        "url": "https://www.openml.org/d/44141",
    },
}


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load raw dataset by name.
    
    Args:
        dataset_name: One of ['Wine', 'Bank', 'PolR', 'SuperconductR', 'BrazilianHousesR']
        
    Returns:
        X: Feature matrix of shape (n_samples, n_features) as float32
        y: Target array of shape (n_samples,) as float32 (or int32 for classification)
        metadata: Dictionary containing dataset characteristics
    """
    # Normalize dataset name lookup (case insensitive / alias handling)
    normalized = None
    for key in DATASET_REGISTRY:
        if dataset_name.lower().replace("_", "") == key.lower().replace("_", ""):
            normalized = key
            break

    if normalized is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Allowed datasets: {list(DATASET_REGISTRY.keys())}"
        )

    info = DATASET_REGISTRY[normalized]
    X_df, y_ser = fetch_openml(data_id=info["openml_id"], return_X_y=True, as_frame=True)

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)

    if info["task_type"] == "classification":
        mapping = info["label_mapping"]
        # Map labels to 0 and 1
        y_mapped = y_ser.map(mapping)
        if y_mapped.isna().any():
            # Fallback to factorize if string labels don't match mapping exactly
            y_mapped, _ = pd.factorize(y_ser)
        y = np.asarray(y_mapped, dtype=np.int32)
    else:
        y = np.asarray(y_ser, dtype=np.float32)

    metadata = {
        "dataset_name": normalized,
        "task_type": info["task_type"],
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_col": info["target_col"],
        "openml_id": info["openml_id"],
        "url": info["url"],
        "feature_names": feature_names,
    }

    return X, y, metadata
