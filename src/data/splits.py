"""
Data splitting utilities for EEG replication.

Implements the exact 70% train / 9% validation / 21% test split protocol
established by Grinsztajn et al. (2022) and adopted in Pisztora & Li (AAAI 2024).
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_train_val_test_splits(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.09,
    test_ratio: float = 0.21,
) -> Dict[str, np.ndarray]:
    """
    Split dataset into train (70%), validation (9%), and test (21%) partitions.
    
    Stratification is applied for classification tasks to preserve class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
        task_type: 'classification' or 'regression'
        seed: Random seed for deterministic reproducibility
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.09)
        test_ratio: Proportion for testing (default 0.21)
        
    Returns:
        Dictionary containing:
            'X_train', 'y_train',
            'X_val', 'y_val',
            'X_test', 'y_test',
            'indices_train', 'indices_val', 'indices_test'
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    n_samples = len(X)
    indices = np.arange(n_samples)

    stratify_target = y if task_type == "classification" else None

    # Step 1: Split into train_val (79%) and test (21%)
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_target,
    )

    # Step 2: Split train_val into train (70% of total) and val (9% of total)
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    stratify_train_val = y[idx_train_val] if task_type == "classification" else None

    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_rel_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_train_val,
    )

    return {
        "X_train": X[idx_train],
        "y_train": y[idx_train],
        "X_val": X[idx_val],
        "y_val": y[idx_val],
        "X_test": X[idx_test],
        "y_test": y[idx_test],
        "indices_train": idx_train,
        "indices_val": idx_val,
        "indices_test": idx_test,
    }
