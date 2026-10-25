# -*- coding: utf-8 -*-
"""
Perturbation utilities for tabular data stress tests in EEG.

Ensures that:
1. Continuous features scaled in [-1, 1] receive controlled additive perturbations.
2. Perturbations respect boundary limits [-1, 1].
3. Categorical and binary indicator columns are strictly preserved without continuous noise.
4. Seeds are fully deterministic and reproducible.
"""

import numpy as np
import pandas as pd


def generate_tabular_perturbations(X, categorical_indicator, epsilon, seed=0, distribution='uniform'):
    """
    Applies controlled additive perturbation of strength epsilon to continuous features in X.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (expected normalized in [-1, 1]).
    categorical_indicator : dict
        Mapping of {feature_name: is_categorical_boolean}.
    epsilon : float
        Perturbation magnitude. epsilon=0.0 returns an exact copy.
    seed : int
        Random seed for reproducibility.
    distribution : str
        'uniform': delta ~ U(-epsilon, epsilon)
        'gaussian': delta ~ N(0, (epsilon/2)^2) clipped to [-epsilon, epsilon]

    Returns:
    --------
    X_pert : pd.DataFrame or np.ndarray (same type as X)
        Perturbed feature matrix clipped to [-1, 1].
    """
    if epsilon <= 0.0:
        return X.copy()

    rng = np.random.RandomState(seed)

    if isinstance(X, pd.DataFrame):
        X_pert = X.copy()
        for col in X.columns:
            is_cat = categorical_indicator.get(col, False)
            if not is_cat:
                col_vals = X[col].to_numpy(dtype=np.float32)
                if distribution == 'uniform':
                    noise = rng.uniform(-epsilon, epsilon, size=len(col_vals))
                elif distribution == 'gaussian':
                    noise = rng.normal(0.0, epsilon / 2.0, size=len(col_vals))
                    noise = np.clip(noise, -epsilon, epsilon)
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")
                X_pert[col] = np.clip(col_vals + noise, -1.0, 1.0)
        return X_pert
    else:
        # Array input
        X_arr = np.array(X, dtype=np.float32, copy=True)
        # If categorical_indicator is a dict with column names, assume all columns continuous if length doesn't match
        is_cat_list = list(categorical_indicator.values()) if len(categorical_indicator) == X_arr.shape[1] else [False] * X_arr.shape[1]
        for j in range(X_arr.shape[1]):
            if not is_cat_list[j]:
                if distribution == 'uniform':
                    noise = rng.uniform(-epsilon, epsilon, size=X_arr.shape[0])
                elif distribution == 'gaussian':
                    noise = rng.normal(0.0, epsilon / 2.0, size=X_arr.shape[0])
                    noise = np.clip(noise, -epsilon, epsilon)
                else:
                    raise ValueError(f"Unknown distribution: {distribution}")
                X_arr[:, j] = np.clip(X_arr[:, j] + noise, -1.0, 1.0)
        return X_arr
