"""
Data preprocessing transformations for EEG replication.

Strictly adheres to the paper's specification:
- All features and regression response variables are rescaled to [-1, 1].
- Transformations are fitted STRICTLY on the training set to prevent data leakage.
- Validation and test sets are transformed using parameters fitted on train.
"""

from typing import Tuple, Optional
import numpy as np


class EEGDataPreprocessor:
    """
    Min-max preprocessor that maps data to [-1, 1] range using training statistics.
    
    Prevents data leakage by computing min and max strictly on training data.
    """

    def __init__(self, task_type: str, scale_features: bool = True, scale_target: bool = True):
        self.task_type = task_type
        self.scale_features = scale_features
        self.scale_target = scale_target and (task_type == "regression")

        self.x_min: Optional[np.ndarray] = None
        self.x_max: Optional[np.ndarray] = None
        self.y_min: Optional[float] = None
        self.y_max: Optional[float] = None
        self.fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "EEGDataPreprocessor":
        """Fit scaler min and max on training set."""
        X_arr = np.asarray(X_train, dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.float32)

        self.x_min = np.min(X_arr, axis=0)
        self.x_max = np.max(X_arr, axis=0)

        if self.scale_target:
            self.y_min = float(np.min(y_arr))
            self.y_max = float(np.max(y_arr))

        self.fitted = True
        return self

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features to [-1, 1] using fitted training statistics."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming features.")
        if not self.scale_features:
            return np.asarray(X, dtype=np.float32)

        X_arr = np.asarray(X, dtype=np.float32)
        denom = self.x_max - self.x_min
        # Handle zero variance features safely
        safe_denom = np.where(denom == 0.0, 1.0, denom)
        scaled_X = 2.0 * (X_arr - self.x_min) / safe_denom - 1.0
        # If feature was constant, map to 0.0
        scaled_X[:, denom == 0.0] = 0.0
        return np.clip(scaled_X, -1.0, 1.0)

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """Scale regression target to [-1, 1] using fitted training statistics."""
        y_arr = np.asarray(y, dtype=np.float32)
        if not self.scale_target:
            return y_arr

        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming target.")

        denom = self.y_max - self.y_min
        if denom == 0.0:
            return np.zeros_like(y_arr)
        scaled_y = 2.0 * (y_arr - self.y_min) / denom - 1.0
        return np.clip(scaled_y, -1.0, 1.0)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Invert scaling back to original target scale."""
        y_arr = np.asarray(y_scaled, dtype=np.float32)
        if not self.scale_target or not self.fitted:
            return y_arr

        denom = self.y_max - self.y_min
        return (y_arr + 1.0) * denom / 2.0 + self.y_min

    def fit_transform(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform training features and target."""
        self.fit(X_train, y_train)
        return self.transform_features(X_train), self.transform_target(y_train)

    def transform_splits(
        self, splits: dict
    ) -> dict:
        """Transform train, val, and test splits with fitted preprocessor."""
        if not self.fitted:
            self.fit(splits["X_train"], splits["y_train"])

        return {
            "X_train": self.transform_features(splits["X_train"]),
            "y_train": self.transform_target(splits["y_train"]),
            "X_val": self.transform_features(splits["X_val"]),
            "y_val": self.transform_target(splits["y_val"]),
            "X_test": self.transform_features(splits["X_test"]),
            "y_test": self.transform_target(splits["y_test"]),
            "indices_train": splits["indices_train"],
            "indices_val": splits["indices_val"],
            "indices_test": splits["indices_test"],
        }
