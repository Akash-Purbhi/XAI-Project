"""
Black-box models for EEG replication.

Classification:
- Gradient Boosting Trees Classifier (GradientBoostingClassifier)
- Neural Network Classifier (TabWRN-28)

Regression:
- Gradient Boosting Trees Regressor (GradientBoostingRegressor)
- Neural Network Regressor (TabWRN-28)
"""

from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from src.models.tab_wrn import TabWRNModel


class GradientBoostingClassifierModel:
    """Gradient Boosting Trees Classifier black-box model."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        subsample: float = 0.8,
        random_state: int = 0,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifierModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))


class GradientBoostingRegressorModel:
    """Gradient Boosting Trees Regressor black-box model."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        subsample: float = 0.8,
        random_state: int = 0,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressorModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))
