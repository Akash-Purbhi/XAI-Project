"""
Intrinsically explainable glass-box models for EEG replication.

Classification:
- Logistic Regression (L1-regularized or L2-regularized)
- Classification Tree (DecisionTreeClassifier)

Regression:
- Linear Regression (Lasso / Ridge / Ordinary Least Squares)
- Regression Tree (DecisionTreeRegressor)
"""

from typing import Optional, Dict, Any, Union
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class LogisticRegressionModel:
    """Logistic Regression glass-box model with probabilistic output."""

    def __init__(self, C: float = 1.0, penalty: str = "l2", solver: str = "lbfgs", random_state: int = 0, **kwargs):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            random_state=self.random_state,
            max_iter=2000,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))


class ClassificationTreeModel:
    """Decision Tree Classifier glass-box model."""

    def __init__(
        self,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        max_leaf_nodes: Optional[int] = None,
        random_state: int = 0,
        **kwargs,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassificationTreeModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))


class LinearRegressionModel:
    """Linear regression / regularized linear glass-box model."""

    def __init__(self, alpha: float = 0.01, model_type: str = "lasso", random_state: int = 0, **kwargs):
        self.alpha = alpha
        self.model_type = model_type
        self.random_state = random_state
        self.kwargs = kwargs

        if model_type == "lasso":
            self.model = Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=2000, **kwargs)
        elif model_type == "ridge":
            self.model = Ridge(alpha=self.alpha, random_state=self.random_state, **kwargs)
        else:
            self.model = LinearRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))


class RegressionTreeModel:
    """Decision Tree Regressor glass-box model."""

    def __init__(
        self,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        max_leaf_nodes: Optional[int] = None,
        random_state: int = 0,
        **kwargs,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTreeModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(X, y))
