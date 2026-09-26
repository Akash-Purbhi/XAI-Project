"""
Model factory and hyperparameter tuning for EEG replication.

Implements 4-fold cross-validation tuning on training data for tree and linear/boosting models,
and validation-set tuning for neural network models, strictly following the paper:
"Hyperparameter tuning for all models is done using 4-fold cross-validation, with the
exception of the neural network tuning which is done using the validation set."
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV

from src.models.glass_box import (
    LogisticRegressionModel,
    ClassificationTreeModel,
    LinearRegressionModel,
    RegressionTreeModel,
)
from src.models.black_box import (
    GradientBoostingClassifierModel,
    GradientBoostingRegressorModel,
    TabWRNModel,
)


def get_default_param_grid(model_name: str) -> Dict[str, list]:
    """Return compact, effective hyperparameter grid for tuning."""
    grid = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "ClassificationTree": {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "max_leaf_nodes": [8, 16, 32, 64, None],
        },
        "LinearRegression": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        },
        "RegressionTree": {
            "max_depth": [8, 12, None],
            "min_samples_split": [2, 5, 10],
            "max_leaf_nodes": [32, 64, None],
        },
        "GradientBoostingClassifier": {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [64, 100, 128],
            "max_depth": [3, 4, 6],
            "subsample": [0.8, 1.0],
        },
        "GradientBoostingRegressor": {
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 128],
            "max_depth": [5, 6],
            "subsample": [0.8, 1.0],
        },
    }
    return grid.get(model_name, {})


def tune_and_fit_model(
    model_name: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    seed: int = 0,
    cv_folds: int = 4,
    tune: bool = True,
    **fixed_kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune (via 4-fold CV on train) and fit a model on the full training set.
    
    Args:
        model_name: Model identifier string
        task_type: 'classification' or 'regression'
        X_train: Preprocessed training features
        y_train: Preprocessed training targets
        X_val: Preprocessed validation features (for neural net or validation checks)
        y_val: Preprocessed validation targets
        seed: Random seed
        cv_folds: Number of folds for cross validation (default 4)
        tune: Whether to run grid search tuning
        fixed_kwargs: Additional fixed kwargs
        
    Returns:
        fitted_model: The fitted model instance
        best_params: Dictionary of chosen hyperparameters
    """
    if model_name == "TabWRN":
        # Neural network tuning uses validation set per paper
        model = TabWRNModel(task_type=task_type, random_state=seed, **fixed_kwargs)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return model, {"learning_rate": model.learning_rate, "base_nodes": model.base_nodes}

    param_grid = get_default_param_grid(model_name)
    if not tune or not param_grid:
        # Fit with defaults
        model = _instantiate_model(model_name, task_type, seed=seed, **fixed_kwargs)
        model.fit(X_train, y_train)
        return model, fixed_kwargs

    # Map model name to base sklearn estimator for GridSearchCV
    base_estimator, scoring = _get_sklearn_base_estimator(model_name, task_type, seed=seed)

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Wrap in our model wrapper
    wrapped_model = _wrap_sklearn_estimator(model_name, best_estimator)
    return wrapped_model, best_params


def _instantiate_model(model_name: str, task_type: str, seed: int = 0, **kwargs):
    if model_name == "LogisticRegression":
        return LogisticRegressionModel(random_state=seed, **kwargs)
    elif model_name == "ClassificationTree":
        return ClassificationTreeModel(random_state=seed, **kwargs)
    elif model_name == "LinearRegression":
        return LinearRegressionModel(random_state=seed, **kwargs)
    elif model_name == "RegressionTree":
        return RegressionTreeModel(random_state=seed, **kwargs)
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifierModel(random_state=seed, **kwargs)
    elif model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressorModel(random_state=seed, **kwargs)
    elif model_name == "TabWRN":
        return TabWRNModel(task_type=task_type, random_state=seed, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _get_sklearn_base_estimator(model_name: str, task_type: str, seed: int = 0):
    from sklearn.linear_model import LogisticRegression, Lasso
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    if model_name == "LogisticRegression":
        return LogisticRegression(random_state=seed, max_iter=2000), "accuracy"
    elif model_name == "ClassificationTree":
        return DecisionTreeClassifier(random_state=seed), "accuracy"
    elif model_name == "LinearRegression":
        return Lasso(random_state=seed, max_iter=2000), "neg_mean_squared_error"
    elif model_name == "RegressionTree":
        return DecisionTreeRegressor(random_state=seed), "neg_mean_squared_error"
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(random_state=seed), "accuracy"
    elif model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(random_state=seed), "neg_mean_squared_error"
    else:
        raise ValueError(f"No sklearn base estimator for {model_name}")


def _wrap_sklearn_estimator(model_name: str, estimator: Any):
    class SklearnWrapper:
        def __init__(self, raw_est):
            self.model = raw_est

        def fit(self, X, y):
            self.model.fit(X, y)
            return self

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

        def score(self, X, y):
            return float(self.model.score(X, y))

    return SklearnWrapper(estimator)
