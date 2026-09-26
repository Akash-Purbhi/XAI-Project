from src.allocator.sufficiency import (
    compute_loss,
    compute_regression_epsilon,
    compute_sufficiency_indicators,
    categorize_sufficiency,
    compute_sufficient_performance,
)
from src.allocator.desirability import (
    compute_desirability_score,
    compute_normalized_ranking,
    compute_training_category_quantiles,
    predict_sufficiency_categories,
)
from src.allocator.random_allocator import evaluate_random_allocator
from src.allocator.oracle_allocator import evaluate_oracle_allocator
from src.allocator.learned_allocator import (
    build_augmented_features,
    EEGAllocator,
    evaluate_eeg_ensemble,
)

__all__ = [
    "compute_loss",
    "compute_regression_epsilon",
    "compute_sufficiency_indicators",
    "categorize_sufficiency",
    "compute_sufficient_performance",
    "compute_desirability_score",
    "compute_normalized_ranking",
    "compute_training_category_quantiles",
    "predict_sufficiency_categories",
    "evaluate_random_allocator",
    "evaluate_oracle_allocator",
    "build_augmented_features",
    "EEGAllocator",
    "evaluate_eeg_ensemble",
]
