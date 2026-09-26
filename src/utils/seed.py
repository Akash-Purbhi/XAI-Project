"""
Deterministic random seed management for reproducibility.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 0) -> None:
    """Set random seed across all libraries to ensure deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
