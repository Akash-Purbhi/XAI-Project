"""
I/O utilities for saving and loading machine-readable CSV/JSON experiment artifacts.
"""

import os
import json
from typing import Dict, Any
import pandas as pd


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to formatted JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file into dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Save pandas DataFrame to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
