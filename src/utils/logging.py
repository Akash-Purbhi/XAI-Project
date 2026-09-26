"""
Experiment logging configuration.
"""

import os
import sys
import logging
from datetime import datetime


def setup_logger(log_name: str, log_dir: str = "experiments/phase1/logs") -> logging.Logger:
    """Set up logger that outputs to both console and a log file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear previous handlers

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
