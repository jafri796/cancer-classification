"""
Cross-validation utilities for PCam center-region detection.
"""

from typing import List, Tuple
import logging
import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def create_cv_splits(
    x_path: str,
    y_path: str,
    n_folds: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified CV splits based on labels in PCam H5.
    """
    with h5py.File(y_path, "r") as f:
        labels = f["y"][:, 0, 0, 0].astype(int)

    indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    splits = []
    for train_idx, val_idx in skf.split(indices, labels):
        splits.append((train_idx, val_idx))

    logger.info(f"Created {n_folds} stratified folds from {y_path}")
    return splits