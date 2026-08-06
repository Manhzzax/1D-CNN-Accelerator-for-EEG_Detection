"""Validation-only operating-point selection for sealed test evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def select_threshold_max_balanced_accuracy(targets, probabilities, grid=None):
    """Pick a threshold using validation labels only.

    Grid defaults to 0.05..0.95 inclusive with step 0.01, matching common
    offline screening grids without touching the test partition.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probabilities.ndim != 1 or len(targets) != len(probabilities):
        raise ValueError("targets and probabilities must be 1-D arrays of equal length")
    if len(np.unique(targets)) < 2:
        raise ValueError("Validation set must contain both classes to select a threshold")

    if grid is None:
        grid = np.round(np.arange(0.05, 0.950001, 0.01), 4)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in grid:
        predictions = (probabilities >= float(threshold)).astype(np.int64)
        score = float(balanced_accuracy_score(targets, predictions))
        # Prefer higher threshold on ties to reduce false positives slightly.
        if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
            best_score = score
            best_threshold = float(threshold)
    return {
        "threshold": best_threshold,
        "validation_balanced_accuracy": best_score,
        "grid_min": float(grid[0]),
        "grid_max": float(grid[-1]),
        "grid_step": float(grid[1] - grid[0]) if len(grid) > 1 else 0.0,
        "objective": "max_validation_balanced_accuracy",
    }
