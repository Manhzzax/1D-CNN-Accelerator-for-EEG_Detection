"""Tests for validation-only threshold selection."""

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from src.operating_point import select_threshold_max_balanced_accuracy


@unittest.skipIf(np is None, "numpy is required")
class OperatingPointTests(unittest.TestCase):
    def test_selects_higher_threshold_when_scores_are_calibrated(self):
        # Negatives clustered near 0.2, positives near 0.8 → mid thresholds work.
        targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.15, 0.25, 0.75, 0.85, 0.9, 0.8])
        result = select_threshold_max_balanced_accuracy(targets, probabilities)
        self.assertGreaterEqual(result["threshold"], 0.3)
        self.assertLessEqual(result["threshold"], 0.7)
        self.assertGreaterEqual(result["validation_balanced_accuracy"], 0.99)

    def test_rejects_single_class_validation(self):
        with self.assertRaises(ValueError):
            select_threshold_max_balanced_accuracy(np.zeros(5), np.linspace(0.1, 0.9, 5))


if __name__ == "__main__":
    unittest.main()

