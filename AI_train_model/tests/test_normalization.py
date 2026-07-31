"""Tests for leakage-safe fixed-window normalization."""

import unittest

import numpy as np

from src.normalization import window_channel_zscore


class WindowNormalizationTests(unittest.TestCase):
    def test_each_window_channel_has_zero_mean_and_unit_variance(self):
        windows = np.arange(2 * 17 * 512, dtype=np.float32).reshape(2, 17, 512)
        normalized = window_channel_zscore(windows)
        np.testing.assert_allclose(normalized.mean(axis=2), 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=2), 1.0, atol=1e-6)

    def test_single_window_shape_is_preserved(self):
        window = np.ones((17, 512), dtype=np.float32)
        self.assertEqual(window_channel_zscore(window).shape, (17, 512))


if __name__ == "__main__":
    unittest.main()
