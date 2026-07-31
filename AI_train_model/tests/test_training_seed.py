"""Tests for explicit reproducible training-seed parsing."""

import os
import unittest
from unittest.mock import patch

from scripts.run_train import _training_seed


class TrainingSeedTests(unittest.TestCase):
    def test_environment_seed_overrides_configured_seed(self):
        with patch.dict(os.environ, {"CHBMIT_TRAIN_SEED": "314"}, clear=False):
            self.assertEqual(_training_seed({"data": {"seed": 42}}), 314)

    def test_negative_seed_is_rejected(self):
        with patch.dict(os.environ, {"CHBMIT_TRAIN_SEED": "-1"}, clear=False):
            with self.assertRaises(ValueError):
                _training_seed({"data": {"seed": 42}})


if __name__ == "__main__":
    unittest.main()
