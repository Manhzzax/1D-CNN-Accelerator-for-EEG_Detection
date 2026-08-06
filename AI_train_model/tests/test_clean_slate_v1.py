"""Unit tests for the clean-slate v1 config contract."""

import os
import unittest
from pathlib import Path
from unittest import mock

from src.runtime_config import apply_runtime_overrides


ROOT = Path(__file__).resolve().parents[1]
CLEAN_CONFIG = ROOT / "config" / "clean_slate_v1.yaml"


class CleanSlateV1Tests(unittest.TestCase):
    def test_clean_slate_yaml_file_exists_with_contract_markers(self):
        text = CLEAN_CONFIG.read_text(encoding="utf-8")
        self.assertIn("protocol_output_dir: \"chbmit_protocol_clean_slate_v1\"", text)
        self.assertIn("prepared_output_dir: \"chbmit_prepared_raw_5s_clean_v1\"", text)
        self.assertIn("train: 0.60", text)
        self.assertIn("val: 0.20", text)
        self.assertIn("test: 0.20", text)
        self.assertIn("filter_mode: \"causal_iir\"", text)
        self.assertIn("window_sec: 5", text)
        self.assertIn("architecture: \"hierarchical_separable_1dcnn\"", text)
        self.assertIn("input_length: 1280", text)
        self.assertIn("primary_success_metric: \"test_balanced_window_accuracy\"", text)
        self.assertIn("primary_success_threshold: 0.95", text)

    def test_split_ratio_env_override(self):
        config = {
            "data": {"split_ratios": {"train": 0.7, "val": 0.1, "test": 0.2}},
            "preprocessing": {
                "sample_rate_hz": 256,
                "stride_sec": 1,
                "window_sec": 5,
            },
            "model": {"input_length": 1280},
        }
        with mock.patch.dict(os.environ, {"CHBMIT_SPLIT_RATIOS": "0.6,0.2,0.2"}, clear=False):
            # Clear window override so only split ratios apply.
            env = {k: v for k, v in os.environ.items() if k != "CHBMIT_WINDOW_SEC"}
            env["CHBMIT_SPLIT_RATIOS"] = "0.6,0.2,0.2"
            with mock.patch.dict(os.environ, env, clear=True):
                apply_runtime_overrides(config)
        self.assertEqual(config["data"]["split_ratios"]["train"], 0.6)
        self.assertEqual(config["data"]["split_ratios"]["val"], 0.2)
        self.assertEqual(config["data"]["split_ratios"]["test"], 0.2)


if __name__ == "__main__":
    unittest.main()
