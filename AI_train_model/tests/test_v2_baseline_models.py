"""Shape and budget checks for newly frozen V2 baseline model definitions."""

import unittest

import torch

from src.model import build_model, load_config


class V2BaselineModelTests(unittest.TestCase):
    def test_new_v2_baselines_accept_five_second_windows(self):
        config = load_config()["model"]
        config["input_length"] = 1280
        sample = torch.randn(2, 17, 1280)
        for architecture in ("v2_bandpower_linear", "v2_vanilla_1dcnn", "v2_deep_matched_1dcnn"):
            with self.subTest(architecture=architecture):
                model = build_model(architecture, config)
                self.assertEqual(tuple(model(sample).shape), (2, 2))

    def test_plain_cnn_baseline_budgets_are_under_25k(self):
        config = load_config()["model"]
        for architecture in ("v2_vanilla_1dcnn", "v2_deep_matched_1dcnn"):
            model = build_model(architecture, config)
            self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 25_000)


if __name__ == "__main__":
    unittest.main()
