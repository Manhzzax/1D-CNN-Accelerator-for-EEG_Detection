"""Boundary tests for V2.6 score-replay diagnostics."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.v26_score_replay import validate_v26_score_replay_config


def _config():
    return {
        "version": "v2.6.1-score-replay-diagnostic-only",
        "allowed_folds": ["00", "01", "02"],
        "sealed_temporal_blocks": [5, 6],
        "candidate_score_subdirectories": {"C1": "v22_development", "H2": "v24_development", "G1": "v25_development"},
        "preprocessing": {
            "sample_rate_hz": 256, "window_sec": 5.0, "stride_sec": 1.0,
            "filter_mode": "causal_iir", "normalization": "train_channel_zscore",
        },
        "evaluation": {
            "primary_far_per_hour": 0.5, "refractory_sec": 30,
            "threshold_grid": {"minimum": 0.85, "maximum": 0.999, "step": 0.001},
            "temporal_policies": [[3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16], [9, 18], [10, 20]],
        },
        "score_cache": {
            "default_mode": "reuse_verified_existing_run_scores_only",
            "rescore_missing_requires_explicit_flag": True,
            "batch_size": 128,
        },
        "prohibited_actions": [
            "model_training", "hyperparameter_selection", "threshold_selection", "temporal_policy_selection",
            "candidate_selection", "block_5_access", "block_6_access", "final_training", "quantization_calibration",
            "tensor_export", "fpga_performance_claim",
        ],
    }


class V26ScoreReplayTests(unittest.TestCase):
    def test_config_rejects_changed_policy_grid(self):
        config = _config()
        config["evaluation"]["temporal_policies"] = [[1, 1]]
        with self.assertRaises(ValueError):
            validate_v26_score_replay_config(config)

    def test_config_rejects_missing_sealed_block(self):
        config = _config()
        config["sealed_temporal_blocks"] = [6]
        with self.assertRaises(ValueError):
            validate_v26_score_replay_config(config)

    def test_config_rejects_implicit_rescore_mode(self):
        config = _config()
        config["score_cache"]["rescore_missing_requires_explicit_flag"] = False
        with self.assertRaises(ValueError):
            validate_v26_score_replay_config(config)

    def test_filter_rejects_artifact_outside_frozen_set(self):
        from research_v2.v26_score_replay import _filter_records

        with self.assertRaises(ValueError):
            _filter_records([{"artifact": "frozen_run"}], ["other_run"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_temporal_oracle_never_returns_ineligible_policy(self):
        import numpy as np
        from research_v2.v26_score_replay import _temporal_oracle

        config = _config()
        scores = {
            "probabilities": np.asarray([0.99, 0.99, 0.99, 0.99]),
            "start_samples": np.asarray([0, 1, 2, 3]),
            "record_offsets": np.asarray([0, 4]),
            "records": [{"sample_count": 10000, "seizure_intervals": [[2, 4]]}],
        }
        result = _temporal_oracle(scores, config)
        if result["target_feasible"]:
            self.assertLessEqual(result["best_sensitivity_at_target"]["false_alarms_per_hour"], 0.5)


if __name__ == "__main__":
    unittest.main()
