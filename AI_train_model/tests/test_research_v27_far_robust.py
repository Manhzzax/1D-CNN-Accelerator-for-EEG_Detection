"""Tests for the V2.7 multiplicity-aware FAR calibration diagnostic."""

import importlib.util
import unittest

from research_v2.v27_far_robust import (
    select_far_robust_calibration_policy,
    simultaneous_one_sided_confidence,
    validate_v27_far_robust_config,
)


def _config():
    return {
        "version": "v2.7.0-far-robust-calibration-diagnostic-only",
        "candidate_comparators": ["C1", "H2"],
        "allowed_folds": ["00", "01", "02"],
        "sealed_temporal_blocks": [5, 6],
        "candidate_score_subdirectories": {"C1": "v22_development", "H2": "v24_development"},
        "evaluation": {
            "primary_far_per_hour": 0.5, "refractory_sec": 30, "window_sec": 5.0, "sample_rate_hz": 256,
            "declared_operating_point_count": 1200, "family_wise_confidence": 0.95,
            "multiplicity_control": "bonferroni_over_all_predeclared_threshold_policy_pairs",
            "calibration_rule": "maximize_sensitivity_subject_to_one_sided_exact_garwood_far_upper_bound_at_or_below_target",
        },
        "prohibited_actions": [
            "model_training", "hyperparameter_selection", "candidate_selection", "retrospective_temporal_policy_selection",
            "block_5_access", "block_6_access", "final_training", "quantization_calibration", "tensor_export",
            "fpga_performance_claim",
        ],
    }


class V27FarRobustTests(unittest.TestCase):
    def test_config_rejects_unadjusted_calibration_rule(self):
        config = _config()
        config["evaluation"]["multiplicity_control"] = "none"
        with self.assertRaises(ValueError):
            validate_v27_far_robust_config(config)

    def test_family_confidence_is_bonferroni_adjusted(self):
        confidence = simultaneous_one_sided_confidence(_config())
        self.assertAlmostEqual(confidence, 1.0 - 0.05 / 1200)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "requires scipy")
    def test_selection_rejects_when_simultaneous_bound_exceeds_target(self):
        config = _config()
        sweep = _sweep(false_alarms=1, interictal_hours=1.0)
        selected, _ = select_far_robust_calibration_policy(sweep, config)
        self.assertIsNone(selected)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "requires scipy")
    def test_selection_uses_calibration_upper_bound(self):
        config = _config()
        sweep = _sweep(false_alarms=0, interictal_hours=100.0)
        selected, annotated = select_far_robust_calibration_policy(sweep, config)
        self.assertIsNotNone(selected)
        self.assertTrue(all(item["calibration_far_upper_bound_target_met"] for item in annotated))


def _sweep(false_alarms, interictal_hours):
    return [
        {
            "false_alarms": false_alarms, "interictal_hours": interictal_hours, "event_sensitivity": 0.5,
            "false_alarms_per_hour": false_alarms / interictal_hours, "threshold": round(0.85 + index * 0.001, 3),
            "policy_name": f"{positive}_of_{decision}", "positive_windows": positive,
            "decision_window_windows": decision, "median_detection_delay_sec": 1.0,
        }
        for positive, decision in ((3, 6), (4, 8), (5, 10), (6, 12), (7, 14), (8, 16), (9, 18), (10, 20))
        for index in range(150)
    ]


if __name__ == "__main__":
    unittest.main()
