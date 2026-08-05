"""Tests for the V2.6 artifact-only diagnostic boundary."""

import json
import tempfile
import unittest
from pathlib import Path

from research_v2.v26_diagnostics import build_operating_point_atlas, validate_v26_diagnostic_config


def _config():
    return {
        "version": "v2.6.0-diagnostic-only",
        "consumed_development_folds": ["00", "01", "02"],
        "sealed_temporal_blocks": [5, 6],
        "training_seeds": [7, 42, 123, 314, 2718],
        "event_endpoint": {"primary_far_per_hour": 0.5},
        "artifact_contract": {
            "required_files": [
                "model_spec.json", "provenance.json", "training_summary.json",
                "validation_window_metrics.json", "temporal_confirmation.json", "calibration_policy_sweep.json",
            ],
            "expected_parameter_count": 57446,
            "expected_precision": "amp_fp16_train_fp32_evaluate",
        },
        "candidates": [
            {"id": "C1", "candidate_id": "C1", "artifact_template": "c1_f{fold}_s{seed}"},
            {"id": "H2", "candidate_id": "H2", "artifact_template": "h2_f{fold}_s{seed}"},
            {"id": "G1", "candidate_id": "G1", "artifact_template": "g1_f{fold}_s{seed}"},
        ],
        "reporting": {"top_patient_groups": 3, "diagnostic_limit": "test"},
        "prohibited_actions": [
            "model_training", "hyperparameter_selection", "threshold_selection", "temporal_policy_selection",
            "block_5_access", "block_6_access", "final_training", "quantization_calibration", "tensor_export",
            "fpga_performance_claim",
        ],
    }


def _artifact(path, candidate_id, seed, temporal_far):
    path.mkdir(parents=True)
    files = {
        "model_spec.json": {"parameter_count": 57446},
        "provenance.json": {
            "candidate_id": candidate_id, "training_seed": seed,
            "precision": "amp_fp16_train_fp32_evaluate", "checkpoint_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        },
        "training_summary.json": {},
        "validation_window_metrics.json": {"balanced_accuracy": 0.8, "auroc": 0.9},
        "temporal_confirmation.json": {
            "policy_selection_status": "feasible_calibration_policy_selected",
            "selected_calibration_policy": {"policy_name": "4_of_8", "threshold": 0.99, "false_alarms_per_hour": 0.4},
            "temporal_evaluation": {
                "event_sensitivity": 0.7, "false_alarms_per_hour": temporal_far,
                "detected_events": 7, "total_events": 10, "median_detection_delay_sec": 12.0,
            },
            "temporal_uncertainty": {"per_patient_group": {
                "subject_01": {"false_alarms": 8, "interictal_hours": 4.0, "detected_events": 1, "total_events": 2},
                "subject_02": {"false_alarms": 2, "interictal_hours": 4.0, "detected_events": 6, "total_events": 8},
            }},
        },
        "calibration_policy_sweep.json": [],
    }
    for name, payload in files.items():
        (path / name).write_text(json.dumps(payload), encoding="utf-8")


class V26DiagnosticTests(unittest.TestCase):
    def test_config_rejects_open_block(self):
        config = _config()
        config["sealed_temporal_blocks"] = [6]
        with self.assertRaises(ValueError):
            validate_v26_diagnostic_config(config)

    def test_artifact_atlas_reports_transfer_and_concentration(self):
        config = _config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            artifacts = root / "artifacts"
            for candidate in config["candidates"]:
                for fold in config["consumed_development_folds"]:
                    for seed in config["training_seeds"]:
                        _artifact(
                            artifacts / candidate["artifact_template"].format(fold=fold, seed=seed),
                            candidate["candidate_id"], seed, 0.8,
                        )
            report = build_operating_point_atlas(config_path, artifacts, root / "report")
            summary = report["candidate_fold_summaries"]["C1"]["00"]
            self.assertEqual(summary["temporal_far_target_passes"], 0)
            self.assertAlmostEqual(summary["temporal_minus_calibration_far"]["mean"], 0.4)
            self.assertEqual(summary["patient_group_false_alarm_concentration"]["top_patient_groups"][0]["patient_group"], "subject_01")
            self.assertTrue((root / "report" / "v26_operating_point_atlas.md").is_file())


if __name__ == "__main__":
    unittest.main()
