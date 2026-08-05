"""Tests for the sealed patient-group V2.1 forward protocol."""

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from research_v2.protocol import canonical_json_hash, validate_protocol_config
from research_v2.statistics import patient_group_cluster_bootstrap
from research_v2.v21 import assign_patient_group_blocks, audit_v21, create_final_freeze, verify_final_freeze, write_v21_artifacts


def _config():
    return {
        "version": "v2.1.0",
        "dataset": {"case_ids": 24, "patient_groups": 23},
        "preprocessing": {"filter_mode": "causal_iir", "window_sec": 5, "stride_sec": 1},
        "labels": {"rule": "causal_window_endpoint"},
        "split": {
            "strategy": "patient_group_cumulative_duration_forward_chaining", "base_block_count": 7,
            "confirmation_folds": 3, "final_test_status": "sealed_until_final_freeze",
            "patient_grouping": {
                "case_to_patient_group": {"chb01": "subject_01_21", "chb21": "subject_01_21"},
                "session_order": {"subject_01_21": ["chb21", "chb01"]},
                "session_order_evidence": "test",
            },
            "feasibility_gate": {
                "minimum_union_seizures": 20, "minimum_seizure_contributing_patient_groups": 5,
                "minimum_nonictal_replay_hours": 24.0,
            },
        },
        "evaluation": {"primary_far_per_hour": 0.5},
        "training": {
            "training_seeds": [7, 42, 123, 314, 2718], "deployment_export_seed": 42,
            "max_epochs": 50, "min_epochs": 12, "early_stopping_patience": 12,
            "frozen_candidates": {
                "B4_dilated_lightseizure_like": {
                    "architecture": "dilated_hierarchical_separable_1dcnn", "learning_rate": 0.001,
                    "weight_decay": 0.0005,
                },
            },
        },
        "hardware": {
            "quantization_primary": "symmetric_int16_weights_activations_int32_bias_accumulator",
            "input_layout": "NCT", "input_shape": [1, 17, 1280],
        },
    }


def _row(case_id, index):
    return {
        "recording_id": f"{case_id}/{case_id}_{index:02d}.edf", "case_id": case_id,
        "edf_path": f"/{case_id}_{index:02d}.edf", "sample_count": "14400", "sampling_rate_hz": "1",
        "seizure_intervals_json": "[[10, 20]]", "seizure_count": "10",
    }


def _rows():
    rows = [_row("chb01", index) for index in range(4)] + [_row("chb21", index) for index in range(3)]
    for case_id in ("chb02", "chb03", "chb04", "chb05"):
        rows.extend(_row(case_id, index) for index in range(7))
    return rows


class V21ProtocolTests(unittest.TestCase):
    def test_patient_group_merge_and_duration_blocks(self):
        assigned = assign_patient_group_blocks(_rows(), _config()["split"])
        subject_rows = sorted(
            (row for row in assigned if row["patient_group"] == "subject_01_21"),
            key=lambda row: row["patient_recording_order"],
        )
        self.assertEqual(len(subject_rows), 7)
        self.assertEqual([row["case_id"] for row in subject_rows[:3]], ["chb21"] * 3)
        self.assertEqual([row["case_id"] for row in subject_rows[3:]], ["chb01"] * 4)
        self.assertEqual(sorted({row["temporal_block"] for row in subject_rows}), list(range(7)))

    def test_confirmation_gate_and_sealed_final_block(self):
        audit = audit_v21(_rows(), _config())
        self.assertTrue(audit["valid"])
        self.assertEqual(len(audit["confirmation_folds"]), 3)
        for fold in audit["confirmation_folds"]:
            self.assertGreaterEqual(fold["validation_union"]["seizures"], 20)
            self.assertGreaterEqual(fold["validation_union"]["patient_groups"], 5)
            self.assertTrue(all(row["temporal_block"] < 5 for row in fold["rows"] if row["split"] != "future"))
        final_test = [row for row in audit["final_holdout"]["rows"] if row["split"] == "test"]
        self.assertTrue(final_test)
        self.assertTrue(all(row["temporal_block"] == 6 for row in final_test))

    def test_freeze_hash_blocks_manifest_or_protocol_changes(self):
        decision = {
            "candidate_id": "B4_dilated_lightseizure_like", "architecture": "dilated_hierarchical_separable_1dcnn",
            "learning_rate": 0.001, "weight_decay": 0.0005, "seed_schedule": [7, 42, 123, 314, 2718],
            "deployment_export_seed": 42, "threshold_policy_search": {},
            "quantization": _config()["hardware"]["quantization_primary"], "hardware_interface": _config()["hardware"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, manifest, decision_path, freeze = root / "protocol.json", root / "final.csv", root / "decision.json", root / "freeze.json"
            protocol.write_text(json.dumps(_config()), encoding="utf-8")
            manifest.write_text("recording_id\nexample\n", encoding="utf-8")
            decision_path.write_text(json.dumps(decision), encoding="utf-8")
            create_final_freeze(protocol, manifest, decision_path, freeze)
            self.assertEqual(verify_final_freeze(freeze, protocol, manifest)["decision"]["deployment_export_seed"], 42)
            manifest.write_text("recording_id\nchanged\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_final_freeze(freeze, protocol, manifest)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_patient_group_bootstrap_does_not_count_sessions_twice(self):
        interval = patient_group_cluster_bootstrap({"subject_01_21": (1, 2), "subject_02": (1, 2)}, replicates=100)
        self.assertEqual(interval.replicates, 100)
        self.assertAlmostEqual(interval.mean_difference, 0.5)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_event_bootstrap_excludes_groups_without_events(self):
        from research_v2.v21_evaluation import _cluster_interval_or_unavailable

        interval = _cluster_interval_or_unavailable({"subject_02": (1, 2)})
        self.assertFalse(interval["estimable"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_no_far_eligible_policy_is_a_rejection_not_an_exception(self):
        from research_v2.v21_evaluation import select_calibration_policy

        scores = {
            "probabilities": __import__("numpy").ones(6), "start_samples": __import__("numpy").arange(6),
            "record_offsets": __import__("numpy").asarray([0, 6]),
            "records": [{"sample_count": 100, "seizure_intervals": [[1, 5]]}],
        }
        config = {
            "dataset": {"sample_rate_hz": 1}, "preprocessing": {"window_sec": 1, "stride_sec": 1, "bandpass_hz": [0.5, 1.0], "notch_hz": 1, "filter_mode": "causal_iir"},
            "evaluation": {"refractory_sec": 0, "primary_far_per_hour": 0.0, "temporal_policies": [[1, 1]], "threshold_grid": {"minimum": 0.9, "maximum": 0.9, "step": 0.1}},
        }
        selected, sweep = select_calibration_policy(scores, config)
        self.assertIsNone(selected)
        self.assertEqual(len(sweep), 1)

    def test_v21_config_contract(self):
        validate_protocol_config(_config())
        self.assertEqual(canonical_json_hash(_config()), canonical_json_hash(_config()))

    def test_v22_is_development_only_and_cannot_open_final_holdout(self):
        protocol_path = Path(__file__).resolve().parents[1] / "research_v2" / "configs" / "protocol_v2_2.json"
        config = json.loads(protocol_path.read_text(encoding="utf-8"))
        validate_protocol_config(config)
        self.assertEqual(config["split"]["final_test_status"], "sealed_v22_development_only")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = root / "decision.json"
            manifest = root / "final.csv"
            decision.write_text("{}", encoding="utf-8")
            manifest.write_text("recording_id\nexample\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                create_final_freeze(protocol_path, manifest, decision, root / "freeze.json")

    def test_v23_is_development_only_and_has_a_frozen_mining_contract(self):
        protocol_path = Path(__file__).resolve().parents[1] / "research_v2" / "configs" / "protocol_v2_3.json"
        config = json.loads(protocol_path.read_text(encoding="utf-8"))
        validate_protocol_config(config)
        mining = config["hard_negative_mining"]
        self.assertEqual(config["split"]["final_test_status"], "sealed_v23_development_only")
        self.assertEqual(mining["hard_negative_to_positive_ratio"], 0.10)
        self.assertEqual(mining["sampling_multiplier"], 3.0)
        self.assertEqual(set(mining["source_runs"]), {"00", "01", "02"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision, manifest = root / "decision.json", root / "final.csv"
            decision.write_text("{}", encoding="utf-8")
            manifest.write_text("recording_id\nexample\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                create_final_freeze(protocol_path, manifest, decision, root / "freeze.json")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
    def test_v23_miner_requires_full_clean_alarm_context_and_new_window(self):
        import numpy as np
        from research_v2.v23_hard_negative import select_policy_aligned_candidates

        starts = np.arange(7, dtype=np.int64)
        probabilities = np.asarray([0.1, 0.96, 0.96, 0.1, 0.1, 0.1, 0.1])
        candidates, diagnostics = select_policy_aligned_candidates(
            starts, probabilities, set(starts.tolist()), set(), "r", "subject_01", 1,
            0.95, 2, 3, 3,
        )
        self.assertEqual(diagnostics["clean_false_alarm_contexts"], 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["start_sample"], 1)
        guarded, _ = select_policy_aligned_candidates(
            starts, probabilities, {0, 2, 3, 4, 5, 6}, set(), "r", "subject_01", 1,
            0.95, 2, 3, 3,
        )
        self.assertEqual(guarded, [])
        duplicate, _ = select_policy_aligned_candidates(
            starts, probabilities, set(starts.tolist()), {("r", 1), ("r", 2)}, "r", "subject_01", 1,
            0.95, 2, 3, 3,
        )
        self.assertEqual(duplicate, [])


if __name__ == "__main__":
    unittest.main()
