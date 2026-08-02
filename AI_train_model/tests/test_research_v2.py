"""Unit tests for the isolated V2 scientific contract."""

import unittest

import numpy as np

from research_v2.events import event_metrics_from_records, match_alarms_to_events, temporal_alarms
from research_v2.folds import select_feasible_protocol
from research_v2.protocol import causal_window_index, validate_protocol_config
from research_v2.statistics import paired_bootstrap_interval, promotion_decision


class V2ProtocolTests(unittest.TestCase):
    def test_causal_endpoint_labels_onset_but_not_offset(self):
        positives, normals, all_starts = causal_window_index(
            sample_count=40,
            seizure_intervals=[(10, 20)],
            window_samples=5,
            stride_samples=5,
            guard_samples=0,
        )
        np.testing.assert_array_equal(all_starts, [0, 5, 10, 15, 20, 25, 30, 35])
        np.testing.assert_array_equal(positives, [5, 10])
        np.testing.assert_array_equal(normals, [0, 15, 20, 25, 30, 35])

    def test_guard_excludes_only_nonictal_endpoints(self):
        positives, normals, _ = causal_window_index(
            sample_count=60,
            seizure_intervals=[(20, 30)],
            window_samples=5,
            stride_samples=5,
            guard_samples=5,
        )
        np.testing.assert_array_equal(positives, [15, 20])
        np.testing.assert_array_equal(normals, [0, 5, 30, 35, 40, 45, 50, 55])

    def test_one_alarm_can_detect_only_one_event(self):
        matches, unmatched = match_alarms_to_events([15], [(10, 20), (20, 30)])
        self.assertEqual([match["detected"] for match in matches], [True, False])
        self.assertEqual(unmatched, [])

    def test_unmatched_alarm_counts_as_false_alarm(self):
        metrics = event_metrics_from_records([
            {"sample_count": 3600, "seizure_intervals": [(100, 200)], "alarms": [50, 120, 180]},
        ], sample_rate=1)
        self.assertEqual(metrics["detected_events"], 1)
        self.assertEqual(metrics["false_alarms"], 2)
        self.assertAlmostEqual(metrics["false_alarms_per_hour"], 2.0 / (3500.0 / 3600.0))

    def test_temporal_alarm_respects_refractory(self):
        alarms = temporal_alarms([1, 2, 3, 4], [0.9, 0.9, 0.9, 0.9], 0.8, 3, 1, 1)
        self.assertEqual(alarms, [1, 4])

    def test_five_fold_fallback_is_explicit(self):
        rows = []
        for case_id in ("a", "b", "c"):
            for index in range(6):
                rows.append({"recording_id": f"{case_id}_{index}", "case_id": case_id, "seizure_count": "1"})
        artifact, selected = select_feasible_protocol(rows, requested_folds=5, fallback_folds=3)
        self.assertIn(artifact["selected_outer_folds"], (3, 5))
        self.assertTrue(selected["valid"])

    def test_invalid_protocol_rejects_offline_filter(self):
        config = {
            "preprocessing": {"filter_mode": "zero_phase", "window_sec": 5, "stride_sec": 1},
            "labels": {"rule": "causal_window_endpoint"},
            "split": {"requested_outer_folds": 5, "fallback_outer_folds": 3},
            "evaluation": {"primary_far_per_hour": 0.5},
            "training": {"training_seeds": [7, 42, 123, 314, 2718], "max_epochs": 50, "min_epochs": 12, "early_stopping_patience": 12},
        }
        with self.assertRaises(ValueError):
            validate_protocol_config(config)

    def test_large_model_requires_statistical_and_pareto_gate(self):
        interval = paired_bootstrap_interval([0.80, 0.81, 0.79], [0.84, 0.85, 0.83], replicates=200)
        rejected = promotion_decision(interval, interval, 0.01, False, 30_000)
        self.assertFalse(rejected["promoted"])
        promoted = promotion_decision(interval, interval, 0.01, True, 30_000)
        self.assertTrue(promoted["promoted"])


if __name__ == "__main__":
    unittest.main()
