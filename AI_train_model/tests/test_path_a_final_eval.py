import json
import tempfile
import unittest
from pathlib import Path

from research_path_a_final.aggregate import aggregate_event_artifacts, aggregate_window_artifacts


def _window(case_id, balanced, sensitivity):
    return {
        "evaluation_kind": "exploratory_test_probe",
        "source_run_id": f"ps_a12_{case_id}_s42",
        "balanced_test_diagnostic": {
            "positive_windows": 100,
            "negative_windows": 100,
            "metrics": {"balanced_accuracy": balanced, "sensitivity": sensitivity},
        },
        "test_window_counts": {"positive": 100, "negative": 1000, "total": 1100},
        "test_prevalence_metrics": {
            "accuracy": 0.9, "balanced_accuracy": balanced, "sensitivity": sensitivity,
        },
    }


def _event(case_id, detected, total, alarms, hours):
    return {
        "evaluation_kind": "exploratory_test_replay",
        "case_id": case_id,
        "test_event_metrics": {
            "detected_events": detected,
            "total_events": total,
            "false_alarms": alarms,
            "interictal_hours": hours,
            "detection_delays_sec": [4.0] * detected,
            "threshold": 0.5,
            "positive_windows": 1,
            "decision_window_windows": 1,
            "policy_name": "single_window",
        },
    }


class PathAFinalEvaluationTests(unittest.TestCase):
    def test_window_aggregation_merges_chb01_and_chb21(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for case, balanced, sensitivity in (("chb01", 0.8, 0.7), ("chb21", 0.9, 0.9), ("chb02", 0.6, 0.5)):
                path = root / f"{case}.json"
                path.write_text(json.dumps(_window(case, balanced, sensitivity)), encoding="utf-8")
                paths.append(path)
            result = aggregate_window_artifacts(paths, replicates=100, seed=1)
            self.assertFalse(result["final_claim_eligible"])
            self.assertEqual(result["case_count"], 3)
            self.assertEqual(result["patient_group_count"], 2)
            merged = next(row for row in result["per_patient_group"] if row["patient_group"] == "subject_01_21")
            self.assertEqual(merged["case_ids"], ["chb01", "chb21"])

    def test_event_aggregation_pools_counts_by_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for case, detected, total, alarms, hours in (("chb01", 1, 2, 2, 1.0), ("chb21", 2, 2, 1, 1.0), ("chb02", 1, 2, 3, 2.0)):
                path = root / f"{case}.json"
                path.write_text(json.dumps(_event(case, detected, total, alarms, hours)), encoding="utf-8")
                paths.append(path)
            result = aggregate_event_artifacts(paths, replicates=100, seed=1)
            self.assertEqual(result["patient_group_count"], 2)
            self.assertAlmostEqual(result["pooled_event_sensitivity"]["estimate"], 4.0 / 6.0)
            self.assertAlmostEqual(result["pooled_false_alarms_per_hour"]["estimate"], 6.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
