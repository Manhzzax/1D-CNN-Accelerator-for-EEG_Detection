"""Regression tests for the patient-held-out causal evaluation protocol."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.chbmit_patient_split import create_patient_heldout_split_plan
from src.chbmit_preparation import filter_eeg


class PatientHeldoutProtocolTests(unittest.TestCase):
    def test_case_01_and_case_21_remain_in_one_patient_group(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit_dir = Path(temporary_directory) / "audit"
            protocol_dir = Path(temporary_directory) / "protocol"
            audit_dir.mkdir()
            rows = []
            for case_id, seizure_count in (
                ("chb01", 2), ("chb21", 3), ("chb02", 5), ("chb03", 4),
                ("chb04", 3), ("chb05", 2), ("chb06", 1),
            ):
                rows.append({
                    "case_id": case_id,
                    "recording_id": f"{case_id}/{case_id}_01.edf",
                    "seizure_count": seizure_count,
                })
            with (audit_dir / "recording_manifest.csv").open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            plan, manifest_path = create_patient_heldout_split_plan(
                audit_dir, protocol_dir, (0.6, 0.2, 0.2), seed=42
            )
            with manifest_path.open("r", newline="", encoding="utf-8") as input_file:
                planned_rows = list(csv.DictReader(input_file))

            by_case = {row["case_id"]: row for row in planned_rows}
            self.assertEqual(by_case["chb01"]["patient_group"], "subject_01_21")
            self.assertEqual(by_case["chb01"]["split"], by_case["chb21"]["split"])
            self.assertEqual(plan["strategy"], "patient_group_disjoint_stratified_holdout")
            self.assertTrue(all(plan["aggregate"][split]["seizures"] > 0 for split in ("train", "val", "test")))

    def test_causal_filter_prefix_is_invariant_to_future_samples(self):
        random_state = np.random.default_rng(42)
        signal = random_state.normal(size=(17, 1024)).astype(np.float32)
        full_filtered = filter_eeg(signal, 256, 0.5, 45.0, 60.0, filter_mode="causal_iir")
        prefix_filtered = filter_eeg(signal[:, :512], 256, 0.5, 45.0, 60.0, filter_mode="causal_iir")
        np.testing.assert_allclose(full_filtered[:, :512], prefix_filtered, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
