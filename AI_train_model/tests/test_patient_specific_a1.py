"""Unit tests for Path A patient-specific planning."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.chbmit_patient_specific import create_patient_specific_split_plans


def _write_audit(audit_dir: Path, cases):
    path = audit_dir / "recording_manifest.csv"
    fieldnames = [
        "recording_id", "case_id", "edf_path", "sample_count", "seizure_count", "seizure_intervals_json",
    ]
    rows = []
    for case_id, recs in cases.items():
        for index, seizures in enumerate(recs):
            rows.append({
                "recording_id": f"{case_id}/{case_id}_{index:02d}.edf",
                "case_id": case_id,
                "edf_path": f"/tmp/{case_id}_{index:02d}.edf",
                "sample_count": "2560",
                "seizure_count": str(seizures),
                "seizure_intervals_json": "[[1.0, 5.0]]" if seizures else "[]",
            })
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PatientSpecificA1Tests(unittest.TestCase):
    def test_plans_eligible_cases_and_skips_thin_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit"
            out = root / "protocol"
            audit.mkdir()
            _write_audit(audit, {
                "chb02": [1, 0, 1, 0, 1, 0],  # eligible
                "chb03": [1, 0],              # too few recordings
                "chb04": [0, 0, 0, 0],        # no seizures
            })
            cohort = create_patient_specific_split_plans(audit, out, [0.6, 0.2, 0.2])
            self.assertIn("chb02", cohort["eligible_cases"])
            self.assertTrue((out / "chb02" / "recording_split_manifest.csv").is_file())
            skipped_ids = {item["case_id"] for item in cohort["skipped_cases"]}
            self.assertIn("chb03", skipped_ids)
            self.assertIn("chb04", skipped_ids)


if __name__ == "__main__":
    unittest.main()
