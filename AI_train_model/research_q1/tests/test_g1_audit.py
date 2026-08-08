import tempfile
import unittest
from pathlib import Path

from research_q1.src.g1_audit import (
    AuditFailure, CASE_IDS, channel_kind, intervals_equal, normalize_label_candidate,
    parse_summary, require_dataset_root, subject_identity,
)


class G1AuditTests(unittest.TestCase):
    def test_chb01_and_chb21_share_biological_identity(self):
        self.assertEqual(subject_identity("chb01"), ("subject_01_21", "subject_01_21"))
        self.assertEqual(subject_identity("chb21"), ("subject_01_21", "subject_01_21"))
        self.assertNotEqual(subject_identity("chb02"), subject_identity("chb01"))

    def test_root_validation_requires_snapshot_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(AuditFailure):
                require_dataset_root(root)
            for name in ("RECORDS", "RECORDS-WITH-SEIZURES", "SHA256SUMS.txt", "SUBJECT-INFO", "ANNOTATORS"):
                (root / name).write_text("x\n", encoding="utf-8")
            for case in CASE_IDS: (root / case).mkdir()
            require_dataset_root(root)

    def test_summary_parser_preserves_intervals_and_rejects_unpaired_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chb01-summary.txt"
            path.write_text("File Name: chb01_03.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 12.5 seconds\nSeizure End Time: 20.0 seconds\n", encoding="utf-8")
            self.assertEqual(parse_summary(path)["chb01_03.edf"]["intervals"], [[12.5, 20.0]])
            path.write_text("File Name: chb01_03.edf\nSeizure Start Time: 12 seconds\n", encoding="utf-8")
            with self.assertRaises(AuditFailure): parse_summary(path)

    def test_channel_census_candidates_do_not_select_a_montage(self):
        self.assertEqual(normalize_label_candidate("EEG Fp1-REF"), "FP1")
        self.assertEqual(channel_kind("ECG"), "ECG")
        self.assertEqual(channel_kind("VNS"), "VNS")
        self.assertEqual(channel_kind("EEG Fp1-REF"), "likely_EEG")

    def test_annotation_comparison_has_documented_tolerance(self):
        self.assertTrue(intervals_equal([[1.0, 2.0]], [[1.0005, 2.0005]], 0.001))
        self.assertFalse(intervals_equal([[1.0, 2.0]], [[1.01, 2.0]], 0.001))

