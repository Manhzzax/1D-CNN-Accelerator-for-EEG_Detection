import tempfile
import unittest
from pathlib import Path

from eegkv.audit import (
    AuditError, CASE_IDS, _channel_kind, _identity, _label_candidate,
    _parse_summary, _require_root, _same_intervals,
)


class G1AuditTests(unittest.TestCase):
    def test_chb01_and_chb21_share_subject_and_split_group(self):
        self.assertEqual(_identity("chb01/chb01_03.edf")[0], "subject_01_21")
        self.assertEqual(_identity("chb21/chb21_03.edf")[0], "subject_01_21")
        self.assertEqual(_identity("chb01/chb01_03.edf")[4], _identity("chb21/chb21_03.edf")[4])

    def test_dataset_root_contract_requires_all_snapshot_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(AuditError): _require_root(root)
            for name in ("RECORDS", "RECORDS-WITH-SEIZURES", "SHA256SUMS.txt", "SUBJECT-INFO", "ANNOTATORS"):
                (root / name).write_text("x\n", encoding="utf-8")
            for case in CASE_IDS: (root / case).mkdir()
            _require_root(root)

    def test_summary_annotations_do_not_silently_repair_unpaired_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chb01-summary.txt"
            path.write_text("File Name: chb01_03.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 2.5 seconds\nSeizure End Time: 7.5 seconds\n", encoding="utf-8")
            self.assertEqual(_parse_summary(path)["chb01_03.edf"]["intervals"], [[2.5, 7.5]])
            path.write_text("File Name: chb01_03.edf\nSeizure Start Time: 2.5 seconds\n", encoding="utf-8")
            with self.assertRaises(AuditError): _parse_summary(path)

    def test_channel_census_preserves_raw_label_and_only_proposes_spelling(self):
        self.assertEqual(_label_candidate("EEG Fp1-REF"), "FP1")
        self.assertEqual(_channel_kind("ECG"), "ECG")
        self.assertEqual(_channel_kind("VNS"), "VNS")
        self.assertEqual(_channel_kind("EEG Fp1-REF"), "likely_EEG")

    def test_annotation_tolerance_is_explicit(self):
        self.assertTrue(_same_intervals([[1.0, 2.0]], [[1.0005, 2.0005]], 0.001))
        self.assertFalse(_same_intervals([[1.0, 2.0]], [[1.01, 2.0]], 0.001))

