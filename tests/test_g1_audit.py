"""G1A tests use generated temporary fixtures, never real CHB-MIT data."""

import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from eegkv.audit import (
    AuditError, CASE_IDS, MANIFEST_SCHEMA_VERSION, _channel_kind, _identity,
    _label_candidate, _parse_summary, _require_root, _same_intervals,
    run_g1_audit, run_g1_preflight,
)
from eegkv.cli import main


def _field(value: str, width: int) -> bytes:
    return value.encode("ascii").ljust(width)[:width]


def _write_synthetic_edf(path: Path) -> None:
    """Create a minimal one-channel EDF fixture; no CHB-MIT file is used."""
    fixed = b"".join((
        _field("0", 8), _field("synthetic", 80), _field("G1A fixture", 80),
        _field("01.01.01", 8), _field("01.01.01", 8), _field("512", 8),
        _field("", 44), _field("1", 8), _field("1", 8), _field("1", 4),
    ))
    signal = b"".join((
        _field("EEG Fp1-REF", 16), _field("", 80), _field("uV", 8),
        _field("-100", 8), _field("100", 8), _field("-32768", 8),
        _field("32767", 8), _field("", 80), _field("1", 8), _field("", 32),
    ))
    path.write_bytes(fixed + signal + b"\x00\x00")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksum_manifest(root: Path, *, omit: set[str] | None = None, extra: bool = False) -> None:
    omit = omit or set()
    relevant = {"RECORDS", "RECORDS-WITH-SEIZURES", "SUBJECT-INFO", "ANNOTATORS"}
    relevant |= {f"{case}/{case}-summary.txt" for case in CASE_IDS}
    relevant |= {"chb01/chb01_01.edf", "chb01/chb01_01.edf.seizures"}
    lines = []
    for relative in sorted(relevant - omit):
        lines.append(f"{_sha256(root / relative)}  {relative}")
    if extra:
        lines.append(f"{'0' * 64}  unexpected.txt")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthetic_snapshot(root: Path) -> None:
    for name in ("SUBJECT-INFO", "ANNOTATORS"):
        (root / name).write_text("synthetic\n", encoding="utf-8")
    for case in CASE_IDS:
        directory = root / case
        directory.mkdir()
        summary = ""
        if case == "chb01":
            summary = (
                "File Name: chb01_01.edf\nNumber of Seizures in File: 1\n"
                "Seizure Start Time: 0.25 seconds\nSeizure End Time: 0.75 seconds\n"
            )
        (directory / f"{case}-summary.txt").write_text(summary, encoding="utf-8")
    (root / "RECORDS").write_text("chb01/chb01_01.edf\n", encoding="utf-8")
    (root / "RECORDS-WITH-SEIZURES").write_text("chb01/chb01_01.edf\n", encoding="utf-8")
    _write_synthetic_edf(root / "chb01/chb01_01.edf")
    (root / "chb01/chb01_01.edf.seizures").write_text("synthetic annotation\n", encoding="utf-8")
    _write_checksum_manifest(root)


class G1AuditTests(unittest.TestCase):
    def test_chb01_and_chb21_share_subject_and_split_group(self):
        self.assertEqual(_identity("chb01/chb01_03.edf")[0], "subject_01_21")
        self.assertEqual(_identity("chb21/chb21_03.edf")[0], "subject_01_21")
        self.assertEqual(_identity("chb01/chb01_03.edf")[4], _identity("chb21/chb21_03.edf")[4])

    def test_dataset_root_contract_requires_all_snapshot_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(AuditError):
                _require_root(root)
            for name in ("RECORDS", "RECORDS-WITH-SEIZURES", "SHA256SUMS.txt", "SUBJECT-INFO", "ANNOTATORS"):
                (root / name).write_text("x\n", encoding="utf-8")
            for case in CASE_IDS:
                (root / case).mkdir()
            _require_root(root)

    def test_summary_annotations_do_not_silently_repair_unpaired_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chb01-summary.txt"
            path.write_text("File Name: chb01_03.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 2.5 seconds\nSeizure End Time: 7.5 seconds\n", encoding="utf-8")
            self.assertEqual(_parse_summary(path)["chb01_03.edf"]["intervals"], [[2.5, 7.5]])
            path.write_text("File Name: chb01_03.edf\nSeizure Start Time: 2.5 seconds\n", encoding="utf-8")
            with self.assertRaises(AuditError):
                _parse_summary(path)

    def test_channel_census_preserves_raw_label_and_only_proposes_spelling(self):
        self.assertEqual(_label_candidate("EEG Fp1-REF"), "FP1")
        self.assertEqual(_channel_kind("ECG"), "ECG")
        self.assertEqual(_channel_kind("VNS"), "VNS")
        self.assertEqual(_channel_kind("EEG Fp1-REF"), "likely_EEG")

    def test_annotation_tolerance_is_explicit(self):
        self.assertTrue(_same_intervals([[1.0, 2.0]], [[1.0005, 2.0005]], 0.001))
        self.assertFalse(_same_intervals([[1.0, 2.0]], [[1.01, 2.0]], 0.001))

    def test_preflight_passes_generated_fixture_without_disclosing_its_path(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"CHBMIT_RAW_DIR": temporary}):
            _synthetic_snapshot(Path(temporary))
            result = run_g1_preflight()
            self.assertEqual(result["preflight_status"], "passed")
            self.assertEqual(result["physical_edf_count"], 1)
            self.assertNotIn(temporary, json.dumps(result))

    def test_preflight_command_emits_one_json_object_for_generated_fixture(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"CHBMIT_RAW_DIR": temporary}):
            _synthetic_snapshot(Path(temporary))
            output = io.StringIO()
            with redirect_stdout(output):
                main(["preflight-g1"])
            result = json.loads(output.getvalue())
            self.assertEqual(result["preflight_status"], "passed")

    def test_preflight_detects_duplicate_seizure_manifest_entry(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"CHBMIT_RAW_DIR": temporary}):
            root = Path(temporary); _synthetic_snapshot(root)
            (root / "RECORDS-WITH-SEIZURES").write_text("chb01/chb01_01.edf\nchb01/chb01_01.edf\n", encoding="utf-8")
            _write_checksum_manifest(root)
            result = run_g1_preflight()
            self.assertIn("chb01/chb01_01.edf", result["anomalies"]["duplicate_seizure_manifest_entries"])
            self.assertEqual(result["preflight_status"], "failed")

    def test_preflight_detects_inventory_and_machine_annotation_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"CHBMIT_RAW_DIR": temporary}):
            root = Path(temporary); _synthetic_snapshot(root)
            _write_synthetic_edf(root / "chb02/unlisted.edf")
            (root / "chb02/unlisted.edf.seizures").write_text("x", encoding="utf-8")
            result = run_g1_preflight()
            self.assertIn("chb02/unlisted.edf", result["anomalies"]["unexpected_edfs"])
            self.assertIn("chb02/unlisted.edf", result["anomalies"]["machine_annotations_outside_records_with_seizures"])

    def test_preflight_detects_missing_machine_annotation_and_checksum_coverage(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"CHBMIT_RAW_DIR": temporary}):
            root = Path(temporary); _synthetic_snapshot(root)
            (root / "chb01/chb01_01.edf.seizures").unlink()
            _write_checksum_manifest(root, omit={"chb01/chb01_01.edf", "chb01/chb01_01.edf.seizures"}, extra=True)
            result = run_g1_preflight()
            self.assertIn("chb01/chb01_01.edf", result["anomalies"]["records_with_seizures_missing_machine_annotations"])
            self.assertIn("chb01/chb01_01.edf", result["anomalies"]["checksum_coverage"]["missing_checksum_entries"])
            self.assertIn("unexpected.txt", result["anomalies"]["checksum_coverage"]["unexpected_checksum_entries"])

    def test_audit_compares_summary_machine_intervals_and_writes_portable_provenance(self):
        header = {
            "sampling_rate_hz": 4.0, "sampling_rates_hz": [4.0], "duration_s": 1.0,
            "num_samples": 4, "num_samples_by_channel": [4], "original_channel_count": 1,
            "original_channel_labels": ["EEG Fp1-REF"], "physical_dimensions": ["uV"],
        }
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as output, patch.dict(os.environ, {"CHBMIT_RAW_DIR": temporary}):
            root = Path(temporary); _synthetic_snapshot(root)
            with patch("eegkv.audit._edf_header", return_value=header), patch("eegkv.audit._machine_intervals", return_value=[[0.25, 0.75]]):
                result = run_g1_audit(Path(output))
            self.assertEqual(result["audit_status"], "passed")
            row = json.loads((Path(output) / "manifests/chbmit_recordings.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["manifest_schema_version"], MANIFEST_SCHEMA_VERSION)
            provenance = (Path(output) / "reports/provenance_shareable.json").read_text(encoding="utf-8")
            self.assertNotIn(temporary, provenance)
            self.assertEqual(json.loads(provenance)["digest_cache_computed_file_count"], 31)

    def test_audit_fails_when_summary_and_machine_intervals_disagree(self):
        header = {"sampling_rate_hz": 4.0, "sampling_rates_hz": [4.0], "duration_s": 1.0, "num_samples": 4, "num_samples_by_channel": [4], "original_channel_count": 1, "original_channel_labels": ["EEG Fp1-REF"], "physical_dimensions": ["uV"]}
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as output, patch.dict(os.environ, {"CHBMIT_RAW_DIR": temporary}):
            _synthetic_snapshot(Path(temporary))
            with patch("eegkv.audit._edf_header", return_value=header), patch("eegkv.audit._machine_intervals", return_value=[[0.0, 0.5]]):
                with self.assertRaises(AuditError):
                    run_g1_audit(Path(output))
