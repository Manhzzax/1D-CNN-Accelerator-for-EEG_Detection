#!/usr/bin/env python3
"""Run the Q1 G1 read-only CHB-MIT audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research_q1.src.g1_audit import AuditFailure, run_audit


def print_summary(summary: dict) -> None:
    print("G1-AUDIT")
    for key, value in summary.items():
        print(f"{key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1 G1 read-only CHB-MIT audit")
    parser.add_argument("--replace", action="store_true", help="replace existing small G1 output artifacts")
    parser.add_argument("--skip-checksum-verification", action="store_true", help="record skipped checksum verification; intended only for local fixture debugging")
    args = parser.parse_args()
    raw_root = os.environ.get("CHBMIT_RAW_DIR")
    if not raw_root:
        raise SystemExit("CHBMIT_RAW_DIR is required; G1 never uses a fallback dataset path.")
    try:
        summary = run_audit(Path(raw_root), PROJECT_ROOT / "research_q1", replace=args.replace, verify_checksums=not args.skip_checksum_verification)
    except AuditFailure as error:
        report_path = PROJECT_ROOT / "research_q1/reports/data_audit.json"
        summary = {"audit_status": "failed", "error": str(error), "report": str(report_path), "tests": os.environ.get("G1_TEST_STATUS", "not_run")}
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary.update({key: report.get(key) for key in ("records_count", "physical_edf_count", "records_with_seizures_count", "parsed_seizure_containing_record_count", "parsed_seizure_event_count", "case_directory_count", "biological_subject_group_count", "total_recording_duration_s", "sample_rate_summary_hz", "channel_pattern_count", "anomalies", "created_files", "tracked_files_modified_before_run")})
        print_summary(summary)
        raise SystemExit(1)
    summary["tests"] = os.environ.get("G1_TEST_STATUS", "not_run")
    print_summary(summary)


if __name__ == "__main__":
    main()
