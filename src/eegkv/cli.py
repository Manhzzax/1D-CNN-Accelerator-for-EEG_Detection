"""Public Q1 command line interface; G1 is the only enabled stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import AuditError, run_g1_audit, run_g1_preflight


def _print_summary(value: dict) -> None:
    print("G1-AUDIT")
    for key in ("audit_status", "records_count", "physical_edf_count", "records_with_seizures_count", "parsed_seizure_containing_record_count", "parsed_seizure_event_count", "case_directory_count", "biological_subject_group_count", "total_recording_duration_s", "sample_rate_summary_hz", "channel_pattern_count", "parquet_status", "anomalies", "created_files", "tracked_files_modified_before_run"):
        if key in value: print(f"{key}={value[key]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eegkv", description="Q1 patient-independent EEG research")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-g1", help="run the read-only CHB-MIT G1 audit")
    audit.add_argument("--output-root", type=Path, default=Path("artifacts/g1"))
    audit.add_argument("--replace", action="store_true", help="replace existing G1 audit artifacts")
    audit.add_argument("--skip-checksum-verification", action="store_true", help="fixture-only debugging; never use for the server audit")
    commands.add_parser("preflight-g1", help="read-only SERVER-02 snapshot preflight; prints JSON and writes nothing")
    args = parser.parse_args(argv)
    if args.command == "preflight-g1":
        try:
            result = run_g1_preflight()
        except AuditError as error:
            print(json.dumps({"preflight_status": "failed", "error": str(error)}, sort_keys=True))
            raise SystemExit(1)
        print(json.dumps(result, sort_keys=True))
        if result["preflight_status"] != "passed":
            raise SystemExit(1)
        return
    try:
        summary = run_g1_audit(args.output_root, replace=args.replace, verify_checksums=not args.skip_checksum_verification)
    except AuditError as error:
        report = args.output_root / "reports/data_audit.json"
        summary = {"audit_status": "failed", "error": str(error), "report": str(report)}
        if report.is_file(): summary.update(json.loads(report.read_text(encoding="utf-8")))
        _print_summary(summary)
        raise SystemExit(1)
    _print_summary(summary)


if __name__ == "__main__":
    main()
