"""Read-only inventory, annotation and channel audit for the Q1 track."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_ROOT_FILES = ("RECORDS", "RECORDS-WITH-SEIZURES", "SHA256SUMS.txt", "SUBJECT-INFO", "ANNOTATORS")
CASE_IDS = tuple(f"chb{index:02d}" for index in range(1, 25))
MANIFEST_COLUMNS = (
    "dataset", "dataset_version", "subject_id", "case_id", "session_id", "recording_id", "relative_path",
    "sampling_rate_hz", "sampling_rates_hz", "duration_s", "num_samples", "num_samples_by_channel",
    "original_channel_count", "original_channel_labels", "physical_dimensions", "seizure_intervals_s",
    "split_group", "file_size_bytes", "file_sha256", "source_manifest_entry", "annotation_source", "recording_boundary",
)


class AuditFailure(RuntimeError):
    """An invariant of the source snapshot did not hold."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_dataset_root(root: Path) -> None:
    if not root.is_dir():
        raise AuditFailure(f"CHBMIT_RAW_DIR is not an accessible directory: {root}")
    missing = [name for name in REQUIRED_ROOT_FILES if not (root / name).is_file()]
    missing += [case for case in CASE_IDS if not (root / case).is_dir()]
    if missing:
        raise AuditFailure("CHB-MIT snapshot is incomplete; missing: " + ", ".join(missing))


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="strict").splitlines() if line.strip()]


def subject_identity(case_id: str) -> tuple[str, str]:
    if case_id in {"chb01", "chb21"}:
        return "subject_01_21", "subject_01_21"
    if case_id not in CASE_IDS:
        raise AuditFailure(f"Invalid case identifier: {case_id}")
    return f"subject_{case_id[3:]}", f"subject_{case_id[3:]}"


def parse_edf_header(path: Path) -> dict[str, Any]:
    """Read fixed EDF header bytes only; no waveform data is read or modified."""
    def field(data: bytes) -> str:
        return data.decode("latin-1", errors="replace").strip()
    with path.open("rb") as source:
        fixed = source.read(256)
        if len(fixed) != 256:
            raise AuditFailure(f"Truncated EDF fixed header: {path}")
        try:
            header_bytes = int(field(fixed[184:192]))
            number_records = int(field(fixed[236:244]))
            record_duration = float(field(fixed[244:252]))
            signals = int(field(fixed[252:256]))
        except ValueError as error:
            raise AuditFailure(f"Invalid EDF fixed header fields: {path}") from error
        if signals <= 0 or record_duration <= 0 or header_bytes < 256 + signals * 256:
            raise AuditFailure(f"Invalid EDF signal/header layout: {path}")
        remainder = source.read(header_bytes - 256)
    if len(remainder) != header_bytes - 256:
        raise AuditFailure(f"Truncated EDF signal header: {path}")
    cursor = 0
    def fields(width: int) -> list[str]:
        nonlocal cursor
        end = cursor + signals * width
        values = [field(remainder[index:index + width]) for index in range(cursor, end, width)]
        cursor = end
        return values
    labels = fields(16); fields(80); dimensions = fields(8); fields(8); fields(8); fields(8); fields(8); fields(80)
    samples_per_record = [int(value or "0") for value in fields(8)]
    fields(32)
    rates = [value / record_duration for value in samples_per_record]
    sample_counts = [value * number_records if number_records >= 0 else None for value in samples_per_record]
    duration = number_records * record_duration if number_records >= 0 else None
    uniform_rate = rates[0] if rates and all(value == rates[0] for value in rates) else None
    uniform_count = sample_counts[0] if sample_counts and all(value == sample_counts[0] for value in sample_counts) else None
    return {
        "sampling_rate_hz": uniform_rate, "sampling_rates_hz": rates, "duration_s": duration,
        "num_samples": uniform_count, "num_samples_by_channel": sample_counts,
        "original_channel_count": signals, "original_channel_labels": labels, "physical_dimensions": dimensions,
        "number_data_records": number_records, "data_record_duration_s": record_duration,
    }


def parse_summary(path: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    current: str | None = None; starts: list[float] = []; ends: list[float] = []; declared: int | None = None
    def save() -> None:
        if current is not None:
            if len(starts) != len(ends):
                raise AuditFailure(f"Unpaired seizure boundaries in {path} for {current}")
            annotations[current.lower()] = {"intervals": [[start, end] for start, end in zip(starts, ends)], "declared_count": declared}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = re.match(r"File Name:\s*(\S+\.edf)\s*$", line, re.IGNORECASE)
        if match:
            save(); current = match.group(1); starts, ends, declared = [], [], None; continue
        match = re.match(r"Number of Seizures in File:\s*(\d+)\s*$", line, re.IGNORECASE)
        if match and current:
            declared = int(match.group(1)); continue
        match = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", line, re.IGNORECASE)
        if match and current:
            starts.append(float(match.group(1))); continue
        match = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", line, re.IGNORECASE)
        if match and current:
            ends.append(float(match.group(1)))
    save()
    return annotations


def load_summary_annotations(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for case in CASE_IDS:
        summaries = sorted((root / case).glob(f"{case}-summary.txt"))
        if len(summaries) != 1:
            raise AuditFailure(f"Expected exactly one summary file for {case}, found {len(summaries)}")
        for filename, details in parse_summary(summaries[0]).items():
            result[f"{case}/{filename}"] = details
    return result


def parse_machine_annotations(edf_path: Path, rate: float | None) -> list[list[float]] | None:
    candidates = ((Path(str(edf_path) + ".seizures"), "seizures"), (edf_path.with_suffix(".seizure"), "seizure"))
    available = [(path, extension) for path, extension in candidates if path.is_file()]
    if not available:
        return None
    if rate is None:
        raise AuditFailure(f"Cannot read annotation samples with mixed EDF rates: {edf_path}")
    try:
        import wfdb
    except ImportError as error:
        raise AuditFailure("wfdb is required because machine-readable seizure annotations are present") from error
    annotation_path, extension = available[0]
    annotation = wfdb.rdann(str(edf_path), extension)
    starts: list[int] = []; intervals: list[list[float]] = []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol == "[": starts.append(int(sample))
        elif symbol == "]":
            if not starts: raise AuditFailure(f"Unpaired seizure end marker: {annotation_path}")
            start = starts.pop(0); end = int(sample)
            if end <= start: raise AuditFailure(f"Non-positive machine annotation: {annotation_path}")
            intervals.append([start / rate, end / rate])
    if starts: raise AuditFailure(f"Unpaired seizure start marker: {annotation_path}")
    return intervals


def normalize_label_candidate(label: str) -> str:
    return re.sub(r"\s+", "", label.upper().replace("EEG", "").replace("-REF", "").strip())


def channel_kind(label: str) -> str:
    value = label.upper()
    if "ECG" in value or "EKG" in value: return "ECG"
    if "VNS" in value: return "VNS"
    if any(token in value for token in ("DUMMY", "PHOTIC", "MARKER", "ANNOT", "EVENT", "STATUS")): return "dummy_or_placeholder"
    if value.startswith("EEG") or re.search(r"\b(FP|F|C|P|O|T|A)[0-9Z]", value): return "likely_EEG"
    return "unknown"


def intervals_equal(left: list[list[float]], right: list[list[float]], tolerance: float) -> bool:
    return len(left) == len(right) and all(abs(a - c) <= tolerance and abs(b - d) <= tolerance for (a, b), (c, d) in zip(left, right))


def git_value(repository: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repository), *args], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def package_versions() -> dict[str, str]:
    result = {}
    for package in ("wfdb", "pandas", "pyarrow", "fastparquet"):
        try: result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result[package] = "not_installed"
    return result


def parse_checksum_entries(path: Path) -> dict[str, str]:
    entries = {}
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if not line.strip(): continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line)
        if not match: raise AuditFailure(f"Malformed SHA256SUMS entry at line {number}")
        digest, relative = match.group(1).lower(), match.group(2)
        if relative in entries: raise AuditFailure(f"Duplicate SHA256SUMS path: {relative}")
        entries[relative] = digest
    return entries


def verify_checksums(root: Path, entries: dict[str, str]) -> dict[str, Any]:
    failures = []
    for relative, expected in entries.items():
        target = root / relative
        if not target.is_file(): failures.append({"path": relative, "reason": "missing"})
        elif file_sha256(target) != expected: failures.append({"path": relative, "reason": "sha256_mismatch"})
    return {"status": "passed" if not failures else "failed", "entry_count": len(entries), "failures": failures}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_output_paths(q1_root: Path, replace: bool) -> dict[str, Path]:
    paths = {
        "csv": q1_root / "manifests/chbmit_recordings.csv", "jsonl": q1_root / "manifests/chbmit_recordings.jsonl",
        "parquet": q1_root / "manifests/chbmit_recordings.parquet", "audit": q1_root / "reports/data_audit.json",
        "provenance": q1_root / "reports/provenance.json", "census": q1_root / "reports/channel_census.csv",
        "patterns": q1_root / "reports/channel_patterns.json", "anomalies": q1_root / "reports/anomalies.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not replace: raise AuditFailure("G1 outputs already exist; use --replace only after preserving prior audit artifacts: " + ", ".join(existing))
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def run_audit(root: Path, q1_root: Path, *, replace: bool, verify_checksums: bool) -> dict[str, Any]:
    require_dataset_root(root)
    config = json.loads((q1_root / "configs/g1_audit_v1.json").read_text(encoding="utf-8"))
    outputs = validate_output_paths(q1_root, replace)
    records = read_lines(root / "RECORDS"); seizure_records = read_lines(root / "RECORDS-WITH-SEIZURES")
    duplicate_manifest_paths = sorted(path for path, count in Counter(records).items() if count > 1)
    if duplicate_manifest_paths: raise AuditFailure("Duplicate RECORDS entries: " + ", ".join(duplicate_manifest_paths))
    record_set = set(records); seizure_set = set(seizure_records)
    physical_edfs = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".edf"}
    missing_edfs = sorted(path for path in record_set if not (root / path).is_file())
    unexpected_edfs = sorted(physical_edfs - record_set)
    outside_records_with_seizures = sorted(seizure_set - record_set)
    summaries = load_summary_annotations(root)
    checksum_entries = parse_checksum_entries(root / "SHA256SUMS.txt")
    checksum_report = verify_checksums(root, checksum_entries) if verify_checksums else {"status": "skipped", "entry_count": len(checksum_entries), "failures": []}
    anomalies: dict[str, Any] = {"missing_edfs": missing_edfs, "unexpected_edfs": unexpected_edfs, "records_with_seizures_outside_records": outside_records_with_seizures, "unreadable_edfs": [], "duplicate_recording_ids": [], "duplicate_file_sha256": [], "annotation_discrepancies": [], "invalid_intervals": []}
    manifest: list[dict[str, Any]] = []; checksum_to_recordings: defaultdict[str, list[str]] = defaultdict(list)
    census: dict[str, dict[str, Any]] = {}; patterns: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {"recordings": [], "cases": set(), "labels": []})
    parsed_seizure_records: set[str] = set(); event_count = 0; total_duration = 0.0
    for entry in records:
        relative = Path(entry).as_posix(); parts = relative.split("/")
        if len(parts) != 2 or parts[0] not in CASE_IDS or not relative.lower().endswith(".edf"):
            raise AuditFailure(f"RECORDS entry has invalid CHB-MIT identity: {entry}")
        case_id, filename = parts; recording_id = f"{case_id}/{Path(filename).stem}"; subject_id, split_group = subject_identity(case_id)
        edf_path = root / relative
        try: header = parse_edf_header(edf_path)
        except Exception as error:
            anomalies["unreadable_edfs"].append({"relative_path": relative, "error": f"{type(error).__name__}: {error}"}); continue
        duration = header["duration_s"]
        summary = summaries.get(relative.lower())
        if summary is None:
            anomalies["annotation_discrepancies"].append({"relative_path": relative, "reason": "summary_record_missing"}); summary = {"intervals": [], "declared_count": None}
        intervals = summary["intervals"]
        if summary["declared_count"] is None or summary["declared_count"] != len(intervals):
            anomalies["annotation_discrepancies"].append({"relative_path": relative, "reason": "summary_declared_count_mismatch", "declared_count": summary["declared_count"], "parsed_count": len(intervals)})
        for onset, offset in intervals:
            if onset < 0 or offset <= onset or duration is None or offset > duration + config["annotation_tolerance_seconds"]:
                anomalies["invalid_intervals"].append({"relative_path": relative, "onset_s": onset, "offset_s": offset, "duration_s": duration})
        machine = parse_machine_annotations(edf_path, header["sampling_rate_hz"])
        if machine is not None and not intervals_equal(intervals, machine, config["annotation_tolerance_seconds"]):
            anomalies["annotation_discrepancies"].append({"relative_path": relative, "reason": "summary_machine_annotation_mismatch", "summary": intervals, "machine": machine})
        annotation_source = "summary_and_machine_verified" if machine is not None else "summary"
        if bool(intervals) != (relative in seizure_set): anomalies["annotation_discrepancies"].append({"relative_path": relative, "reason": "records_with_seizures_mismatch", "listed": relative in seizure_set, "parsed_interval_count": len(intervals)})
        digest = file_sha256(edf_path); checksum_to_recordings[digest].append(recording_id)
        if duration is not None: total_duration += duration
        if intervals: parsed_seizure_records.add(relative); event_count += len(intervals)
        raw_labels = header["original_channel_labels"]; fingerprint = hashlib.sha256(json.dumps(raw_labels, ensure_ascii=False).encode()).hexdigest()
        patterns[fingerprint]["recordings"].append(recording_id); patterns[fingerprint]["cases"].add(case_id); patterns[fingerprint]["labels"] = raw_labels
        duplicates = sorted(label for label, count in Counter(raw_labels).items() if count > 1)
        if duplicates: anomalies.setdefault("duplicate_labels_within_recordings", []).append({"recording_id": recording_id, "labels": duplicates})
        for label in set(raw_labels):
            item = census.setdefault(label, {"original_label": label, "normalized_spelling_candidate": normalize_label_candidate(label), "likely_kind": channel_kind(label), "recording_ids": set(), "case_ids": set()})
            item["recording_ids"].add(recording_id); item["case_ids"].add(case_id)
        manifest.append({"dataset": config["dataset"], "dataset_version": config["dataset_version"], "subject_id": subject_id, "case_id": case_id, "session_id": case_id, "recording_id": recording_id, "relative_path": relative, "sampling_rate_hz": header["sampling_rate_hz"], "sampling_rates_hz": header["sampling_rates_hz"], "duration_s": duration, "num_samples": header["num_samples"], "num_samples_by_channel": header["num_samples_by_channel"], "original_channel_count": header["original_channel_count"], "original_channel_labels": raw_labels, "physical_dimensions": header["physical_dimensions"], "seizure_intervals_s": intervals, "split_group": split_group, "file_size_bytes": edf_path.stat().st_size, "file_sha256": digest, "source_manifest_entry": entry, "annotation_source": annotation_source, "recording_boundary": "independent_edf_no_cross_recording_continuity"})
    ids = [row["recording_id"] for row in manifest]
    anomalies["duplicate_recording_ids"] = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    allowed_duplicates = set(config["documented_duplicate_sha256_exceptions"])
    anomalies["duplicate_file_sha256"] = [{"sha256": digest, "recording_ids": values} for digest, values in checksum_to_recordings.items() if len(values) > 1 and digest not in allowed_duplicates]
    required_failures = any(anomalies[key] for key in ("missing_edfs", "unexpected_edfs", "records_with_seizures_outside_records", "unreadable_edfs", "duplicate_recording_ids", "duplicate_file_sha256", "annotation_discrepancies", "invalid_intervals")) or checksum_report["status"] == "failed"
    manifest.sort(key=lambda row: row["relative_path"])
    csv_rows = [{field: json.dumps(row[field], ensure_ascii=False) if isinstance(row[field], list) else row[field] for field in MANIFEST_COLUMNS} for row in manifest]
    write_csv(outputs["csv"], csv_rows, MANIFEST_COLUMNS)
    with outputs["jsonl"].open("w", encoding="utf-8") as destination:
        for row in manifest: destination.write(json.dumps(row, sort_keys=True) + "\n")
    parquet_status = "not_attempted"
    try:
        import pandas as pd
        pd.DataFrame(csv_rows).to_parquet(outputs["parquet"], index=False); parquet_status = "written"
    except (ImportError, ValueError): parquet_status = "not_written_optional_engine_unavailable"
    pattern_rows = [{"fingerprint": key, "recording_count": len(value["recordings"]), "case_count": len(value["cases"]), "recording_ids": sorted(value["recordings"]), "case_ids": sorted(value["cases"]), "original_channel_labels": value["labels"]} for key, value in sorted(patterns.items())]
    census_rows = [{"original_label": value["original_label"], "normalized_spelling_candidate": value["normalized_spelling_candidate"], "likely_kind": value["likely_kind"], "recording_count": len(value["recording_ids"]), "case_count": len(value["case_ids"])} for value in census.values()]
    write_csv(outputs["census"], sorted(census_rows, key=lambda row: (-row["recording_count"], row["original_label"])), tuple(census_rows[0].keys()) if census_rows else ("original_label", "normalized_spelling_candidate", "likely_kind", "recording_count", "case_count"))
    write_json(outputs["patterns"], pattern_rows); write_json(outputs["anomalies"], anomalies)
    repository = q1_root.parents[1]
    provenance = {"git_commit": git_value(repository, "rev-parse", "HEAD"), "git_branch": git_value(repository, "branch", "--show-current"), "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(), "python_version": sys.version, "platform": platform.platform(), "package_versions": package_versions(), "chbmit_raw_dir": str(root), "records_sha256": file_sha256(root / "RECORDS"), "records_with_seizures_sha256": file_sha256(root / "RECORDS-WITH-SEIZURES"), "sha256sums_sha256": file_sha256(root / "SHA256SUMS.txt"), "checksum_verification": checksum_report, "known_metadata_discrepancy": config["known_metadata_discrepancy"]}
    write_json(outputs["provenance"], provenance)
    report = {"records_count": len(records), "physical_edf_count": len(physical_edfs), "records_with_seizures_count": len(seizure_records), "parsed_seizure_containing_record_count": len(parsed_seizure_records), "parsed_seizure_event_count": event_count, "case_directory_count": len(CASE_IDS), "biological_subject_group_count": len({row["subject_id"] for row in manifest}), "total_recording_duration_s": total_duration, "sample_rate_summary_hz": dict(Counter(str(row["sampling_rate_hz"]) for row in manifest)), "channel_pattern_count": len(pattern_rows), "parquet_status": parquet_status, "audit_status": "failed" if required_failures else "passed", "known_metadata_discrepancy": config["known_metadata_discrepancy"], "anomalies": anomalies, "created_files": [str(path.relative_to(q1_root)) for path in outputs.values() if path.exists()], "tracked_files_modified_before_run": git_value(repository, "status", "--porcelain")}
    write_json(outputs["audit"], report)
    if required_failures: raise AuditFailure(f"G1 audit failed; inspect {outputs['audit']}")
    return {"repository_path": str(repository), "branch": provenance["git_branch"], "git_commit": provenance["git_commit"], "dataset_root": str(root), "records_count": len(records), "physical_edf_count": len(physical_edfs), "records_with_seizures_count": len(seizure_records), "parsed_seizure_containing_record_count": len(parsed_seizure_records), "parsed_seizure_event_count": event_count, "case_directory_count": len(CASE_IDS), "biological_subject_group_count": len({row["subject_id"] for row in manifest}), "total_recording_duration_s": round(total_duration, 3), "sha256_verification_status": checksum_report["status"], "checksum_entry_count": checksum_report["entry_count"], "sample_rate_summary": report["sample_rate_summary_hz"], "channel_pattern_count": len(pattern_rows), "anomalies": "none", "tests": "run research_q1/tests before server audit", "created_files": report["created_files"], "tracked_files_modified": git_value(repository, "status", "--porcelain") or "none"}
