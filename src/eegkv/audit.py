"""G1 read-only CHB-MIT inventory, annotation and channel audit."""

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
CASE_IDS = tuple(f"chb{number:02d}" for number in range(1, 25))
MANIFEST_FIELDS = ("dataset", "dataset_version", "subject_id", "case_id", "session_id", "recording_id", "relative_path", "sampling_rate_hz", "sampling_rates_hz", "duration_s", "num_samples", "num_samples_by_channel", "original_channel_count", "original_channel_labels", "physical_dimensions", "seizure_intervals_s", "split_group", "file_size_bytes", "file_sha256", "source_manifest_entry", "annotation_source", "recording_boundary")


class AuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_root(root: Path) -> None:
    if not root.is_dir():
        raise AuditError(f"CHBMIT_RAW_DIR is not a directory: {root}")
    missing = [name for name in REQUIRED_ROOT_FILES if not (root / name).is_file()] + [case for case in CASE_IDS if not (root / case).is_dir()]
    if missing:
        raise AuditError("Incomplete CHB-MIT snapshot; missing " + ", ".join(missing))


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="strict").splitlines() if line.strip()]


def _identity(relative_path: str) -> tuple[str, str, str, str, str]:
    parts = Path(relative_path).as_posix().split("/")
    if len(parts) != 2 or parts[0] not in CASE_IDS or not parts[1].lower().endswith(".edf"):
        raise AuditError(f"Invalid RECORDS identity: {relative_path}")
    case_id = parts[0]
    subject_id = "subject_01_21" if case_id in {"chb01", "chb21"} else f"subject_{case_id[3:]}"
    return subject_id, case_id, case_id, f"{case_id}/{Path(parts[1]).stem}", subject_id


def _edf_header(path: Path) -> dict[str, Any]:
    """Read EDF metadata with pyedflib only; never load samples."""
    try:
        import pyedflib
    except ImportError as error:  # pragma: no cover
        raise AuditError("pyedflib is required by the declared project dependency contract") from error
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = list(reader.getSignalLabels()); count = reader.signals_in_file
        rates = [float(reader.getSampleFrequency(index)) for index in range(count)]
        samples = [int(value) for value in reader.getNSamples()]
        dimensions = [str(reader.getPhysicalDimension(index)) for index in range(count)]
        duration = float(reader.getFileDuration())
        uniform_rate = rates[0] if rates and all(rate == rates[0] for rate in rates) else None
        uniform_samples = samples[0] if samples and all(value == samples[0] for value in samples) else None
        return {"sampling_rate_hz": uniform_rate, "sampling_rates_hz": rates, "duration_s": duration, "num_samples": uniform_samples, "num_samples_by_channel": samples, "original_channel_count": count, "original_channel_labels": labels, "physical_dimensions": dimensions}
    finally:
        reader.close()


def _parse_summary(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}; current = None; starts: list[float] = []; ends: list[float] = []; declared = None
    def save() -> None:
        if current is not None:
            if len(starts) != len(ends): raise AuditError(f"Unpaired seizure boundaries in {path}: {current}")
            result[current.lower()] = {"declared_count": declared, "intervals": [[start, end] for start, end in zip(starts, ends)]}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip(); match = re.match(r"File Name:\s*(\S+\.edf)\s*$", line, re.I)
        if match: save(); current, starts, ends, declared = match.group(1), [], [], None; continue
        match = re.match(r"Number of Seizures in File:\s*(\d+)\s*$", line, re.I)
        if match and current: declared = int(match.group(1)); continue
        match = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", line, re.I)
        if match and current: starts.append(float(match.group(1))); continue
        match = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", line, re.I)
        if match and current: ends.append(float(match.group(1)))
    save(); return result


def _all_summary_annotations(root: Path) -> dict[str, dict[str, Any]]:
    annotations = {}
    for case in CASE_IDS:
        paths = list((root / case).glob(f"{case}-summary.txt"))
        if len(paths) != 1: raise AuditError(f"Expected one summary file for {case}, found {len(paths)}")
        annotations.update({f"{case}/{name}": value for name, value in _parse_summary(paths[0]).items()})
    return annotations


def _machine_intervals(edf_path: Path, sampling_rate: float | None) -> list[list[float]] | None:
    candidates = [(Path(str(edf_path) + ".seizures"), "seizures"), (edf_path.with_suffix(".seizure"), "seizure")]
    found = [(path, extension) for path, extension in candidates if path.is_file()]
    if not found: return None
    if sampling_rate is None: raise AuditError(f"Machine annotation requires a uniform EDF sampling rate: {edf_path}")
    try:
        import wfdb
    except ImportError as error:  # pragma: no cover
        raise AuditError("wfdb is required because machine-readable seizure annotations exist") from error
    annotation = wfdb.rdann(str(edf_path), found[0][1]); starts: list[int] = []; intervals = []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol == "[": starts.append(int(sample))
        elif symbol == "]":
            if not starts: raise AuditError(f"Unpaired machine annotation end: {found[0][0]}")
            onset, offset = starts.pop(0), int(sample)
            if offset <= onset: raise AuditError(f"Non-positive machine annotation: {found[0][0]}")
            intervals.append([onset / sampling_rate, offset / sampling_rate])
    if starts: raise AuditError(f"Unpaired machine annotation start: {found[0][0]}")
    return intervals


def _same_intervals(first: list[list[float]], second: list[list[float]], tolerance: float) -> bool:
    return len(first) == len(second) and all(abs(a - c) <= tolerance and abs(b - d) <= tolerance for (a, b), (c, d) in zip(first, second))


def _label_candidate(label: str) -> str:
    return re.sub(r"\s+", "", label.upper().replace("EEG", "").replace("-REF", "").strip())


def _channel_kind(label: str) -> str:
    value = label.upper()
    if "ECG" in value or "EKG" in value: return "ECG"
    if "VNS" in value: return "VNS"
    if any(token in value for token in ("DUMMY", "PHOTIC", "MARKER", "ANNOT", "EVENT", "STATUS")): return "dummy_or_placeholder"
    if value.startswith("EEG") or re.search(r"\b(FP|F|C|P|O|T|A)[0-9Z]", value): return "likely_EEG"
    return "unknown"


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repository), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def _checksum_entries(path: Path) -> dict[str, str]:
    entries = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if not raw.strip(): continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", raw)
        if not match: raise AuditError(f"Malformed SHA256SUMS entry at line {line_no}")
        if match.group(2) in entries: raise AuditError(f"Duplicate SHA256SUMS path: {match.group(2)}")
        entries[match.group(2)] = match.group(1).lower()
    return entries


def _verify_checksums(root: Path, entries: dict[str, str]) -> dict[str, Any]:
    failures = []
    for relative, expected in entries.items():
        target = root / relative
        if not target.is_file(): failures.append({"path": relative, "reason": "missing"})
        elif _sha256(target) != expected: failures.append({"path": relative, "reason": "sha256_mismatch"})
    return {"status": "passed" if not failures else "failed", "entry_count": len(entries), "failures": failures}


def run_g1_audit(output_root: Path, *, replace: bool = False, verify_checksums: bool = True) -> dict[str, Any]:
    raw_value = os.environ.get("CHBMIT_RAW_DIR")
    if not raw_value: raise AuditError("CHBMIT_RAW_DIR is required; no fallback dataset path is permitted")
    root = Path(raw_value); _require_root(root)
    repository = Path(__file__).resolve().parents[2]
    config = json.loads((repository / "configs/chbmit_g1_audit_v1.json").read_text(encoding="utf-8"))
    paths = {"manifest_csv": output_root / "manifests/chbmit_recordings.csv", "manifest_jsonl": output_root / "manifests/chbmit_recordings.jsonl", "manifest_parquet": output_root / "manifests/chbmit_recordings.parquet", "audit": output_root / "reports/data_audit.json", "provenance": output_root / "reports/provenance.json", "census": output_root / "reports/channel_census.csv", "patterns": output_root / "reports/channel_patterns.json", "anomalies": output_root / "reports/anomalies.json"}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not replace: raise AuditError("G1 output already exists; refuse overwrite without --replace: " + ", ".join(existing))
    for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
    records = _lines(root / "RECORDS"); seizure_records = _lines(root / "RECORDS-WITH-SEIZURES")
    if len(records) != len(set(records)): raise AuditError("RECORDS contains duplicate entries")
    physical = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".edf"}
    anomalies: dict[str, Any] = {"missing_edfs": sorted(set(records) - physical), "unexpected_edfs": sorted(physical - set(records)), "records_with_seizures_outside_records": sorted(set(seizure_records) - set(records)), "unreadable_edfs": [], "duplicate_recording_ids": [], "duplicate_file_sha256": [], "annotation_discrepancies": [], "invalid_intervals": [], "duplicate_labels_within_recordings": []}
    summaries, entries = _all_summary_annotations(root), _checksum_entries(root / "SHA256SUMS.txt")
    checksum = _verify_checksums(root, entries) if verify_checksums else {"status": "skipped", "entry_count": len(entries), "failures": []}
    manifest: list[dict[str, Any]] = []; digest_records: defaultdict[str, list[str]] = defaultdict(list); parsed_positive = set(); event_count = 0; total_duration = 0.0
    census: dict[str, dict[str, Any]] = {}; patterns: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {"labels": [], "recordings": [], "cases": set()})
    for entry in records:
        subject_id, case_id, session_id, recording_id, split_group = _identity(entry); path = root / entry
        try: header = _edf_header(path)
        except Exception as error:
            anomalies["unreadable_edfs"].append({"relative_path": entry, "error": f"{type(error).__name__}: {error}"}); continue
        summary = summaries.get(entry.lower())
        if summary is None: summary = {"declared_count": None, "intervals": []}; anomalies["annotation_discrepancies"].append({"relative_path": entry, "reason": "summary_record_missing"})
        intervals = summary["intervals"]
        if summary["declared_count"] is None or summary["declared_count"] != len(intervals): anomalies["annotation_discrepancies"].append({"relative_path": entry, "reason": "summary_declared_count_mismatch", "declared_count": summary["declared_count"], "parsed_count": len(intervals)})
        for onset, offset in intervals:
            if onset < 0 or offset <= onset or offset > header["duration_s"] + config["annotation_tolerance_seconds"]: anomalies["invalid_intervals"].append({"relative_path": entry, "onset_s": onset, "offset_s": offset, "duration_s": header["duration_s"]})
        machine = _machine_intervals(path, header["sampling_rate_hz"])
        if machine is not None and not _same_intervals(intervals, machine, config["annotation_tolerance_seconds"]): anomalies["annotation_discrepancies"].append({"relative_path": entry, "reason": "summary_machine_annotation_mismatch", "summary": intervals, "machine": machine})
        if bool(intervals) != (entry in seizure_records): anomalies["annotation_discrepancies"].append({"relative_path": entry, "reason": "records_with_seizures_mismatch", "listed": entry in seizure_records, "parsed_interval_count": len(intervals)})
        digest = _sha256(path); digest_records[digest].append(recording_id)
        if intervals: parsed_positive.add(entry); event_count += len(intervals)
        total_duration += header["duration_s"]
        labels = header["original_channel_labels"]; fingerprint = hashlib.sha256(json.dumps(labels, ensure_ascii=False).encode()).hexdigest(); patterns[fingerprint]["labels"] = labels; patterns[fingerprint]["recordings"].append(recording_id); patterns[fingerprint]["cases"].add(case_id)
        duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
        if duplicates: anomalies["duplicate_labels_within_recordings"].append({"recording_id": recording_id, "labels": duplicates})
        for label in set(labels):
            item = census.setdefault(label, {"original_label": label, "normalized_spelling_candidate": _label_candidate(label), "likely_kind": _channel_kind(label), "recordings": set(), "cases": set()}); item["recordings"].add(recording_id); item["cases"].add(case_id)
        manifest.append({"dataset": config["dataset"], "dataset_version": config["dataset_version"], "subject_id": subject_id, "case_id": case_id, "session_id": session_id, "recording_id": recording_id, "relative_path": entry, **header, "seizure_intervals_s": intervals, "split_group": split_group, "file_size_bytes": path.stat().st_size, "file_sha256": digest, "source_manifest_entry": entry, "annotation_source": "summary_and_machine_verified" if machine is not None else "summary", "recording_boundary": "independent_edf_no_cross_recording_continuity"})
    anomalies["duplicate_recording_ids"] = sorted(value for value, count in Counter(row["recording_id"] for row in manifest).items() if count > 1)
    allowed = set(config["documented_duplicate_sha256_exceptions"]); anomalies["duplicate_file_sha256"] = [{"sha256": key, "recording_ids": value} for key, value in digest_records.items() if len(value) > 1 and key not in allowed]
    manifest.sort(key=lambda row: row["relative_path"])
    csv_rows = [{key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], list) else row[key] for key in MANIFEST_FIELDS} for row in manifest]
    _write_csv(paths["manifest_csv"], csv_rows, MANIFEST_FIELDS)
    with paths["manifest_jsonl"].open("w", encoding="utf-8") as output:
        for row in manifest: output.write(json.dumps(row, sort_keys=True) + "\n")
    parquet_status = "not_written_optional_engine_unavailable"
    try:
        import pandas as pd
        pd.DataFrame(csv_rows).to_parquet(paths["manifest_parquet"], index=False); parquet_status = "written"
    except (ImportError, ValueError): pass
    census_rows = [{"original_label": item["original_label"], "normalized_spelling_candidate": item["normalized_spelling_candidate"], "likely_kind": item["likely_kind"], "recording_count": len(item["recordings"]), "case_count": len(item["cases"])} for item in census.values()]
    _write_csv(paths["census"], sorted(census_rows, key=lambda row: (-row["recording_count"], row["original_label"])), ("original_label", "normalized_spelling_candidate", "likely_kind", "recording_count", "case_count"))
    _write_json(paths["patterns"], [{"fingerprint": key, "recording_count": len(value["recordings"]), "case_count": len(value["cases"]), "recording_ids": sorted(value["recordings"]), "case_ids": sorted(value["cases"]), "original_channel_labels": value["labels"]} for key, value in sorted(patterns.items())]); _write_json(paths["anomalies"], anomalies)
    versions = {}
    for package in ("pyedflib", "wfdb", "pandas", "pyarrow", "fastparquet"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = "not_installed"
    provenance = {"git_commit": _git(repository, "rev-parse", "HEAD"), "git_branch": _git(repository, "branch", "--show-current"), "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(), "python_version": sys.version, "platform": platform.platform(), "package_versions": versions, "chbmit_raw_dir": str(root), "records_sha256": _sha256(root / "RECORDS"), "records_with_seizures_sha256": _sha256(root / "RECORDS-WITH-SEIZURES"), "sha256sums_sha256": _sha256(root / "SHA256SUMS.txt"), "checksum_verification": checksum, "known_metadata_discrepancy": config["known_metadata_discrepancy"]}
    _write_json(paths["provenance"], provenance)
    hard_keys = ("missing_edfs", "unexpected_edfs", "records_with_seizures_outside_records", "unreadable_edfs", "duplicate_recording_ids", "duplicate_file_sha256", "annotation_discrepancies", "invalid_intervals")
    failed = checksum["status"] == "failed" or any(anomalies[key] for key in hard_keys)
    report = {"audit_status": "failed" if failed else "passed", "records_count": len(records), "physical_edf_count": len(physical), "records_with_seizures_count": len(seizure_records), "parsed_seizure_containing_record_count": len(parsed_positive), "parsed_seizure_event_count": event_count, "case_directory_count": len(CASE_IDS), "biological_subject_group_count": len({row["subject_id"] for row in manifest}), "total_recording_duration_s": total_duration, "sample_rate_summary_hz": dict(Counter(str(row["sampling_rate_hz"]) for row in manifest)), "channel_pattern_count": len(patterns), "parquet_status": parquet_status, "known_metadata_discrepancy": config["known_metadata_discrepancy"], "anomalies": anomalies, "created_files": [str(path) for path in paths.values() if path.exists()], "tracked_files_modified_before_run": _git(repository, "status", "--porcelain")}
    _write_json(paths["audit"], report)
    if failed: raise AuditError(f"G1 audit failed; inspect {paths['audit']}")
    return report
