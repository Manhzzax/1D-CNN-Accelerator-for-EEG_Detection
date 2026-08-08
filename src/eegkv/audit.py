"""G1A read-only CHB-MIT snapshot audit and server handoff contracts.

This module never reads EDF signal samples.  It is intentionally limited to
metadata, manifests, annotations, and cryptographic checksums.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_ROOT_FILES = (
    "RECORDS",
    "RECORDS-WITH-SEIZURES",
    "SHA256SUMS.txt",
    "SUBJECT-INFO",
    "ANNOTATORS",
)
CASE_IDS = tuple(f"chb{number:02d}" for number in range(1, 25))
MANIFEST_SCHEMA_VERSION = "g1a-recording-manifest/v1"
MANIFEST_FIELDS = (
    "manifest_schema_version", "dataset_snapshot_id", "dataset", "dataset_version",
    "subject_id", "case_id", "session_id", "recording_id", "relative_path",
    "sampling_rate_hz", "sampling_rates_hz", "duration_s", "num_samples",
    "num_samples_by_channel", "original_channel_count", "original_channel_labels",
    "physical_dimensions", "seizure_intervals_s", "split_group", "file_size_bytes",
    "file_sha256", "source_manifest_entry", "annotation_source", "recording_boundary",
)


class AuditError(RuntimeError):
    """A dataset snapshot violates the G1A contract."""


class _DigestCache:
    """Memoize file digests so verification and manifest creation hash once."""

    def __init__(self) -> None:
        self._values: dict[Path, str] = {}
        self.computed_count = 0

    def sha256(self, path: Path) -> str:
        path = path.resolve()
        if path not in self._values:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            self._values[path] = digest.hexdigest()
            self.computed_count += 1
        return self._values[path]


def _require_root(root: Path) -> None:
    if not root.is_dir():
        raise AuditError("CHBMIT_RAW_DIR is not a directory")
    missing = [name for name in REQUIRED_ROOT_FILES if not (root / name).is_file()]
    missing += [case for case in CASE_IDS if not (root / case).is_dir()]
    if missing:
        raise AuditError("Incomplete CHB-MIT snapshot; missing " + ", ".join(missing))


def _safe_output_root(raw_root: Path, output_root: Path) -> Path:
    """Resolve paths before any output is created and keep raw data read-only."""
    resolved_raw = raw_root.resolve(strict=True)
    resolved_output = output_root.resolve(strict=False)
    try:
        resolved_output.relative_to(resolved_raw)
    except ValueError:
        return resolved_output
    raise AuditError("G1 output_root must not equal or reside inside CHBMIT_RAW_DIR")


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="strict").splitlines() if line.strip()]


def _relative(value: str) -> str:
    value = value.replace("\\", "/")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("/"):
        raise AuditError(f"Unsafe snapshot-relative path: {value}")
    return path.as_posix()


def _identity(relative_path: str) -> tuple[str, str, str, str, str]:
    parts = _relative(relative_path).split("/")
    if len(parts) != 2 or parts[0] not in CASE_IDS or not parts[1].lower().endswith(".edf"):
        raise AuditError(f"Invalid RECORDS identity: {relative_path}")
    case_id = parts[0]
    subject_id = "subject_01_21" if case_id in {"chb01", "chb21"} else f"subject_{case_id[3:]}"
    return subject_id, case_id, case_id, f"{case_id}/{Path(parts[1]).stem}", subject_id


def _edf_header(path: Path) -> dict[str, Any]:
    """Read EDF header metadata only; never read any signal sample."""
    try:
        import pyedflib
    except ImportError as error:  # pragma: no cover - declared dependency on server
        raise AuditError("pyedflib is required for EDF-header auditing") from error
    reader = pyedflib.EdfReader(str(path))
    try:
        count = reader.signals_in_file
        labels = list(reader.getSignalLabels())
        rates = [float(reader.getSampleFrequency(index)) for index in range(count)]
        samples = [int(value) for value in reader.getNSamples()]
        dimensions = [str(reader.getPhysicalDimension(index)) for index in range(count)]
        duration = float(reader.getFileDuration())
    finally:
        reader.close()
    uniform_rate = rates[0] if rates and all(rate == rates[0] for rate in rates) else None
    uniform_samples = samples[0] if samples and all(value == samples[0] for value in samples) else None
    return {
        "sampling_rate_hz": uniform_rate, "sampling_rates_hz": rates,
        "duration_s": duration, "num_samples": uniform_samples,
        "num_samples_by_channel": samples, "original_channel_count": count,
        "original_channel_labels": labels, "physical_dimensions": dimensions,
    }


def _parse_summary(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    current: str | None = None
    starts: list[float] = []
    ends: list[float] = []
    declared: int | None = None

    def save() -> None:
        if current is None:
            return
        if len(starts) != len(ends):
            raise AuditError(f"Unpaired seizure boundaries in {path.name}: {current}")
        key = current.lower()
        if key in result:
            raise AuditError(f"Duplicate summary record in {path.name}: {current}")
        result[key] = {"declared_count": declared, "intervals": [[start, end] for start, end in zip(starts, ends)]}

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = re.match(r"File Name:\s*(\S+\.edf)\s*$", line, re.I)
        if match:
            save(); current, starts, ends, declared = match.group(1), [], [], None
            continue
        match = re.match(r"Number of Seizures in File:\s*(\d+)\s*$", line, re.I)
        if match and current:
            declared = int(match.group(1)); continue
        match = re.match(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", line, re.I)
        if match and current:
            starts.append(float(match.group(1))); continue
        match = re.match(r"Seizure(?:\s+\d+)?\s+End Time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", line, re.I)
        if match and current:
            ends.append(float(match.group(1)))
    save()
    return result


def _all_summary_annotations(root: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for case in CASE_IDS:
        paths = list((root / case).glob(f"{case}-summary.txt"))
        if len(paths) != 1:
            raise AuditError(f"Expected one summary file for {case}, found {len(paths)}")
        for name, value in _parse_summary(paths[0]).items():
            key = f"{case}/{name}"
            if key in annotations:
                raise AuditError(f"Duplicate summary identity: {key}")
            annotations[key] = value
    return annotations


def _machine_annotation_paths(root: Path) -> dict[str, tuple[Path, str]]:
    found: dict[str, tuple[Path, str]] = {}
    for suffix, extension in ((".edf.seizures", "seizures"), (".edf.seizure", "seizure")):
        for path in root.rglob(f"*{suffix}"):
            relative = path.relative_to(root).as_posix()
            edf_relative = relative[:-len(f".{extension}")]
            if edf_relative in found:
                raise AuditError(f"Multiple machine annotations for {edf_relative}")
            found[edf_relative] = (path, extension)
    return found


def _machine_intervals(edf_path: Path, annotation: tuple[Path, str], sampling_rate: float | None) -> list[list[float]]:
    if sampling_rate is None:
        raise AuditError(f"Machine annotation requires a uniform EDF sampling rate: {edf_path.name}")
    try:
        import wfdb
    except ImportError as error:  # pragma: no cover - declared dependency on server
        raise AuditError("wfdb is required for machine-readable seizure annotations") from error
    _, extension = annotation
    value = wfdb.rdann(str(edf_path), extension)
    starts: list[int] = []
    intervals: list[list[float]] = []
    for sample, symbol in zip(value.sample, value.symbol):
        if symbol == "[":
            starts.append(int(sample))
        elif symbol == "]":
            if not starts:
                raise AuditError(f"Unpaired machine annotation end: {annotation[0].name}")
            onset, offset = starts.pop(0), int(sample)
            if offset <= onset:
                raise AuditError(f"Non-positive machine annotation: {annotation[0].name}")
            intervals.append([onset / sampling_rate, offset / sampling_rate])
    if starts:
        raise AuditError(f"Unpaired machine annotation start: {annotation[0].name}")
    return intervals


def _same_intervals(first: list[list[float]], second: list[list[float]], tolerance: float) -> bool:
    return len(first) == len(second) and all(
        abs(a - c) <= tolerance and abs(b - d) <= tolerance
        for (a, b), (c, d) in zip(first, second)
    )


def _label_candidate(label: str) -> str:
    return re.sub(r"\s+", "", label.upper().replace("EEG", "").replace("-REF", "").strip())


def _channel_kind(label: str) -> str:
    value = label.upper()
    if "ECG" in value or "EKG" in value:
        return "ECG"
    if "VNS" in value:
        return "VNS"
    if any(token in value for token in ("DUMMY", "PHOTIC", "MARKER", "ANNOT", "EVENT", "STATUS")):
        return "dummy_or_placeholder"
    if value.startswith("EEG") or re.search(r"\b(FP|F|C|P|O|T|A)[0-9Z]", value):
        return "likely_EEG"
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
    entries: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        if not raw.strip():
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", raw)
        if not match:
            raise AuditError(f"Malformed SHA256SUMS entry at line {line_no}")
        relative = _relative(match.group(2))
        if relative in entries:
            raise AuditError(f"Duplicate SHA256SUMS path: {relative}")
        entries[relative] = match.group(1).lower()
    return entries


def _relevant_checksum_paths(root: Path, records: list[str], machine_paths: dict[str, tuple[Path, str]]) -> set[str]:
    # Root-level source documents (for example the official PDF) are part of
    # the snapshot too. SHA256SUMS.txt cannot safely checksum itself.
    root_files = {path.name for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"}
    summaries = {f"{case}/{case}-summary.txt" for case in CASE_IDS}
    machine = {path.relative_to(root).as_posix() for path, _ in machine_paths.values()}
    return root_files | set(records) | summaries | machine


def _checksum_coverage(root: Path, entries: dict[str, str], required: set[str]) -> dict[str, list[str]]:
    # SHA256SUMS.txt cannot safely checksum itself; all other G1A-relevant files must be covered.
    return {
        "missing_checksum_entries": sorted(required - set(entries)),
        "checksum_entries_outside_g1a_scope": sorted(set(entries) - required),
        "missing_files_referenced_by_checksum": sorted(relative for relative in entries if not (root / relative).is_file()),
    }


def _verify_checksums(root: Path, entries: dict[str, str], cache: _DigestCache) -> dict[str, Any]:
    failures = []
    for relative, expected in sorted(entries.items()):
        target = root / relative
        if not target.is_file():
            failures.append({"path": relative, "reason": "missing"})
        elif cache.sha256(target) != expected:
            failures.append({"path": relative, "reason": "sha256_mismatch"})
    return {"status": "passed" if not failures else "failed", "entry_count": len(entries), "failures": failures}


def _snapshot_id(config: dict[str, Any]) -> str:
    value = config.get("dataset_snapshot_id")
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]+", value):
        raise AuditError("Config requires a portable dataset_snapshot_id")
    return value


def _inventory_state(root: Path, records: list[str], seizure_records: list[str]) -> dict[str, Any]:
    record_set, seizure_set = set(records), set(seizure_records)
    physical = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".edf"}
    machine_paths = _machine_annotation_paths(root)
    return {
        "physical_edfs": physical,
        "machine_paths": machine_paths,
        "anomalies": {
            "duplicate_records_entries": sorted(value for value, count in Counter(records).items() if count > 1),
            "duplicate_seizure_manifest_entries": sorted(value for value, count in Counter(seizure_records).items() if count > 1),
            "missing_edfs": sorted(record_set - physical),
            "unexpected_edfs": sorted(physical - record_set),
            "records_with_seizures_outside_records": sorted(seizure_set - record_set),
            "machine_annotations_outside_records_with_seizures": sorted(set(machine_paths) - seizure_set),
            "records_with_seizures_missing_machine_annotations": sorted(seizure_set - set(machine_paths)),
        },
    }


def run_g1_preflight() -> dict[str, Any]:
    """Read-only, artifact-free check for SERVER-02 before the full audit."""
    raw_value = os.environ.get("CHBMIT_RAW_DIR")
    if not raw_value:
        raise AuditError("CHBMIT_RAW_DIR is required; no fallback dataset path is permitted")
    root = Path(raw_value)
    _require_root(root)
    records, seizure_records = _lines(root / "RECORDS"), _lines(root / "RECORDS-WITH-SEIZURES")
    state = _inventory_state(root, records, seizure_records)
    entries = _checksum_entries(root / "SHA256SUMS.txt")
    coverage = _checksum_coverage(root, entries, _relevant_checksum_paths(root, records, state["machine_paths"]))
    anomalies = {**state["anomalies"], "checksum_coverage": coverage}
    failed = any(value for key, value in anomalies.items() if key != "checksum_coverage") or any(
        coverage[key] for key in ("missing_checksum_entries", "missing_files_referenced_by_checksum")
    )
    return {
        "preflight_schema_version": "g1a-server-preflight/v1",
        "preflight_status": "failed" if failed else "passed",
        "dataset_root_binding": "CHBMIT_RAW_DIR (value intentionally omitted)",
        "records_count": len(records),
        "records_with_seizures_count": len(seizure_records),
        "physical_edf_count": len(state["physical_edfs"]),
        "machine_annotation_count": len(state["machine_paths"]),
        "checksum_entry_count": len(entries),
        "anomalies": anomalies,
    }


def run_g1_audit(output_root: Path, *, replace: bool = False) -> dict[str, Any]:
    raw_value = os.environ.get("CHBMIT_RAW_DIR")
    if not raw_value:
        raise AuditError("CHBMIT_RAW_DIR is required; no fallback dataset path is permitted")
    root = Path(raw_value); _require_root(root)
    root = root.resolve(strict=True)
    repository = Path(__file__).resolve().parents[2]
    repository_status_at_start = _git(repository, "status", "--porcelain")
    output_root = _safe_output_root(root, output_root)
    config = json.loads((repository / "configs/chbmit_g1_audit_v1.json").read_text(encoding="utf-8"))
    paths = {
        "manifest_csv": output_root / "manifests/chbmit_recordings.csv",
        "manifest_jsonl": output_root / "manifests/chbmit_recordings.jsonl",
        "manifest_parquet": output_root / "manifests/chbmit_recordings.parquet",
        "audit": output_root / "reports/data_audit.json",
        "provenance": output_root / "reports/provenance_shareable.json",
        "census": output_root / "reports/channel_census.csv",
        "patterns": output_root / "reports/channel_patterns.json",
        "anomalies": output_root / "reports/anomalies.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not replace:
        raise AuditError("G1 output already exists; refuse overwrite without --replace: " + ", ".join(existing))
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    records, seizure_records = _lines(root / "RECORDS"), _lines(root / "RECORDS-WITH-SEIZURES")
    state = _inventory_state(root, records, seizure_records)
    anomalies: dict[str, Any] = {
        **state["anomalies"], "unreadable_edfs": [], "duplicate_recording_ids": [],
        "duplicate_file_sha256": [], "annotation_discrepancies": [], "invalid_intervals": [],
        "duplicate_labels_within_recordings": [],
    }
    entries = _checksum_entries(root / "SHA256SUMS.txt")
    anomalies["checksum_coverage"] = _checksum_coverage(root, entries, _relevant_checksum_paths(root, records, state["machine_paths"]))
    cache = _DigestCache()
    checksum = _verify_checksums(root, entries, cache)
    snapshot_id = _snapshot_id(config)
    summaries = _all_summary_annotations(root)
    anomalies["summary_records_outside_records"] = sorted(set(summaries) - set(records))
    anomalies["records_missing_summary_records"] = sorted(set(records) - set(summaries))
    manifest: list[dict[str, Any]] = []
    digest_records: defaultdict[str, list[str]] = defaultdict(list)
    parsed_positive: set[str] = set()
    event_count, total_duration = 0, 0.0
    census: dict[str, dict[str, Any]] = {}
    patterns: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {"labels": [], "recordings": [], "cases": set()})

    for entry in records:
        try:
            subject_id, case_id, session_id, recording_id, split_group = _identity(entry)
        except AuditError as error:
            anomalies["unreadable_edfs"].append({"relative_path": entry, "error": str(error)})
            continue
        path = root / entry
        if not path.is_file():
            continue
        try:
            header = _edf_header(path)
        except Exception as error:
            anomalies["unreadable_edfs"].append({"relative_path": entry, "error": f"{type(error).__name__}: {error}"})
            continue
        summary = summaries.get(entry.lower())
        if summary is None:
            summary = {"declared_count": None, "intervals": []}
            anomalies["annotation_discrepancies"].append({"relative_path": entry, "reason": "summary_record_missing"})
        intervals = summary["intervals"]
        if summary["declared_count"] is None or summary["declared_count"] != len(intervals):
            anomalies["annotation_discrepancies"].append({"relative_path": entry, "reason": "summary_declared_count_mismatch", "declared_count": summary["declared_count"], "parsed_count": len(intervals)})
        for onset, offset in intervals:
            if onset < 0 or offset <= onset or offset > header["duration_s"] + config["annotation_tolerance_seconds"]:
                anomalies["invalid_intervals"].append({"relative_path": entry, "onset_s": onset, "offset_s": offset, "duration_s": header["duration_s"]})
        machine = None
        if entry in state["machine_paths"]:
            machine = _machine_intervals(path, state["machine_paths"][entry], header["sampling_rate_hz"])
            if not _same_intervals(intervals, machine, config["annotation_tolerance_seconds"]):
                anomalies["annotation_discrepancies"].append({"relative_path": entry, "reason": "summary_machine_annotation_mismatch", "summary": intervals, "machine": machine})
        if bool(intervals) != (entry in set(seizure_records)):
            anomalies["annotation_discrepancies"].append({"relative_path": entry, "reason": "records_with_seizures_mismatch", "listed": entry in set(seizure_records), "parsed_interval_count": len(intervals)})
        digest = cache.sha256(path); digest_records[digest].append(recording_id)
        if intervals:
            parsed_positive.add(entry); event_count += len(intervals)
        total_duration += header["duration_s"]
        labels = header["original_channel_labels"]
        fingerprint = hashlib.sha256(json.dumps(labels, ensure_ascii=False).encode()).hexdigest()
        patterns[fingerprint]["labels"] = labels; patterns[fingerprint]["recordings"].append(recording_id); patterns[fingerprint]["cases"].add(case_id)
        duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
        if duplicates:
            anomalies["duplicate_labels_within_recordings"].append({"recording_id": recording_id, "labels": duplicates})
        for label in set(labels):
            item = census.setdefault(label, {"original_label": label, "normalized_spelling_candidate": _label_candidate(label), "likely_kind": _channel_kind(label), "recordings": set(), "cases": set()})
            item["recordings"].add(recording_id); item["cases"].add(case_id)
        manifest.append({
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION, "dataset_snapshot_id": snapshot_id,
            "dataset": config["dataset"], "dataset_version": config["dataset_version"],
            "subject_id": subject_id, "case_id": case_id, "session_id": session_id,
            "recording_id": recording_id, "relative_path": entry, **header,
            "seizure_intervals_s": intervals, "split_group": split_group,
            "file_size_bytes": path.stat().st_size, "file_sha256": digest,
            "source_manifest_entry": entry,
            "annotation_source": "summary_and_machine_verified" if machine is not None else "summary",
            "recording_boundary": "independent_edf_no_cross_recording_continuity",
        })

    anomalies["duplicate_recording_ids"] = sorted(value for value, count in Counter(row["recording_id"] for row in manifest).items() if count > 1)
    allowed = set(config["documented_duplicate_sha256_exceptions"])
    anomalies["duplicate_file_sha256"] = [
        {"sha256": digest, "recording_ids": ids} for digest, ids in digest_records.items()
        if len(ids) > 1 and digest not in allowed
    ]
    manifest.sort(key=lambda row: row["relative_path"])
    csv_rows = [{key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], list) else row[key] for key in MANIFEST_FIELDS} for row in manifest]
    _write_csv(paths["manifest_csv"], csv_rows, MANIFEST_FIELDS)
    with paths["manifest_jsonl"].open("w", encoding="utf-8") as output:
        for row in manifest:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    parquet_status = "not_written_optional_engine_unavailable"
    try:
        import pandas as pd
        pd.DataFrame(csv_rows).to_parquet(paths["manifest_parquet"], index=False)
        parquet_status = "written"
    except (ImportError, ValueError):
        pass
    census_rows = [{"original_label": item["original_label"], "normalized_spelling_candidate": item["normalized_spelling_candidate"], "likely_kind": item["likely_kind"], "recording_count": len(item["recordings"]), "case_count": len(item["cases"])} for item in census.values()]
    _write_csv(paths["census"], sorted(census_rows, key=lambda row: (-row["recording_count"], row["original_label"])), ("original_label", "normalized_spelling_candidate", "likely_kind", "recording_count", "case_count"))
    _write_json(paths["patterns"], [{"fingerprint": key, "recording_count": len(value["recordings"]), "case_count": len(value["cases"]), "recording_ids": sorted(value["recordings"]), "case_ids": sorted(value["cases"]), "original_channel_labels": value["labels"]} for key, value in sorted(patterns.items())])
    _write_json(paths["anomalies"], anomalies)
    versions: dict[str, str] = {}
    for package in ("pyedflib", "wfdb", "pandas", "pyarrow", "fastparquet"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    provenance = {
        "provenance_schema_version": "g1a-shareable-provenance/v1",
        "git_commit": _git(repository, "rev-parse", "HEAD"), "git_branch": _git(repository, "branch", "--show-current"),
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(), "python_version": sys.version,
        "package_versions": versions, "dataset_root_binding": "CHBMIT_RAW_DIR (value intentionally omitted)",
        "dataset_snapshot_id": snapshot_id, "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "records_sha256": cache.sha256(root / "RECORDS"),
        "records_with_seizures_sha256": cache.sha256(root / "RECORDS-WITH-SEIZURES"),
        "sha256sums_sha256": cache.sha256(root / "SHA256SUMS.txt"),
        "checksum_verification": checksum, "checksum_coverage": anomalies["checksum_coverage"],
        "repository_status_at_start": repository_status_at_start,
        "known_metadata_discrepancy": config["known_metadata_discrepancy"],
        "digest_cache_computed_file_count": cache.computed_count,
    }
    _write_json(paths["provenance"], provenance)
    hard_keys = (
        "duplicate_records_entries", "duplicate_seizure_manifest_entries", "missing_edfs", "unexpected_edfs",
        "records_with_seizures_outside_records", "machine_annotations_outside_records_with_seizures",
        "records_with_seizures_missing_machine_annotations", "unreadable_edfs", "duplicate_recording_ids",
        "duplicate_file_sha256", "annotation_discrepancies", "invalid_intervals", "checksum_coverage",
        "summary_records_outside_records", "records_missing_summary_records",
    )
    failed = checksum["status"] == "failed" or any(
        anomalies[key] if key != "checksum_coverage" else any(
            anomalies[key][coverage_key]
            for coverage_key in ("missing_checksum_entries", "missing_files_referenced_by_checksum")
        )
        for key in hard_keys
    )
    report = {
        "audit_status": "failed" if failed else "passed", "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_snapshot_id": snapshot_id, "records_count": len(records), "physical_edf_count": len(state["physical_edfs"]),
        "records_with_seizures_count": len(seizure_records), "parsed_seizure_containing_record_count": len(parsed_positive),
        "parsed_seizure_event_count": event_count, "case_directory_count": len(CASE_IDS),
        "biological_subject_group_count": len({row["subject_id"] for row in manifest}),
        "total_recording_duration_s": total_duration,
        "sample_rate_summary_hz": dict(Counter(str(row["sampling_rate_hz"]) for row in manifest)),
        "channel_pattern_count": len(patterns), "parquet_status": parquet_status,
        "known_metadata_discrepancy": config["known_metadata_discrepancy"], "anomalies": anomalies,
        "created_files": [str(path.relative_to(output_root)) for path in paths.values() if path.exists()],
        "repository_status_at_start": repository_status_at_start,
    }
    _write_json(paths["audit"], report)
    if failed:
        raise AuditError(f"G1 audit failed; inspect {paths['audit']}")
    return report
