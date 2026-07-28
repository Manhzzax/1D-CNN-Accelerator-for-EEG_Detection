"""Audit CHB-MIT EDF headers and seizure annotations without loading waveforms."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import mne


CANONICAL_BIPOLAR_18 = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
)


def normalize_channel_name(name):
    """Normalize common CHB-MIT EDF channel-name variants for auditing only."""
    normalized = name.upper().strip()
    normalized = re.sub(r"^EEG\s*", "", normalized)
    normalized = re.sub(r"\s*(?:-REF|-LE)$", "", normalized)
    return normalized.replace(" ", "")


def parse_summary_annotations(summary_path):
    """Return {edf_file_name: [(start_seconds, end_seconds), ...]} from a case summary."""
    annotations = {}
    current_file = None
    starts = []
    ends = []

    def save_current():
        if current_file is None:
            return
        annotations[current_file] = [
            (start, end) for start, end in zip(starts, ends) if end > start
        ]

    with summary_path.open("r", encoding="utf-8", errors="replace") as summary_file:
        for raw_line in summary_file:
            line = raw_line.strip()
            file_match = re.match(r"File Name:\s*(\S+\.edf)\s*$", line, re.IGNORECASE)
            if file_match:
                save_current()
                current_file = file_match.group(1)
                starts = []
                ends = []
                continue

            start_match = re.match(
                r"Seizure\s+\d+\s+Start Time:\s*(\d+)\s*seconds",
                line,
                re.IGNORECASE,
            )
            if start_match and current_file:
                starts.append(int(start_match.group(1)))
                continue

            end_match = re.match(
                r"Seizure\s+\d+\s+End Time:\s*(\d+)\s*seconds",
                line,
                re.IGNORECASE,
            )
            if end_match and current_file:
                ends.append(int(end_match.group(1)))

    save_current()
    return annotations


def load_all_annotations(raw_dataset_dir):
    """Read each case summary once and key annotations by RECORDS-style path."""
    annotations = {}
    root = Path(raw_dataset_dir)
    for summary_path in sorted(root.glob("chb*/chb*-summary.txt")):
        case_id = summary_path.parent.name
        for file_name, intervals in parse_summary_annotations(summary_path).items():
            annotations[f"{case_id}/{file_name}"] = intervals
    return annotations


def load_records(raw_dataset_dir):
    records_path = Path(raw_dataset_dir) / "RECORDS"
    if not records_path.is_file():
        raise FileNotFoundError(f"Official RECORDS manifest is missing: {records_path}")
    with records_path.open("r", encoding="utf-8", errors="replace") as records_file:
        return [line.strip() for line in records_file if line.strip()]


def find_local_edf_paths(raw_dataset_dir):
    root = Path(raw_dataset_dir)
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.edf")
        if path.is_file()
    }


def inspect_edf_header(edf_path):
    """Read only EDF metadata and always release the file handle."""
    raw = None
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
        channel_names = list(raw.ch_names)
        sfreq = float(raw.info["sfreq"])
        n_samples = int(raw.n_times)
        return {
            "sampling_rate_hz": sfreq,
            "sample_count": n_samples,
            "duration_sec": round(n_samples / sfreq, 6),
            "channel_count": len(channel_names),
            "channel_names": channel_names,
            "normalized_channels": [normalize_channel_name(name) for name in channel_names],
            "error": "",
        }
    except Exception as exc:
        return {
            "sampling_rate_hz": "",
            "sample_count": "",
            "duration_sec": "",
            "channel_count": "",
            "channel_names": [],
            "normalized_channels": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if raw is not None:
            raw.close()


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_chbmit_audit(raw_dataset_dir, output_dir):
    """Create reproducible manifest and channel audit artifacts for every RECORDS EDF."""
    root = Path(raw_dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw CHB-MIT directory does not exist: {root}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    records = load_records(root)
    record_set = set(records)
    local_set = find_local_edf_paths(root)
    missing_local = sorted(record_set - local_set)
    local_only = sorted(local_set - record_set)
    annotations = load_all_annotations(root)

    manifest_rows = []
    channel_presence = Counter()
    sampling_rates = Counter()
    channel_counts = Counter()
    header_errors = []
    total_annotated_seizures = 0

    print(f"Auditing {len(records)} EDF files listed by RECORDS...")
    for index, record_id in enumerate(records, start=1):
        edf_path = root / record_id
        header = inspect_edf_header(edf_path)
        intervals = annotations.get(record_id, [])
        total_annotated_seizures += len(intervals)
        normalized_channels = header.pop("normalized_channels")

        if header["error"]:
            header_errors.append({"recording_id": record_id, "error": header["error"]})
        else:
            sampling_rates[str(header["sampling_rate_hz"])] += 1
            channel_counts[str(header["channel_count"])] += 1
            for channel_name in set(normalized_channels):
                channel_presence[channel_name] += 1

        case_id = record_id.split("/", maxsplit=1)[0]
        manifest_rows.append({
            "recording_id": record_id,
            "case_id": case_id,
            "edf_path": str(edf_path),
            "sampling_rate_hz": header["sampling_rate_hz"],
            "sample_count": header["sample_count"],
            "duration_sec": header["duration_sec"],
            "channel_count": header["channel_count"],
            "channel_names_json": json.dumps(header["channel_names"]),
            "seizure_intervals_json": json.dumps(intervals),
            "seizure_count": len(intervals),
            "header_error": header["error"],
        })

        if index % 25 == 0 or index == len(records):
            print(f"  Header audit: {index}/{len(records)}")

    write_csv(
        output_path / "recording_manifest.csv",
        manifest_rows,
        [
            "recording_id", "case_id", "edf_path", "sampling_rate_hz", "sample_count",
            "duration_sec", "channel_count", "channel_names_json", "seizure_intervals_json",
            "seizure_count", "header_error",
        ],
    )

    channel_rows = []
    for channel_name, count in sorted(channel_presence.items(), key=lambda item: (-item[1], item[0])):
        channel_rows.append({
            "normalized_channel": channel_name,
            "recording_count": count,
            "coverage_percent": round(100.0 * count / len(records), 3),
            "is_canonical_bipolar_18": channel_name in CANONICAL_BIPOLAR_18,
        })
    write_csv(
        output_path / "channel_presence.csv",
        channel_rows,
        ["normalized_channel", "recording_count", "coverage_percent", "is_canonical_bipolar_18"],
    )

    summary = {
        "dataset_root": str(root),
        "records_manifest_count": len(records),
        "local_edf_count": len(local_set),
        "missing_local_count": len(missing_local),
        "local_only_count": len(local_only),
        "header_success_count": len(records) - len(header_errors),
        "header_error_count": len(header_errors),
        "sampling_rate_distribution_hz": dict(sorted(sampling_rates.items())),
        "channel_count_distribution": dict(sorted(channel_counts.items())),
        "summary_annotation_seizure_count": total_annotated_seizures,
        "canonical_bipolar_18_coverage": {
            channel_name: channel_presence[channel_name] for channel_name in CANONICAL_BIPOLAR_18
        },
        "channels_present_in_every_readable_recording": sorted(
            channel_name
            for channel_name, count in channel_presence.items()
            if count == len(records) - len(header_errors)
        ),
        "missing_local": missing_local,
        "local_only": local_only,
        "header_errors": header_errors,
    }
    with (output_path / "audit_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    verified = not missing_local and not local_only and not header_errors
    print(f"Manifest records: {len(records)} | local EDF: {len(local_set)}")
    print(f"Header errors: {len(header_errors)} | annotated seizures: {total_annotated_seizures}")
    print(f"Audit artifacts: {output_path}")
    print(f"Audit result: {'PASS' if verified else 'REVIEW REQUIRED'}")
    return verified
