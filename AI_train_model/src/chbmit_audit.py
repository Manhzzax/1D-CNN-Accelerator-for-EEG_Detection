"""Audit CHB-MIT EDF headers and seizure annotations without loading waveforms."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

from .chbmit_montage import (
    CANONICAL_BIPOLAR_17,
    normalize_channel_name,
    resolve_canonical_bipolar_17,
)


CANONICAL_BIPOLAR_18_DIRECT = CANONICAL_BIPOLAR_17[:14] + ("T8-P8",) + CANONICAL_BIPOLAR_17[14:]


def parse_summary_annotations(summary_path):
    """Return seizure intervals and declared counts from a case summary."""
    annotations = {}
    current_file = None
    starts = []
    ends = []
    declared_count = None

    def save_current():
        if current_file is None:
            return
        annotations[current_file] = {
            "intervals": [(start, end) for start, end in zip(starts, ends) if end > start],
            "declared_count": declared_count,
        }

    with summary_path.open("r", encoding="utf-8", errors="replace") as summary_file:
        for raw_line in summary_file:
            line = raw_line.strip()
            file_match = re.match(r"File Name:\s*(\S+\.edf)\s*$", line, re.IGNORECASE)
            if file_match:
                save_current()
                current_file = file_match.group(1)
                starts = []
                ends = []
                declared_count = None
                continue

            count_match = re.match(r"Number of Seizures in File:\s*(\d+)\s*$", line, re.IGNORECASE)
            if count_match and current_file:
                declared_count = int(count_match.group(1))
                continue

            start_match = re.match(
                r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds",
                line,
                re.IGNORECASE,
            )
            if start_match and current_file:
                starts.append(int(start_match.group(1)))
                continue

            end_match = re.match(
                r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds",
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
        for file_name, details in parse_summary_annotations(summary_path).items():
            annotations[f"{case_id}/{file_name}"] = details
    return annotations


def load_records(raw_dataset_dir):
    records_path = Path(raw_dataset_dir) / "RECORDS"
    if not records_path.is_file():
        raise FileNotFoundError(f"Official RECORDS manifest is missing: {records_path}")
    with records_path.open("r", encoding="utf-8", errors="replace") as records_file:
        return [line.strip() for line in records_file if line.strip()]


def load_records_with_seizures(raw_dataset_dir):
    records_path = Path(raw_dataset_dir) / "RECORDS-WITH-SEIZURES"
    if not records_path.is_file():
        raise FileNotFoundError(f"Seizure RECORDS manifest is missing: {records_path}")
    with records_path.open("r", encoding="utf-8", errors="replace") as records_file:
        return [line.strip() for line in records_file if line.strip()]


def find_local_edf_paths(raw_dataset_dir):
    root = Path(raw_dataset_dir)
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.edf")
        if path.is_file()
    }


def parse_edf_seizure_annotations(edf_path, sampling_rate_hz):
    """Read the PhysioNet WFDB '[' and ']' seizure markers beside an EDF file."""
    annotation_path = Path(f"{edf_path}.seizures")
    if not annotation_path.is_file():
        return None

    import wfdb

    annotation = wfdb.rdann(str(edf_path), "seizures")
    starts = []
    intervals = []
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        if symbol == "[":
            starts.append(int(sample))
        elif symbol == "]":
            if not starts:
                raise ValueError(f"Unpaired seizure end marker in {annotation_path}")
            start_sample = starts.pop(0)
            end_sample = int(sample)
            if end_sample <= start_sample:
                raise ValueError(f"Non-positive seizure interval in {annotation_path}")
            intervals.append((
                round(start_sample / sampling_rate_hz, 6),
                round(end_sample / sampling_rate_hz, 6),
            ))

    if starts:
        raise ValueError(f"Unpaired seizure start marker in {annotation_path}")
    return intervals


def inspect_edf_header(edf_path):
    """Read only EDF metadata and always release the file handle."""
    import mne

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
    seizure_records = load_records_with_seizures(root)
    seizure_record_set = set(seizure_records)
    local_set = find_local_edf_paths(root)
    missing_local = sorted(record_set - local_set)
    local_only = sorted(local_set - record_set)
    seizure_records_outside_records = sorted(seizure_record_set - record_set)
    annotations = load_all_annotations(root)

    manifest_rows = []
    channel_presence = Counter()
    sampling_rates = Counter()
    channel_counts = Counter()
    canonical_17_direct_counts = Counter()
    canonical_17_reconstructed_counts = Counter()
    canonical_17_complete_records = []
    canonical_17_incomplete_records = []
    header_errors = []
    total_declared_seizures = 0
    total_summary_seizures = 0
    total_primary_seizures = 0
    annotation_file_errors = []
    summary_annotation_discrepancies = []
    seizure_manifest_discrepancies = []
    summary_missing_records = []

    print(f"Auditing {len(records)} EDF files listed by RECORDS...")
    for index, record_id in enumerate(records, start=1):
        edf_path = root / record_id
        header = inspect_edf_header(edf_path)
        annotation = annotations.get(record_id, {"intervals": [], "declared_count": None})
        summary_intervals = annotation["intervals"]
        declared_seizure_count = annotation["declared_count"]
        is_seizure_record = record_id in seizure_record_set
        external_intervals = None
        external_error = ""
        if not header["error"]:
            try:
                external_intervals = parse_edf_seizure_annotations(
                    edf_path,
                    header["sampling_rate_hz"],
                )
            except Exception as exc:
                external_error = f"{type(exc).__name__}: {exc}"
                annotation_file_errors.append({"recording_id": record_id, "error": external_error})

        intervals = external_intervals if external_intervals is not None else summary_intervals
        label_source = "edf_seizures" if external_intervals is not None else "summary"
        total_summary_seizures += len(summary_intervals)
        total_primary_seizures += len(intervals)
        if declared_seizure_count is not None:
            total_declared_seizures += declared_seizure_count
        if declared_seizure_count is None:
            summary_missing_records.append(record_id)

        if declared_seizure_count is not None and declared_seizure_count != len(summary_intervals):
            summary_annotation_discrepancies.append({
                "recording_id": record_id,
                "declared_seizure_count": declared_seizure_count,
                "summary_interval_count": len(summary_intervals),
                "reason": "declared_count_differs_from_summary_intervals",
            })
        if external_intervals is not None and external_intervals != summary_intervals:
            summary_annotation_discrepancies.append({
                "recording_id": record_id,
                "summary_intervals": summary_intervals,
                "edf_seizures_intervals": external_intervals,
                "reason": "summary_intervals_differ_from_edf_seizures",
            })
        if is_seizure_record != bool(intervals):
            seizure_manifest_discrepancies.append({
                "recording_id": record_id,
                "is_listed_in_records_with_seizures": is_seizure_record,
                "primary_interval_count": len(intervals),
                "label_source": label_source,
            })
        normalized_channels = header.pop("normalized_channels")

        if header["error"]:
            header_errors.append({"recording_id": record_id, "error": header["error"]})
        else:
            sampling_rates[str(header["sampling_rate_hz"])] += 1
            channel_counts[str(header["channel_count"])] += 1
            for channel_name in set(normalized_channels):
                channel_presence[channel_name] += 1
            channel_resolution = resolve_canonical_bipolar_17(header["channel_names"])
            missing_channels = [
                channel_name
                for channel_name, (mode, _) in channel_resolution.items()
                if mode == "missing"
            ]
            if missing_channels:
                canonical_17_incomplete_records.append({
                    "recording_id": record_id,
                    "missing_channels": missing_channels,
                })
            else:
                canonical_17_complete_records.append(record_id)
            for channel_name, (mode, _) in channel_resolution.items():
                if mode == "direct":
                    canonical_17_direct_counts[channel_name] += 1
                elif mode == "difference":
                    canonical_17_reconstructed_counts[channel_name] += 1
        if header["error"]:
            channel_resolution = {
                channel_name: ("missing", ()) for channel_name in CANONICAL_BIPOLAR_17
            }

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
            "canonical_bipolar_17_resolution_json": json.dumps({
                channel_name: {"mode": mode, "indices": list(indices)}
                for channel_name, (mode, indices) in channel_resolution.items()
            }),
            "seizure_intervals_json": json.dumps(intervals),
            "label_source": label_source,
            "edf_seizures_intervals_json": json.dumps(external_intervals),
            "edf_seizures_error": external_error,
            "declared_seizure_count": declared_seizure_count,
            "seizure_count": len(intervals),
            "is_listed_in_records_with_seizures": is_seizure_record,
            "header_error": header["error"],
        })

        if index % 25 == 0 or index == len(records):
            print(f"  Header audit: {index}/{len(records)}")

    write_csv(
        output_path / "recording_manifest.csv",
        manifest_rows,
        [
            "recording_id", "case_id", "edf_path", "sampling_rate_hz", "sample_count",
            "duration_sec", "channel_count", "channel_names_json", "canonical_bipolar_17_resolution_json",
            "seizure_intervals_json",
            "label_source", "edf_seizures_intervals_json", "edf_seizures_error",
            "declared_seizure_count", "seizure_count", "is_listed_in_records_with_seizures", "header_error",
        ],
    )

    channel_rows = []
    for channel_name, count in sorted(channel_presence.items(), key=lambda item: (-item[1], item[0])):
        channel_rows.append({
            "normalized_channel": channel_name,
            "recording_count": count,
            "coverage_percent": round(100.0 * count / len(records), 3),
            "is_canonical_bipolar_18_direct": channel_name in CANONICAL_BIPOLAR_18_DIRECT,
        })
    write_csv(
        output_path / "channel_presence.csv",
        channel_rows,
        ["normalized_channel", "recording_count", "coverage_percent", "is_canonical_bipolar_18_direct"],
    )

    summary = {
        "dataset_root": str(root),
        "records_manifest_count": len(records),
        "local_edf_count": len(local_set),
        "missing_local_count": len(missing_local),
        "local_only_count": len(local_only),
        "records_with_seizures_count": len(seizure_records),
        "records_with_seizures_outside_records": seizure_records_outside_records,
        "header_success_count": len(records) - len(header_errors),
        "header_error_count": len(header_errors),
        "sampling_rate_distribution_hz": dict(sorted(sampling_rates.items())),
        "channel_count_distribution": dict(sorted(channel_counts.items())),
        "summary_declared_seizure_count": total_declared_seizures,
        "summary_parsed_seizure_count": total_summary_seizures,
        "primary_label_source": "edf_seizures_when_available_else_summary",
        "primary_parsed_seizure_count": total_primary_seizures,
        "edf_seizures_error_count": len(annotation_file_errors),
        "edf_seizures_errors": annotation_file_errors,
        "summary_annotation_discrepancy_count": len(summary_annotation_discrepancies),
        "summary_annotation_discrepancies": summary_annotation_discrepancies,
        "records_with_seizures_discrepancy_count": len(seizure_manifest_discrepancies),
        "records_with_seizures_discrepancies": seizure_manifest_discrepancies,
        "summary_missing_record_count": len(summary_missing_records),
        "summary_missing_records": summary_missing_records,
        "canonical_bipolar_18_direct_coverage": {
            channel_name: channel_presence[channel_name] for channel_name in CANONICAL_BIPOLAR_18_DIRECT
        },
        "canonical_bipolar_17_direct_coverage": {
            channel_name: canonical_17_direct_counts[channel_name] for channel_name in CANONICAL_BIPOLAR_17
        },
        "canonical_bipolar_17_reconstructed_coverage": {
            channel_name: canonical_17_reconstructed_counts[channel_name] for channel_name in CANONICAL_BIPOLAR_17
        },
        "canonical_bipolar_17_complete_record_count": len(canonical_17_complete_records),
        "canonical_bipolar_17_incomplete_records": canonical_17_incomplete_records,
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

    verified = (
        not missing_local
        and not local_only
        and not seizure_records_outside_records
        and not header_errors
        and not annotation_file_errors
        and total_primary_seizures == total_declared_seizures
        and not canonical_17_incomplete_records
    )
    print(f"Manifest records: {len(records)} | local EDF: {len(local_set)}")
    print(
        "Header errors: "
        f"{len(header_errors)} | declared seizures: {total_declared_seizures} | "
        f"summary seizures: {total_summary_seizures} | primary seizures: {total_primary_seizures} | "
        f"seizure records: {len(seizure_records)} | annotation-file errors: {len(annotation_file_errors)} | "
        f"summary/annotation discrepancies: {len(summary_annotation_discrepancies)} | "
        f"seizure-manifest discrepancies: {len(seizure_manifest_discrepancies)} | "
        f"summary-missing records: {len(summary_missing_records)} | "
        f"canonical-17 complete: {len(canonical_17_complete_records)}/{len(records)}"
    )
    print(f"Audit artifacts: {output_path}")
    print(f"Audit result: {'PASS' if verified else 'REVIEW REQUIRED'}")
    return verified
