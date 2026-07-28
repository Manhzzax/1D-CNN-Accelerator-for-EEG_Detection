"""Chronological, recording-grouped split planning for CHB-MIT seizure detection."""

import csv
import json
from collections import defaultdict
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")


def _split_counts(records, first_boundary, second_boundary):
    groups = (
        records[:first_boundary],
        records[first_boundary:second_boundary],
        records[second_boundary:],
    )
    return [
        {"recordings": len(group), "seizures": sum(int(row["seizure_count"]) for row in group)}
        for group in groups
    ]


def plan_case_split(records, ratios):
    """Choose chronological recording boundaries while preserving seizure events in every split."""
    if len(records) < 3:
        raise ValueError(f"Case {records[0]['case_id']} has fewer than three recordings")

    total_recordings = len(records)
    total_seizures = sum(int(row["seizure_count"]) for row in records)
    desired_sizes = [ratio * total_recordings for ratio in ratios]
    desired_seizures = [ratio * total_seizures for ratio in ratios]
    require_all_splits_to_have_seizure = total_seizures >= len(SPLIT_NAMES)
    best = None

    for first_boundary in range(1, total_recordings - 1):
        for second_boundary in range(first_boundary + 1, total_recordings):
            counts = _split_counts(records, first_boundary, second_boundary)
            recording_error = sum(
                (counts[index]["recordings"] - desired_sizes[index]) ** 2
                for index in range(len(SPLIT_NAMES))
            )
            seizure_error = sum(
                (counts[index]["seizures"] - desired_seizures[index]) ** 2
                for index in range(len(SPLIT_NAMES))
            )
            missing_seizure_splits = sum(
                count["seizures"] == 0 for count in counts
            )
            constraint_penalty = 1_000_000 * missing_seizure_splits if require_all_splits_to_have_seizure else 0
            score = constraint_penalty + recording_error + 0.25 * seizure_error
            candidate = (score, first_boundary, second_boundary, counts)
            if best is None or candidate[0] < best[0]:
                best = candidate

    _, first_boundary, second_boundary, counts = best
    return first_boundary, second_boundary, counts, require_all_splits_to_have_seizure


def load_audit_manifest(audit_dir):
    manifest_path = Path(audit_dir) / "recording_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Audit manifest is missing: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise ValueError("Audit manifest contains no recordings")
    return rows


def create_chronological_split_plan(audit_dir, output_dir, split_ratios):
    """Write a per-recording train/validation/test plan before waveform preprocessing."""
    if len(split_ratios) != len(SPLIT_NAMES) or abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must contain train, validation, and test values that sum to 1")

    rows = load_audit_manifest(audit_dir)
    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)

    planned_rows = []
    case_summaries = []
    for case_id, case_rows in sorted(cases.items()):
        first_boundary, second_boundary, counts, required = plan_case_split(case_rows, split_ratios)
        boundaries = (first_boundary, second_boundary)
        for index, row in enumerate(case_rows):
            split_index = 0 if index < boundaries[0] else 1 if index < boundaries[1] else 2
            planned_row = dict(row)
            planned_row["recording_order_in_case"] = index
            planned_row["split"] = SPLIT_NAMES[split_index]
            planned_rows.append(planned_row)

        case_summaries.append({
            "case_id": case_id,
            "total_recordings": len(case_rows),
            "total_seizures": sum(int(row["seizure_count"]) for row in case_rows),
            "require_seizure_in_each_split": required,
            "train": counts[0],
            "val": counts[1],
            "test": counts[2],
        })

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_manifest_path = output_path / "recording_split_manifest.csv"
    fieldnames = list(planned_rows[0].keys())
    with split_manifest_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(planned_rows)

    aggregate = {
        split_name: {
            "recordings": sum(row["split"] == split_name for row in planned_rows),
            "seizures": sum(int(row["seizure_count"]) for row in planned_rows if row["split"] == split_name),
        }
        for split_name in SPLIT_NAMES
    }
    cases_without_full_event_coverage = [
        summary["case_id"]
        for summary in case_summaries
        if summary["require_seizure_in_each_split"]
        and any(summary[split_name]["seizures"] == 0 for split_name in SPLIT_NAMES)
    ]
    plan = {
        "strategy": "casewise_chronological_recording_groups",
        "split_ratios": dict(zip(SPLIT_NAMES, split_ratios)),
        "recording_count": len(planned_rows),
        "case_count": len(case_summaries),
        "aggregate": aggregate,
        "cases_without_full_event_coverage": cases_without_full_event_coverage,
        "cases": case_summaries,
    }
    with (output_path / "split_plan_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(plan, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    return plan, split_manifest_path
