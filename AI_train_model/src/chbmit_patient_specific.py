"""Patient-specific chronological split planning for high-accuracy Path A.

Each CHB-MIT case receives its own train/val/test recording split. Models are
trained and evaluated independently; the headline metric is the unweighted mean
of per-case sealed test balanced window accuracy.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .chbmit_split import SPLIT_NAMES, plan_case_split


def create_patient_specific_split_plans(audit_dir, output_dir, split_ratios):
    """Write one locked chronological protocol directory per eligible case."""
    if len(split_ratios) != 3 or abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError("split_ratios must be train,val,test summing to 1")

    audit_path = Path(audit_dir)
    manifest_path = audit_path / "recording_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Audit manifest is missing: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Audit manifest contains no recordings")

    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    cohort = {
        "strategy": "patient_specific_casewise_chronological_recording_groups",
        "split_ratios": dict(zip(SPLIT_NAMES, split_ratios)),
        "cases": {},
        "eligible_cases": [],
        "skipped_cases": [],
    }

    for case_id, case_rows in sorted(cases.items()):
        # Preserve audit order as chronological order within the case.
        total_seizures = sum(int(row["seizure_count"]) for row in case_rows)
        if len(case_rows) < 3:
            cohort["skipped_cases"].append({
                "case_id": case_id,
                "reason": "fewer_than_three_recordings",
                "recordings": len(case_rows),
                "seizures": total_seizures,
            })
            continue
        if total_seizures < 2:
            cohort["skipped_cases"].append({
                "case_id": case_id,
                "reason": "fewer_than_two_seizure_annotations",
                "recordings": len(case_rows),
                "seizures": total_seizures,
            })
            continue

        try:
            first_boundary, second_boundary, counts, required = plan_case_split(
                case_rows, split_ratios
            )
        except ValueError as error:
            cohort["skipped_cases"].append({
                "case_id": case_id,
                "reason": str(error),
                "recordings": len(case_rows),
                "seizures": total_seizures,
            })
            continue

        # Require at least one seizure in train and test for a meaningful sealed test.
        if counts[0]["seizures"] < 1 or counts[2]["seizures"] < 1:
            cohort["skipped_cases"].append({
                "case_id": case_id,
                "reason": "no_seizure_in_train_or_test_after_split",
                "recordings": len(case_rows),
                "seizures": total_seizures,
                "split_counts": counts,
            })
            continue

        case_dir = root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        planned_rows = []
        boundaries = (first_boundary, second_boundary)
        for index, row in enumerate(case_rows):
            split_index = 0 if index < boundaries[0] else 1 if index < boundaries[1] else 2
            planned = dict(row)
            planned["recording_order_in_case"] = index
            planned["split"] = SPLIT_NAMES[split_index]
            planned_rows.append(planned)

        fieldnames = list(planned_rows[0].keys())
        with (case_dir / "recording_split_manifest.csv").open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(planned_rows)

        case_summary = {
            "case_id": case_id,
            "total_recordings": len(case_rows),
            "total_seizures": total_seizures,
            "require_seizure_in_each_split": required,
            "boundaries": {
                "train_end_exclusive": first_boundary,
                "val_end_exclusive": second_boundary,
            },
            "train": counts[0],
            "val": counts[1],
            "test": counts[2],
        }
        with (case_dir / "split_plan_summary.json").open("w", encoding="utf-8") as output_file:
            json.dump(case_summary, output_file, indent=2, sort_keys=True)
            output_file.write("\n")

        cohort["cases"][case_id] = case_summary
        cohort["eligible_cases"].append(case_id)

    with (root / "cohort_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(cohort, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    return cohort
