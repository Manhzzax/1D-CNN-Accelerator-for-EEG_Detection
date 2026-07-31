"""Patient-group-disjoint CHB-MIT split planning.

The public dataset represents chb01 and chb21 as separate cases even though
they are recording sessions from the same participant. They must remain in one
evaluation group for any patient-held-out claim.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")
_CASE_TO_PATIENT_GROUP = {
    "chb01": "subject_01_21",
    "chb21": "subject_01_21",
}


def patient_group_for_case(case_id):
    """Return the independent patient group for a CHB-MIT case identifier."""
    return _CASE_TO_PATIENT_GROUP.get(case_id, f"subject_{case_id.removeprefix('chb')}")


def _load_audit_rows(audit_dir):
    manifest_path = Path(audit_dir) / "recording_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Audit manifest is missing: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Audit manifest contains no recordings")
    return rows


def _group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[patient_group_for_case(row["case_id"])].append(row)
    return grouped


def _split_statistics(assignments, groups):
    statistics = {
        split_name: {"patient_groups": 0, "cases": 0, "recordings": 0, "seizures": 0}
        for split_name in SPLIT_NAMES
    }
    for group_id, split_name in assignments.items():
        rows = groups[group_id]
        statistics[split_name]["patient_groups"] += 1
        statistics[split_name]["cases"] += len({row["case_id"] for row in rows})
        statistics[split_name]["recordings"] += len(rows)
        statistics[split_name]["seizures"] += sum(int(row["seizure_count"]) for row in rows)
    return statistics


def _objective(assignments, groups, ratios):
    statistics = _split_statistics(assignments, groups)
    total_recordings = sum(len(rows) for rows in groups.values())
    total_seizures = sum(int(row["seizure_count"]) for rows in groups.values() for row in rows)
    score = 0.0
    for split_name, ratio in zip(SPLIT_NAMES, ratios):
        record_target = max(1.0, total_recordings * ratio)
        seizure_target = max(1.0, total_seizures * ratio)
        observed = statistics[split_name]
        score += ((observed["recordings"] - record_target) / record_target) ** 2
        score += 2.0 * ((observed["seizures"] - seizure_target) / seizure_target) ** 2
        if observed["patient_groups"] == 0 or observed["seizures"] == 0:
            score += 1_000_000.0
    return score


def _initial_assignments(groups, ratios, seed):
    """Greedily assign seizure-rich groups, then repair via local search."""
    randomizer = random.Random(seed)
    group_ids = list(groups)
    randomizer.shuffle(group_ids)
    group_ids.sort(
        key=lambda group_id: sum(int(row["seizure_count"]) for row in groups[group_id]),
        reverse=True,
    )
    assignments = {}
    for index, group_id in enumerate(group_ids):
        if index < len(SPLIT_NAMES):
            assignments[group_id] = SPLIT_NAMES[index]
            continue
        candidates = []
        for split_name in SPLIT_NAMES:
            trial = dict(assignments)
            trial[group_id] = split_name
            candidates.append((_objective(trial, groups, ratios), split_name))
        assignments[group_id] = min(candidates)[1]
    return assignments


def _refine_assignments(assignments, groups, ratios):
    """Use deterministic single-group moves and pair swaps to reduce imbalance."""
    assignments = dict(assignments)
    while True:
        baseline = _objective(assignments, groups, ratios)
        best_score = baseline
        best_assignments = None
        statistics = _split_statistics(assignments, groups)
        for group_id, current_split in assignments.items():
            if statistics[current_split]["patient_groups"] <= 1:
                continue
            for candidate_split in SPLIT_NAMES:
                if candidate_split == current_split:
                    continue
                trial = dict(assignments)
                trial[group_id] = candidate_split
                score = _objective(trial, groups, ratios)
                if score + 1e-12 < best_score:
                    best_score, best_assignments = score, trial
        group_ids = sorted(assignments)
        for left_index, left_group in enumerate(group_ids):
            for right_group in group_ids[left_index + 1:]:
                if assignments[left_group] == assignments[right_group]:
                    continue
                trial = dict(assignments)
                trial[left_group], trial[right_group] = trial[right_group], trial[left_group]
                score = _objective(trial, groups, ratios)
                if score + 1e-12 < best_score:
                    best_score, best_assignments = score, trial
        if best_assignments is None:
            return assignments
        assignments = best_assignments


def create_patient_heldout_split_plan(audit_dir, output_dir, split_ratios, seed):
    """Create a patient-group-disjoint train/validation/test manifest.

    The manifest has the same shape as the legacy planner's output, so all
    downstream preprocessing and evaluation code can consume it unchanged.
    """
    if len(split_ratios) != len(SPLIT_NAMES) or abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must contain train, validation, and test values that sum to 1")
    rows = _load_audit_rows(audit_dir)
    groups = _group_rows(rows)
    if len(groups) < len(SPLIT_NAMES):
        raise ValueError("Patient-held-out planning requires at least three independent patient groups")

    assignments = _refine_assignments(_initial_assignments(groups, split_ratios, seed), groups, split_ratios)
    statistics = _split_statistics(assignments, groups)
    if any(statistics[split_name]["seizures"] == 0 for split_name in SPLIT_NAMES):
        raise RuntimeError("Unable to allocate at least one seizure to every patient-held-out split")

    planned_rows = []
    group_summaries = []
    for group_id in sorted(groups):
        group_rows = sorted(groups[group_id], key=lambda row: row["recording_id"])
        split_name = assignments[group_id]
        group_summaries.append({
            "patient_group": group_id,
            "cases": sorted({row["case_id"] for row in group_rows}),
            "recordings": len(group_rows),
            "seizures": sum(int(row["seizure_count"]) for row in group_rows),
            "split": split_name,
        })
        for row in group_rows:
            planned_row = dict(row)
            planned_row["patient_group"] = group_id
            planned_row["split"] = split_name
            planned_rows.append(planned_row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "recording_split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(planned_rows[0].keys()))
        writer.writeheader()
        writer.writerows(planned_rows)

    plan = {
        "strategy": "patient_group_disjoint_stratified_holdout",
        "patient_group_rule": "chb01 and chb21 are assigned to subject_01_21; every other case is one group",
        "split_ratios": dict(zip(SPLIT_NAMES, split_ratios)),
        "seed": int(seed),
        "recording_count": len(planned_rows),
        "patient_group_count": len(groups),
        "aggregate": statistics,
        "groups": group_summaries,
    }
    with (output_path / "split_plan_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(plan, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return plan, manifest_path
