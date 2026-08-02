"""Blocked forward-chaining fold generation and feasibility auditing."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .protocol import canonical_json_hash, save_json


SPLITS = ("train", "val", "test", "future")


def load_recording_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {"recording_id", "case_id", "seizure_count"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    return rows


def _ordered_cases(rows: list[dict]) -> dict[str, list[dict]]:
    cases: dict[str, list[dict]] = defaultdict(list)
    for source_order, row in enumerate(rows):
        copy = dict(row)
        copy["_source_order"] = source_order
        cases[copy["case_id"]].append(copy)
    for case_rows in cases.values():
        case_rows.sort(key=lambda row: int(row.get("recording_order_in_case", row["_source_order"])))
    return dict(cases)


def _fold_assignment(case_rows: list[dict], fold_index: int, fold_count: int) -> list[dict]:
    """Assign a case chronology to train/validation/test/future for one fold.

    Blocks are ordinal rather than duration-balanced.  This preserves complete
    recordings and makes all evaluation records later than the records used for
    fitting or threshold selection.  Short cases may not contribute to every
    partition; feasibility is checked at the aggregate cohort level and reported
    per case instead of silently redistributing recordings.
    """
    block_count = fold_count + 2
    size = len(case_rows)
    assigned = []
    for order, original in enumerate(case_rows):
        block = min(block_count - 1, (order * block_count) // size)
        split = "train" if block <= fold_index else "val" if block == fold_index + 1 else "test" if block == fold_index + 2 else "future"
        row = dict(original)
        row["recording_order_in_case"] = order
        row["temporal_block"] = block
        row["split"] = split
        assigned.append(row)
    return assigned


def build_forward_fold(rows: list[dict], fold_index: int, fold_count: int) -> list[dict]:
    if not 0 <= fold_index < fold_count:
        raise ValueError("fold_index must be within fold_count")
    return [
        row
        for case_rows in _ordered_cases(rows).values()
        for row in _fold_assignment(case_rows, fold_index, fold_count)
    ]


def _summary(rows: list[dict]) -> dict:
    summary = {name: {"recordings": 0, "seizures": 0, "cases": set()} for name in SPLITS}
    for row in rows:
        entry = summary[row["split"]]
        entry["recordings"] += 1
        entry["seizures"] += int(row["seizure_count"])
        entry["cases"].add(row["case_id"])
    return {
        name: {"recordings": value["recordings"], "seizures": value["seizures"], "cases": len(value["cases"])}
        for name, value in summary.items()
    }


def _chronology_violations(rows: list[dict]) -> list[str]:
    by_case: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["split"] != "future":
            by_case[row["case_id"]][row["split"]].append(int(row["recording_order_in_case"]))
    violations = []
    for case_id, partitions in by_case.items():
        train, val, test = partitions.get("train", []), partitions.get("val", []), partitions.get("test", [])
        if train and val and max(train) >= min(val):
            violations.append(f"{case_id}: train overlaps or follows validation")
        if val and test and max(val) >= min(test):
            violations.append(f"{case_id}: validation overlaps or follows test")
        if train and test and max(train) >= min(test):
            violations.append(f"{case_id}: train overlaps or follows test")
    return violations


def audit_forward_folds(rows: list[dict], fold_count: int) -> dict:
    folds = []
    for fold_index in range(fold_count):
        fold_rows = build_forward_fold(rows, fold_index, fold_count)
        summary = _summary(fold_rows)
        chronology = _chronology_violations(fold_rows)
        valid = (
            summary["train"]["recordings"] > 0
            and summary["val"]["recordings"] > 0
            and summary["test"]["recordings"] > 0
            and summary["val"]["seizures"] > 0
            and summary["test"]["seizures"] > 0
            and not chronology
        )
        folds.append({
            "fold_index": fold_index,
            "valid": valid,
            "summary": summary,
            "chronology_violations": chronology,
            "rows": fold_rows,
        })
    return {
        "fold_count": fold_count,
        "valid": all(fold["valid"] for fold in folds),
        "folds": folds,
    }


def select_feasible_protocol(rows: list[dict], requested_folds: int = 5, fallback_folds: int = 3) -> dict:
    requested = audit_forward_folds(rows, requested_folds)
    if requested["valid"]:
        selected = requested
        fallback_used = False
    else:
        selected = audit_forward_folds(rows, fallback_folds)
        if not selected["valid"]:
            raise ValueError("Neither requested nor fallback temporal fold protocol has ictal event coverage")
        fallback_used = True

    artifact = {
        "strategy": "blocked_forward_chaining_by_case_recording_group",
        "requested_outer_folds": requested_folds,
        "selected_outer_folds": selected["fold_count"],
        "fallback_used": fallback_used,
        "requested_audit": _strip_rows(requested),
        "selected_audit": _strip_rows(selected),
    }
    artifact["protocol_hash"] = canonical_json_hash(artifact)
    return artifact, selected


def _strip_rows(audit: dict) -> dict:
    return {
        "fold_count": audit["fold_count"],
        "valid": audit["valid"],
        "folds": [
            {
                "fold_index": fold["fold_index"],
                "valid": fold["valid"],
                "summary": fold["summary"],
                "chronology_violations": fold["chronology_violations"],
            }
            for fold in audit["folds"]
        ],
    }


def write_protocol_artifacts(output_dir: str | Path, artifact: dict, selected: dict) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "temporal_fold_feasibility.json", artifact)
    for fold in selected["folds"]:
        manifest_path = output / f"fold_{fold['fold_index']:02d}_manifest.csv"
        rows = fold["rows"]
        fields = [key for key in rows[0] if key != "_source_order"]
        with manifest_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{key: row[key] for key in fields} for row in rows])
