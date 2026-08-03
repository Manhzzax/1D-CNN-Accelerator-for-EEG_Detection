"""V2.1 patient-group, duration-based, forward-chaining protocol helpers."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .protocol import canonical_json_hash, file_sha256, save_json


def patient_group_for_case(case_id: str, grouping: dict) -> str:
    return grouping.get("case_to_patient_group", {}).get(case_id, f"subject_{case_id.removeprefix('chb')}")


def _duration_seconds(row: dict) -> float:
    return float(row["sample_count"]) / float(row["sampling_rate_hz"])


def _seizure_seconds(row: dict) -> float:
    return sum(float(end) - float(start) for start, end in json.loads(row["seizure_intervals_json"]))


def _session_rank(case_id: str, group_id: str, grouping: dict) -> int:
    order = grouping.get("session_order", {}).get(group_id, [case_id])
    if case_id not in order:
        raise ValueError(f"Patient group {group_id} lacks explicit session order for {case_id}")
    return order.index(case_id)


def assign_patient_group_blocks(rows: list[dict], split_config: dict) -> list[dict]:
    """Assign immutable whole recordings to duration-balanced chronological blocks."""
    grouping = split_config["patient_grouping"]
    block_count = int(split_config["base_block_count"])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for source_index, source_row in enumerate(rows):
        row = dict(source_row)
        row["patient_group"] = patient_group_for_case(row["case_id"], grouping)
        row["_source_index"] = source_index
        grouped[row["patient_group"]].append(row)

    assigned: list[dict] = []
    for group_id, group_rows in grouped.items():
        group_rows.sort(key=lambda row: (
            _session_rank(row["case_id"], group_id, grouping),
            int(row.get("recording_order_in_case", row["_source_index"])),
        ))
        total_duration = sum(_duration_seconds(row) for row in group_rows)
        if total_duration <= 0:
            raise ValueError(f"Patient group {group_id} has no positive EEG duration")
        target = total_duration / block_count
        block, duration_in_block = 0, 0.0
        for patient_order, row in enumerate(group_rows):
            duration = _duration_seconds(row)
            if block < block_count - 1 and duration_in_block > 0.0:
                keep_error = abs((duration_in_block + duration) - target)
                close_error = abs(duration_in_block - target)
                if close_error <= keep_error:
                    block += 1
                    duration_in_block = 0.0
            row["patient_recording_order"] = patient_order
            row["recording_duration_sec"] = duration
            row["temporal_block"] = block
            duration_in_block += duration
            assigned.append(row)
    return sorted(assigned, key=lambda row: row["_source_index"])


def _partition_summary(rows: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    for row in rows:
        group = groups.setdefault(row["patient_group"], {
            "patient_group": row["patient_group"], "cases": set(), "recordings": 0,
            "eeg_seconds": 0.0, "nonictal_seconds": 0.0, "seizures": 0,
        })
        duration = float(row["recording_duration_sec"])
        group["cases"].add(row["case_id"])
        group["recordings"] += 1
        group["eeg_seconds"] += duration
        group["nonictal_seconds"] += duration - _seizure_seconds(row)
        group["seizures"] += int(row["seizure_count"])
    distribution = []
    for group in groups.values():
        distribution.append({
            "patient_group": group["patient_group"], "cases": sorted(group["cases"]),
            "recordings": group["recordings"], "eeg_hours": group["eeg_seconds"] / 3600.0,
            "nonictal_replay_hours": group["nonictal_seconds"] / 3600.0, "seizures": group["seizures"],
        })
    distribution.sort(key=lambda value: value["patient_group"])
    return {
        "recordings": sum(value["recordings"] for value in distribution),
        "patient_groups": len(distribution),
        "cases": len({case for value in distribution for case in value["cases"]}),
        "eeg_hours": sum(value["eeg_hours"] for value in distribution),
        "nonictal_replay_hours": sum(value["nonictal_replay_hours"] for value in distribution),
        "seizures": sum(value["seizures"] for value in distribution),
        "seizure_contributing_patient_groups": sum(value["seizures"] > 0 for value in distribution),
        "patient_group_seizure_distribution": distribution,
    }


def build_confirmation_folds(rows: list[dict], split_config: dict) -> list[dict]:
    assigned = assign_patient_group_blocks(rows, split_config)
    fold_count = int(split_config["confirmation_folds"])
    folds = []
    for fold_index in range(fold_count):
        fold_rows = []
        for source in assigned:
            row = {key: value for key, value in source.items() if key != "_source_index"}
            block = int(row["temporal_block"])
            row["split"] = "train" if block <= fold_index else "val" if block == fold_index + 1 else "temporal_eval" if block == fold_index + 2 else "future"
            fold_rows.append(row)
        summaries = {name: _partition_summary([row for row in fold_rows if row["split"] == name]) for name in ("train", "val", "temporal_eval", "future")}
        union_rows = [row for row in fold_rows if row["split"] in ("val", "temporal_eval")]
        union = _partition_summary(union_rows)
        gate = split_config["feasibility_gate"]
        valid = (
            summaries["val"]["seizures"] > 0 and summaries["temporal_eval"]["seizures"] > 0
            and union["seizures"] >= int(gate["minimum_union_seizures"])
            and union["seizure_contributing_patient_groups"] >= int(gate["minimum_seizure_contributing_patient_groups"])
            and union["nonictal_replay_hours"] >= float(gate["minimum_nonictal_replay_hours"])
        )
        folds.append({"fold_index": fold_index, "rows": fold_rows, "summary": summaries, "validation_union": union, "valid": valid})
    return folds


def build_final_holdout(rows: list[dict], split_config: dict) -> dict:
    assigned = assign_patient_group_blocks(rows, split_config)
    final_rows = []
    for source in assigned:
        row = {key: value for key, value in source.items() if key != "_source_index"}
        block = int(row["temporal_block"])
        row["split"] = "train" if block <= 4 else "val" if block == 5 else "test" if block == 6 else "future"
        final_rows.append(row)
    summary = {name: _partition_summary([row for row in final_rows if row["split"] == name]) for name in ("train", "val", "test", "future")}
    if summary["test"]["seizures"] == 0 or summary["test"]["patient_groups"] == 0:
        raise ValueError("Sealed final block 6 has no seizure coverage")
    return {"rows": final_rows, "summary": summary}


def audit_v21(rows: list[dict], config: dict) -> dict:
    split_config = config["split"]
    folds = build_confirmation_folds(rows, split_config)
    final_holdout = build_final_holdout(rows, split_config)
    return {
        "protocol_version": config["version"], "strategy": split_config["strategy"],
        "base_block_count": split_config["base_block_count"], "confirmation_folds": folds,
        "final_holdout": final_holdout, "valid": all(fold["valid"] for fold in folds),
    }


def audit_session_timestamps(rows: list[dict], config: dict) -> dict:
    """Verify configured multi-session patient-group order from EDF headers."""
    import mne

    grouping = config["split"]["patient_grouping"]
    case_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        case_rows[row["case_id"]].append(row)
    findings = []
    for group_id, sessions in grouping.get("session_order", {}).items():
        if len(sessions) < 2:
            continue
        session_times = []
        for case_id in sessions:
            dates = []
            for row in case_rows.get(case_id, []):
                raw = mne.io.read_raw_edf(row["edf_path"], preload=False, verbose="ERROR")
                try:
                    value = raw.info.get("meas_date")
                    if value is not None:
                        dates.append(value.isoformat())
                finally:
                    raw.close()
            if not dates:
                raise ValueError(f"Cannot verify configured session order: {case_id} has no EDF acquisition timestamps")
            session_times.append({"case_id": case_id, "first": min(dates), "last": max(dates), "recordings": len(case_rows[case_id])})
        for earlier, later in zip(session_times, session_times[1:]):
            if earlier["last"] > later["first"]:
                raise ValueError(f"Configured session order conflicts with EDF timestamps: {earlier['case_id']} then {later['case_id']}")
        findings.append({"patient_group": group_id, "sessions": session_times, "verified": True})
    return {"evidence": grouping["session_order_evidence"], "groups": findings}


def write_v21_artifacts(output_dir: str | Path, audit: dict, config: dict) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {}
    for fold in audit["confirmation_folds"]:
        path = output / f"confirmation_f{fold['fold_index']:02d}_manifest.csv"
        _write_manifest(path, fold["rows"])
        manifest_hashes[path.name] = file_sha256(path)
    final_path = output / "final_holdout_manifest.csv"
    _write_manifest(final_path, audit["final_holdout"]["rows"])
    manifest_hashes[final_path.name] = file_sha256(final_path)
    report = {
        "protocol_hash": canonical_json_hash(config), "manifest_hashes": manifest_hashes,
        "audit": _strip_rows(audit),
    }
    report["audit_hash"] = canonical_json_hash(report)
    save_json(output / "v21_split_audit.json", report)
    return report


def _write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty manifest: {path}")
    fields = [key for key in rows[0] if not key.startswith("_")]
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])


def _strip_rows(audit: dict) -> dict:
    return {
        "protocol_version": audit["protocol_version"], "strategy": audit["strategy"],
        "base_block_count": audit["base_block_count"], "valid": audit["valid"],
        "confirmation_folds": [{
            "fold_index": fold["fold_index"], "valid": fold["valid"],
            "summary": fold["summary"], "validation_union": fold["validation_union"],
        } for fold in audit["confirmation_folds"]],
        "final_holdout": {"summary": audit["final_holdout"]["summary"]},
        "session_audit": audit.get("session_audit"),
    }


def create_final_freeze(protocol_path: str | Path, manifest_path: str | Path, decision_path: str | Path, output_path: str | Path) -> dict:
    """Seal all final choices before block-6 tensors can be materialized."""
    with Path(decision_path).open("r", encoding="utf-8") as source:
        decision = json.load(source)
    required = {
        "candidate_id", "architecture", "learning_rate", "weight_decay", "seed_schedule",
        "deployment_export_seed", "threshold_policy_search", "quantization", "hardware_interface",
    }
    missing = required.difference(decision)
    if missing:
        raise ValueError(f"Final decision is missing required frozen fields: {sorted(missing)}")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    training = protocol["training"]
    candidate = training["frozen_candidates"].get(decision["candidate_id"])
    if candidate is None:
        raise ValueError(f"Final decision names a candidate outside the frozen V2.1 set: {decision['candidate_id']}")
    if decision["architecture"] != candidate["architecture"]:
        raise ValueError("Final decision architecture differs from the predeclared candidate")
    if float(decision["learning_rate"]) != float(candidate["learning_rate"]) or float(decision["weight_decay"]) != float(candidate["weight_decay"]):
        raise ValueError("Final decision optimizer settings differ from the predeclared candidate")
    if list(decision["seed_schedule"]) != list(training["training_seeds"]):
        raise ValueError("Final decision must retain the five predeclared seeds")
    if int(decision["deployment_export_seed"]) != int(training["deployment_export_seed"]):
        raise ValueError("Final decision must retain the predeclared deployment export seed")
    if decision["quantization"] != protocol["hardware"]["quantization_primary"]:
        raise ValueError("Final decision quantization policy differs from the V2.1 hardware contract")
    if decision["hardware_interface"] != protocol["hardware"]:
        raise ValueError("Final decision hardware interface differs from the V2.1 protocol")
    freeze = {
        "protocol_hash": canonical_json_hash(protocol), "final_manifest_sha256": file_sha256(manifest_path),
        "decision_sha256": file_sha256(decision_path), "decision": decision,
        "final_test_status": "authorized_for_one_frozen_batch",
    }
    freeze["freeze_hash"] = canonical_json_hash(freeze)
    save_json(output_path, freeze)
    return freeze


def verify_final_freeze(freeze_path: str | Path, protocol_path: str | Path, manifest_path: str | Path) -> dict:
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if freeze.get("final_test_status") != "authorized_for_one_frozen_batch":
        raise ValueError("Final holdout is not authorized")
    if freeze.get("protocol_hash") != canonical_json_hash(protocol):
        raise ValueError("Final freeze protocol hash mismatch")
    if freeze.get("final_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Final freeze manifest hash mismatch")
    if freeze.get("freeze_hash") != canonical_json_hash({key: value for key, value in freeze.items() if key != "freeze_hash"}):
        raise ValueError("Final freeze hash mismatch")
    return freeze
