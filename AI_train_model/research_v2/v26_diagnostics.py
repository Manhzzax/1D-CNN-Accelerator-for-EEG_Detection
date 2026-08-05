"""Artifact-only operating-point diagnostics for consumed V2.1 folds.

This module deliberately reads packaged JSON artifacts only. It neither opens
EEG files nor checkpoints, creates score streams, selects a policy, or trains a
model. Its role is to make the calibration-to-future transfer failure auditable
before any separately preregistered intervention is proposed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .protocol import canonical_json_hash, load_json, save_json


def validate_v26_diagnostic_config(config: dict[str, Any]) -> None:
    """Reject a diagnostic config that could silently become a training plan."""
    if config.get("version") != "v2.6.0-diagnostic-only":
        raise ValueError("V2.6 diagnostics require version v2.6.0-diagnostic-only")
    if list(config.get("consumed_development_folds", [])) != ["00", "01", "02"]:
        raise ValueError("V2.6 diagnostics are restricted to consumed folds 00, 01, and 02")
    if list(config.get("sealed_temporal_blocks", [])) != [5, 6]:
        raise ValueError("V2.6 diagnostics must keep blocks 5 and 6 sealed")
    if list(config.get("training_seeds", [])) != [7, 42, 123, 314, 2718]:
        raise ValueError("V2.6 diagnostics require the fixed five-seed schedule")
    endpoint = config.get("event_endpoint", {})
    if float(endpoint.get("primary_far_per_hour", 0.0)) != 0.5:
        raise ValueError("V2.6 diagnostics require the 0.5/h FAR target")
    contract = config.get("artifact_contract", {})
    if int(contract.get("expected_parameter_count", 0)) != 57446:
        raise ValueError("V2.6 compares only the fixed 57,446-parameter C1 graph")
    expected_files = {
        "model_spec.json", "provenance.json", "training_summary.json",
        "validation_window_metrics.json", "temporal_confirmation.json",
        "calibration_policy_sweep.json",
    }
    if set(contract.get("required_files", [])) != expected_files:
        raise ValueError("V2.6 artifact contract is incomplete")
    candidates = config.get("candidates", [])
    if [candidate.get("id") for candidate in candidates] != ["C1", "H2", "G1"]:
        raise ValueError("V2.6 requires the C1, H2, and G1 artifact families")
    if any("{fold}" not in candidate.get("artifact_template", "") or "{seed}" not in candidate.get("artifact_template", "") for candidate in candidates):
        raise ValueError("Every V2.6 artifact template must include fold and seed placeholders")
    prohibited = set(config.get("prohibited_actions", []))
    for action in ("model_training", "threshold_selection", "block_5_access", "block_6_access", "tensor_export"):
        if action not in prohibited:
            raise ValueError(f"V2.6 must prohibit {action}")


def _sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _required_artifact(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    required = config["artifact_contract"]["required_files"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete V2.6 artifact {path}: {missing}")
    return {name: load_json(path / name) for name in required if name.endswith(".json")}


def _run_record(path: Path, candidate: dict[str, Any], fold: str, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    payload = _required_artifact(path, config)
    spec = payload["model_spec.json"]
    provenance = payload["provenance.json"]
    confirmation = payload["temporal_confirmation.json"]
    expected_parameters = int(config["artifact_contract"]["expected_parameter_count"])
    if int(spec.get("parameter_count", -1)) != expected_parameters:
        raise ValueError(f"Unexpected parameter count in {path}: {spec.get('parameter_count')}")
    if provenance.get("candidate_id") != candidate["candidate_id"]:
        raise ValueError(f"Artifact candidate mismatch in {path}: {provenance.get('candidate_id')}")
    if int(provenance.get("training_seed", -1)) != seed:
        raise ValueError(f"Artifact seed mismatch in {path}: {provenance.get('training_seed')}")
    if provenance.get("precision") != config["artifact_contract"]["expected_precision"]:
        raise ValueError(f"Unexpected precision contract in {path}: {provenance.get('precision')}")
    if confirmation.get("policy_selection_status") != "feasible_calibration_policy_selected":
        raise ValueError(f"V2.6 requires a feasible calibration policy in {path}")
    selected = confirmation.get("selected_calibration_policy")
    temporal = confirmation.get("temporal_evaluation")
    uncertainty = confirmation.get("temporal_uncertainty", {})
    if not selected or not temporal:
        raise ValueError(f"V2.6 requires selected and temporal metrics in {path}")
    return {
        "artifact": path.name,
        "candidate": candidate["id"],
        "fold": fold,
        "seed": seed,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "manifest_sha256": provenance["split_sha256"],
        "parameter_count": int(spec["parameter_count"]),
        "validation": payload["validation_window_metrics.json"],
        "calibration": selected,
        "temporal": temporal,
        "per_patient_group": uncertainty.get("per_patient_group", {}),
    }


def _summarize_patient_groups(records: list[dict[str, Any]], top_count: int) -> dict[str, Any]:
    aggregate: dict[str, dict[str, float]] = defaultdict(lambda: {
        "false_alarms": 0.0, "interictal_hours": 0.0, "detected_events": 0.0, "total_events": 0.0,
    })
    for record in records:
        for group, values in record["per_patient_group"].items():
            target = aggregate[group]
            for field in target:
                target[field] += float(values.get(field, 0.0))
    total_false_alarms = sum(values["false_alarms"] for values in aggregate.values())
    rows = []
    for group, values in aggregate.items():
        far = values["false_alarms"] / values["interictal_hours"] if values["interictal_hours"] else 0.0
        share = values["false_alarms"] / total_false_alarms if total_false_alarms else 0.0
        rows.append({
            "patient_group": group,
            "false_alarms": int(values["false_alarms"]),
            "interictal_hours": _round(values["interictal_hours"]),
            "false_alarms_per_hour": _round(far),
            "false_alarm_share": _round(share),
            "detected_events": int(values["detected_events"]),
            "total_events": int(values["total_events"]),
        })
    rows.sort(key=lambda row: (-row["false_alarms"], -row["false_alarms_per_hour"], row["patient_group"]))
    shares = [row["false_alarm_share"] for row in rows]
    return {
        "patient_groups": rows,
        "false_alarm_total": int(total_false_alarms),
        "top_group_false_alarm_share": _round(max(shares, default=0.0)),
        "false_alarm_hhi": _round(sum(share * share for share in shares)),
        "top_patient_groups": rows[:top_count],
    }


def _summary(records: list[dict[str, Any]], target_far: float, top_count: int) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty V2.6 record set")
    metric_paths = {
        "window_balanced_accuracy": ("validation", "balanced_accuracy"),
        "window_auroc": ("validation", "auroc"),
        "calibration_far_per_hour": ("calibration", "false_alarms_per_hour"),
        "temporal_event_sensitivity": ("temporal", "event_sensitivity"),
        "temporal_far_per_hour": ("temporal", "false_alarms_per_hour"),
        "temporal_median_delay_sec": ("temporal", "median_detection_delay_sec"),
    }
    metrics = {}
    for name, (section, field) in metric_paths.items():
        values = [float(record[section][field]) for record in records if record[section][field] is not None]
        metrics[name] = {"mean": _round(mean(values)), "sample_std": _round(_sample_std(values)), "count": len(values)}
    policies = Counter(record["calibration"]["policy_name"] for record in records)
    thresholds = [float(record["calibration"]["threshold"]) for record in records]
    transfer_failures = [record for record in records if float(record["temporal"]["false_alarms_per_hour"]) > target_far]
    paired_delta = [
        float(record["temporal"]["false_alarms_per_hour"]) - float(record["calibration"]["false_alarms_per_hour"])
        for record in records
    ]
    return {
        "runs": len(records),
        "metrics": metrics,
        "selected_policy_counts": dict(sorted(policies.items())),
        "selected_threshold": {"minimum": _round(min(thresholds)), "maximum": _round(max(thresholds)), "mean": _round(mean(thresholds))},
        "calibration_feasible_runs": len(records),
        "temporal_far_target_passes": len(records) - len(transfer_failures),
        "temporal_far_target_failures": len(transfer_failures),
        "temporal_minus_calibration_far": {"mean": _round(mean(paired_delta)), "sample_std": _round(_sample_std(paired_delta))},
        "patient_group_false_alarm_concentration": _summarize_patient_groups(records, top_count),
    }


def _build_markdown(config: dict[str, Any], report: dict[str, Any]) -> str:
    target = config["event_endpoint"]["primary_far_per_hour"]
    lines = [
        "# V2.6 Operating-Point Atlas",
        "",
        "## Scope",
        "",
        "This is an artifact-only diagnostic over already consumed V2.1 F00--F02",
        "development replays. It neither trains a model nor selects a threshold or policy.",
        "Blocks 5 and 6 remain sealed.",
        "",
        "## Calibration-To-Temporal Transfer",
        "",
        "Values are means +/- sample standard deviations across the five fixed seeds",
        "within a fold. Fold-by-seed values are not treated as independent patients.",
        "",
        "| Candidate | Fold | Balanced accuracy (%) | AUROC (%) | Event SEN (%) | Calibration FAR/h | Temporal FAR/h | Temporal FAR passes | Mean temporal-calibration FAR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in config["candidates"]:
        for fold in config["consumed_development_folds"]:
            summary = report["candidate_fold_summaries"][candidate["id"]][fold]
            metrics = summary["metrics"]
            lines.append(
                "| {candidate} | F{fold} | {bacc:.2f} +/- {bacc_sd:.2f} | {auroc:.2f} +/- {auroc_sd:.2f} | "
                "{sen:.2f} +/- {sen_sd:.2f} | {cal:.3f} +/- {cal_sd:.3f} | {far:.3f} +/- {far_sd:.3f} | "
                "{passes}/{runs} at <= {target:.1f}/h | {drift:+.3f} |".format(
                    candidate=candidate["id"], fold=fold,
                    bacc=100 * metrics["window_balanced_accuracy"]["mean"], bacc_sd=100 * metrics["window_balanced_accuracy"]["sample_std"],
                    auroc=100 * metrics["window_auroc"]["mean"], auroc_sd=100 * metrics["window_auroc"]["sample_std"],
                    sen=100 * metrics["temporal_event_sensitivity"]["mean"], sen_sd=100 * metrics["temporal_event_sensitivity"]["sample_std"],
                    cal=metrics["calibration_far_per_hour"]["mean"], cal_sd=metrics["calibration_far_per_hour"]["sample_std"],
                    far=metrics["temporal_far_per_hour"]["mean"], far_sd=metrics["temporal_far_per_hour"]["sample_std"],
                    passes=summary["temporal_far_target_passes"], runs=summary["runs"], target=target,
                    drift=summary["temporal_minus_calibration_far"]["mean"],
                )
            )
    lines.extend([
        "",
        "## False-Alarm Concentration",
        "",
        "Top groups aggregate false alarms and replay hours across the five seeds in",
        "one fold. This is a diagnostic aggregation, not a patient-level confidence interval.",
        "",
        "| Candidate | Fold | Top patient group | Top-group FAR/h | Share of false alarms | HHI |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for candidate in config["candidates"]:
        for fold in config["consumed_development_folds"]:
            concentration = report["candidate_fold_summaries"][candidate["id"]][fold]["patient_group_false_alarm_concentration"]
            top = concentration["top_patient_groups"]
            top_row = top[0] if top else {"patient_group": "none", "false_alarms_per_hour": 0.0}
            lines.append(
                f"| {candidate['id']} | F{fold} | {top_row['patient_group']} | "
                f"{top_row['false_alarms_per_hour']:.3f} | {concentration['top_group_false_alarm_share']:.3f} | "
                f"{concentration['false_alarm_hhi']:.3f} |"
            )
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "The committed artifacts establish whether a calibration-feasible operating point",
        "transferred to the next block and whether false alarms are concentrated. They do",
        "not contain full temporal score trajectories, so they cannot distinguish a bad",
        "calibration policy from a representation failure or assign EEG artifact labels.",
        "A future score-replay audit, if approved, must be diagnostic-only, use only the",
        "already consumed F00--F02 recordings, and must not select a threshold, policy,",
        "or new candidate.",
        "",
        "## Integrity",
        "",
        f"- Diagnostic config SHA-256: `{report['config_sha256']}`",
        f"- Artifact records checked: `{len(report['runs'])}`",
        "- No raw EEG, prepared cache, continuous score stream, test result, or hardware",
        "  artifact is created by this command.",
        "",
    ])
    return "\n".join(lines)


def _write_csv(output_dir: Path, records: list[dict[str, Any]]) -> None:
    import csv

    fields = [
        "candidate", "fold", "seed", "artifact", "parameter_count", "window_balanced_accuracy", "window_auroc",
        "calibration_policy", "calibration_threshold", "calibration_far_per_hour", "temporal_event_sensitivity",
        "temporal_far_per_hour", "temporal_detected_events", "temporal_total_events", "temporal_median_delay_sec",
    ]
    with (output_dir / "v26_operating_point_runs.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "candidate": record["candidate"], "fold": record["fold"], "seed": record["seed"],
                "artifact": record["artifact"], "parameter_count": record["parameter_count"],
                "window_balanced_accuracy": record["validation"]["balanced_accuracy"],
                "window_auroc": record["validation"]["auroc"],
                "calibration_policy": record["calibration"]["policy_name"],
                "calibration_threshold": record["calibration"]["threshold"],
                "calibration_far_per_hour": record["calibration"]["false_alarms_per_hour"],
                "temporal_event_sensitivity": record["temporal"]["event_sensitivity"],
                "temporal_far_per_hour": record["temporal"]["false_alarms_per_hour"],
                "temporal_detected_events": record["temporal"]["detected_events"],
                "temporal_total_events": record["temporal"]["total_events"],
                "temporal_median_delay_sec": record["temporal"]["median_detection_delay_sec"],
            })


def collect_v26_artifact_records(config_path: str | Path, artifact_root: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Load the immutable C1/H2/G1 artifact set and verify common folds."""
    config_path, artifact_root = Path(config_path), Path(artifact_root)
    config = load_json(config_path)
    validate_v26_diagnostic_config(config)
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"V2.6 artifact root does not exist: {artifact_root}")
    records = []
    manifest_hashes: dict[str, set[str]] = defaultdict(set)
    for candidate in config["candidates"]:
        for fold in config["consumed_development_folds"]:
            for seed in config["training_seeds"]:
                artifact = artifact_root / candidate["artifact_template"].format(fold=fold, seed=seed)
                record = _run_record(artifact, candidate, fold, int(seed), config)
                records.append(record)
                manifest_hashes[fold].add(record["manifest_sha256"])
    for fold, hashes in manifest_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"V2.6 candidates disagree on the consumed F{fold} manifest hash")
    return config, records, {fold: next(iter(hashes)) for fold, hashes in sorted(manifest_hashes.items())}


def build_operating_point_atlas(config_path: str | Path, artifact_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Validate compact artifacts and summarize operating-point transfer."""
    config_path, artifact_root, output_dir = Path(config_path), Path(artifact_root), Path(output_dir)
    config, records, manifest_hashes = collect_v26_artifact_records(config_path, artifact_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_far = float(config["event_endpoint"]["primary_far_per_hour"])
    top_count = int(config["reporting"]["top_patient_groups"])
    summaries = {
        candidate["id"]: {
            fold: _summary(
                [record for record in records if record["candidate"] == candidate["id"] and record["fold"] == fold],
                target_far,
                top_count,
            )
            for fold in config["consumed_development_folds"]
        }
        for candidate in config["candidates"]
    }
    report = {
        "version": config["version"],
        "config_sha256": canonical_json_hash(config),
        "artifact_root": str(artifact_root),
        "scope": {
            "consumed_development_folds": config["consumed_development_folds"],
            "sealed_temporal_blocks": config["sealed_temporal_blocks"],
            "prohibited_actions": config["prohibited_actions"],
        },
        "manifest_sha256_by_fold": manifest_hashes,
        "runs": records,
        "candidate_fold_summaries": summaries,
        "interpretation_limit": config["reporting"]["diagnostic_limit"],
    }
    save_json(output_dir / "v26_operating_point_atlas.json", report)
    (output_dir / "v26_operating_point_atlas.md").write_text(_build_markdown(config, report), encoding="utf-8")
    _write_csv(output_dir, records)
    return report
