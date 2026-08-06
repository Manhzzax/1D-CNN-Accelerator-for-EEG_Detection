"""V2.7 multiplicity-aware FAR calibration diagnostic.

This module evaluates a preregistered conservative calibration rule over
already consumed V2.1 development folds. It cannot train a model, create a
new score stream, select a model candidate, or access temporal blocks 5 or 6.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .protocol import canonical_json_hash, file_sha256, load_json, save_json
from .statistics import poisson_one_sided_upper_far
from .v26_diagnostics import collect_v26_artifact_records
from .v26_score_replay import _compare_evaluation_inputs, _expected_records


def validate_v27_far_robust_config(config: dict[str, Any]) -> None:
    """Reject any configuration that could turn this diagnostic into a new experiment."""
    if config.get("version") != "v2.7.0-far-robust-calibration-diagnostic-only":
        raise ValueError("V2.7 requires the FAR-robust diagnostic-only version")
    if config.get("candidate_comparators") != ["C1", "H2"]:
        raise ValueError("V2.7 is restricted to frozen C1 and H2 comparators")
    if config.get("allowed_folds") != ["00", "01", "02"]:
        raise ValueError("V2.7 is restricted to already consumed folds 00, 01, and 02")
    if config.get("sealed_temporal_blocks") != [5, 6]:
        raise ValueError("V2.7 must keep blocks 5 and 6 sealed")
    if set(config.get("candidate_score_subdirectories", {})) != {"C1", "H2"}:
        raise ValueError("V2.7 requires C1 and H2 existing score locations only")
    evaluation = config.get("evaluation", {})
    if float(evaluation.get("primary_far_per_hour", 0.0)) != 0.5:
        raise ValueError("V2.7 requires FAR <= 0.5/h")
    if int(evaluation.get("refractory_sec", -1)) != 30 or float(evaluation.get("window_sec", 0.0)) != 5.0:
        raise ValueError("V2.7 must preserve the causal 5-second / 30-second contract")
    if int(evaluation.get("sample_rate_hz", 0)) != 256 or int(evaluation.get("declared_operating_point_count", 0)) != 1200:
        raise ValueError("V2.7 requires the unchanged 1,200-point operating grid")
    if float(evaluation.get("family_wise_confidence", 0.0)) != 0.95:
        raise ValueError("V2.7 requires 95% family-wise confidence")
    if evaluation.get("multiplicity_control") != "bonferroni_over_all_predeclared_threshold_policy_pairs":
        raise ValueError("V2.7 requires Bonferroni control over all predeclared operating points")
    if evaluation.get("calibration_rule") != "maximize_sensitivity_subject_to_one_sided_exact_garwood_far_upper_bound_at_or_below_target":
        raise ValueError("V2.7 calibration rule is not frozen")
    prohibited = set(config.get("prohibited_actions", []))
    for action in ("model_training", "candidate_selection", "retrospective_temporal_policy_selection", "block_5_access", "block_6_access", "tensor_export"):
        if action not in prohibited:
            raise ValueError(f"V2.7 must prohibit {action}")


def simultaneous_one_sided_confidence(config: dict[str, Any]) -> float:
    """Bonferroni-adjust the one-sided exact bound across the frozen grid."""
    evaluation = config["evaluation"]
    alpha_family = 1.0 - float(evaluation["family_wise_confidence"])
    return 1.0 - alpha_family / int(evaluation["declared_operating_point_count"])


def _validate_frozen_sweep_grid(sweep: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Ensure an artifact sweep is exactly the 8 x 150 V2.1 operating grid."""
    expected_count = int(config["evaluation"]["declared_operating_point_count"])
    if len(sweep) != expected_count:
        raise ValueError(f"V2.7 expected {expected_count} calibration operating points, found {len(sweep)}")
    policies = ((3, 6), (4, 8), (5, 10), (6, 12), (7, 14), (8, 16), (9, 18), (10, 20))
    expected = {
        (f"{positive}_of_{decision}", positive, decision, round(0.85 + index * 0.001, 3))
        for positive, decision in policies
        for index in range(150)
    }
    observed = {
        (
            str(item.get("policy_name")), int(item.get("positive_windows", -1)),
            int(item.get("decision_window_windows", -1)), round(float(item.get("threshold", -1.0)), 3),
        )
        for item in sweep
    }
    if observed != expected:
        raise ValueError("V2.7 calibration sweep does not match the frozen 8 x 150 operating grid")


def select_far_robust_calibration_policy(sweep: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select from calibration only, with a simultaneous exact FAR upper bound."""
    _validate_frozen_sweep_grid(sweep, config)
    confidence = simultaneous_one_sided_confidence(config)
    target = float(config["evaluation"]["primary_far_per_hour"])
    annotated = []
    for metric in sweep:
        required = {"false_alarms", "interictal_hours", "event_sensitivity", "false_alarms_per_hour", "threshold", "policy_name", "positive_windows", "decision_window_windows"}
        missing = required - set(metric)
        if missing:
            raise ValueError(f"Calibration sweep metric missing fields: {sorted(missing)}")
        bound = poisson_one_sided_upper_far(int(metric["false_alarms"]), float(metric["interictal_hours"]), confidence)
        item = dict(metric)
        item["calibration_far_upper_bound_per_hour"] = float(bound.upper)
        item["calibration_far_upper_bound_confidence"] = float(confidence)
        item["calibration_far_upper_bound_target_met"] = bool(bound.upper <= target)
        annotated.append(item)
    eligible = [item for item in annotated if item["calibration_far_upper_bound_target_met"]]
    if not eligible:
        return None, annotated
    selected = max(eligible, key=lambda item: (
        float(item["event_sensitivity"]),
        -(float(item["median_detection_delay_sec"]) if item.get("median_detection_delay_sec") is not None else float("inf")),
        -float(item["calibration_far_upper_bound_per_hour"]),
        -float(item["false_alarms_per_hour"]),
    ))
    return selected, annotated


def _temporal_rows(manifest_path: Path) -> list[dict[str, str]]:
    from .v21_evaluation import load_manifest_rows

    rows = load_manifest_rows(manifest_path, "temporal_eval")
    if any(int(row["temporal_block"]) >= 5 for row in rows):
        raise ValueError("V2.7 refuses temporal score data from sealed block 5 or 6")
    return rows


def _load_verified_temporal_scores(record: dict[str, Any], artifact_root: Path, run_root: Path, manifest_root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    from src.event_evaluation import load_scores

    run_id, fold = record["artifact"], record["fold"]
    manifest_path = manifest_root / f"confirmation_f{fold}_manifest.csv"
    if file_sha256(manifest_path) != record["manifest_sha256"]:
        raise ValueError(f"V2.7 manifest hash mismatch for {run_id}")
    rows = _temporal_rows(manifest_path)
    score_dir = run_root / run_id / config["candidate_score_subdirectories"][record["candidate"]]
    temporal_path = score_dir / "continuous_temporal_eval_scores.npz"
    sidecar_path = score_dir / "temporal_confirmation.json"
    if not temporal_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"V2.7 requires existing temporal score and sidecar for {run_id}")
    artifact_confirmation = load_json(artifact_root / run_id / "temporal_confirmation.json")
    _compare_evaluation_inputs(artifact_confirmation, load_json(sidecar_path), run_id)
    scores = load_scores(temporal_path)
    _expected_records(scores, rows, "V2.7 temporal")
    return scores, rows


def _temporal_metric(scores: dict[str, Any], selected: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    from src.event_evaluation import event_metrics

    evaluation = config["evaluation"]
    metric = event_metrics(
        scores, float(selected["threshold"]), int(evaluation["sample_rate_hz"]), float(evaluation["window_sec"]),
        int(evaluation["refractory_sec"]), int(selected["positive_windows"]), int(selected["decision_window_windows"]),
        selected["policy_name"],
    )
    bound = poisson_one_sided_upper_far(
        int(metric["false_alarms"]), float(metric["interictal_hours"]), float(evaluation["family_wise_confidence"]),
    )
    metric["temporal_far_one_sided_95_upper_bound_per_hour"] = float(bound.upper)
    metric["temporal_far_target_met"] = bool(metric["false_alarms_per_hour"] <= float(evaluation["primary_far_per_hour"]))
    return metric


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _summary(records: list[dict[str, Any]], target: float) -> dict[str, Any]:
    selected = [record for record in records if record["selected_calibration_policy"] is not None]
    temporal = [record["temporal_evaluation"] for record in selected]
    return {
        "runs": len(records),
        "calibration_ucb_feasible_runs": len(selected),
        "temporal_evaluations_opened": len(temporal),
        "temporal_far_target_passes": sum(metric["temporal_far_target_met"] for metric in temporal),
        "temporal_event_sensitivity": None if not temporal else {
            "mean": _round(mean(float(metric["event_sensitivity"]) for metric in temporal)),
            "sample_std": _round(_sample_std([float(metric["event_sensitivity"]) for metric in temporal])),
        },
        "temporal_far_per_hour": None if not temporal else {
            "mean": _round(mean(float(metric["false_alarms_per_hour"]) for metric in temporal)),
            "sample_std": _round(_sample_std([float(metric["false_alarms_per_hour"]) for metric in temporal])),
        },
        "temporal_one_sided_95_upper_far_per_hour": None if not temporal else {
            "mean": _round(mean(float(metric["temporal_far_one_sided_95_upper_bound_per_hour"]) for metric in temporal)),
            "sample_std": _round(_sample_std([float(metric["temporal_far_one_sided_95_upper_bound_per_hour"]) for metric in temporal])),
        },
        "decision": "diagnostic_only_no_candidate_or_policy_promotion",
        "target_far_per_hour": target,
    }


def _write_csv(output_dir: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "candidate", "fold", "seed", "artifact", "calibration_ucb_feasible", "policy_name", "threshold",
        "calibration_sensitivity", "calibration_far_per_hour", "calibration_far_upper_bound_per_hour",
        "temporal_opened", "temporal_sensitivity", "temporal_far_per_hour", "temporal_far_target_met",
        "temporal_far_one_sided_95_upper_bound_per_hour",
    ]
    with (output_dir / "v27_far_robust_calibration_runs.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for record in records:
            selected, temporal = record["selected_calibration_policy"], record["temporal_evaluation"]
            writer.writerow({
                "candidate": record["candidate"], "fold": record["fold"], "seed": record["seed"], "artifact": record["artifact"],
                "calibration_ucb_feasible": selected is not None,
                "policy_name": None if selected is None else selected["policy_name"],
                "threshold": None if selected is None else selected["threshold"],
                "calibration_sensitivity": None if selected is None else selected["event_sensitivity"],
                "calibration_far_per_hour": None if selected is None else selected["false_alarms_per_hour"],
                "calibration_far_upper_bound_per_hour": None if selected is None else selected["calibration_far_upper_bound_per_hour"],
                "temporal_opened": temporal is not None,
                "temporal_sensitivity": None if temporal is None else temporal["event_sensitivity"],
                "temporal_far_per_hour": None if temporal is None else temporal["false_alarms_per_hour"],
                "temporal_far_target_met": None if temporal is None else temporal["temporal_far_target_met"],
                "temporal_far_one_sided_95_upper_bound_per_hour": None if temporal is None else temporal["temporal_far_one_sided_95_upper_bound_per_hour"],
            })


def _markdown(config: dict[str, Any], report: dict[str, Any]) -> str:
    lines = [
        "# V2.7 FAR-Robust Calibration Diagnostic",
        "",
        "## Scope",
        "",
        "C1 and H2 are replayed only on the already consumed V2.1 F00--F02",
        "development folds. Calibration selects a policy by a simultaneous one-sided",
        "exact Garwood FAR upper bound; temporal replay is performed once only when",
        "that calibration rule is feasible. Blocks 5 and 6 remain sealed.",
        "",
        "## Results",
        "",
        "| Candidate | Fold | Calibration UCB-feasible | Temporal opened | Temporal FAR passes | Temporal SEN (%) | Temporal FAR/h | Temporal 95% upper FAR/h |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in report["included_candidates"]:
        for fold in report["included_folds"]:
            summary = report["candidate_fold_summaries"][candidate][fold]
            sensitivity = summary["temporal_event_sensitivity"]
            far = summary["temporal_far_per_hour"]
            upper = summary["temporal_one_sided_95_upper_far_per_hour"]
            lines.append(
                "| {candidate} | F{fold} | {feasible}/{runs} | {opened}/{runs} | {passes}/{opened} | {sen} | {far} | {upper} |".format(
                    candidate=candidate, fold=fold, feasible=summary["calibration_ucb_feasible_runs"],
                    opened=summary["temporal_evaluations_opened"], passes=summary["temporal_far_target_passes"], runs=summary["runs"],
                    sen="NR" if sensitivity is None else f"{100 * sensitivity['mean']:.2f} +/- {100 * sensitivity['sample_std']:.2f}",
                    far="NR" if far is None else f"{far['mean']:.3f} +/- {far['sample_std']:.3f}",
                    upper="NR" if upper is None else f"{upper['mean']:.3f} +/- {upper['sample_std']:.3f}",
                )
            )
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "This diagnostic evaluates a conservative calibration rule after observing",
        "V2.6 transfer failure. It does not select a winning candidate, does not create",
        "a final operating policy, and cannot support a clinical or FPGA claim. The",
        "Poisson model and Bonferroni bound address count uncertainty and selection",
        "multiplicity, not temporal dependence or external generalization.",
        "",
        "## Integrity",
        "",
        f"- V2.7 config SHA-256: `{report['config_sha256']}`",
        f"- Source artifact config SHA-256: `{report['artifact_config_sha256']}`",
        f"- Simultaneous one-sided confidence: `{report['simultaneous_one_sided_confidence']:.10f}`",
        "- No model was trained, no new score stream was created, and no block-5/block-6 record was accessed.",
        "",
    ])
    return "\n".join(lines)


def build_far_robust_calibration_report(
    config_path: str | Path,
    artifact_config_path: str | Path,
    artifact_root: str | Path,
    run_root: str | Path,
    manifest_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate the V2.7 calibration-only rule over frozen C1/H2 artifacts."""
    config = load_json(config_path)
    validate_v27_far_robust_config(config)
    artifact_config, all_records, manifest_hashes = collect_v26_artifact_records(artifact_config_path, artifact_root)
    if list(artifact_config["consumed_development_folds"]) != list(config["allowed_folds"]):
        raise ValueError("V2.7 and source artifacts disagree on consumed folds")
    artifact_root, run_root, manifest_root, output_dir = Path(artifact_root), Path(run_root), Path(manifest_root), Path(output_dir)
    records = [record for record in all_records if record["candidate"] in config["candidate_comparators"]]
    if len(records) != len(config["candidate_comparators"]) * len(config["allowed_folds"]) * 5:
        raise ValueError("V2.7 source artifact count is incomplete")
    results = []
    for record in records:
        artifact_dir = artifact_root / record["artifact"]
        sweep = load_json(artifact_dir / "calibration_policy_sweep.json")
        selected, annotated_sweep = select_far_robust_calibration_policy(sweep, config)
        result = dict(record)
        result["calibration_sweep_operating_points"] = len(annotated_sweep)
        result["selected_calibration_policy"] = selected
        result["temporal_evaluation"] = None
        result["temporal_evaluation_status"] = "not_opened_no_simultaneous_ucb_feasible_policy"
        if selected is not None:
            temporal_scores, _ = _load_verified_temporal_scores(record, artifact_root, run_root, manifest_root, config)
            result["temporal_evaluation"] = _temporal_metric(temporal_scores, selected, config)
            result["temporal_evaluation_status"] = "opened_once_after_simultaneous_ucb_calibration_selection"
        results.append(result)
    target = float(config["evaluation"]["primary_far_per_hour"])
    summaries = {
        candidate: {
            fold: _summary([row for row in results if row["candidate"] == candidate and row["fold"] == fold], target)
            for fold in config["allowed_folds"]
        }
        for candidate in config["candidate_comparators"]
    }
    report = {
        "version": config["version"],
        "config_sha256": canonical_json_hash(config),
        "artifact_config_sha256": canonical_json_hash(artifact_config),
        "manifest_sha256_by_fold": manifest_hashes,
        "simultaneous_one_sided_confidence": simultaneous_one_sided_confidence(config),
        "scope": {
            "candidate_comparators": config["candidate_comparators"], "allowed_folds": config["allowed_folds"],
            "sealed_temporal_blocks": config["sealed_temporal_blocks"], "prohibited_actions": config["prohibited_actions"],
        },
        "runs": results,
        "included_candidates": config["candidate_comparators"],
        "included_folds": config["allowed_folds"],
        "candidate_fold_summaries": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "v27_far_robust_calibration.json", report)
    (output_dir / "v27_far_robust_calibration.md").write_text(_markdown(config, report), encoding="utf-8")
    _write_csv(output_dir, results)
    return report
