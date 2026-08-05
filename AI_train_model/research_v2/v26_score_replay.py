"""Replay-only V2.6 score diagnostics for already consumed temporal folds.

The temporal oracle in this module is a counterfactual diagnostic. It is never
a policy-selection mechanism and cannot authorize another candidate, final
training, quantization, tensor export, or block-5/block-6 access.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .protocol import canonical_json_hash, file_sha256, load_json, save_json
from .v26_diagnostics import collect_v26_artifact_records


def validate_v26_score_replay_config(config: dict[str, Any]) -> None:
    """Validate the replay-only boundary separately from model protocols."""
    if config.get("version") != "v2.6.1-score-replay-diagnostic-only":
        raise ValueError("V2.6 score replay requires version v2.6.1-score-replay-diagnostic-only")
    if list(config.get("allowed_folds", [])) != ["00", "01", "02"]:
        raise ValueError("V2.6 score replay is restricted to consumed folds 00, 01, and 02")
    if list(config.get("sealed_temporal_blocks", [])) != [5, 6]:
        raise ValueError("V2.6 score replay must keep blocks 5 and 6 sealed")
    if set(config.get("candidate_score_subdirectories", {})) != {"C1", "H2", "G1"}:
        raise ValueError("V2.6 score replay requires C1, H2, and G1 score subdirectories")
    preprocessing = config.get("preprocessing", {})
    if (
        int(preprocessing.get("sample_rate_hz", 0)) != 256
        or float(preprocessing.get("window_sec", 0.0)) != 5.0
        or float(preprocessing.get("stride_sec", 0.0)) != 1.0
        or preprocessing.get("filter_mode") != "causal_iir"
        or preprocessing.get("normalization") != "train_channel_zscore"
    ):
        raise ValueError("V2.6 score replay must preserve the V2.1 causal 5 s / 1 s contract")
    evaluation = config.get("evaluation", {})
    if float(evaluation.get("primary_far_per_hour", 0.0)) != 0.5 or int(evaluation.get("refractory_sec", -1)) != 30:
        raise ValueError("V2.6 score replay must preserve FAR <= 0.5/h and 30-second refractory")
    threshold = evaluation.get("threshold_grid", {})
    if threshold != {"minimum": 0.85, "maximum": 0.999, "step": 0.001}:
        raise ValueError("V2.6 score replay must preserve the declared threshold grid")
    if evaluation.get("temporal_policies") != [[3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16], [9, 18], [10, 20]]:
        raise ValueError("V2.6 score replay must preserve the declared temporal policy grid")
    prohibited = set(config.get("prohibited_actions", []))
    for action in ("model_training", "threshold_selection", "candidate_selection", "block_5_access", "block_6_access", "tensor_export"):
        if action not in prohibited:
            raise ValueError(f"V2.6 score replay must prohibit {action}")
    score_cache = config.get("score_cache", {})
    if score_cache.get("default_mode") != "reuse_verified_existing_run_scores_only":
        raise ValueError("V2.6 score replay must reuse verified existing score streams by default")
    if score_cache.get("rescore_missing_requires_explicit_flag") is not True:
        raise ValueError("V2.6 score replay must require an explicit rescore flag")
    if int(score_cache.get("batch_size", 0)) < 1:
        raise ValueError("V2.6 score replay cache batch size must be positive")


def _score_preprocessing(config: dict[str, Any]) -> dict[str, float | str]:
    source = config["preprocessing"]
    return {
        "sample_rate_hz": int(source["sample_rate_hz"]),
        "window_sec": float(source["window_sec"]),
        "stride_sec": float(source["stride_sec"]),
        "bandpass_low_hz": float(source["bandpass_low_hz"]),
        "bandpass_high_hz": float(source["bandpass_high_hz"]),
        "notch_hz": float(source["notch_hz"]),
        "filter_mode": source["filter_mode"],
    }


def _policy_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    evaluation = config["evaluation"]
    threshold = evaluation["threshold_grid"]
    values = np.arange(
        float(threshold["minimum"]),
        float(threshold["maximum"]) + float(threshold["step"]) / 2,
        float(threshold["step"]),
    )
    return [
        {
            "policy_name": f"{positive}_of_{decision}",
            "positive_windows": int(positive),
            "decision_window_windows": int(decision),
            "threshold": float(value),
        }
        for positive, decision in evaluation["temporal_policies"]
        for value in values
    ]


def _expected_records(scores: dict[str, Any], rows: list[dict[str, str]], label: str) -> None:
    expected = [row["recording_id"] for row in rows]
    observed = [record["recording_id"] for record in scores["records"]]
    if observed != expected:
        raise ValueError(f"{label} score stream recordings differ from the consumed manifest")


def _consumed_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    from .v21_evaluation import load_manifest_rows

    rows = load_manifest_rows(manifest_path, split)
    for row in rows:
        if int(row["temporal_block"]) >= 5:
            raise ValueError(f"V2.6 score replay refuses block {row['temporal_block']} in {split}")
    return rows


def _compare_evaluation_inputs(artifact_confirmation: dict[str, Any], sidecar_confirmation: dict[str, Any], run_id: str) -> None:
    expected = artifact_confirmation.get("evaluation_inputs", {})
    observed = sidecar_confirmation.get("evaluation_inputs", {})
    for key in ("checkpoint_sha256", "manifest_sha256", "scaler_mean_sha256", "scaler_scale_sha256"):
        if expected.get(key) != observed.get(key):
            raise ValueError(f"Existing score sidecar does not match packaged artifact for {run_id}: {key}")


def _rescore_pair(
    artifact_dir: Path,
    calibration_rows: list[dict[str, str]],
    temporal_rows: list[dict[str, str]],
    cache_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Explicit fallback only; model is loaded from the immutable artifact copy."""
    import numpy as np
    import torch

    from src.event_evaluation import load_scores, save_scores, score_continuous_recordings
    from src.model import build_model_from_run

    cache_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = cache_dir / "continuous_calibration_scores.npz"
    temporal_path = cache_dir / "continuous_temporal_eval_scores.npz"
    metadata_path = cache_dir / "score_cache_metadata.json"
    artifact_provenance = load_json(artifact_dir / "provenance.json")
    if calibration_path.exists() and temporal_path.exists() and metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata.get("checkpoint_sha256") != artifact_provenance["checkpoint_sha256"]:
            raise ValueError(f"Diagnostic score cache checkpoint mismatch: {cache_dir}")
        if metadata.get("config_sha256") != canonical_json_hash(config):
            raise ValueError(f"Diagnostic score cache config mismatch: {cache_dir}")
        if metadata.get("calibration_manifest_sha256") != file_sha256_from_rows(calibration_rows):
            raise ValueError(f"Diagnostic calibration cache manifest mismatch: {cache_dir}")
        if metadata.get("temporal_manifest_sha256") != file_sha256_from_rows(temporal_rows):
            raise ValueError(f"Diagnostic temporal cache manifest mismatch: {cache_dir}")
        calibration, temporal = load_scores(calibration_path), load_scores(temporal_path)
        _expected_records(calibration, calibration_rows, "Cached calibration")
        _expected_records(temporal, temporal_rows, "Cached temporal")
        return calibration, temporal, "v26_diagnostic_cache"
    if calibration_path.exists() or temporal_path.exists() or metadata_path.exists():
        raise RuntimeError(f"Refusing incomplete V2.6 diagnostic score cache: {cache_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_run(artifact_dir).to(device)
    model.load_state_dict(torch.load(artifact_dir / "best_model.pth", map_location=device, weights_only=True))
    scaler_mean = np.load(artifact_dir / "scaler_mean.npy")
    scaler_scale = np.load(artifact_dir / "scaler_scale.npy")
    preprocessing = _score_preprocessing(config)
    use_amp = device.type == "cuda"
    batch_size = int(config["score_cache"]["batch_size"])
    calibration = score_continuous_recordings(
        model, device, calibration_rows, preprocessing, batch_size, use_amp, scaler_mean, scaler_scale,
        normalization_mode="train_channel_zscore",
    )
    temporal = score_continuous_recordings(
        model, device, temporal_rows, preprocessing, batch_size, use_amp, scaler_mean, scaler_scale,
        normalization_mode="train_channel_zscore",
    )
    save_scores(calibration_path, calibration)
    save_scores(temporal_path, temporal)
    save_json(metadata_path, {
        "checkpoint_sha256": artifact_provenance["checkpoint_sha256"],
        "source": "explicit_v26_rescore_from_packaged_artifact",
        "config_sha256": canonical_json_hash(config),
        "calibration_manifest_sha256": file_sha256_from_rows(calibration_rows),
        "temporal_manifest_sha256": file_sha256_from_rows(temporal_rows),
    })
    return calibration, temporal, "explicit_v26_rescore"


def file_sha256_from_rows(rows: list[dict[str, str]]) -> str:
    """Hash only the consumed rows to identify a local diagnostic cache."""
    return canonical_json_hash(rows)


def _load_or_rescore_pair(
    record: dict[str, Any],
    artifact_root: Path,
    run_root: Path,
    manifest_root: Path,
    score_cache_root: Path,
    config: dict[str, Any],
    rescore_missing: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    fold = record["fold"]
    run_id = record["artifact"]
    manifest_path = manifest_root / f"confirmation_f{fold}_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing consumed V2.1 manifest: {manifest_path}")
    if file_sha256(manifest_path) != record["manifest_sha256"]:
        raise ValueError(f"V2.6 manifest hash mismatch for {run_id}")
    calibration_rows = _consumed_rows(manifest_path, "val")
    temporal_rows = _consumed_rows(manifest_path, "temporal_eval")
    score_dir = run_root / run_id / config["candidate_score_subdirectories"][record["candidate"]]
    calibration_path = score_dir / "continuous_calibration_scores.npz"
    temporal_path = score_dir / "continuous_temporal_eval_scores.npz"
    sidecar_path = score_dir / "temporal_confirmation.json"
    if calibration_path.is_file() and temporal_path.is_file() and sidecar_path.is_file():
        from src.event_evaluation import load_scores

        _compare_evaluation_inputs(record["artifact_confirmation"], load_json(sidecar_path), run_id)
        calibration, temporal = load_scores(calibration_path), load_scores(temporal_path)
        _expected_records(calibration, calibration_rows, "Existing calibration")
        _expected_records(temporal, temporal_rows, "Existing temporal")
        return calibration, temporal, "verified_existing_run_scores"
    if not rescore_missing:
        raise FileNotFoundError(
            f"Missing verified existing score pair for {run_id}. Re-run with --rescore-missing only after approving explicit replay."
        )
    return _rescore_pair(
        artifact_root / run_id, calibration_rows, temporal_rows, score_cache_root / run_id, config,
    )


def inspect_score_replay_inventory(
    artifact_config_path: str | Path,
    score_replay_config_path: str | Path,
    artifact_root: str | Path,
    run_root: str | Path,
    manifest_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Check score-pair availability and sidecar provenance without loading NPZ data."""
    replay_config = load_json(score_replay_config_path)
    validate_v26_score_replay_config(replay_config)
    artifact_config, records, manifest_hashes = collect_v26_artifact_records(artifact_config_path, artifact_root)
    if list(artifact_config["consumed_development_folds"]) != list(replay_config["allowed_folds"]):
        raise ValueError("Artifact and score-replay configs disagree on the consumed folds")
    artifact_root, run_root, manifest_root, output_dir = (
        Path(artifact_root), Path(run_root), Path(manifest_root), Path(output_dir),
    )
    inventory = []
    for record in records:
        run_id, fold = record["artifact"], record["fold"]
        manifest_path = manifest_root / f"confirmation_f{fold}_manifest.csv"
        manifest_status = "missing"
        if manifest_path.is_file():
            if file_sha256(manifest_path) != record["manifest_sha256"]:
                manifest_status = "hash_mismatch"
            else:
                try:
                    _consumed_rows(manifest_path, "val")
                    _consumed_rows(manifest_path, "temporal_eval")
                    manifest_status = "verified"
                except (KeyError, ValueError):
                    manifest_status = "invalid_consumed_rows"
        score_dir = run_root / run_id / replay_config["candidate_score_subdirectories"][record["candidate"]]
        calibration_path = score_dir / "continuous_calibration_scores.npz"
        temporal_path = score_dir / "continuous_temporal_eval_scores.npz"
        sidecar_path = score_dir / "temporal_confirmation.json"
        score_files_present = calibration_path.is_file() and temporal_path.is_file()
        sidecar_status = "missing"
        if sidecar_path.is_file():
            try:
                confirmation = load_json(artifact_root / run_id / "temporal_confirmation.json")
                _compare_evaluation_inputs(confirmation, load_json(sidecar_path), run_id)
                sidecar_status = "verified"
            except (ValueError, KeyError, json.JSONDecodeError):
                sidecar_status = "mismatch"
        inventory.append({
            "candidate": record["candidate"], "fold": fold, "seed": record["seed"], "artifact": run_id,
            "manifest_status": manifest_status,
            "calibration_score_present": calibration_path.is_file(),
            "temporal_score_present": temporal_path.is_file(),
            "sidecar_status": sidecar_status,
            "replay_ready": manifest_status == "verified" and score_files_present and sidecar_status == "verified",
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "version": replay_config["version"],
        "artifact_config_sha256": canonical_json_hash(artifact_config),
        "score_replay_config_sha256": canonical_json_hash(replay_config),
        "manifest_sha256_by_fold": manifest_hashes,
        "runs": inventory,
        "ready_runs": sum(row["replay_ready"] for row in inventory),
        "missing_or_invalid_runs": sum(not row["replay_ready"] for row in inventory),
        "scope": {"allowed_folds": replay_config["allowed_folds"], "sealed_temporal_blocks": replay_config["sealed_temporal_blocks"]},
        "limit": "This inventory does not load score arrays, replay metrics, score EEG, or access blocks 5 and 6.",
    }
    save_json(output_dir / "v26_score_replay_inventory.json", result)
    fields = list(inventory[0].keys())
    with (output_dir / "v26_score_replay_inventory.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    return result


def _filter_records(records: list[dict[str, Any]], artifact_ids: list[str] | None) -> list[dict[str, Any]]:
    if not artifact_ids:
        return records
    requested = set(artifact_ids)
    known = {record["artifact"] for record in records}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Requested V2.6 score-replay artifacts are not in the frozen set: {sorted(unknown)}")
    return [record for record in records if record["artifact"] in requested]


def _metric(scores: dict[str, Any], policy: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    from src.event_evaluation import event_metrics

    preprocessing = config["preprocessing"]
    evaluation = config["evaluation"]
    return event_metrics(
        scores, float(policy["threshold"]), int(preprocessing["sample_rate_hz"]), float(preprocessing["window_sec"]),
        int(evaluation["refractory_sec"]), int(policy["positive_windows"]), int(policy["decision_window_windows"]),
        policy["policy_name"],
    )


def _temporal_oracle(scores: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    target = float(config["evaluation"]["primary_far_per_hour"])
    sweep = [_metric(scores, policy, config) for policy in _policy_grid(config)]
    eligible = [metric for metric in sweep if metric["false_alarms_per_hour"] <= target]
    minimum_far = min(sweep, key=lambda metric: (metric["false_alarms_per_hour"], -metric["event_sensitivity"]))
    if not eligible:
        return {
            "target_feasible": False,
            "best_sensitivity_at_target": None,
            "minimum_far_policy": minimum_far,
        }
    best = max(eligible, key=lambda metric: (
        metric["event_sensitivity"],
        -(metric["median_detection_delay_sec"] if metric["median_detection_delay_sec"] is not None else float("inf")),
        -metric["false_alarms_per_hour"],
    ))
    return {
        "target_feasible": True,
        "best_sensitivity_at_target": best,
        "minimum_far_policy": minimum_far,
    }


def _assert_selected_replay_matches_artifact(record: dict[str, Any], temporal_scores: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    replayed = _metric(temporal_scores, record["calibration"], config)
    expected = record["temporal"]
    for field in ("detected_events", "total_events", "false_alarms"):
        if int(replayed[field]) != int(expected[field]):
            raise ValueError(f"Selected-policy replay mismatch for {record['artifact']}: {field}")
    for field in ("event_sensitivity", "false_alarms_per_hour"):
        if not np.isclose(float(replayed[field]), float(expected[field]), rtol=0.0, atol=1e-12):
            raise ValueError(f"Selected-policy replay mismatch for {record['artifact']}: {field}")
    return replayed


def _sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _summarize(records: list[dict[str, Any]], target: float) -> dict[str, Any]:
    selected_far = [float(record["selected_replay"]["false_alarms_per_hour"]) for record in records]
    selected_sensitivity = [float(record["selected_replay"]["event_sensitivity"]) for record in records]
    feasible = [record for record in records if record["temporal_oracle"]["target_feasible"]]
    oracle_far = [float(record["temporal_oracle"]["best_sensitivity_at_target"]["false_alarms_per_hour"]) for record in feasible]
    oracle_sensitivity = [float(record["temporal_oracle"]["best_sensitivity_at_target"]["event_sensitivity"]) for record in feasible]
    gaps = [
        float(record["temporal_oracle"]["best_sensitivity_at_target"]["event_sensitivity"])
        - float(record["selected_replay"]["event_sensitivity"])
        for record in feasible
    ]
    selected_passes = sum(value <= target for value in selected_far)
    if not feasible:
        diagnosis = "representation_limited_at_declared_grid_for_all_replayed_runs"
    elif len(feasible) > selected_passes:
        diagnosis = "calibration_to_temporal_policy_mismatch_possible_but_not_proven"
    else:
        diagnosis = "temporal_target_feasibility_matches_calibration_selected_policy_for_replayed_runs"
    return {
        "runs": len(records),
        "selected_policy_temporal_far_passes": selected_passes,
        "temporal_oracle_target_feasible_runs": len(feasible),
        "selected_policy_temporal_far": {"mean": _round(mean(selected_far)), "sample_std": _round(_sample_std(selected_far))},
        "selected_policy_temporal_sensitivity": {"mean": _round(mean(selected_sensitivity)), "sample_std": _round(_sample_std(selected_sensitivity))},
        "oracle_sensitivity_at_target": None if not oracle_sensitivity else {"mean": _round(mean(oracle_sensitivity)), "sample_std": _round(_sample_std(oracle_sensitivity))},
        "oracle_far_at_target": None if not oracle_far else {"mean": _round(mean(oracle_far)), "sample_std": _round(_sample_std(oracle_far))},
        "oracle_minus_selected_sensitivity": None if not gaps else {"mean": _round(mean(gaps)), "sample_std": _round(_sample_std(gaps))},
        "diagnostic_status": diagnosis,
    }


def _write_csv(output_dir: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "candidate", "fold", "seed", "artifact", "score_source", "selected_policy", "selected_threshold",
        "selected_temporal_sensitivity", "selected_temporal_far_per_hour", "oracle_target_feasible",
        "oracle_policy", "oracle_threshold", "oracle_temporal_sensitivity", "oracle_temporal_far_per_hour",
        "minimum_far_policy", "minimum_far_threshold", "minimum_temporal_far_per_hour",
    ]
    with (output_dir / "v26_score_replay_runs.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for record in records:
            oracle = record["temporal_oracle"]
            best = oracle["best_sensitivity_at_target"]
            minimum = oracle["minimum_far_policy"]
            writer.writerow({
                "candidate": record["candidate"], "fold": record["fold"], "seed": record["seed"],
                "artifact": record["artifact"], "score_source": record["score_source"],
                "selected_policy": record["calibration"]["policy_name"], "selected_threshold": record["calibration"]["threshold"],
                "selected_temporal_sensitivity": record["selected_replay"]["event_sensitivity"],
                "selected_temporal_far_per_hour": record["selected_replay"]["false_alarms_per_hour"],
                "oracle_target_feasible": oracle["target_feasible"],
                "oracle_policy": None if best is None else best["policy_name"],
                "oracle_threshold": None if best is None else best["threshold"],
                "oracle_temporal_sensitivity": None if best is None else best["event_sensitivity"],
                "oracle_temporal_far_per_hour": None if best is None else best["false_alarms_per_hour"],
                "minimum_far_policy": minimum["policy_name"], "minimum_far_threshold": minimum["threshold"],
                "minimum_temporal_far_per_hour": minimum["false_alarms_per_hour"],
            })


def _markdown(config: dict[str, Any], report: dict[str, Any]) -> str:
    target = config["evaluation"]["primary_far_per_hour"]
    lines = [
        "# V2.6 Score-Replay Diagnostic",
        "",
        "## Scope",
        "",
        "This is a counterfactual analysis of score streams created during already",
        "consumed F00--F02 replays. The temporal oracle uses future labels only to",
        "characterize score separability at the declared FAR target. It does not select",
        "a threshold, policy, candidate, or final model. Blocks 5 and 6 remain sealed.",
        "",
        "## Selected Policy Versus Temporal Oracle",
        "",
        "| Candidate | Fold | Selected-policy FAR passes | Oracle FAR-feasible runs | Selected SEN (%) | Selected FAR/h | Oracle SEN at target (%) | Oracle FAR/h | Diagnostic status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for candidate in report["included_candidates"]:
        for fold in report["included_folds"]:
            summary = report["candidate_fold_summaries"][candidate][fold]
            oracle_sensitivity = summary["oracle_sensitivity_at_target"]
            oracle_far = summary["oracle_far_at_target"]
            lines.append(
                "| {candidate} | F{fold} | {selected}/{runs} | {oracle}/{runs} | {sen:.2f} +/- {sen_sd:.2f} | "
                "{far:.3f} +/- {far_sd:.3f} | {oracle_sen} | {oracle_far} | {status} |".format(
                    candidate=candidate, fold=fold,
                    selected=summary["selected_policy_temporal_far_passes"], oracle=summary["temporal_oracle_target_feasible_runs"], runs=summary["runs"],
                    sen=100 * summary["selected_policy_temporal_sensitivity"]["mean"], sen_sd=100 * summary["selected_policy_temporal_sensitivity"]["sample_std"],
                    far=summary["selected_policy_temporal_far"]["mean"], far_sd=summary["selected_policy_temporal_far"]["sample_std"],
                    oracle_sen="NR" if oracle_sensitivity is None else f"{100 * oracle_sensitivity['mean']:.2f} +/- {100 * oracle_sensitivity['sample_std']:.2f}",
                    oracle_far="NR" if oracle_far is None else f"{oracle_far['mean']:.3f} +/- {oracle_far['sample_std']:.3f}",
                    status=summary["diagnostic_status"],
                )
            )
    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        f"- A selected-policy pass means the calibration-selected policy replayed at FAR <= {target:.1f}/h.",
        "- An oracle-feasible run means at least one *counterfactual* policy in the",
        "  unchanged grid would have met the target on that future block. It cannot be",
        "  deployed retrospectively and must not be used to choose a replacement policy.",
        "- If the oracle is infeasible, the fixed score stream has no operating point in",
        "  the declared grid that reaches the target; this is evidence against score",
        "  separability at that grid, not a proof of a physiological cause.",
        "",
        "## Integrity",
        "",
        f"- Artifact-diagnostic config SHA-256: `{report['artifact_config_sha256']}`",
        f"- Score-replay config SHA-256: `{report['score_replay_config_sha256']}`",
        f"- Replayed run records: `{len(report['runs'])}`",
        "- No model was trained and no block-5/block-6 recording was scored.",
        "",
    ])
    return "\n".join(lines)


def build_score_replay_atlas(
    artifact_config_path: str | Path,
    score_replay_config_path: str | Path,
    artifact_root: str | Path,
    run_root: str | Path,
    manifest_root: str | Path,
    score_cache_root: str | Path,
    output_dir: str | Path,
    rescore_missing: bool = False,
    artifact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Produce the V2.6 counterfactual score report without selecting an intervention."""
    replay_config = load_json(score_replay_config_path)
    validate_v26_score_replay_config(replay_config)
    artifact_config, records, manifest_hashes = collect_v26_artifact_records(artifact_config_path, artifact_root)
    if list(artifact_config["consumed_development_folds"]) != list(replay_config["allowed_folds"]):
        raise ValueError("Artifact and score-replay configs disagree on the consumed folds")
    artifact_root, run_root = Path(artifact_root), Path(run_root)
    manifest_root, score_cache_root, output_dir = Path(manifest_root), Path(score_cache_root), Path(output_dir)
    selected_records = _filter_records(records, artifact_ids)
    scored = []
    for record in selected_records:
        record = dict(record)
        record["artifact_confirmation"] = load_json(artifact_root / record["artifact"] / "temporal_confirmation.json")
        calibration, temporal, source = _load_or_rescore_pair(
            record, artifact_root, run_root, manifest_root, score_cache_root, replay_config, rescore_missing,
        )
        _expected_records(calibration, _consumed_rows(manifest_root / f"confirmation_f{record['fold']}_manifest.csv", "val"), "Calibration")
        _expected_records(temporal, _consumed_rows(manifest_root / f"confirmation_f{record['fold']}_manifest.csv", "temporal_eval"), "Temporal")
        record["selected_replay"] = _assert_selected_replay_matches_artifact(record, temporal, replay_config)
        record["temporal_oracle"] = _temporal_oracle(temporal, replay_config)
        record["score_source"] = source
        record.pop("artifact_confirmation")
        scored.append(record)
    target = float(replay_config["evaluation"]["primary_far_per_hour"])
    included_candidates = [candidate for candidate in ("C1", "H2", "G1") if any(record["candidate"] == candidate for record in scored)]
    included_folds = [fold for fold in replay_config["allowed_folds"] if any(record["fold"] == fold for record in scored)]
    summaries = {
        candidate: {
            fold: _summarize(
                [record for record in scored if record["candidate"] == candidate and record["fold"] == fold], target,
            )
            for fold in included_folds
        }
        for candidate in included_candidates
    }
    report = {
        "version": replay_config["version"],
        "artifact_config_sha256": canonical_json_hash(artifact_config),
        "score_replay_config_sha256": canonical_json_hash(replay_config),
        "manifest_sha256_by_fold": manifest_hashes,
        "scope": {
            "allowed_folds": replay_config["allowed_folds"],
            "sealed_temporal_blocks": replay_config["sealed_temporal_blocks"],
            "prohibited_actions": replay_config["prohibited_actions"],
            "rescore_missing": bool(rescore_missing),
            "artifact_filter": sorted(artifact_ids or []),
        },
        "runs": scored,
        "included_candidates": included_candidates,
        "included_folds": included_folds,
        "candidate_fold_summaries": summaries,
        "oracle_warning": replay_config["evaluation"]["oracle_definition"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "v26_score_replay_atlas.json", report)
    (output_dir / "v26_score_replay_atlas.md").write_text(_markdown(replay_config, report), encoding="utf-8")
    _write_csv(output_dir, scored)
    return report
