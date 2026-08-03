"""Calibration-separated V2.1 temporal evaluation utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.event_evaluation import event_metrics, load_scores, save_scores, score_continuous_recordings
from src.model import build_model_from_run
from .protocol import canonical_json_hash, file_sha256
from .statistics import patient_group_cluster_bootstrap, poisson_exact_far_interval


def load_manifest_rows(path: str | Path, split: str) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} rows in {path}")
    return rows


def v21_preprocessing(config: dict) -> dict:
    return {
        "sample_rate_hz": int(config["dataset"]["sample_rate_hz"]),
        "window_sec": float(config["preprocessing"]["window_sec"]),
        "stride_sec": float(config["preprocessing"]["stride_sec"]),
        "bandpass_low_hz": float(config["preprocessing"]["bandpass_hz"][0]),
        "bandpass_high_hz": float(config["preprocessing"]["bandpass_hz"][1]),
        "notch_hz": float(config["preprocessing"]["notch_hz"]),
        "filter_mode": config["preprocessing"]["filter_mode"],
    }


def policy_grid(config: dict) -> list[dict]:
    policies = []
    for positives, decision in config["evaluation"]["temporal_policies"]:
        policies.append({"name": f"{positives}_of_{decision}", "positive_windows": int(positives), "decision_window_windows": int(decision)})
    threshold = config["evaluation"]["threshold_grid"]
    thresholds = np.arange(float(threshold["minimum"]), float(threshold["maximum"]) + float(threshold["step"]) / 2, float(threshold["step"]))
    return [{**policy, "threshold": float(value)} for policy in policies for value in thresholds]


def select_calibration_policy(scores: dict, config: dict) -> tuple[dict, list[dict]]:
    preprocessing = v21_preprocessing(config)
    evaluation = config["evaluation"]
    metrics = []
    for candidate in policy_grid(config):
        result = event_metrics(
            scores, candidate["threshold"], preprocessing["sample_rate_hz"], preprocessing["window_sec"],
            evaluation["refractory_sec"], candidate["positive_windows"], candidate["decision_window_windows"], candidate["name"],
        )
        result["calibration_far_target_met"] = result["false_alarms_per_hour"] <= float(evaluation["primary_far_per_hour"])
        metrics.append(result)
    eligible = [result for result in metrics if result["calibration_far_target_met"]]
    if not eligible:
        raise RuntimeError("No predeclared V2.1 threshold/policy satisfies calibration FAR <= 0.5/h")
    selected = max(eligible, key=lambda value: (
        value["event_sensitivity"], -(value["median_detection_delay_sec"] if value["median_detection_delay_sec"] is not None else float("inf")), -value["false_alarms_per_hour"],
    ))
    return selected, metrics


def _single_record_scores(scores: dict, record_index: int) -> dict:
    """Create a one-record score view so the legacy metric stays the authority."""
    start = int(scores["record_offsets"][record_index])
    end = int(scores["record_offsets"][record_index + 1])
    return {
        "probabilities": scores["probabilities"][start:end],
        "record_indices": np.zeros(end - start, dtype=np.int32),
        "start_samples": scores["start_samples"][start:end],
        "records": [scores["records"][record_index]],
        "record_offsets": np.asarray([0, end - start], dtype=np.int64),
    }


def patient_group_uncertainty(scores: dict, rows: list[dict], policy: dict, config: dict) -> dict:
    """Compute group-clustered CIs without treating sessions/windows as IID."""
    by_recording = {row["recording_id"]: row["patient_group"] for row in rows}
    sensitivity: dict[str, list[int]] = {}
    far: dict[str, list[float]] = {}
    preprocessing = v21_preprocessing(config)
    for record_index, record in enumerate(scores["records"]):
        group = by_recording.get(record["recording_id"])
        if group is None:
            raise ValueError(f"Continuous score recording is absent from its V2.1 manifest: {record['recording_id']}")
        metric = event_metrics(
            _single_record_scores(scores, record_index), policy["threshold"], preprocessing["sample_rate_hz"],
            preprocessing["window_sec"], config["evaluation"]["refractory_sec"],
            policy["positive_windows"], policy["decision_window_windows"], policy["policy_name"],
        )
        event_values = sensitivity.setdefault(group, [0, 0])
        event_values[0] += int(metric["detected_events"])
        event_values[1] += int(metric["total_events"])
        far_values = far.setdefault(group, [0.0, 0.0])
        far_values[0] += int(metric["false_alarms"])
        far_values[1] += float(metric["interictal_hours"])

    seizure_group_sensitivity = {
        group: (values[0], values[1]) for group, values in sensitivity.items() if values[1] > 0
    }
    sensitivity_ci = _cluster_interval_or_unavailable(seizure_group_sensitivity)
    far_ci = patient_group_cluster_bootstrap(
        {group: (int(values[0]), values[1]) for group, values in far.items()}
    )
    total_false_alarms = sum(int(values[0]) for values in far.values())
    total_hours = sum(float(values[1]) for values in far.values())
    poisson_ci = poisson_exact_far_interval(total_false_alarms, total_hours)
    return {
        "independent_patient_groups": len(sensitivity),
        "seizure_contributing_patient_groups": len(seizure_group_sensitivity),
        "event_sensitivity_cluster_bootstrap_95ci": sensitivity_ci,
        "far_per_hour_cluster_bootstrap_95ci": far_ci.__dict__,
        "far_per_hour_poisson_exact_95ci": poisson_ci.__dict__,
        "per_patient_group": {
            group: {
                "detected_events": sensitivity[group][0], "total_events": sensitivity[group][1],
                "false_alarms": int(far[group][0]), "interictal_hours": far[group][1],
            }
            for group in sorted(sensitivity)
        },
    }


def _cluster_interval_or_unavailable(contributions: dict[str, tuple[int, int]]) -> dict:
    """Event sensitivity is undefined for a patient group with zero events."""
    if len(contributions) < 2:
        return {
            "estimable": False,
            "reason": "fewer_than_two_seizure_contributing_patient_groups_in_partition",
            "contributing_patient_groups": len(contributions),
        }
    return {"estimable": True, **patient_group_cluster_bootstrap(contributions).__dict__}


def _load_recovery_scores(path: Path, rows: list[dict]) -> dict:
    scores = load_scores(path)
    expected = [row["recording_id"] for row in rows]
    observed = [record["recording_id"] for record in scores["records"]]
    if observed != expected:
        raise ValueError(f"Recovery scores do not match the V2.1 manifest: {path}")
    return scores


def score_and_evaluate_run(
    run_dir: str | Path,
    prepared_dir: str | Path,
    manifest_path: str | Path,
    config: dict,
    output_dir: str | Path,
    use_amp: bool = False,
    batch_size: int = 128,
    reuse_existing_scores: bool = False,
) -> dict:
    """Select policy on calibration recordings and apply it once to temporal_eval recordings."""
    import torch

    run_dir, prepared_dir, output_dir = Path(run_dir), Path(prepared_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_run(run_dir).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    scaler_mean = np.load(run_dir / "scaler_mean.npy")
    scaler_scale = np.load(run_dir / "scaler_scale.npy")
    preprocessing = v21_preprocessing(config)
    calibration_rows = load_manifest_rows(manifest_path, "val")
    temporal_rows = load_manifest_rows(manifest_path, "temporal_eval")
    calibration_path = output_dir / "continuous_calibration_scores.npz"
    temporal_path = output_dir / "continuous_temporal_eval_scores.npz"
    if reuse_existing_scores:
        calibration_scores = _load_recovery_scores(calibration_path, calibration_rows)
        temporal_scores = _load_recovery_scores(temporal_path, temporal_rows)
    else:
        calibration_scores = score_continuous_recordings(
            model, device, calibration_rows, preprocessing, batch_size, use_amp, scaler_mean, scaler_scale,
            normalization_mode="train_channel_zscore",
        )
        save_scores(calibration_path, calibration_scores)
    selected, sweep = select_calibration_policy(calibration_scores, config)
    if not reuse_existing_scores:
        temporal_scores = score_continuous_recordings(
            model, device, temporal_rows, preprocessing, batch_size, use_amp, scaler_mean, scaler_scale,
            normalization_mode="train_channel_zscore",
        )
        save_scores(temporal_path, temporal_scores)
    temporal_metrics = event_metrics(
        temporal_scores, selected["threshold"], preprocessing["sample_rate_hz"], preprocessing["window_sec"],
        config["evaluation"]["refractory_sec"], selected["positive_windows"], selected["decision_window_windows"], selected["policy_name"],
    )
    temporal_uncertainty = patient_group_uncertainty(temporal_scores, temporal_rows, selected, config)
    payload = {
        "selection_split": "calibration", "evaluation_split": "temporal_eval",
        "selection_rule": config["evaluation"]["calibration_rule"], "selected_calibration_policy": selected,
        "temporal_evaluation": temporal_metrics, "calibration_recordings": len(calibration_rows),
        "temporal_evaluation_recordings": len(temporal_rows), "temporal_uncertainty": temporal_uncertainty,
        "reused_existing_scores": bool(reuse_existing_scores),
        "evaluation_inputs": {
            "protocol_hash": canonical_json_hash(config), "manifest_sha256": file_sha256(manifest_path),
            "checkpoint_sha256": file_sha256(run_dir / "best_model.pth"),
            "scaler_mean_sha256": file_sha256(run_dir / "scaler_mean.npy"),
            "scaler_scale_sha256": file_sha256(run_dir / "scaler_scale.npy"),
        },
    }
    with (output_dir / "temporal_confirmation.json").open("w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, sort_keys=True)
        target.write("\n")
    with (output_dir / "calibration_policy_sweep.json").open("w", encoding="utf-8") as target:
        json.dump(sweep, target, indent=2, sort_keys=True)
        target.write("\n")
    return payload
