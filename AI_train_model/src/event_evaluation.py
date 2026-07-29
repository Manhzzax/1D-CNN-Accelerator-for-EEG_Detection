"""Continuous recording event evaluation for the locked CHB-MIT protocol."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .chbmit_preparation import (
    extract_canonical_bipolar_data,
    filter_eeg,
    intervals_to_samples,
)


def load_split_recordings(protocol_dir, split_name):
    path = Path(protocol_dir) / "recording_split_manifest.csv"
    with path.open("r", newline="", encoding="utf-8") as input_file:
        rows = [row for row in csv.DictReader(input_file) if row["split"] == split_name]
    if not rows:
        raise ValueError(f"No {split_name} recordings in {path}")
    return rows


def score_continuous_recordings(
    model, device, rows, preprocessing, batch_size, use_amp, scaler_mean, scaler_std,
    normalization_mode="train_channel_zscore", recording_normalization=None,
):
    """Run model probabilities for every non-overlapping window in each recording."""
    import mne

    sample_rate = preprocessing["sample_rate_hz"]
    window_samples = int(preprocessing["window_sec"] * sample_rate)
    stride_samples = int(preprocessing["stride_sec"] * sample_rate)
    model.eval()
    all_probabilities = []
    all_record_indices = []
    all_start_samples = []
    record_metadata = []
    record_offsets = [0]

    for record_index, row in enumerate(rows):
        raw = mne.io.read_raw_edf(row["edf_path"], preload=True, verbose="ERROR")
        try:
            if int(round(raw.info["sfreq"])) != sample_rate:
                raise ValueError(f"Unexpected sample rate in {row['recording_id']}: {raw.info['sfreq']}")
            data = extract_canonical_bipolar_data(raw)
        finally:
            raw.close()

        data = filter_eeg(
            data,
            sample_rate,
            preprocessing["bandpass_low_hz"],
            preprocessing["bandpass_high_hz"],
            preprocessing["notch_hz"],
        )
        if normalization_mode == "train_channel_zscore":
            data = (data - scaler_mean[:, None]) / scaler_std[:, None]
        elif normalization_mode == "per_recording_zscore":
            if recording_normalization is None or row["recording_id"] not in recording_normalization:
                raise ValueError(f"Missing saved normalization statistics for {row['recording_id']}")
            stats = recording_normalization[row["recording_id"]]
            recording_mean = np.asarray(stats["mean"], dtype=np.float32)[:, None]
            recording_std = np.asarray(stats["std"], dtype=np.float32)[:, None]
            data = (data - recording_mean) / recording_std
        else:
            raise ValueError(f"Unsupported continuous normalization mode: {normalization_mode}")
        starts = np.arange(0, data.shape[1] - window_samples + 1, stride_samples, dtype=np.int64)
        probabilities = []
        with torch.no_grad():
            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start:batch_start + batch_size]
                batch = np.stack(
                    [data[:, start:start + window_samples] for start in batch_starts], axis=0
                )
                inputs = torch.from_numpy(batch).to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(inputs)
                else:
                    logits = model(inputs)
                probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

        record_probabilities = np.concatenate(probabilities)
        all_probabilities.append(record_probabilities)
        all_record_indices.append(np.full(len(starts), record_index, dtype=np.int32))
        all_start_samples.append(starts)
        record_metadata.append({
            "recording_id": row["recording_id"],
            "sample_count": int(data.shape[1]),
            "seizure_intervals": intervals_to_samples(
                json.loads(row["seizure_intervals_json"]), sample_rate
            ),
        })
        record_offsets.append(record_offsets[-1] + len(starts))
        print(f"  Continuous inference: {record_index + 1}/{len(rows)}")

    return {
        "probabilities": np.concatenate(all_probabilities),
        "record_indices": np.concatenate(all_record_indices),
        "start_samples": np.concatenate(all_start_samples),
        "records": record_metadata,
        "record_offsets": np.asarray(record_offsets, dtype=np.int64),
    }


def _validate_temporal_policy(positive_windows, decision_window_windows):
    if positive_windows < 1 or decision_window_windows < 1:
        raise ValueError("Temporal policy window counts must be positive")
    if positive_windows > decision_window_windows:
        raise ValueError("positive_windows cannot exceed decision_window_windows")


def generate_alarms(
    starts,
    probabilities,
    threshold,
    refractory_samples,
    positive_windows,
    decision_window_windows,
):
    """Confirm an alarm only when the temporal policy is satisfied."""
    _validate_temporal_policy(positive_windows, decision_window_windows)
    threshold_hits = probabilities >= threshold
    hit_count = 0
    alarms = []
    next_allowed = -1

    for index, start in enumerate(starts):
        hit_count += int(threshold_hits[index])
        if index >= decision_window_windows:
            hit_count -= int(threshold_hits[index - decision_window_windows])
        if index + 1 < decision_window_windows or hit_count < positive_windows:
            continue
        if start < next_allowed:
            continue
        alarms.append(int(start))
        next_allowed = int(start) + refractory_samples
    return alarms


def event_metrics(
    scores,
    threshold,
    sample_rate,
    window_sec,
    refractory_sec,
    positive_windows=1,
    decision_window_windows=1,
    policy_name="single_window",
):
    """Compute seizure-event sensitivity, false alarms/hour, and detection delay."""
    window_samples = int(window_sec * sample_rate)
    refractory_samples = int(refractory_sec * sample_rate)
    probabilities = scores["probabilities"]
    start_samples = scores["start_samples"]
    total_events = 0
    detected_events = 0
    false_alarms = 0
    interictal_seconds = 0.0
    delays = []

    for record_index, record in enumerate(scores["records"]):
        offset_start = scores["record_offsets"][record_index]
        offset_end = scores["record_offsets"][record_index + 1]
        record_starts = start_samples[offset_start:offset_end]
        record_probabilities = probabilities[offset_start:offset_end]
        alarms = generate_alarms(
            record_starts,
            record_probabilities,
            threshold,
            refractory_samples,
            positive_windows,
            decision_window_windows,
        )

        intervals = record["seizure_intervals"]
        total_events += len(intervals)
        interictal_seconds += (
            record["sample_count"] - sum(end - start for start, end in intervals)
        ) / sample_rate
        for seizure_start, seizure_end in intervals:
            event_alarms = [
                alarm for alarm in alarms
                if alarm < seizure_end and alarm + window_samples > seizure_start
            ]
            if event_alarms:
                detected_events += 1
                delays.append(max(0.0, (min(event_alarms) - seizure_start) / sample_rate))
        for alarm in alarms:
            if not any(alarm < end and alarm + window_samples > start for start, end in intervals):
                false_alarms += 1

    return {
        "policy_name": policy_name,
        "positive_windows": int(positive_windows),
        "decision_window_windows": int(decision_window_windows),
        "threshold": float(threshold),
        "event_sensitivity": float(detected_events / total_events) if total_events else 0.0,
        "detected_events": detected_events,
        "total_events": total_events,
        "false_alarms": false_alarms,
        "false_alarms_per_hour": float(false_alarms / (interictal_seconds / 3600.0)),
        "median_detection_delay_sec": float(np.median(delays)) if delays else None,
        "mean_detection_delay_sec": float(np.mean(delays)) if delays else None,
        "interictal_hours": float(interictal_seconds / 3600.0),
    }


def _temporal_policies(evaluation):
    policies = evaluation.get(
        "temporal_policies",
        [{"name": "single_window", "positive_windows": 1, "decision_window_windows": 1}],
    )
    if not policies:
        raise ValueError("At least one temporal policy is required")
    normalized = []
    for policy in policies:
        name = str(policy["name"])
        positive_windows = int(policy["positive_windows"])
        decision_window_windows = int(policy["decision_window_windows"])
        _validate_temporal_policy(positive_windows, decision_window_windows)
        normalized.append({
            "name": name,
            "positive_windows": positive_windows,
            "decision_window_windows": decision_window_windows,
        })
    return normalized


def choose_threshold(validation_scores, preprocessing, evaluation):
    thresholds = np.arange(
        evaluation["threshold_min"],
        evaluation["threshold_max"] + evaluation["threshold_step"] / 2,
        evaluation["threshold_step"],
    )
    metrics = []
    for policy in _temporal_policies(evaluation):
        for threshold in thresholds:
            metrics.append(event_metrics(
                validation_scores,
                threshold,
                preprocessing["sample_rate_hz"],
                preprocessing["window_sec"],
                evaluation["refractory_sec"],
                positive_windows=policy["positive_windows"],
                decision_window_windows=policy["decision_window_windows"],
                policy_name=policy["name"],
            ))
    eligible = [
        result for result in metrics
        if result["false_alarms_per_hour"] <= evaluation["target_false_alarms_per_hour"]
    ]
    candidates = eligible or metrics
    selected = max(
        candidates,
        key=lambda result: (
            result["event_sensitivity"],
            -result["false_alarms_per_hour"],
            -(result["median_detection_delay_sec"] or float("inf")),
        ),
    )
    return selected, metrics, bool(eligible)


def save_scores(path, scores):
    np.savez_compressed(
        path,
        probabilities=scores["probabilities"],
        record_indices=scores["record_indices"],
        start_samples=scores["start_samples"],
        record_offsets=scores["record_offsets"],
    )
    with Path(path).with_suffix(".records.json").open("w", encoding="utf-8") as output_file:
        json.dump(scores["records"], output_file, indent=2)
        output_file.write("\n")


def load_scores(path):
    """Load previously scored continuous recordings and their event metadata."""
    score_path = Path(path)
    records_path = score_path.with_suffix(".records.json")
    if not score_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(f"Missing continuous score artifacts for reuse: {score_path}")
    with np.load(score_path, allow_pickle=False) as source:
        required = {"probabilities", "record_indices", "start_samples", "record_offsets"}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"Continuous scores missing fields: {sorted(missing)}")
        scores = {name: np.asarray(source[name]) for name in required}
    with records_path.open("r", encoding="utf-8") as input_file:
        scores["records"] = json.load(input_file)
    if len(scores["record_offsets"]) != len(scores["records"]) + 1:
        raise ValueError("Continuous score record offsets do not match metadata")
    return scores


def write_threshold_sweep(path, metrics):
    fields = list(metrics[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
