"""Descriptive event-level error analysis for a completed experiment run."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .event_evaluation import event_metrics, generate_alarms, load_scores


def _record_subset(scores, record_index):
    start = int(scores["record_offsets"][record_index])
    end = int(scores["record_offsets"][record_index + 1])
    return {
        "probabilities": scores["probabilities"][start:end],
        "record_indices": np.zeros(end - start, dtype=np.int32),
        "start_samples": scores["start_samples"][start:end],
        "record_offsets": np.asarray([0, end - start], dtype=np.int64),
        "records": [scores["records"][record_index]],
    }


def _aggregate(rows):
    total_events = sum(row["total_events"] for row in rows)
    detected_events = sum(row["detected_events"] for row in rows)
    false_alarms = sum(row["false_alarms"] for row in rows)
    interictal_hours = sum(row["interictal_hours"] for row in rows)
    return {
        "recordings": len(rows),
        "total_events": total_events,
        "detected_events": detected_events,
        "event_sensitivity": detected_events / total_events if total_events else None,
        "false_alarms": false_alarms,
        "interictal_hours": interictal_hours,
        "false_alarms_per_hour": false_alarms / interictal_hours if interictal_hours else None,
    }


def _write_alarm_and_event_tables(output_dir, split_name, scores, selected, preprocessing, evaluation):
    sample_rate = preprocessing["sample_rate_hz"]
    window_samples = int(preprocessing["window_sec"] * sample_rate)
    refractory_samples = int(evaluation["refractory_sec"] * sample_rate)
    alarm_rows = []
    event_rows = []

    def max_hits_in_window(hits, window_size):
        running_total = 0
        maximum = 0
        for index, hit in enumerate(hits):
            running_total += int(hit)
            if index >= window_size:
                running_total -= int(hits[index - window_size])
            maximum = max(maximum, running_total)
        return maximum

    for record_index, record in enumerate(scores["records"]):
        offset_start = int(scores["record_offsets"][record_index])
        offset_end = int(scores["record_offsets"][record_index + 1])
        starts = scores["start_samples"][offset_start:offset_end]
        probabilities = scores["probabilities"][offset_start:offset_end]
        alarms = generate_alarms(
            starts,
            probabilities,
            selected["threshold"],
            refractory_samples,
            selected["positive_windows"],
            selected["decision_window_windows"],
        )
        intervals = record["seizure_intervals"]
        case_id = record["recording_id"].split("/", 1)[0]

        for alarm in alarms:
            alarm_index = int(np.searchsorted(starts, alarm))
            overlapping_events = [
                event_index
                for event_index, (start, end) in enumerate(intervals)
                if alarm < end and alarm + window_samples > start
            ]
            alarm_rows.append({
                "case_id": case_id,
                "recording_id": record["recording_id"],
                "alarm_start_sample": alarm,
                "alarm_start_sec": alarm / sample_rate,
                "alarm_probability": float(probabilities[alarm_index]),
                "is_false_alarm": not bool(overlapping_events),
                "overlapping_event_index": overlapping_events[0] if overlapping_events else None,
            })

        for event_index, (seizure_start, seizure_end) in enumerate(intervals):
            overlapping_windows = (starts < seizure_end) & (starts + window_samples > seizure_start)
            event_probabilities = probabilities[overlapping_windows]
            event_threshold_hits = event_probabilities >= selected["threshold"]
            matching_alarms = [
                alarm for alarm in alarms
                if alarm < seizure_end and alarm + window_samples > seizure_start
            ]
            first_alarm = min(matching_alarms) if matching_alarms else None
            event_rows.append({
                "case_id": case_id,
                "recording_id": record["recording_id"],
                "event_index": event_index,
                "seizure_start_sec": seizure_start / sample_rate,
                "seizure_end_sec": seizure_end / sample_rate,
                "seizure_duration_sec": (seizure_end - seizure_start) / sample_rate,
                "overlapping_window_count": int(len(event_probabilities)),
                "max_ictal_probability": float(event_probabilities.max()),
                "ictal_windows_at_threshold": int(event_threshold_hits.sum()),
                "max_ictal_hits_in_decision_window": int(max_hits_in_window(
                    event_threshold_hits,
                    selected["decision_window_windows"],
                )),
                "detected": first_alarm is not None,
                "first_alarm_sec": first_alarm / sample_rate if first_alarm is not None else None,
                "detection_delay_sec": (
                    max(0.0, (first_alarm - seizure_start) / sample_rate)
                    if first_alarm is not None else None
                ),
            })

    alarm_path = output_dir / f"event_diagnostics_{split_name}_alarms.csv"
    alarm_fields = [
        "case_id", "recording_id", "alarm_start_sample", "alarm_start_sec",
        "alarm_probability", "is_false_alarm", "overlapping_event_index",
    ]
    with alarm_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=alarm_fields)
        writer.writeheader()
        writer.writerows(alarm_rows)
    event_path = output_dir / f"event_diagnostics_{split_name}_events.csv"
    event_fields = [
        "case_id", "recording_id", "event_index", "seizure_start_sec", "seizure_end_sec",
        "seizure_duration_sec", "overlapping_window_count", "max_ictal_probability",
        "ictal_windows_at_threshold", "max_ictal_hits_in_decision_window", "detected",
        "first_alarm_sec", "detection_delay_sec",
    ]
    with event_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(event_rows)
    return alarm_path, event_path, alarm_rows, event_rows


def analyze_event_run(run_output_dir, split_name, preprocessing, evaluation):
    """Write per-recording and per-case metrics from an already completed event run."""
    output_dir = Path(run_output_dir)
    event_summary_path = output_dir / "event_metrics.json"
    temporal_summary_path = output_dir / "temporal_score_tcn_summary.json"
    score_path = output_dir / f"continuous_{split_name}_scores.npz"
    if event_summary_path.is_file():
        summary_path = event_summary_path
    elif temporal_summary_path.is_file():
        # The temporal TCN is evaluated in its own mode but writes the same
        # threshold-selection contract needed for descriptive diagnostics.
        summary_path = temporal_summary_path
    else:
        raise FileNotFoundError(
            f"Missing event summary: expected {event_summary_path} or {temporal_summary_path}"
        )

    with summary_path.open("r", encoding="utf-8") as input_file:
        event_summary = json.load(input_file)
    selected = event_summary["threshold_selection"]
    scores = load_scores(score_path)
    per_recording = []
    for record_index, record in enumerate(scores["records"]):
        result = event_metrics(
            _record_subset(scores, record_index),
            selected["threshold"],
            sample_rate=preprocessing["sample_rate_hz"],
            window_sec=preprocessing["window_sec"],
            refractory_sec=evaluation["refractory_sec"],
            positive_windows=selected["positive_windows"],
            decision_window_windows=selected["decision_window_windows"],
            policy_name=selected["policy_name"],
        )
        result["recording_id"] = record["recording_id"]
        result["case_id"] = record["recording_id"].split("/", 1)[0]
        per_recording.append(result)

    alarm_path, event_path, alarm_rows, event_rows = _write_alarm_and_event_tables(
        output_dir, split_name, scores, selected, preprocessing, evaluation
    )

    per_recording.sort(key=lambda row: (-row["false_alarms_per_hour"], row["recording_id"]))
    recording_path = output_dir / f"event_diagnostics_{split_name}_per_recording.csv"
    fields = [
        "case_id", "recording_id", "total_events", "detected_events", "event_sensitivity",
        "false_alarms", "interictal_hours", "false_alarms_per_hour",
        "median_detection_delay_sec", "mean_detection_delay_sec", "policy_name",
        "positive_windows", "decision_window_windows", "threshold",
    ]
    with recording_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_recording)

    grouped = defaultdict(list)
    for row in per_recording:
        grouped[row["case_id"]].append(row)
    per_case = []
    for case_id, rows in grouped.items():
        aggregate = _aggregate(rows)
        aggregate["case_id"] = case_id
        per_case.append(aggregate)
    per_case.sort(key=lambda row: (-row["false_alarms_per_hour"], row["case_id"]))
    case_path = output_dir / f"event_diagnostics_{split_name}_per_case.csv"
    with case_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(per_case[0].keys()))
        writer.writeheader()
        writer.writerows(per_case)

    summary = {
        "split": split_name,
        "analysis_use": "descriptive_only; do not use test diagnostics to tune the model",
        "selected_policy": selected,
        "aggregate": _aggregate(per_recording),
        "recordings_with_false_alarms": sum(row["false_alarms"] > 0 for row in per_recording),
        "false_alarm_count": sum(row["is_false_alarm"] for row in alarm_rows),
        "missed_event_count": sum(not row["detected"] for row in event_rows),
        "top_false_alarm_recordings": per_recording[:10],
        "per_recording_csv": str(recording_path),
        "per_case_csv": str(case_path),
        "alarms_csv": str(alarm_path),
        "events_csv": str(event_path),
    }
    summary_path = output_dir / f"event_diagnostics_{split_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return summary
