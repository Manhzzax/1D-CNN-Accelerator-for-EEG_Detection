"""Descriptive event-level error analysis for a completed experiment run."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .event_evaluation import event_metrics, load_scores


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


def analyze_event_run(run_output_dir, split_name, preprocessing, evaluation):
    """Write per-recording and per-case metrics from an already completed event run."""
    output_dir = Path(run_output_dir)
    event_summary_path = output_dir / "event_metrics.json"
    score_path = output_dir / f"continuous_{split_name}_scores.npz"
    if not event_summary_path.is_file():
        raise FileNotFoundError(f"Missing event summary: {event_summary_path}")

    with event_summary_path.open("r", encoding="utf-8") as input_file:
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
        "top_false_alarm_recordings": per_recording[:10],
        "per_recording_csv": str(recording_path),
        "per_case_csv": str(case_path),
    }
    summary_path = output_dir / f"event_diagnostics_{split_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return summary
