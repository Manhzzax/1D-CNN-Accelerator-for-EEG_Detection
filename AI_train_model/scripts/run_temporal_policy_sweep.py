"""Fine validation-only threshold and temporal-confirmation sweep."""

import csv
import json
import os
import sys

import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config
from src.event_evaluation import event_metrics, load_scores
from src.utils import get_outputs_dir


POLICIES = [
    {"name": "3_of_6", "positive_windows": 3, "decision_window_windows": 6},
    {"name": "4_of_8", "positive_windows": 4, "decision_window_windows": 8},
    {"name": "5_of_10", "positive_windows": 5, "decision_window_windows": 10},
    {"name": "6_of_12", "positive_windows": 6, "decision_window_windows": 12},
    {"name": "7_of_14", "positive_windows": 7, "decision_window_windows": 14},
    {"name": "8_of_16", "positive_windows": 8, "decision_window_windows": 16},
    {"name": "9_of_18", "positive_windows": 9, "decision_window_windows": 18},
    {"name": "10_of_20", "positive_windows": 10, "decision_window_windows": 20},
]


def _thresholds():
    minimum = float(os.environ.get("CHBMIT_TEMPORAL_THRESHOLD_MIN", "0.850"))
    maximum = float(os.environ.get("CHBMIT_TEMPORAL_THRESHOLD_MAX", "0.999"))
    step = float(os.environ.get("CHBMIT_TEMPORAL_THRESHOLD_STEP", "0.001"))
    if not 0.0 < minimum <= maximum < 1.0 or step <= 0.0:
        raise ValueError("Temporal threshold range must satisfy 0 < min <= max < 1 and step > 0")
    return np.arange(minimum, maximum + step / 2.0, step)


def _rank_key(result):
    delay = result["median_detection_delay_sec"]
    delay = float("inf") if delay is None else delay
    return (
        result["event_sensitivity"],
        -delay,
        -result["false_alarms_per_hour"],
    )


def main():
    config = load_config()
    source_run_id = os.environ.get("CHBMIT_TEMPORAL_SOURCE_RUN_ID")
    if not source_run_id:
        raise ValueError("Set CHBMIT_TEMPORAL_SOURCE_RUN_ID to a completed validation trial")
    output_run_id = os.environ.get("CHBMIT_RUN_ID")
    if not output_run_id or output_run_id == source_run_id:
        raise ValueError("Set CHBMIT_RUN_ID to a new temporal sweep artifact run")

    source_dir = get_outputs_dir(source_run_id)
    output_dir = get_outputs_dir(output_run_id)
    score_path = os.path.join(source_dir, "continuous_val_scores.npz")
    window_path = os.path.join(source_dir, "validation_window_metrics.json")
    if not os.path.isfile(score_path):
        raise FileNotFoundError(f"Missing validation continuous scores: {score_path}")
    if not os.path.isfile(window_path):
        raise FileNotFoundError(f"Missing validation window metrics: {window_path}")
    with open(window_path, "r", encoding="utf-8") as input_file:
        window_metrics = json.load(input_file)
    scores = load_scores(score_path)
    preprocessing = config["preprocessing"]
    evaluation = config["evaluation"]
    target_far = float(evaluation["target_false_alarms_per_hour"])

    print("=" * 60)
    print("FINE VALIDATION TEMPORAL-POLICY SWEEP")
    print("=" * 60)
    print(f"Source run: {source_run_id} | Validation events: {sum(len(record['seizure_intervals']) for record in scores['records'])}")
    results = []
    thresholds = _thresholds()
    for policy in POLICIES:
        print(f"  Sweeping {policy['name']} across {len(thresholds)} thresholds")
        for threshold in thresholds:
            result = event_metrics(
                scores,
                float(threshold),
                preprocessing["sample_rate_hz"],
                preprocessing["window_sec"],
                evaluation["refractory_sec"],
                positive_windows=policy["positive_windows"],
                decision_window_windows=policy["decision_window_windows"],
                policy_name=policy["name"],
            )
            result["target_far_met"] = result["false_alarms_per_hour"] <= target_far
            result["source_validation_window_accuracy"] = window_metrics["accuracy"]
            result["source_validation_balanced_accuracy"] = window_metrics["balanced_accuracy"]
            result["source_validation_ictal_f1"] = window_metrics["f1"]
            results.append(result)

    eligible = [result for result in results if result["target_far_met"]]
    if not eligible:
        raise RuntimeError("No temporal policy/threshold candidate satisfies the validation FAR target")
    eligible.sort(key=_rank_key, reverse=True)
    results.sort(
        key=lambda result: (result["target_far_met"], *_rank_key(result)), reverse=True
    )
    os.makedirs(output_dir, exist_ok=True)
    fields = list(results[0].keys())
    with open(os.path.join(output_dir, "temporal_policy_validation_sweep.csv"), "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    selected = eligible[0]
    benchmark_pass = bool(
        selected["event_sensitivity"] >= 0.90
        and selected["false_alarms_per_hour"] <= target_far
        and selected["median_detection_delay_sec"] is not None
        and selected["median_detection_delay_sec"] <= 10.0
        and window_metrics["accuracy"] >= 0.90
        and window_metrics["balanced_accuracy"] >= 0.90
        and window_metrics["f1"] >= 0.85
    )
    summary = {
        "source_run_id": source_run_id,
        "evaluated_split": "val",
        "source_validation_window_metrics": window_metrics,
        "threshold_range": {
            "minimum": float(thresholds[0]),
            "maximum": float(thresholds[-1]),
            "step": float(thresholds[1] - thresholds[0]) if len(thresholds) > 1 else None,
        },
        "policies": POLICIES,
        "eligible_candidates": len(eligible),
        "selected": selected,
        "benchmark_pass_validation": benchmark_pass,
    }
    with open(os.path.join(output_dir, "temporal_policy_selection.json"), "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(
        f"Selected validation policy: {selected['policy_name']} "
        f"({selected['positive_windows']}/{selected['decision_window_windows']}) | "
        f"threshold={selected['threshold']:.3f} | sensitivity={selected['event_sensitivity']:.4f} | "
        f"FAR/h={selected['false_alarms_per_hour']:.4f} | delay={selected['median_detection_delay_sec']}"
    )
    print(f"Sweep: {os.path.join(output_dir, 'temporal_policy_validation_sweep.csv')}")
    print("No model training or test recording evaluation was performed.")


if __name__ == "__main__":
    main()
