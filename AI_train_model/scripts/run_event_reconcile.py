"""Recreate event metrics solely from an experiment's persisted score artifacts."""

import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config
from src.event_evaluation import choose_threshold, event_metrics, load_scores, write_threshold_sweep
from src.utils import outputs_dir


def main():
    config = load_config()
    validation_scores = load_scores(os.path.join(outputs_dir, "continuous_val_scores.npz"))
    test_scores = load_scores(os.path.join(outputs_dir, "continuous_test_scores.npz"))
    selected, sweep, target_met = choose_threshold(
        validation_scores, config["preprocessing"], config["evaluation"]
    )
    test_result = event_metrics(
        test_scores,
        selected["threshold"],
        config["preprocessing"]["sample_rate_hz"],
        config["preprocessing"]["window_sec"],
        config["evaluation"]["refractory_sec"],
        positive_windows=selected["positive_windows"],
        decision_window_windows=selected["decision_window_windows"],
        policy_name=selected["policy_name"],
    )
    summary_path = os.path.join(outputs_dir, "event_metrics.json")
    existing = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as input_file:
            existing = json.load(input_file)
    existing.update({
        "threshold_selection": selected,
        "target_false_alarms_per_hour_met_on_validation": target_met,
        "test_event_metrics": test_result,
        "reconciled_from_saved_score_artifacts": True,
    })
    write_threshold_sweep(os.path.join(outputs_dir, "validation_threshold_sweep.csv"), sweep)
    with open(summary_path, "w", encoding="utf-8") as output_file:
        json.dump(existing, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(
        f"Reconciled validation policy: {selected['policy_name']} "
        f"({selected['positive_windows']}/{selected['decision_window_windows']}) | "
        f"threshold: {selected['threshold']:.3f} | target FAR met: {target_met}"
    )
    print(f"Reconciled test event metrics: {json.dumps(test_result, sort_keys=True)}")


if __name__ == "__main__":
    main()
