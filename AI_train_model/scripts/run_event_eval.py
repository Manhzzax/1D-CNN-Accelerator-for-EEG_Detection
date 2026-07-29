import json
import os
import sys

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config
from src.event_evaluation import (
    choose_threshold,
    load_split_recordings,
    save_scores,
    score_continuous_recordings,
    event_metrics,
    write_threshold_sweep,
)
from src.model import EEG1DCNN
from src.utils import get_outputs_dir, outputs_dir


def main():
    config = load_config()
    protocol_dir = os.path.join(project_dir, "data", config["data"]["protocol_output_dir"])
    source_run_id = os.environ.get(
        "CHBMIT_MODEL_RUN_ID", config["evaluation"].get("source_model_run_id", "")
    )
    source_outputs_dir = get_outputs_dir(source_run_id)
    model_path = os.path.join(source_outputs_dir, "best_model.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Missing best_model.pth. Run training before event evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["training"].get("use_amp", False) and device.type == "cuda"
    model = EEG1DCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    batch_size = config["evaluation"]["continuous_batch_size"]
    scaler_mean = np.load(os.path.join(source_outputs_dir, "scaler_mean.npy"))
    scaler_std = np.load(os.path.join(source_outputs_dir, "scaler_scale.npy"))
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 60)
    print("RUNNING CONTINUOUS EVENT-LEVEL EVALUATION")
    print("=" * 60)
    validation_rows = load_split_recordings(protocol_dir, "val")
    validation_scores = score_continuous_recordings(
        model, device, validation_rows, config["preprocessing"], batch_size, use_amp, scaler_mean, scaler_std
    )
    selected, sweep, target_met = choose_threshold(
        validation_scores, config["preprocessing"], config["evaluation"]
    )
    save_scores(os.path.join(outputs_dir, "continuous_val_scores.npz"), validation_scores)
    write_threshold_sweep(os.path.join(outputs_dir, "validation_threshold_sweep.csv"), sweep)
    print(
        f"Selected validation policy: {selected['policy_name']} "
        f"({selected['positive_windows']}/{selected['decision_window_windows']}) | "
        f"threshold: {selected['threshold']:.3f} | target FAR met: {target_met}"
    )

    test_rows = load_split_recordings(protocol_dir, "test")
    test_scores = score_continuous_recordings(
        model, device, test_rows, config["preprocessing"], batch_size, use_amp, scaler_mean, scaler_std
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
    save_scores(os.path.join(outputs_dir, "continuous_test_scores.npz"), test_scores)
    summary = {
        "source_model_outputs_dir": source_outputs_dir,
        "artifact_outputs_dir": outputs_dir,
        "threshold_selection": selected,
        "target_false_alarms_per_hour_met_on_validation": target_met,
        "test_event_metrics": test_result,
    }
    with open(os.path.join(outputs_dir, "event_metrics.json"), "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"Test event metrics: {json.dumps(test_result, sort_keys=True)}")


if __name__ == "__main__":
    main()
