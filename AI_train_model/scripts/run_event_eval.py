import json
import os
import sys

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config, load_normalization_spec
from src.event_evaluation import (
    choose_threshold,
    load_scores,
    load_split_recordings,
    save_scores,
    score_continuous_recordings,
    event_metrics,
    write_threshold_sweep,
)
from src.model import build_model_from_run
from src.utils import get_outputs_dir, outputs_dir


def _load_or_score(
    split_name, source_outputs_dir, artifact_outputs_dir, model, device, rows, config, use_amp,
    scaler_mean, scaler_std, normalization_mode, recording_normalization,
):
    source_score_path = os.path.join(source_outputs_dir, f"continuous_{split_name}_scores.npz")
    can_reuse = (
        config["evaluation"].get("reuse_source_continuous_scores", False)
        and source_outputs_dir != artifact_outputs_dir
        and os.path.isfile(source_score_path)
    )
    if can_reuse:
        print(f"Reusing source {split_name} continuous scores: {source_score_path}")
        return load_scores(source_score_path), True
    return score_continuous_recordings(
        model,
        device,
        rows,
        config["preprocessing"],
        config["evaluation"]["continuous_batch_size"],
        use_amp,
        scaler_mean,
        scaler_std,
        normalization_mode,
        recording_normalization,
    ), False


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
    model = build_model_from_run(source_outputs_dir).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    scaler_mean = np.load(os.path.join(source_outputs_dir, "scaler_mean.npy"))
    scaler_std = np.load(os.path.join(source_outputs_dir, "scaler_scale.npy"))
    normalization_mode = load_normalization_spec(source_outputs_dir)["mode"]
    recording_normalization = None
    if normalization_mode == "per_recording_zscore":
        stats_path = os.path.join(source_outputs_dir, "recording_normalization.json")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Missing recording normalization artifact: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as input_file:
            recording_normalization = json.load(input_file)
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 60)
    print("RUNNING CONTINUOUS EVENT-LEVEL EVALUATION")
    print("=" * 60)
    requested_splits = {
        split.strip() for split in os.environ.get("CHBMIT_EVENT_EVAL_SPLITS", "val,test").split(",") if split.strip()
    }
    if not requested_splits or not requested_splits <= {"val", "test"}:
        raise ValueError("CHBMIT_EVENT_EVAL_SPLITS must be val, test, or val,test")
    if "val" not in requested_splits:
        raise ValueError("Validation scores are required to select a threshold and temporal policy")
    validation_rows = load_split_recordings(protocol_dir, "val")
    validation_scores, reused_validation_scores = _load_or_score(
        "val", source_outputs_dir, outputs_dir, model, device, validation_rows, config, use_amp, scaler_mean, scaler_std,
        normalization_mode, recording_normalization,
    )
    validation_score_path = os.path.join(outputs_dir, "continuous_val_scores.npz")
    # Select thresholds from the exact persisted score representation used by
    # diagnostics and later reproducibility checks.
    save_scores(validation_score_path, validation_scores)
    validation_scores = load_scores(validation_score_path)
    selected, sweep, target_met = choose_threshold(
        validation_scores, config["preprocessing"], config["evaluation"]
    )
    write_threshold_sweep(os.path.join(outputs_dir, "validation_threshold_sweep.csv"), sweep)
    print(
        f"Selected validation policy: {selected['policy_name']} "
        f"({selected['positive_windows']}/{selected['decision_window_windows']}) | "
        f"threshold: {selected['threshold']:.3f} | target FAR met: {target_met}"
    )

    test_result = None
    reused_test_scores = None
    if "test" in requested_splits:
        test_rows = load_split_recordings(protocol_dir, "test")
        test_scores, reused_test_scores = _load_or_score(
            "test", source_outputs_dir, outputs_dir, model, device, test_rows, config, use_amp, scaler_mean, scaler_std,
            normalization_mode, recording_normalization,
        )
        test_score_path = os.path.join(outputs_dir, "continuous_test_scores.npz")
        save_scores(test_score_path, test_scores)
        test_scores = load_scores(test_score_path)
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
    summary = {
        "source_model_outputs_dir": source_outputs_dir,
        "artifact_outputs_dir": outputs_dir,
        "reused_source_continuous_scores": {
            "validation": reused_validation_scores,
            "test": reused_test_scores,
        },
        "evaluated_splits": sorted(requested_splits),
        "threshold_selection": selected,
        "target_false_alarms_per_hour_met_on_validation": target_met,
        "test_event_metrics": test_result,
    }
    with open(os.path.join(outputs_dir, "event_metrics.json"), "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    if test_result is None:
        print(f"Validation event metrics: {json.dumps(selected, sort_keys=True)}")
    else:
        print(f"Test event metrics: {json.dumps(test_result, sort_keys=True)}")


if __name__ == "__main__":
    main()
