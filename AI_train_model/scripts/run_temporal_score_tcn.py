"""Train and evaluate a causal TCN over a frozen 1-second CNN score stream."""

import os
import sys
from pathlib import Path

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config
from src.event_evaluation import (
    choose_threshold,
    event_metrics,
    load_scores,
    load_split_recordings,
    save_scores,
    score_continuous_recordings,
    write_threshold_sweep,
)
from src.model import EEG1DCNN
from src.temporal_score_tcn import (
    TemporalScoreTCN,
    adjust_scores_with_tcn,
    build_context_dataset,
    train_tcn,
    write_json,
)
from src.utils import get_outputs_dir, outputs_dir, set_seed


def _load_or_score(source_output_dir, split_name, model, device, rows, config, use_amp, mean, std):
    path = os.path.join(source_output_dir, f"continuous_{split_name}_scores.npz")
    if os.path.isfile(path):
        print(f"Reusing source {split_name} scores: {path}")
        return load_scores(path)
    return score_continuous_recordings(
        model,
        device,
        rows,
        config["preprocessing"],
        config["evaluation"]["continuous_batch_size"],
        use_amp,
        mean,
        std,
    )


def main():
    config = load_config()
    options = config["temporal_score_tcn"]
    source_run_id = os.environ.get("CHBMIT_SOURCE_MODEL_RUN_ID", options["source_model_run_id"])
    source_output_dir = get_outputs_dir(source_run_id)
    if source_output_dir == outputs_dir:
        raise ValueError("Set CHBMIT_RUN_ID to a new output run before training the temporal TCN")
    os.makedirs(outputs_dir, exist_ok=True)
    set_seed(config["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["training"].get("use_amp", False) and device.type == "cuda"

    base_model_path = os.path.join(source_output_dir, "best_model.pth")
    base_model = EEG1DCNN().to(device)
    base_model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
    scaler_mean = np.load(os.path.join(source_output_dir, "scaler_mean.npy"))
    scaler_std = np.load(os.path.join(source_output_dir, "scaler_scale.npy"))
    protocol_dir = os.path.join(project_dir, "data", config["data"]["protocol_output_dir"])

    print("=" * 60)
    print("TRAINING CAUSAL TEMPORAL SCORE-TCN")
    print("=" * 60)
    train_score_cache = os.path.join(outputs_dir, "base_train_scores.npz")
    if os.path.isfile(train_score_cache) and Path(train_score_cache).with_suffix(".records.json").is_file():
        print(f"Reusing cached train scores: {train_score_cache}")
        train_scores = load_scores(train_score_cache)
    else:
        train_scores = _load_or_score(
            source_output_dir, "train", base_model, device, load_split_recordings(protocol_dir, "train"),
            config, use_amp, scaler_mean, scaler_std,
        )
        save_scores(train_score_cache, train_scores)
    val_scores = _load_or_score(
        source_output_dir, "val", base_model, device, load_split_recordings(protocol_dir, "val"),
        config, use_amp, scaler_mean, scaler_std,
    )
    train_x, train_y, train_summary = build_context_dataset(
        train_scores,
        config["preprocessing"],
        options["context_windows"],
        options["random_normal_to_seizure_ratio"],
        options["hard_normal_to_seizure_ratio"],
        config["data"]["seed"],
    )
    val_x, val_y, val_summary = build_context_dataset(
        val_scores,
        config["preprocessing"],
        options["context_windows"],
        options["random_normal_to_seizure_ratio"],
        options["hard_normal_to_seizure_ratio"],
        config["data"]["seed"] + 1,
    )
    tcn = TemporalScoreTCN(options["context_windows"], options["hidden_channels"], options["dropout"]).to(device)
    checkpoint_path = os.path.join(outputs_dir, "temporal_score_tcn.pth")
    training_result = train_tcn(tcn, device, train_x, train_y, val_x, val_y, options, checkpoint_path, use_amp)
    tcn.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    adjusted_val_scores = adjust_scores_with_tcn(
        tcn, device, val_scores, options["context_windows"], options["batch_size"], use_amp
    )
    selected, sweep, target_met = choose_threshold(adjusted_val_scores, config["preprocessing"], config["evaluation"])
    save_scores(os.path.join(outputs_dir, "continuous_val_scores.npz"), adjusted_val_scores)
    write_threshold_sweep(os.path.join(outputs_dir, "validation_threshold_sweep.csv"), sweep)
    print(
        f"Selected validation policy: {selected['policy_name']} "
        f"({selected['positive_windows']}/{selected['decision_window_windows']}) | "
        f"threshold: {selected['threshold']:.3f} | target FAR met: {target_met}"
    )

    test_scores = _load_or_score(
        source_output_dir, "test", base_model, device, load_split_recordings(protocol_dir, "test"),
        config, use_amp, scaler_mean, scaler_std,
    )
    adjusted_test_scores = adjust_scores_with_tcn(
        tcn, device, test_scores, options["context_windows"], options["batch_size"], use_amp
    )
    test_result = event_metrics(
        adjusted_test_scores,
        selected["threshold"],
        config["preprocessing"]["sample_rate_hz"],
        config["preprocessing"]["window_sec"],
        config["evaluation"]["refractory_sec"],
        positive_windows=selected["positive_windows"],
        decision_window_windows=selected["decision_window_windows"],
        policy_name=selected["policy_name"],
    )
    save_scores(os.path.join(outputs_dir, "continuous_test_scores.npz"), adjusted_test_scores)
    summary = {
        "source_model_run_id": source_run_id,
        "source_model_path": base_model_path,
        "context_options": options,
        "train_dataset": train_summary,
        "validation_dataset": val_summary,
        "training": training_result.__dict__,
        "threshold_selection": selected,
        "target_false_alarms_per_hour_met_on_validation": target_met,
        "test_event_metrics": test_result,
    }
    write_json(os.path.join(outputs_dir, "temporal_score_tcn_summary.json"), summary)
    print(f"Test event metrics: {test_result}")


if __name__ == "__main__":
    main()
