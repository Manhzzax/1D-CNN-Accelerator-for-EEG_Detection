"""Replay one Path A checkpoint over continuous validation and test EDF files.

This runner intentionally labels every result as exploratory because the
historic Path A A1.2 test cohort was already used in the architecture ladder.
It measures the missing event-level quantities without relabelling that test
cohort as a final clinical benchmark.
"""

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
    event_metrics,
    load_split_recordings,
    save_scores,
    score_continuous_recordings,
    write_threshold_sweep,
)
from src.feature_representation import load_feature_spec
from src.model import build_model_from_run
from src.utils import get_outputs_dir


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false")


def _case_id():
    value = os.environ.get("CHBMIT_CASE_ID", "").strip()
    if not value:
        raise ValueError("CHBMIT_CASE_ID is required")
    if not value.startswith("chb") or not value[3:].isdigit():
        raise ValueError("CHBMIT_CASE_ID must be a CHB-MIT case id such as chb05")
    return value


def _score(model, device, rows, config, use_amp, mean, scale, normalization, feature_spec):
    return score_continuous_recordings(
        model, device, rows, config["preprocessing"], config["evaluation"]["continuous_batch_size"],
        use_amp, mean, scale, normalization, None, feature_spec,
    )


def main():
    if not _env_bool("CHBMIT_PATH_A_REPLAY_ALLOW_EXPOSED_TEST", False):
        raise ValueError(
            "Set CHBMIT_PATH_A_REPLAY_ALLOW_EXPOSED_TEST=true to acknowledge that this is exploratory replay, not final testing"
        )
    case_id = _case_id()
    source_run_id = os.environ.get("CHBMIT_CHECKPOINT_SOURCE_RUN_ID", "").strip()
    if not source_run_id:
        raise ValueError("CHBMIT_CHECKPOINT_SOURCE_RUN_ID is required")
    config = load_config()
    protocol_root = config["data"]["patient_specific_protocol_root"]
    protocol_dir = os.path.join(project_dir, "data", protocol_root, case_id)
    source_dir = get_outputs_dir(source_run_id)
    output_dir = get_outputs_dir()
    if os.path.abspath(source_dir) == os.path.abspath(output_dir):
        raise ValueError("CHBMIT_RUN_ID must name a new event-replay artifact directory")
    checkpoint = os.path.join(source_dir, "best_model.pth")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not os.path.isfile(os.path.join(protocol_dir, "recording_split_manifest.csv")):
        raise FileNotFoundError(f"Missing case protocol: {protocol_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = _env_bool("CHBMIT_PATH_A_REPLAY_USE_AMP", False) and device.type == "cuda"
    model = build_model_from_run(source_dir).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    mean = np.load(os.path.join(source_dir, "scaler_mean.npy"))
    scale = np.load(os.path.join(source_dir, "scaler_scale.npy"))
    normalization = load_normalization_spec(source_dir)["mode"]
    if normalization != "train_channel_zscore":
        raise ValueError("Path A event replay requires train_channel_zscore")
    feature_spec = load_feature_spec(source_dir)
    if feature_spec["name"] != "raw":
        raise ValueError("Path A event replay currently supports raw EEG checkpoints only")

    os.makedirs(output_dir, exist_ok=True)
    print(f"PATH A EXPLORATORY EVENT REPLAY: {case_id} | source={source_run_id} | device={device}")
    val_scores = _score(model, device, load_split_recordings(protocol_dir, "val"), config, use_amp, mean, scale, normalization, feature_spec)
    save_scores(os.path.join(output_dir, "continuous_val_scores.npz"), val_scores)
    selected, sweep, far_target_met = choose_threshold(val_scores, config["preprocessing"], config["evaluation"])
    write_threshold_sweep(os.path.join(output_dir, "validation_threshold_sweep.csv"), sweep)
    test_scores = _score(model, device, load_split_recordings(protocol_dir, "test"), config, use_amp, mean, scale, normalization, feature_spec)
    save_scores(os.path.join(output_dir, "continuous_test_scores.npz"), test_scores)
    test_result = event_metrics(
        test_scores, selected["threshold"], config["preprocessing"]["sample_rate_hz"],
        config["preprocessing"]["window_sec"], config["evaluation"]["refractory_sec"],
        positive_windows=selected["positive_windows"], decision_window_windows=selected["decision_window_windows"],
        policy_name=selected["policy_name"], include_detection_delays=True,
    )
    result = {
        "evaluation_kind": "exploratory_test_replay",
        "final_claim_eligible": False,
        "selection_warning": "The A1.2 test cohort was exposed during the architecture ladder; this replay supplies descriptive event metrics only.",
        "case_id": case_id,
        "source_run_id": source_run_id,
        "inference": {"device": str(device), "use_amp": use_amp, "precision_label": "AMP" if use_amp else "FP32"},
        "threshold_selection": selected,
        "target_false_alarms_per_hour_met_on_validation": far_target_met,
        "test_event_metrics": test_result,
    }
    path = os.path.join(output_dir, "patient_specific_event_replay.json")
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(
        f"Test events: {test_result['detected_events']}/{test_result['total_events']} | "
        f"FAR/h={test_result['false_alarms_per_hour']:.4f} | saved={path}"
    )


if __name__ == "__main__":
    main()
