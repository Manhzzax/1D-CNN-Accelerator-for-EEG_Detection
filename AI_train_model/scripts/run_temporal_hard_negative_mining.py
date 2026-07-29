"""Build a train-only dataset from persistent alarm-like interictal contexts."""

import json
import os
import sys

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config
from src.model import build_model_from_run
from src.temporal_hard_negative_mining import mine_temporal_hard_negative_windows
from src.utils import get_outputs_dir


def main():
    config = load_config()
    mining = config["temporal_hard_negative_mining"]
    source_run_id = os.environ.get("CHBMIT_SOURCE_RUN_ID", mining["source_run_id"])
    source_outputs_dir = get_outputs_dir(source_run_id)
    source_model_path = os.path.join(source_outputs_dir, "best_model.pth")
    if not os.path.isfile(source_model_path):
        raise FileNotFoundError(f"Missing source checkpoint: {source_model_path}")
    cache_run_id = os.environ.get("CHBMIT_TEMPORAL_HARDNEG_CACHE_RUN_ID", mining["score_cache_run_id"])
    cache_path = os.path.join(get_outputs_dir(cache_run_id), "base_train_scores.npz") if cache_run_id else ""
    if cache_run_id:
        cache_summary_path = os.path.join(get_outputs_dir(cache_run_id), "temporal_score_tcn_summary.json")
        if os.path.isfile(cache_summary_path):
            with open(cache_summary_path, "r", encoding="utf-8") as input_file:
                cache_summary = json.load(input_file)
            if cache_summary.get("source_model_run_id") != source_run_id:
                raise ValueError(
                    f"Score cache run {cache_run_id} was built from "
                    f"{cache_summary.get('source_model_run_id')!r}, not {source_run_id!r}"
                )
        else:
            # A cache without its provenance summary is not safe to reuse.
            cache_path = ""
    output_name = os.environ.get("CHBMIT_TEMPORAL_HARDNEG_OUTPUT_DIR", mining["output_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["training"].get("use_amp", False) and device.type == "cuda"
    model = build_model_from_run(source_outputs_dir).to(device)
    model.load_state_dict(torch.load(source_model_path, map_location=device, weights_only=True))
    scaler_mean = np.load(os.path.join(source_outputs_dir, "scaler_mean.npy"))
    scaler_std = np.load(os.path.join(source_outputs_dir, "scaler_scale.npy"))

    print("=" * 60)
    print("MINING PERSISTENT TRAIN-ONLY HARD NEGATIVES")
    print("=" * 60)
    summary = mine_temporal_hard_negative_windows(
        protocol_dir=os.path.join(project_dir, "data", config["data"]["protocol_output_dir"]),
        source_prepared_dir=os.path.join(project_dir, "data", config["data"]["prepared_output_dir"]),
        output_dir=os.path.join(project_dir, "data", output_name),
        source_score_cache=cache_path,
        model=model,
        device=device,
        preprocessing=config["preprocessing"],
        batch_size=mining["candidate_batch_size"],
        use_amp=use_amp,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        hard_negative_to_seizure_ratio=float(mining["hard_negative_to_seizure_ratio"]),
        threshold=float(mining["threshold"]),
        decision_windows=int(mining["decision_window_windows"]),
        min_hits=int(mining["min_hits_in_context"]),
        min_separation_sec=float(mining["min_separation_sec"]),
        allow_fewer_than_target=bool(mining["allow_fewer_than_target"]),
        hard_negative_sampling_multiplier=float(mining["hard_negative_sampling_multiplier"]),
        seed=config["data"]["seed"],
        source_model_path=source_model_path,
    )
    print(
        f"Temporal hard-negative dataset: {summary['positive_windows']} ictal + "
        f"{summary['source_normal_windows']} source normals + {summary['hard_negative_windows']} hard negatives | "
        f"persistent episodes: {summary['persistent_candidate_episodes']}"
    )


if __name__ == "__main__":
    main()
