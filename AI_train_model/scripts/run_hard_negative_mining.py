"""Create a train-only hard-negative CHB-MIT dataset from a trained source model."""

import os
import sys

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config
from src.hard_negative_mining import mine_hard_negative_windows
from src.model import EEG1DCNN
from src.utils import get_outputs_dir


def main():
    config = load_config()
    mining = config["hard_negative_mining"]
    source_run_id = os.environ.get("CHBMIT_SOURCE_RUN_ID", mining.get("source_run_id", ""))
    source_outputs_dir = get_outputs_dir(source_run_id)
    source_model_path = os.path.join(source_outputs_dir, "best_model.pth")
    if not os.path.isfile(source_model_path):
        raise FileNotFoundError(f"Missing source checkpoint: {source_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["training"].get("use_amp", False) and device.type == "cuda"
    model = EEG1DCNN().to(device)
    model.load_state_dict(torch.load(source_model_path, map_location=device, weights_only=True))

    source_prepared_dir = os.path.join(project_dir, "data", config["data"]["prepared_output_dir"])
    output_dir = os.path.join(project_dir, "data", config["data"]["hard_negative_output_dir"])
    protocol_dir = os.path.join(project_dir, "data", config["data"]["protocol_output_dir"])
    scaler_mean = np.load(os.path.join(source_outputs_dir, "scaler_mean.npy"))
    scaler_std = np.load(os.path.join(source_outputs_dir, "scaler_scale.npy"))

    print("=" * 60)
    print("MINING TRAIN-ONLY HARD NEGATIVES")
    print("=" * 60)
    summary = mine_hard_negative_windows(
        protocol_dir=protocol_dir,
        source_prepared_dir=source_prepared_dir,
        output_dir=output_dir,
        model=model,
        device=device,
        preprocessing=config["preprocessing"],
        batch_size=mining["candidate_batch_size"],
        use_amp=use_amp,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        normal_to_seizure_ratio=float(mining["normal_to_seizure_ratio"]),
        seed=config["data"]["seed"],
        source_model_path=source_model_path,
    )
    print(
        f"Hard-negative dataset: {summary['positive_windows']} ictal + "
        f"{summary['hard_negative_windows']} hard negatives | "
        f"candidates scored: {summary['normal_candidates_scored']}"
    )


if __name__ == "__main__":
    main()
