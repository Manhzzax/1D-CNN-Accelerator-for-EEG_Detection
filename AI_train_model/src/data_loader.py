"""Load locked CHB-MIT splits and fit normalization on training data only."""

import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .utils import outputs_dir


src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
config_path = os.path.join(project_dir, "config", "config.yaml")


def load_config():
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


class EEGDataset(Dataset):
    """Tensor dataset with canonical `(samples, channels, time)` EEG inputs."""

    def __init__(self, x, y):
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def _load_prepared_split(prepared_dir, split_name, expected_channels, expected_length):
    path = os.path.join(prepared_dir, f"chbmit_{split_name}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Prepared {split_name} split is missing: {path}. Run `python main.py --mode preprocess`."
        )

    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "recording_id", "start_sample", "channels"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Prepared split {path} is missing fields: {sorted(missing)}")
        x = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        recording_ids = np.asarray(data["recording_id"])
        channels = np.asarray(data["channels"]).astype(str)

    if x.ndim != 3 or x.shape[1:] != (expected_channels, expected_length):
        raise ValueError(
            f"Unexpected {split_name} shape {x.shape}; expected (N, {expected_channels}, {expected_length})"
        )
    if len(x) != len(y) or len(y) != len(recording_ids):
        raise ValueError(f"Inconsistent sample counts in {path}")
    if len(channels) != expected_channels:
        raise ValueError(f"Prepared {split_name} channel count does not match config")
    return x, y, recording_ids, channels, path


def _scale_split(x, mean, std):
    return (x - mean[None, :, None]) / std[None, :, None]


def get_train_val_test_datasets():
    """Load immutable recording-grouped splits and prevent split/normalization leakage."""
    config = load_config()
    data_config = config["data"]
    model_config = config["model"]
    prepared_dir_name = os.environ.get(
        "CHBMIT_PREPARED_OUTPUT_DIR", data_config["prepared_output_dir"]
    )
    prepared_dir = os.path.join(project_dir, "data", prepared_dir_name)
    expected_channels = model_config["input_channels"]
    expected_length = model_config["input_length"]

    train_x, train_y, train_records, channels, _ = _load_prepared_split(
        prepared_dir, "train", expected_channels, expected_length
    )
    val_x, val_y, val_records, val_channels, _ = _load_prepared_split(
        prepared_dir, "val", expected_channels, expected_length
    )
    test_x, test_y, test_records, test_channels, _ = _load_prepared_split(
        prepared_dir, "test", expected_channels, expected_length
    )
    if not (np.array_equal(channels, val_channels) and np.array_equal(channels, test_channels)):
        raise ValueError("Prepared splits do not share an identical channel order")

    train_record_set = set(train_records.tolist())
    val_record_set = set(val_records.tolist())
    test_record_set = set(test_records.tolist())
    if train_record_set & val_record_set or train_record_set & test_record_set or val_record_set & test_record_set:
        raise ValueError("Recording leakage detected between prepared splits")

    mean = train_x.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.finfo(np.float32).eps)
    train_x = _scale_split(train_x, mean, std).astype(np.float32, copy=False)
    val_x = _scale_split(val_x, mean, std).astype(np.float32, copy=False)
    test_x = _scale_split(test_x, mean, std).astype(np.float32, copy=False)

    os.makedirs(outputs_dir, exist_ok=True)
    np.save(os.path.join(outputs_dir, "scaler_mean.npy"), mean)
    np.save(os.path.join(outputs_dir, "scaler_scale.npy"), std)
    split_summary = {
        "prepared_output_dir": prepared_dir_name,
        "channels": channels.tolist(),
        "train": {"samples": len(train_y), "ictal": int(train_y.sum()), "recordings": len(train_record_set)},
        "val": {"samples": len(val_y), "ictal": int(val_y.sum()), "recordings": len(val_record_set)},
        "test": {"samples": len(test_y), "ictal": int(test_y.sum()), "recordings": len(test_record_set)},
    }
    with open(os.path.join(outputs_dir, "data_split_summary.json"), "w", encoding="utf-8") as output_file:
        json.dump(split_summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print("Loaded locked prepared splits:")
    for split_name, split in split_summary.items():
        if split_name == "channels":
            continue
        print(
            f"  {split_name}: {split['samples']} windows | {split['ictal']} ictal | "
            f"{split['recordings']} recordings"
        )

    return EEGDataset(train_x, train_y), EEGDataset(val_x, val_y), EEGDataset(test_x, test_y)
