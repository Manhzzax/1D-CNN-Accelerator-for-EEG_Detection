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

    def __init__(self, x, y, sampling_weights=None):
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        if sampling_weights is None:
            sampling_weights = np.ones(len(y), dtype=np.float32)
        self.sampling_weights = torch.from_numpy(
            np.ascontiguousarray(sampling_weights, dtype=np.float32)
        )

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
        sampling_weights = (
            np.asarray(data["sampling_weight"], dtype=np.float32)
            if "sampling_weight" in data.files
            else np.ones(len(y), dtype=np.float32)
        )

    if x.ndim != 3 or x.shape[1:] != (expected_channels, expected_length):
        raise ValueError(
            f"Unexpected {split_name} shape {x.shape}; expected (N, {expected_channels}, {expected_length})"
        )
    if len(x) != len(y) or len(y) != len(recording_ids) or len(y) != len(sampling_weights):
        raise ValueError(f"Inconsistent sample counts in {path}")
    if not np.all(np.isfinite(sampling_weights)) or np.any(sampling_weights <= 0):
        raise ValueError(f"Sampling weights in {path} must be finite and positive")
    if len(channels) != expected_channels:
        raise ValueError(f"Prepared {split_name} channel count does not match config")
    return x, y, recording_ids, channels, sampling_weights, path


def _scale_split(x, mean, std):
    return (x - mean[None, :, None]) / std[None, :, None]


def _scale_per_recording(x, recording_ids):
    """Normalize each recording independently without using seizure labels."""
    scaled = np.empty_like(x, dtype=np.float32)
    recording_stats = {}
    for recording_id in np.unique(recording_ids):
        indices = np.flatnonzero(recording_ids == recording_id)
        recording = x[indices]
        mean = recording.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        std = recording.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
        std = np.maximum(std, np.finfo(np.float32).eps)
        scaled[indices] = _scale_split(recording, mean, std)
        recording_stats[str(recording_id)] = {"mean": mean.tolist(), "std": std.tolist()}
    return scaled, recording_stats


def get_normalization_mode(config):
    mode = os.environ.get(
        "CHBMIT_NORMALIZATION_MODE", config["preprocessing"].get("normalization_mode", "train_channel_zscore")
    )
    if mode not in {"train_channel_zscore", "per_recording_zscore"}:
        raise ValueError(f"Unsupported CHB-MIT normalization mode: {mode}")
    return mode


def load_normalization_spec(output_dir):
    path = os.path.join(output_dir, "normalization_spec.json")
    if not os.path.isfile(path):
        return {"mode": "train_channel_zscore", "source": "legacy_default"}
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


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

    train_x, train_y, train_records, channels, train_weights, _ = _load_prepared_split(
        prepared_dir, "train", expected_channels, expected_length
    )
    val_x, val_y, val_records, val_channels, val_weights, _ = _load_prepared_split(
        prepared_dir, "val", expected_channels, expected_length
    )
    test_x, test_y, test_records, test_channels, test_weights, _ = _load_prepared_split(
        prepared_dir, "test", expected_channels, expected_length
    )
    if not (np.array_equal(channels, val_channels) and np.array_equal(channels, test_channels)):
        raise ValueError("Prepared splits do not share an identical channel order")

    train_record_set = set(train_records.tolist())
    val_record_set = set(val_records.tolist())
    test_record_set = set(test_records.tolist())
    if train_record_set & val_record_set or train_record_set & test_record_set or val_record_set & test_record_set:
        raise ValueError("Recording leakage detected between prepared splits")

    normalization_mode = get_normalization_mode(config)
    if normalization_mode == "train_channel_zscore":
        mean = train_x.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        std = train_x.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
        std = np.maximum(std, np.finfo(np.float32).eps)
        train_x = _scale_split(train_x, mean, std).astype(np.float32, copy=False)
        val_x = _scale_split(val_x, mean, std).astype(np.float32, copy=False)
        test_x = _scale_split(test_x, mean, std).astype(np.float32, copy=False)
    else:
        # Each split is transformed per recording, independently and without labels.
        train_x, train_recording_stats = _scale_per_recording(train_x, train_records)
        val_x, val_recording_stats = _scale_per_recording(val_x, val_records)
        test_x, test_recording_stats = _scale_per_recording(test_x, test_records)
        recording_stats = {**train_recording_stats, **val_recording_stats, **test_recording_stats}
        mean = np.zeros(expected_channels, dtype=np.float32)
        std = np.ones(expected_channels, dtype=np.float32)

    os.makedirs(outputs_dir, exist_ok=True)
    np.save(os.path.join(outputs_dir, "scaler_mean.npy"), mean)
    np.save(os.path.join(outputs_dir, "scaler_scale.npy"), std)
    with open(os.path.join(outputs_dir, "normalization_spec.json"), "w", encoding="utf-8") as output_file:
        json.dump({
            "mode": normalization_mode,
            "scope": "per_recording_unlabeled" if normalization_mode == "per_recording_zscore" else "train_only",
        }, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    if normalization_mode == "per_recording_zscore":
        with open(os.path.join(outputs_dir, "recording_normalization.json"), "w", encoding="utf-8") as output_file:
            json.dump(recording_stats, output_file, sort_keys=True)
            output_file.write("\n")
    split_summary = {
        "prepared_output_dir": prepared_dir_name,
        "normalization_mode": normalization_mode,
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
        if split_name in {"channels", "prepared_output_dir", "normalization_mode"}:
            continue
        print(
            f"  {split_name}: {split['samples']} windows | {split['ictal']} ictal | "
            f"{split['recordings']} recordings"
        )

    return (
        EEGDataset(train_x, train_y, train_weights),
        EEGDataset(val_x, val_y, val_weights),
        EEGDataset(test_x, test_y, test_weights),
    )
