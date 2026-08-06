"""Load locked CHB-MIT splits and fit normalization on training data only."""

import json
import os
import re

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .chbmit_patient_split import patient_group_for_case
from .feature_representation import load_feature_spec, save_feature_spec
from .normalization import window_channel_zscore
from .runtime_config import apply_runtime_overrides
from .utils import get_outputs_dir


src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
config_path = os.path.join(project_dir, "config", "config.yaml")


def load_config():
    """Load YAML config; optional CHBMIT_CONFIG_PATH selects an alternate file."""
    explicit_config = os.environ.get("CHBMIT_CONFIG_PATH")
    resolved_path = (
        os.path.abspath(explicit_config) if explicit_config else config_path
    )
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Config file is missing: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as config_file:
        return apply_runtime_overrides(yaml.safe_load(config_file))


def get_protocol_output_dir_name(config):
    """Return a validated protocol directory name, optionally overridden at runtime."""
    output_name = os.environ.get(
        "CHBMIT_PROTOCOL_OUTPUT_DIR", config["data"]["protocol_output_dir"]
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        raise ValueError(
            "CHBMIT_PROTOCOL_OUTPUT_DIR must contain only letters, digits, underscores, or hyphens"
        )
    return output_name


class EEGDataset(Dataset):
    """Tensor dataset with canonical `(samples, channels, time)` EEG inputs."""

    def __init__(self, x, y, sampling_weights=None, domain_labels=None):
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        if sampling_weights is None:
            sampling_weights = np.ones(len(y), dtype=np.float32)
        self.sampling_weights = torch.from_numpy(
            np.ascontiguousarray(sampling_weights, dtype=np.float32)
        )
        self.domain_labels = None
        if domain_labels is not None:
            if len(domain_labels) != len(y):
                raise ValueError("Domain-label count must match the dataset sample count")
            self.domain_labels = torch.from_numpy(
                np.ascontiguousarray(domain_labels, dtype=np.int64)
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.domain_labels is not None:
            return self.x[index], self.y[index], self.domain_labels[index]
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


def _load_frozen_train_scaler(prepared_dir, expected_channels):
    """Load an optional train-only reference scaler for a derived training cache."""
    path = os.path.join(prepared_dir, "frozen_train_scaler.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=False) as source:
        if {"mean", "scale"} - set(source.files):
            raise ValueError(f"Frozen train scaler is missing mean/scale: {path}")
        mean = np.asarray(source["mean"], dtype=np.float32)
        std = np.asarray(source["scale"], dtype=np.float32)
    if mean.shape != (expected_channels,) or std.shape != (expected_channels,):
        raise ValueError(f"Frozen train scaler has unexpected shape: {path}")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError(f"Frozen train scaler contains invalid values: {path}")
    return mean, std, path


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
    if mode not in {"train_channel_zscore", "per_recording_zscore", "window_channel_zscore"}:
        raise ValueError(f"Unsupported CHB-MIT normalization mode: {mode}")
    return mode


def patient_group_labels(recording_ids):
    """Encode training recording IDs as subject groups for source-only DG training."""
    groups = [patient_group_for_case(str(recording_id).split("/", maxsplit=1)[0]) for recording_id in recording_ids]
    group_to_index = {group_id: index for index, group_id in enumerate(sorted(set(groups)))}
    return np.asarray([group_to_index[group_id] for group_id in groups], dtype=np.int64), group_to_index


def load_normalization_spec(output_dir):
    path = os.path.join(output_dir, "normalization_spec.json")
    if not os.path.isfile(path):
        return {"mode": "train_channel_zscore", "source": "legacy_default"}
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def get_train_val_test_datasets(include_test=True):
    """Load immutable splits and optionally avoid loading an unopened test partition."""
    config = load_config()
    data_config = config["data"]
    model_config = config["model"]
    prepared_dir_name = os.environ.get(
        "CHBMIT_PREPARED_OUTPUT_DIR", data_config["prepared_output_dir"]
    )
    explicit_prepared_dir = os.environ.get("CHBMIT_PREPARED_DIR")
    prepared_dir = (
        os.path.abspath(explicit_prepared_dir)
        if explicit_prepared_dir
        else os.path.join(project_dir, "data", prepared_dir_name)
    )
    feature_spec = load_feature_spec(prepared_dir)
    expected_channels = model_config["input_channels"]
    expected_length = model_config["input_length"]
    expected_feature_shape = [expected_channels, expected_length]
    if feature_spec.get("input_shape") != expected_feature_shape:
        raise ValueError(
            "Prepared feature representation does not match the configured input shape: "
            f"{feature_spec.get('input_shape')} vs {expected_feature_shape}. "
            "Re-run preprocessing with the same CHBMIT_WINDOW_SEC used for training."
        )

    train_x, train_y, train_records, channels, train_weights, _ = _load_prepared_split(
        prepared_dir, "train", expected_channels, expected_length
    )
    val_x, val_y, val_records, val_channels, val_weights, _ = _load_prepared_split(
        prepared_dir, "val", expected_channels, expected_length
    )
    if include_test:
        test_x, test_y, test_records, test_channels, test_weights, _ = _load_prepared_split(
            prepared_dir, "test", expected_channels, expected_length
        )
    else:
        test_x = test_y = test_records = test_channels = test_weights = None
    if not np.array_equal(channels, val_channels):
        raise ValueError("Prepared splits do not share an identical channel order")
    if include_test and not np.array_equal(channels, test_channels):
        raise ValueError("Prepared splits do not share an identical channel order")

    train_record_set = set(train_records.tolist())
    val_record_set = set(val_records.tolist())
    test_record_set = set(test_records.tolist()) if include_test else set()
    if train_record_set & val_record_set or train_record_set & test_record_set or val_record_set & test_record_set:
        raise ValueError("Recording leakage detected between prepared splits")

    normalization_mode = get_normalization_mode(config)
    normalization_reference = "derived_train_windows"
    if normalization_mode == "train_channel_zscore":
        frozen_scaler = _load_frozen_train_scaler(prepared_dir, expected_channels)
        if frozen_scaler is None:
            mean = train_x.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
            std = train_x.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
            std = np.maximum(std, np.finfo(np.float32).eps)
        else:
            mean, std, frozen_scaler_path = frozen_scaler
            normalization_reference = f"frozen_source_train_scaler:{frozen_scaler_path}"
        train_x = _scale_split(train_x, mean, std).astype(np.float32, copy=False)
        val_x = _scale_split(val_x, mean, std).astype(np.float32, copy=False)
        if include_test:
            test_x = _scale_split(test_x, mean, std).astype(np.float32, copy=False)
    elif normalization_mode == "per_recording_zscore":
        # Each split is transformed per recording, independently and without labels.
        train_x, train_recording_stats = _scale_per_recording(train_x, train_records)
        val_x, val_recording_stats = _scale_per_recording(val_x, val_records)
        if include_test:
            test_x, test_recording_stats = _scale_per_recording(test_x, test_records)
            recording_stats = {**train_recording_stats, **val_recording_stats, **test_recording_stats}
        else:
            recording_stats = {**train_recording_stats, **val_recording_stats}
        mean = np.zeros(expected_channels, dtype=np.float32)
        std = np.ones(expected_channels, dtype=np.float32)
    else:
        train_x = window_channel_zscore(train_x)
        val_x = window_channel_zscore(val_x)
        if include_test:
            test_x = window_channel_zscore(test_x)
        mean = np.zeros(expected_channels, dtype=np.float32)
        std = np.ones(expected_channels, dtype=np.float32)

    run_outputs_dir = get_outputs_dir()
    os.makedirs(run_outputs_dir, exist_ok=True)
    save_feature_spec(run_outputs_dir, feature_spec)
    np.save(os.path.join(run_outputs_dir, "scaler_mean.npy"), mean)
    np.save(os.path.join(run_outputs_dir, "scaler_scale.npy"), std)
    with open(os.path.join(run_outputs_dir, "normalization_spec.json"), "w", encoding="utf-8") as output_file:
        json.dump({
            "mode": normalization_mode,
            "scope": {
                "train_channel_zscore": "train_only",
                "per_recording_zscore": "per_recording_unlabeled",
                "window_channel_zscore": "within_current_input_window",
            }[normalization_mode],
            "reference": normalization_reference,
        }, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    if normalization_mode == "per_recording_zscore":
        with open(os.path.join(run_outputs_dir, "recording_normalization.json"), "w", encoding="utf-8") as output_file:
            json.dump(recording_stats, output_file, sort_keys=True)
            output_file.write("\n")
    split_summary = {
        "prepared_output_dir": prepared_dir,
        "normalization_mode": normalization_mode,
        "normalization_reference": normalization_reference,
        "feature_representation": feature_spec,
        "channels": channels.tolist(),
        "train": {"samples": len(train_y), "ictal": int(train_y.sum()), "recordings": len(train_record_set)},
        "val": {"samples": len(val_y), "ictal": int(val_y.sum()), "recordings": len(val_record_set)},
        "test": (
            {"samples": len(test_y), "ictal": int(test_y.sum()), "recordings": len(test_record_set)}
            if include_test
            else {"loaded": False, "reason": "test_evaluation_skipped"}
        ),
    }
    with open(os.path.join(run_outputs_dir, "data_split_summary.json"), "w", encoding="utf-8") as output_file:
        json.dump(split_summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print("Loaded locked prepared splits:")
    for split_name, split in split_summary.items():
        if split_name not in {"train", "val", "test"}:
            continue
        if split.get("loaded") is False:
            print("  test: not loaded (test evaluation skipped)")
        else:
            print(
                f"  {split_name}: {split['samples']} windows | {split['ictal']} ictal | "
                f"{split['recordings']} recordings"
            )

    train_domain_labels, train_domain_mapping = patient_group_labels(train_records)
    split_summary["train_patient_groups"] = {
        "count": len(train_domain_mapping),
        "mapping": train_domain_mapping,
    }
    with open(os.path.join(run_outputs_dir, "data_split_summary.json"), "w", encoding="utf-8") as output_file:
        json.dump(split_summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    return (
        EEGDataset(train_x, train_y, train_weights, domain_labels=train_domain_labels),
        EEGDataset(val_x, val_y, val_weights),
        EEGDataset(test_x, test_y, test_weights) if include_test else None,
    )
