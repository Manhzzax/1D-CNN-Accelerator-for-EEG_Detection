"""Causal fold preparation and continuous FP32 inference."""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import CANONICAL_CHANNELS


def _dependencies():
    try:
        import numpy as np
        import pyedflib
        from scipy import signal
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("numpy, scipy, and pyedflib are required; install project dependencies") from error
    return np, pyedflib, signal


def read_causal_recording(row: dict):
    np, pyedflib, signal = _dependencies()
    indices = row["canonical_channel_indices"]
    if indices is None or len(indices) != len(CANONICAL_CHANNELS):
        raise ValueError(f"{row['recording_id']} does not meet the strict montage contract")
    reader = pyedflib.EdfReader(row["path"])
    try:
        data = np.vstack([reader.readSignal(index) for index in indices]).astype("float32")
    finally:
        reader.close()
    rate = float(row["sampling_rate_hz"])
    if rate != 256:
        raise ValueError(f"{row['recording_id']} is not sampled at 256 Hz")
    band = signal.butter(4, [0.5, 45.0], btype="bandpass", fs=rate, output="sos")
    notch_b, notch_a = signal.iirnotch(60.0, 30.0, fs=rate)
    return signal.lfilter(notch_b, notch_a, signal.sosfilt(band, data, axis=1), axis=1).astype("float32")


def label_window(start: float, end: float, intervals: list[list[float]], guard_seconds: float = 30.0) -> int | None:
    center = (start + end) / 2
    if any(interval_start <= center <= interval_end for interval_start, interval_end in intervals):
        return 1
    if all(end <= interval_start - guard_seconds or start >= interval_end + guard_seconds for interval_start, interval_end in intervals):
        return 0
    return None


def fit_train_normalization(rows: list[dict]):
    np, _, _ = _dependencies()
    total, squared, count = np.zeros(19, dtype="float64"), np.zeros(19, dtype="float64"), 0
    for row in rows:
        values = read_causal_recording(row).astype("float64")
        total += values.sum(axis=1); squared += (values * values).sum(axis=1); count += values.shape[1]
    if not count:
        raise ValueError("Cannot fit normalization without training recordings")
    mean = total / count
    scale = np.sqrt(np.maximum(squared / count - mean * mean, 1e-12))
    return mean.astype("float32"), scale.astype("float32")


def materialize_windows(rows: list[dict], mean, scale):
    np, _, _ = _dependencies()
    xs, ys = [], []
    width, stride = 1024, 256
    for row in rows:
        values = (read_causal_recording(row) - mean[:, None]) / scale[:, None]
        intervals = row["seizure_intervals_seconds"]
        for offset in range(0, values.shape[1] - width + 1, stride):
            label = label_window(offset / 256.0, (offset + width) / 256.0, intervals)
            if label is not None:
                xs.append(values[:, offset:offset + width]); ys.append(label)
    if not xs:
        raise ValueError("No eligible windows after applying event labels and guard interval")
    return np.stack(xs).astype("float32"), np.asarray(ys, dtype="int64")


def prepare_fold(rows: list[dict], fold: dict, output: Path) -> dict:
    np, _, _ = _dependencies()
    by_id = {row["recording_id"]: row for row in rows}
    train_rows = [by_id[item] for item in fold["recordings"]["train"]]
    validation_rows = [by_id[item] for item in fold["recordings"]["validation"]]
    mean, scale = fit_train_normalization(train_rows)
    x_train, y_train = materialize_windows(train_rows, mean, scale)
    x_val, y_val = materialize_windows(validation_rows, mean, scale)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(output / "validation.npz", x=x_val, y=y_val)
    np.savez(output / "normalization.npz", mean=mean, scale=scale)
    summary = {"outer_test_subject": fold["outer_test_subject"], "train_windows": int(len(y_train)), "validation_windows": int(len(y_val)), "normalization_fit": "training_recordings_only"}
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def infer_continuous(rows: list[dict], fold: dict, normalization: Path, checkpoint: Path, output: Path) -> dict:
    try:
        import numpy as np
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("numpy and torch are required; install project dependencies") from error
    from .models import build_reference_model
    by_id = {row["recording_id"]: row for row in rows}
    norm = np.load(normalization); mean, scale = norm["mean"], norm["scale"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_reference_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"]); model.eval()
    output.parent.mkdir(parents=True, exist_ok=True); written, replay_seconds = 0, 0.0
    with output.open("w", encoding="utf-8", newline="\n") as destination, torch.no_grad():
        for recording_id in fold["recordings"]["test"]:
            row = by_id[recording_id]; values = (read_causal_recording(row) - mean[:, None]) / scale[:, None]; replay_seconds += values.shape[1] / 256.0
            for offset in range(0, values.shape[1] - 1024 + 1, 256):
                tensor = torch.from_numpy(values[:, offset:offset + 1024][None]).to(device)
                probability = float(torch.softmax(model(tensor), dim=1)[0, 1].cpu())
                destination.write(json.dumps({"recording_id": recording_id, "timestamp_seconds": offset / 256.0, "seizure_probability": probability}) + "\n"); written += 1
    return {"subject_id": fold["outer_test_subject"], "windows": written, "replay_seconds": replay_seconds, "output": str(output)}

