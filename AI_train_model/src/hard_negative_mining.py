"""Mine difficult interictal windows using only the locked training recordings."""

import csv
import hashlib
import heapq
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .chbmit_preparation import (
    create_window_index,
    extract_canonical_bipolar_data,
    filter_eeg,
    intervals_to_samples,
)


def _load_train_rows(protocol_dir):
    manifest_path = Path(protocol_dir) / "recording_split_manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8") as input_file:
        rows = [row for row in csv.DictReader(input_file) if row["split"] == "train"]
    if not rows:
        raise ValueError(f"No training recordings in {manifest_path}")
    return rows


def _recording_data(row, preprocessing):
    import mne

    sample_rate = preprocessing["sample_rate_hz"]
    raw = mne.io.read_raw_edf(row["edf_path"], preload=True, verbose="ERROR")
    try:
        if int(round(raw.info["sfreq"])) != sample_rate:
            raise ValueError(f"Unexpected sample rate in {row['recording_id']}: {raw.info['sfreq']}")
        data = extract_canonical_bipolar_data(raw)
    finally:
        raw.close()
    return filter_eeg(
        data,
        sample_rate,
        preprocessing["bandpass_low_hz"],
        preprocessing["bandpass_high_hz"],
        preprocessing["notch_hz"],
    )


def _normal_starts(row, sample_count, preprocessing):
    sample_rate = preprocessing["sample_rate_hz"]
    intervals = intervals_to_samples(json.loads(row["seizure_intervals_json"]), sample_rate)
    _, normal_starts = create_window_index(
        sample_count,
        intervals,
        int(preprocessing["window_sec"] * sample_rate),
        int(preprocessing["stride_sec"] * sample_rate),
        int(preprocessing["interictal_guard_sec"] * sample_rate),
    )
    return normal_starts


def _score_windows(model, device, data, starts, window_samples, batch_size, use_amp, mean, std):
    scores = []
    normalized = (data - mean[:, None]) / std[:, None]
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start:batch_start + batch_size]
            batch = np.stack([
                normalized[:, start:start + window_samples] for start in batch_starts
            ], axis=0)
            inputs = torch.from_numpy(batch).to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(inputs)
            else:
                logits = model(inputs)
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def _load_positive_windows(prepared_dir):
    train_path = Path(prepared_dir) / "chbmit_train.npz"
    with np.load(train_path, allow_pickle=False) as source:
        x = np.asarray(source["X"], dtype=np.float32)
        y = np.asarray(source["y"], dtype=np.int64)
        recording_ids = np.asarray(source["recording_id"])
        starts = np.asarray(source["start_sample"], dtype=np.int64)
        channels = np.asarray(source["channels"])
    positive = y == 1
    if not positive.any():
        raise ValueError(f"No ictal windows found in {train_path}")
    return x[positive], recording_ids[positive], starts[positive], channels


def _copy_fixed_splits(source_dir, target_dir):
    for filename in ("chbmit_val.npz", "chbmit_test.npz", "test_continuous_recordings.csv"):
        source_path = Path(source_dir) / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing prepared split artifact: {source_path}")
        shutil.copy2(source_path, Path(target_dir) / filename)


def mine_hard_negative_windows(
    protocol_dir,
    source_prepared_dir,
    output_dir,
    model,
    device,
    preprocessing,
    batch_size,
    use_amp,
    scaler_mean,
    scaler_std,
    normal_to_seizure_ratio,
    seed,
    source_model_path,
):
    """Create a new train split with top-scoring non-seizure windows from train only."""
    if normal_to_seizure_ratio <= 0:
        raise ValueError("normal_to_seizure_ratio must be positive")
    source_dir = Path(source_prepared_dir)
    target_dir = Path(output_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"Hard-negative output already exists and is non-empty: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    positive_x, positive_records, positive_starts, channels = _load_positive_windows(source_dir)
    target_count = int(round(len(positive_x) * normal_to_seizure_ratio))
    rows = _load_train_rows(protocol_dir)
    sample_rate = preprocessing["sample_rate_hz"]
    window_samples = int(preprocessing["window_sec"] * sample_rate)
    heap = []
    candidate_count = 0
    insertion_index = 0

    model.eval()
    for row_index, row in enumerate(rows):
        data = _recording_data(row, preprocessing)
        starts = _normal_starts(row, data.shape[1], preprocessing)
        probabilities = _score_windows(
            model, device, data, starts, window_samples, batch_size, use_amp, scaler_mean, scaler_std
        )
        candidate_count += len(starts)
        for start, probability in zip(starts, probabilities):
            candidate = (float(probability), insertion_index, row_index, int(start))
            insertion_index += 1
            if len(heap) < target_count:
                heapq.heappush(heap, candidate)
            elif candidate[0] > heap[0][0]:
                heapq.heapreplace(heap, candidate)
        print(f"  Scored training recordings: {row_index + 1}/{len(rows)}")

    if len(heap) != target_count:
        raise RuntimeError(f"Expected {target_count} hard negatives, found only {len(heap)}")

    selected_by_row = defaultdict(list)
    for score, _, row_index, start in heap:
        selected_by_row[row_index].append((start, score))

    normal_signals = []
    normal_records = []
    normal_starts = []
    normal_scores = []
    for row_index, selected in sorted(selected_by_row.items()):
        row = rows[row_index]
        data = _recording_data(row, preprocessing)
        for start, score in sorted(selected):
            normal_signals.append(data[:, start:start + window_samples].copy())
            normal_records.append(row["recording_id"])
            normal_starts.append(start)
            normal_scores.append(score)
        print(f"  Extracted hard negatives: {len(normal_signals)}/{target_count}")

    x = np.concatenate((positive_x, np.stack(normal_signals, axis=0)), axis=0)
    y = np.concatenate((
        np.ones(len(positive_x), dtype=np.int64),
        np.zeros(len(normal_signals), dtype=np.int64),
    ))
    recording_ids = np.concatenate((positive_records, np.asarray(normal_records)))
    starts = np.concatenate((positive_starts, np.asarray(normal_starts, dtype=np.int64)))
    order = np.random.default_rng(seed).permutation(len(y))
    np.savez_compressed(
        target_dir / "chbmit_train.npz",
        X=x[order],
        y=y[order],
        recording_id=recording_ids[order],
        start_sample=starts[order],
        channels=channels,
        split="train",
    )
    _copy_fixed_splits(source_dir, target_dir)

    summary = {
        "source_prepared_dir": str(source_dir),
        "source_model_path": str(source_model_path),
        "source_model_sha256": hashlib.sha256(Path(source_model_path).read_bytes()).hexdigest(),
        "normal_to_seizure_ratio": normal_to_seizure_ratio,
        "positive_windows": int(len(positive_x)),
        "hard_negative_windows": int(len(normal_signals)),
        "normal_candidates_scored": int(candidate_count),
        "hard_negative_score_min": float(min(normal_scores)),
        "hard_negative_score_max": float(max(normal_scores)),
        "channels": channels.astype(str).tolist(),
    }
    with (target_dir / "hard_negative_mining_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return summary
