"""Leakage-safe CHB-MIT waveform preparation from locked recording splits."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter, sosfilt, sosfiltfilt

from .feature_representation import save_feature_spec, transform_windows
from .chbmit_montage import CANONICAL_BIPOLAR_17, resolve_canonical_bipolar_17


SPLIT_NAMES = ("train", "val", "test")


def load_locked_split_manifest(protocol_dir):
    path = Path(protocol_dir) / "recording_split_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing locked split manifest: {path}")
    with path.open("r", newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows or {row["split"] for row in rows} != set(SPLIT_NAMES):
        raise ValueError("Split manifest is empty or does not contain train/val/test")
    return rows


def intervals_to_samples(intervals, sample_rate):
    return [
        (int(round(start_sec * sample_rate)), int(round(end_sec * sample_rate)))
        for start_sec, end_sec in intervals
        if end_sec > start_sec
    ]


def create_window_index(sample_count, intervals, window_samples, stride_samples, guard_samples):
    """Return full-ictal and guard-excluded interictal window start positions."""
    starts = np.arange(0, sample_count - window_samples + 1, stride_samples, dtype=np.int64)
    ends = starts + window_samples
    ictal = np.zeros(starts.shape[0], dtype=bool)
    guarded = np.zeros(starts.shape[0], dtype=bool)

    for start, end in intervals:
        ictal |= (starts >= start) & (ends <= end)
        guarded_start = max(0, start - guard_samples)
        guarded_end = min(sample_count, end + guard_samples)
        guarded |= (ends > guarded_start) & (starts < guarded_end)

    return starts[ictal], starts[~guarded]


def extract_canonical_bipolar_data(raw):
    """Return 17 canonical channels, reconstructing bipolar signals when required."""
    resolution = resolve_canonical_bipolar_17(raw.ch_names)
    missing = [channel for channel, (mode, _) in resolution.items() if mode == "missing"]
    if missing:
        raise ValueError(f"Cannot resolve canonical montage: {missing}")

    source = raw.get_data()
    channels = []
    for channel_name in CANONICAL_BIPOLAR_17:
        mode, indices = resolution[channel_name]
        if mode == "direct":
            channels.append(source[indices[0]])
        else:
            channels.append(source[indices[0]] - source[indices[1]])
    return np.asarray(channels, dtype=np.float32)


def filter_eeg(data, sample_rate, low_cut_hz, high_cut_hz, notch_hz, filter_mode="zero_phase"):
    """Filter a complete recording before windowing.

    ``zero_phase`` is retained only for historical offline ablations. ``causal_iir``
    uses each sample and its past only, so it is the required mode for the new
    streaming/patient-held-out protocol.
    """
    bandpass = butter(
        4,
        [low_cut_hz, high_cut_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    if filter_mode == "zero_phase":
        filtered = sosfiltfilt(bandpass, data, axis=1)
        if notch_hz:
            notch_b, notch_a = iirnotch(notch_hz, 30.0, fs=sample_rate)
            filtered = filtfilt(notch_b, notch_a, filtered, axis=1)
    elif filter_mode == "causal_iir":
        filtered = sosfilt(bandpass, data, axis=1)
        if notch_hz:
            notch_b, notch_a = iirnotch(notch_hz, 30.0, fs=sample_rate)
            filtered = lfilter(notch_b, notch_a, filtered, axis=1)
    else:
        raise ValueError(f"Unsupported filter mode: {filter_mode}")
    return np.asarray(filtered * 1_000_000.0, dtype=np.float32)


@dataclass
class WindowReservoir:
    target_size: int
    random_state: np.random.Generator
    signals: list = field(default_factory=list)
    metadata: list = field(default_factory=list)
    seen: int = 0

    def add(self, signal, metadata):
        self.seen += 1
        if len(self.signals) < self.target_size:
            self.signals.append(signal.copy())
            self.metadata.append(metadata)
            return
        replacement_index = int(self.random_state.integers(0, self.seen))
        if replacement_index < self.target_size:
            self.signals[replacement_index] = signal.copy()
            self.metadata[replacement_index] = metadata


@dataclass
class SplitCollector:
    normal_target: int
    random_state: np.random.Generator
    positive_signals: list = field(default_factory=list)
    positive_metadata: list = field(default_factory=list)
    normal_reservoir: WindowReservoir = field(init=False)

    def __post_init__(self):
        self.normal_reservoir = WindowReservoir(self.normal_target, self.random_state)

    def add_positive(self, signal, metadata):
        self.positive_signals.append(signal.copy())
        self.positive_metadata.append(metadata)

    def add_normal(self, signal, metadata):
        self.normal_reservoir.add(signal, metadata)

    def save(self, output_path, split_name, feature_spec, feature_batch_size):
        signals = self.positive_signals + self.normal_reservoir.signals
        labels = np.concatenate((
            np.ones(len(self.positive_signals), dtype=np.int64),
            np.zeros(len(self.normal_reservoir.signals), dtype=np.int64),
        ))
        metadata = self.positive_metadata + self.normal_reservoir.metadata
        if not signals:
            raise ValueError(f"No prepared windows for split {split_name}")

        order = self.random_state.permutation(len(signals))
        x = np.stack(signals, axis=0)[order]
        x = transform_windows(x, feature_spec, batch_size=feature_batch_size)
        y = labels[order]
        recording_ids = np.asarray([metadata[index]["recording_id"] for index in order])
        start_samples = np.asarray([metadata[index]["start_sample"] for index in order], dtype=np.int64)
        np.savez_compressed(
            output_path,
            X=x,
            y=y,
            recording_id=recording_ids,
            start_sample=start_samples,
            channels=np.asarray(CANONICAL_BIPOLAR_17),
            split=split_name,
        )
        return {
            "positive_windows": len(self.positive_signals),
            "normal_windows": len(self.normal_reservoir.signals),
            "normal_candidates_seen": self.normal_reservoir.seen,
            "path": str(output_path),
        }


def calculate_normal_targets(rows, preprocessing):
    sample_rate = preprocessing["sample_rate_hz"]
    window_samples = int(preprocessing["window_sec"] * sample_rate)
    stride_samples = int(preprocessing["stride_sec"] * sample_rate)
    guard_samples = int(preprocessing["interictal_guard_sec"] * sample_rate)
    counts = {split_name: {"positive": 0, "normal_candidates": 0} for split_name in SPLIT_NAMES}

    for row in rows:
        intervals = intervals_to_samples(json.loads(row["seizure_intervals_json"]), sample_rate)
        positives, normals = create_window_index(
            int(row["sample_count"]), intervals, window_samples, stride_samples, guard_samples
        )
        counts[row["split"]]["positive"] += len(positives)
        counts[row["split"]]["normal_candidates"] += len(normals)

    ratios = preprocessing["normal_to_seizure_ratio"]
    for split_name, split_counts in counts.items():
        requested = int(round(split_counts["positive"] * ratios[split_name]))
        split_counts["normal_target"] = min(requested, split_counts["normal_candidates"])
    return counts


def prepare_chbmit_windows(protocol_dir, output_dir, preprocessing, seed, feature_spec=None):
    """Prepare sampled windows without changing the locked recording split."""
    protocol_path = Path(protocol_dir)
    rows = load_locked_split_manifest(protocol_path)
    sample_rate = preprocessing["sample_rate_hz"]
    window_samples = int(preprocessing["window_sec"] * sample_rate)
    stride_samples = int(preprocessing["stride_sec"] * sample_rate)
    guard_samples = int(preprocessing["interictal_guard_sec"] * sample_rate)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if feature_spec is None:
        feature_spec = {"name": "raw", "input_shape": [17, window_samples]}
    feature_batch_size = int(preprocessing.get("feature_transform_batch_size", 512))
    if feature_batch_size < 1:
        raise ValueError("feature_transform_batch_size must be positive")

    counts = calculate_normal_targets(rows, preprocessing)
    collectors = {
        split_name: SplitCollector(
            normal_target=counts[split_name]["normal_target"],
            random_state=np.random.default_rng(seed + index),
        )
        for index, split_name in enumerate(SPLIT_NAMES)
    }

    print(f"Preparing {len(rows)} recordings from the locked split manifest...")
    import mne

    for index, row in enumerate(rows, start=1):
        edf_path = Path(row["edf_path"])
        if not edf_path.is_file():
            raise FileNotFoundError(f"EDF listed by split manifest is missing: {edf_path}")

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        try:
            if int(round(raw.info["sfreq"])) != sample_rate:
                raise ValueError(f"Unexpected sample rate in {edf_path}: {raw.info['sfreq']}")
            data = extract_canonical_bipolar_data(raw)
        finally:
            raw.close()

        data = filter_eeg(
            data,
            sample_rate,
            preprocessing["bandpass_low_hz"],
            preprocessing["bandpass_high_hz"],
            preprocessing["notch_hz"],
            preprocessing.get("filter_mode", "zero_phase"),
        )
        intervals = intervals_to_samples(json.loads(row["seizure_intervals_json"]), sample_rate)
        positive_starts, normal_starts = create_window_index(
            data.shape[1], intervals, window_samples, stride_samples, guard_samples
        )
        collector = collectors[row["split"]]
        for start_sample in positive_starts:
            collector.add_positive(
                data[:, start_sample:start_sample + window_samples],
                {"recording_id": row["recording_id"], "start_sample": int(start_sample)},
            )
        for start_sample in normal_starts:
            collector.add_normal(
                data[:, start_sample:start_sample + window_samples],
                {"recording_id": row["recording_id"], "start_sample": int(start_sample)},
            )

        if index % 10 == 0 or index == len(rows):
            print(f"  Prepared recordings: {index}/{len(rows)}")

    split_outputs = {}
    for split_name in SPLIT_NAMES:
        split_outputs[split_name] = collectors[split_name].save(
            output_path / f"chbmit_{split_name}.npz", split_name, feature_spec, feature_batch_size
        )

    save_feature_spec(output_path, feature_spec)

    with (output_path / "test_continuous_recordings.csv").open("w", newline="", encoding="utf-8") as output_file:
        test_rows = [row for row in rows if row["split"] == "test"]
        writer = csv.DictWriter(output_file, fieldnames=list(test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_rows)

    summary = {
        "protocol_manifest": str(protocol_path / "recording_split_manifest.csv"),
        "protocol_summary": str(protocol_path / "split_plan_summary.json"),
        "channels": list(CANONICAL_BIPOLAR_17),
        "sample_rate_hz": sample_rate,
        "window_sec": preprocessing["window_sec"],
        "stride_sec": preprocessing["stride_sec"],
        "interictal_guard_sec": preprocessing["interictal_guard_sec"],
        "bandpass_hz": [preprocessing["bandpass_low_hz"], preprocessing["bandpass_high_hz"]],
        "notch_hz": preprocessing["notch_hz"],
        "filter_mode": preprocessing.get("filter_mode", "zero_phase"),
        "normal_to_seizure_ratio": preprocessing["normal_to_seizure_ratio"],
        "feature_representation": feature_spec,
        "prepass_counts": counts,
        "outputs": split_outputs,
        "continuous_test_manifest": str(output_path / "test_continuous_recordings.csv"),
    }
    with (output_path / "preparation_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return summary
