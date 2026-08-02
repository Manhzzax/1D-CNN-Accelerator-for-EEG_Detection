"""Causal, fold-local CHB-MIT window preparation for V2.

This builder deliberately does not call the legacy three-way preparation code:
the V2 fold manifests contain a per-fold future partition and use causal-endpoint
labels.  It writes the same compact NPZ field names expected by the existing
training dataset loader, allowing the model code to be reused after a V2 fold
is frozen.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .protocol import causal_window_index, canonical_json_hash, file_sha256, save_json


ACTIVE_SPLITS = ("train", "val", "test")


@dataclass
class _Reservoir:
    target_size: int
    rng: object
    windows: list = field(default_factory=list)
    metadata: list = field(default_factory=list)
    candidates_seen: int = 0

    def add(self, window, metadata) -> None:
        self.candidates_seen += 1
        if len(self.windows) < self.target_size:
            self.windows.append(window.copy())
            self.metadata.append(dict(metadata))
            return
        replacement = int(self.rng.integers(0, self.candidates_seen))
        if replacement < self.target_size:
            self.windows[replacement] = window.copy()
            self.metadata[replacement] = dict(metadata)


@dataclass
class _Collector:
    normal_target: int
    rng: object
    positives: list = field(default_factory=list)
    positive_metadata: list = field(default_factory=list)
    normals: _Reservoir = field(init=False)

    def __post_init__(self) -> None:
        self.normals = _Reservoir(self.normal_target, self.rng)

    def add_positive(self, window, metadata) -> None:
        self.positives.append(window.copy())
        self.positive_metadata.append(dict(metadata))


def _load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows or not {"split", "edf_path", "seizure_intervals_json"}.issubset(rows[0]):
        raise ValueError("Fold manifest is missing required V2 recording fields")
    return rows


def _count_targets(rows: list[dict], config: dict) -> dict:
    preprocessing = config["preprocessing"]
    sample_rate = int(config["dataset"]["sample_rate_hz"])
    window_samples = int(float(preprocessing["window_sec"]) * sample_rate)
    stride_samples = int(float(preprocessing["stride_sec"]) * sample_rate)
    guard_samples = int(float(preprocessing["interictal_guard_sec"]) * sample_rate)
    ratios = config["window_sampling"]["normal_to_positive_ratio"]
    result = {split: {"positive": 0, "normal_candidates": 0} for split in ACTIVE_SPLITS}
    for row in rows:
        split = row["split"]
        if split not in ACTIVE_SPLITS:
            continue
        intervals = json.loads(row["seizure_intervals_json"])
        sample_intervals = [(round(float(start) * sample_rate), round(float(end) * sample_rate)) for start, end in intervals]
        positives, normals, _ = causal_window_index(
            int(row["sample_count"]), sample_intervals, window_samples, stride_samples, guard_samples
        )
        result[split]["positive"] += len(positives)
        result[split]["normal_candidates"] += len(normals)
    for split, counts in result.items():
        requested = round(counts["positive"] * float(ratios[split]))
        counts["normal_target"] = min(int(requested), counts["normal_candidates"])
    return result


def _save_split(output: Path, split: str, collector: _Collector, rng) -> dict:
    import numpy as np
    from src.chbmit_montage import CANONICAL_BIPOLAR_17

    windows = collector.positives + collector.normals.windows
    metadata = collector.positive_metadata + collector.normals.metadata
    if not windows:
        raise ValueError(f"V2 fold has no windows for {split}")
    labels = np.concatenate((
        np.ones(len(collector.positives), dtype=np.int64),
        np.zeros(len(collector.normals.windows), dtype=np.int64),
    ))
    order = rng.permutation(len(windows))
    np.savez_compressed(
        output / f"chbmit_{split}.npz",
        X=np.stack(windows, axis=0)[order].astype(np.float32, copy=False),
        y=labels[order],
        recording_id=np.asarray([metadata[index]["recording_id"] for index in order]),
        start_sample=np.asarray([metadata[index]["start_sample"] for index in order], dtype=np.int64),
        channels=np.asarray(CANONICAL_BIPOLAR_17),
        split=split,
    )
    return {
        "positive_windows": len(collector.positives),
        "normal_windows": len(collector.normals.windows),
        "normal_candidates_seen": collector.normals.candidates_seen,
        "path": str(output / f"chbmit_{split}.npz"),
    }


def prepare_fold_windows(fold_manifest: str | Path, output_dir: str | Path, config: dict) -> dict:
    """Write clean classifier windows and a full test recording manifest for one fold."""
    import mne
    import numpy as np
    from src.chbmit_preparation import extract_canonical_bipolar_data, filter_eeg
    from src.feature_representation import save_feature_spec

    rows = _load_rows(fold_manifest)
    preprocessing = config["preprocessing"]
    sample_rate = int(config["dataset"]["sample_rate_hz"])
    window_samples = int(float(preprocessing["window_sec"]) * sample_rate)
    stride_samples = int(float(preprocessing["stride_sec"]) * sample_rate)
    guard_samples = int(float(preprocessing["interictal_guard_sec"]) * sample_rate)
    targets = _count_targets(rows, config)
    sampling_seed = int(config["training"]["dataset_sampling_seed"])
    collectors = {
        split: _Collector(targets[split]["normal_target"], np.random.default_rng(sampling_seed + index))
        for index, split in enumerate(ACTIVE_SPLITS)
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    active_rows = [row for row in rows if row["split"] in ACTIVE_SPLITS]
    for record_index, row in enumerate(active_rows, start=1):
        raw = mne.io.read_raw_edf(row["edf_path"], preload=True, verbose="ERROR")
        try:
            if int(round(raw.info["sfreq"])) != sample_rate:
                raise ValueError(f"Unexpected sample rate in {row['recording_id']}")
            data = extract_canonical_bipolar_data(raw)
        finally:
            raw.close()
        data = filter_eeg(
            data, sample_rate, preprocessing["bandpass_hz"][0], preprocessing["bandpass_hz"][1],
            preprocessing["notch_hz"], preprocessing["filter_mode"],
        )
        intervals_seconds = json.loads(row["seizure_intervals_json"])
        intervals = [(round(float(start) * sample_rate), round(float(end) * sample_rate)) for start, end in intervals_seconds]
        positives, normals, _ = causal_window_index(data.shape[1], intervals, window_samples, stride_samples, guard_samples)
        collector = collectors[row["split"]]
        for start in positives:
            collector.add_positive(data[:, start:start + window_samples], {"recording_id": row["recording_id"], "start_sample": int(start)})
        for start in normals:
            collector.normals.add(data[:, start:start + window_samples], {"recording_id": row["recording_id"], "start_sample": int(start)})
        if record_index % 10 == 0 or record_index == len(active_rows):
            print(f"  V2 prepared recordings: {record_index}/{len(active_rows)}")

    outputs = {
        split: _save_split(output, split, collectors[split], np.random.default_rng(sampling_seed + 100 + index))
        for index, split in enumerate(ACTIVE_SPLITS)
    }
    save_feature_spec(output, {"name": "raw", "input_shape": [17, window_samples]})
    test_rows = [row for row in rows if row["split"] == "test"]
    with (output / "continuous_test_recordings.csv").open("w", newline="", encoding="utf-8") as target:
        fields = list(test_rows[0])
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(test_rows)
    summary = {
        "protocol": "research_v2_causal_endpoint",
        "fold_manifest": str(fold_manifest),
        "fold_manifest_sha256": file_sha256(fold_manifest),
        "config_hash": canonical_json_hash(config),
        "window_samples": window_samples,
        "sampling_seed": sampling_seed,
        "targets": targets,
        "outputs": outputs,
        "continuous_test_manifest": str(output / "continuous_test_recordings.csv"),
    }
    save_json(output / "preparation_summary.json", summary)
    return summary
