"""Protocol primitives shared by V2 preparation and evaluation.

The legacy pipeline keeps its historical full-window labels.  This module is
intentionally separate: V2 labels an input by the causal end timestamp of the
window, which is the decision time exposed to the online detector.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

def canonical_json_hash(value: object) -> str:
    """Return a stable SHA-256 digest for JSON-compatible protocol artifacts."""
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def save_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def normalize_intervals(intervals: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """Validate and sort non-overlapping half-open sample intervals."""
    normalized = sorted((int(start), int(end)) for start, end in intervals)
    previous_end = -1
    for start, end in normalized:
        if start < 0 or end <= start:
            raise ValueError("Intervals must be non-negative half-open ranges with end > start")
        if start < previous_end:
            raise ValueError("Seizure intervals must not overlap")
        previous_end = end
    return normalized


def causal_window_index(
    sample_count: int,
    seizure_intervals: Iterable[Sequence[int]],
    window_samples: int,
    stride_samples: int,
    guard_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive, clean-normal, and all window starts for causal labels.

    The endpoint is ``start + window_samples`` and an interval is half-open:
    ``[seizure_start, seizure_end)``.  A window ending exactly at onset is
    therefore the first positive decision; a window ending exactly at offset is
    non-ictal.  Guarding is applied to endpoints and never removes a positive.
    """
    import numpy as np

    if sample_count < 0 or window_samples < 1 or stride_samples < 1 or guard_samples < 0:
        raise ValueError("Invalid window-index arguments")
    if sample_count < window_samples:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    intervals = normalize_intervals(seizure_intervals)
    starts = np.arange(0, sample_count - window_samples + 1, stride_samples, dtype=np.int64)
    endpoints = starts + window_samples
    positive = np.zeros(starts.shape[0], dtype=bool)
    guarded = np.zeros(starts.shape[0], dtype=bool)
    for seizure_start, seizure_end in intervals:
        positive |= (endpoints >= seizure_start) & (endpoints < seizure_end)
        guarded |= (endpoints >= seizure_start - guard_samples) & (endpoints < seizure_end + guard_samples)

    normal = ~(positive | guarded)
    return starts[positive], starts[normal], starts


def validate_protocol_config(config: dict) -> None:
    """Reject protocol states that would invalidate the V2 scientific claim."""
    preprocessing = config["preprocessing"]
    labels = config["labels"]
    split = config["split"]
    evaluation = config["evaluation"]
    training = config["training"]

    if preprocessing["filter_mode"] != "causal_iir":
        raise ValueError("V2 requires causal_iir preprocessing")
    if labels["rule"] != "causal_window_endpoint":
        raise ValueError("V2 requires causal_window_endpoint labels")
    if float(preprocessing["window_sec"]) != 5.0 or float(preprocessing["stride_sec"]) != 1.0:
        raise ValueError("The V2 primary protocol is fixed at 5-second windows and 1-second stride")
    if int(split["requested_outer_folds"]) != 5 or int(split["fallback_outer_folds"]) != 3:
        raise ValueError("V2 requires a five-fold feasibility audit with three-fold fallback")
    if float(evaluation["primary_far_per_hour"]) != 0.5:
        raise ValueError("V2 primary event endpoint is sensitivity at FAR <= 0.5/h")
    expected_seeds = [7, 42, 123, 314, 2718]
    if list(training["training_seeds"]) != expected_seeds:
        raise ValueError(f"V2 training seeds are fixed: {expected_seeds}")
    if training["max_epochs"] != 50 or training["min_epochs"] != 12 or training["early_stopping_patience"] != 12:
        raise ValueError("V2 training budget is fixed at 50/12/12 epochs")
