"""One-to-one causal event matching for the V2 full-recording replay."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .protocol import normalize_intervals


def temporal_alarms(
    decision_times: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    refractory_samples: int,
    positive_windows: int = 1,
    decision_window_windows: int = 1,
) -> list[int]:
    """Create causal alarms with an explicit trailing positive-count policy."""
    if not 0 < threshold <= 1 or refractory_samples < 0:
        raise ValueError("Invalid alarm threshold or refractory period")
    if not 1 <= positive_windows <= decision_window_windows:
        raise ValueError("Invalid temporal policy")
    times = list(map(int, decision_times))
    scores = np.asarray(list(probabilities), dtype=np.float64)
    if len(times) != len(scores):
        raise ValueError("decision_times and probabilities must have equal length")
    hits = scores >= threshold
    alarms, hit_count, next_allowed = [], 0, -1
    for index, decision_time in enumerate(times):
        hit_count += int(hits[index])
        if index >= decision_window_windows:
            hit_count -= int(hits[index - decision_window_windows])
        if index + 1 < decision_window_windows or hit_count < positive_windows or decision_time < next_allowed:
            continue
        alarms.append(decision_time)
        next_allowed = decision_time + refractory_samples
    return alarms


def match_alarms_to_events(
    alarms: Iterable[int], seizure_intervals: Iterable[Sequence[int]]
) -> tuple[list[dict], list[int]]:
    """Match at most one alarm to each seizure and never reuse an alarm."""
    events = normalize_intervals(seizure_intervals)
    unmatched = sorted(map(int, alarms))
    matches = []
    for event_index, (start, end) in enumerate(events):
        match_index = next((index for index, alarm in enumerate(unmatched) if start <= alarm < end), None)
        if match_index is None:
            matches.append({"event_index": event_index, "detected": False, "delay_samples": None})
            continue
        alarm = unmatched.pop(match_index)
        matches.append({"event_index": event_index, "detected": True, "delay_samples": alarm - start})
    return matches, unmatched


def event_metrics_from_records(records: Iterable[dict], sample_rate: float) -> dict:
    """Aggregate already-generated alarms over full recordings.

    ``records`` contains complete seizure intervals and all alarms, including
    alarms near seizure boundaries.  Therefore the FAR denominator remains the
    true non-ictal duration rather than a sampled 1:10 window subset.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    total_events = detected_events = false_alarms = 0
    interictal_samples = 0
    delays = []
    for record in records:
        intervals = normalize_intervals(record["seizure_intervals"])
        matches, unmatched = match_alarms_to_events(record["alarms"], intervals)
        total_events += len(intervals)
        detected_events += sum(match["detected"] for match in matches)
        false_alarms += len(unmatched)
        interictal_samples += int(record["sample_count"]) - sum(end - start for start, end in intervals)
        delays.extend(match["delay_samples"] / sample_rate for match in matches if match["detected"])
    hours = interictal_samples / sample_rate / 3600.0
    return {
        "event_sensitivity": detected_events / total_events if total_events else 0.0,
        "detected_events": detected_events,
        "total_events": total_events,
        "false_alarms": false_alarms,
        "false_alarms_per_hour": false_alarms / hours if hours else 0.0,
        "interictal_hours": hours,
        "median_detection_delay_sec": float(np.median(delays)) if delays else None,
        "mean_detection_delay_sec": float(np.mean(delays)) if delays else None,
        "alarm_matching": "one_to_one_first_alarm_per_event",
    }
