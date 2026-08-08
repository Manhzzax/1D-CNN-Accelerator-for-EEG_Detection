"""Continuous event construction and one-to-one event scoring."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True, order=True)
class Event:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("An event must have positive duration")


def merge_events(events: list[Event], gap_seconds: float = 90.0) -> list[Event]:
    merged: list[Event] = []
    for event in sorted(events):
        if merged and event.start - merged[-1].end < gap_seconds:
            merged[-1] = Event(merged[-1].start, max(merged[-1].end, event.end))
        else:
            merged.append(event)
    return merged


def split_long_events(events: list[Event], maximum_seconds: float = 300.0) -> list[Event]:
    output: list[Event] = []
    for event in events:
        start = event.start
        while start < event.end:
            end = min(start + maximum_seconds, event.end)
            output.append(Event(start, end))
            start = end
    return output


def probabilities_to_events(
    points: list[tuple[float, float]], threshold: float, window_seconds: float = 4.0, merge_gap_seconds: float = 90.0
) -> list[Event]:
    """Convert timestamped window probabilities to merged positive event spans."""
    positives = [Event(timestamp, timestamp + window_seconds) for timestamp, probability in sorted(points) if probability >= threshold]
    return merge_events(positives, merge_gap_seconds)


def score_events(
    reference: list[Event], predicted: list[Event], before_seconds: float = 30.0,
    after_seconds: float = 60.0, merge_gap_seconds: float = 90.0, split_seconds: float = 300.0,
) -> dict:
    """Score events with tolerance-expanded one-to-one matching.

    This is a transparent SzCORE-compatible starting implementation; every
    match is returned for independent audit and regression testing.
    """
    references = split_long_events(sorted(reference), split_seconds)
    predictions = merge_events(sorted(predicted), merge_gap_seconds)
    unmatched = set(range(len(references)))
    matches: list[dict] = []
    false_alarms: list[Event] = []
    for prediction in predictions:
        candidates = [
            index for index in unmatched
            if prediction.end >= references[index].start - before_seconds
            and prediction.start <= references[index].end + after_seconds
        ]
        if not candidates:
            false_alarms.append(prediction)
            continue
        index = min(candidates, key=lambda item: abs(prediction.start - references[item].start))
        unmatched.remove(index)
        matches.append({
            "reference_index": index,
            "prediction": {"start": prediction.start, "end": prediction.end},
            "delay_seconds": max(0.0, prediction.start - references[index].start),
        })
    tp, fp, fn = len(matches), len(false_alarms), len(unmatched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    delays = [match["delay_seconds"] for match in matches]
    return {
        "true_positive_events": tp, "false_positive_events": fp, "false_negative_events": fn,
        "precision": precision, "sensitivity": sensitivity, "event_f1": f1,
        "median_detection_delay_seconds": median(delays) if delays else None,
        "mean_detection_delay_seconds": sum(delays) / len(delays) if delays else None,
        "matches": matches,
        "false_alarm_events": [{"start": event.start, "end": event.end} for event in false_alarms],
    }


def false_positives_per_day(false_positive_events: int, replay_seconds: float) -> float:
    return false_positive_events * 86400.0 / replay_seconds if replay_seconds > 0 else 0.0

