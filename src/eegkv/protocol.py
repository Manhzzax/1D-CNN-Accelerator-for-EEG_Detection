"""Immutable G0--G2 data and identity contract."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

CANONICAL_CHANNELS = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ",
    "C4", "T8", "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)

_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "FPZ": "FPZ", "FP1": "FP1", "FP2": "FP2", "FZ": "FZ",
    "CZ": "CZ", "PZ": "PZ",
}


def normalize_channel(label: str) -> str:
    """Normalize an EDF lead label without deriving a new EEG channel."""
    value = label.upper().replace("EEG", "").replace("-REF", "").replace(" ", "")
    return _ALIASES.get(value, value)


def canonical_channel_indices(labels: Iterable[str]) -> list[int] | None:
    """Return strict canonical ordering, or ``None`` when a channel is absent.

    Bipolar derivations such as ``FP1-F7`` are intentionally not accepted as a
    substitute for a named canonical electrode channel.
    """
    normalized = [normalize_channel(label) for label in labels]
    if len(set(normalized)) != len(normalized):
        return None
    try:
        return [normalized.index(channel) for channel in CANONICAL_CHANNELS]
    except ValueError:
        return None


def patient_group(case_id: str) -> str:
    case = case_id.lower()
    if case in {"chb01", "chb21"}:
        return "subject_01_21"
    if not case.startswith("chb") or not case[3:].isdigit():
        raise ValueError(f"Invalid CHB-MIT case id: {case_id}")
    return f"subject_{int(case[3:]):02d}"


def deterministic_validation_groups(
    outer_test_group: str, candidate_groups: Iterable[str], count: int = 4, seed: int = 20260808
) -> list[str]:
    """Choose validation groups deterministically without consulting metrics."""
    groups = sorted(set(candidate_groups) - {outer_test_group})
    if len(groups) < count:
        raise ValueError("Not enough non-test patient groups for validation")
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{outer_test_group}:{group}".encode()).hexdigest())
    return sorted(ranked[:count])

