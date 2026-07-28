"""Canonical CHB-MIT bipolar montage resolution for training and deployment."""

import re


CANONICAL_BIPOLAR_17 = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "P8-O2", "FZ-CZ", "CZ-PZ",
)

_REFERENTIAL_SUFFIXES = ("REF", "LE", "CS2")
_ELECTRODE_ALIASES = {"01": "O1"}


def normalize_channel_name(name):
    """Normalize label decoration while preserving a direct bipolar channel name."""
    normalized = name.upper().strip()
    normalized = re.sub(r"^EEG\s*", "", normalized)
    return normalized.replace(" ", "")


def referential_electrode_name(channel_name):
    """Return the electrode for a common-reference channel, otherwise None."""
    normalized = normalize_channel_name(channel_name)
    if normalized in _ELECTRODE_ALIASES:
        return _ELECTRODE_ALIASES[normalized]
    if re.fullmatch(r"[A-Z][A-Z0-9]*", normalized):
        return normalized

    suffix_pattern = "|".join(_REFERENTIAL_SUFFIXES)
    match = re.fullmatch(rf"([A-Z][A-Z0-9]*)-(?:{suffix_pattern})", normalized)
    if match:
        return _ELECTRODE_ALIASES.get(match.group(1), match.group(1))
    return None


def resolve_canonical_bipolar_17(channel_names):
    """Resolve direct or re-referenced source channels for the 17-channel montage.

    Each result is `(mode, indices)`, where mode is `direct`, `difference`, or
    `missing`. For `difference`, the signal is `indices[0] - indices[1]`.
    """
    direct_channels = {}
    referential_channels = {}

    for index, channel_name in enumerate(channel_names):
        normalized = normalize_channel_name(channel_name)
        if normalized in CANONICAL_BIPOLAR_17:
            direct_channels.setdefault(normalized, index)

        electrode = referential_electrode_name(channel_name)
        if electrode:
            referential_channels.setdefault(electrode, index)

    resolved = {}
    for bipolar_channel in CANONICAL_BIPOLAR_17:
        if bipolar_channel in direct_channels:
            resolved[bipolar_channel] = ("direct", (direct_channels[bipolar_channel],))
            continue

        left, right = bipolar_channel.split("-", maxsplit=1)
        if left in referential_channels and right in referential_channels:
            resolved[bipolar_channel] = (
                "difference",
                (referential_channels[left], referential_channels[right]),
            )
        else:
            resolved[bipolar_channel] = ("missing", ())

    return resolved
