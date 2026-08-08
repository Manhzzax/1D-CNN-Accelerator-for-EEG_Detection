"""EDF discovery and strict manifest construction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .protocol import CANONICAL_CHANNELS, canonical_channel_indices, patient_group


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_summary(root: Path) -> dict[str, list[list[float]]]:
    """Parse CHB-MIT ``*-summary.txt`` seizure intervals into seconds."""
    result: dict[str, list[list[float]]] = {}
    for summary in root.rglob("*-summary.txt"):
        current: str | None = None
        start: float | None = None
        for raw in summary.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            match = re.match(r"File Name:\s*(\S+\.edf)", line, flags=re.I)
            if match:
                current = match.group(1).lower()
                result.setdefault(current, [])
                start = None
                continue
            start_match = re.match(r"Seizure\s+Start\s+Time:\s*([0-9.]+)\s*seconds", line, flags=re.I)
            if start_match:
                start = float(start_match.group(1))
                continue
            end_match = re.match(r"Seizure\s+End\s+Time:\s*([0-9.]+)\s*seconds", line, flags=re.I)
            if end_match and current is not None and start is not None:
                end = float(end_match.group(1))
                if end > start:
                    result[current].append([start, end])
                start = None
    return result


def _edf_metadata(path: Path) -> tuple[list[str], float, float]:
    try:
        import pyedflib
    except ImportError as error:  # pragma: no cover - dependency guarded by packaging
        raise RuntimeError("pyedflib is required; install the project dependencies first") from error
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = list(reader.getSignalLabels())
        rates = [float(reader.getSampleFrequency(index)) for index in range(reader.signals_in_file)]
        if not rates or len(set(rates)) != 1:
            raise ValueError("EDF does not have one common sampling rate")
        return labels, rates[0], float(reader.getFileDuration())
    finally:
        reader.close()


def build_manifest(edf_root: Path) -> list[dict]:
    summaries = parse_summary(edf_root)
    rows: list[dict] = []
    for edf in sorted(edf_root.rglob("*.edf")):
        case_match = re.search(r"(chb\d{2})", edf.as_posix(), flags=re.I)
        if not case_match:
            continue
        case_id = case_match.group(1).lower()
        labels, sample_rate, duration = _edf_metadata(edf)
        indices = canonical_channel_indices(labels)
        rows.append({
            "dataset": "chbmit-v1.0.0",
            "subject_id": patient_group(case_id),
            "split_group": patient_group(case_id),
            "case_id": case_id,
            "session_id": case_id,
            "recording_id": edf.name.lower(),
            "path": str(edf.resolve()),
            "sha256": sha256(edf),
            "sampling_rate_hz": sample_rate,
            "duration_seconds": duration,
            "source_channels": labels,
            "canonical_channel_indices": indices,
            "channel_coverage": "complete" if indices is not None else "incomplete",
            "seizure_intervals_seconds": summaries.get(edf.name.lower(), []),
        })
    return rows


def write_jsonl(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_manifest(rows: list[dict]) -> None:
    seen_checksums: set[str] = set()
    for row in rows:
        if row["channel_coverage"] != "complete":
            continue
        if row["sampling_rate_hz"] != 256:
            raise ValueError(f"{row['recording_id']} is not sampled at 256 Hz")
        if row["canonical_channel_indices"] is None or len(row["canonical_channel_indices"]) != len(CANONICAL_CHANNELS):
            raise ValueError(f"{row['recording_id']} violates the strict 19-channel contract")
        if row["sha256"] in seen_checksums:
            raise ValueError(f"Duplicate recording checksum: {row['recording_id']}")
        seen_checksums.add(row["sha256"])

