"""Machine-readable per-subject aggregation."""

from __future__ import annotations

import json
from pathlib import Path


def aggregate_subject_results(paths: list[Path]) -> dict:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not rows:
        raise ValueError("No per-subject results supplied")
    metric_names = ("event_f1", "precision", "sensitivity", "false_positives_per_day")
    macro = {name: sum(row[name] for row in rows) / len(rows) for name in metric_names}
    return {"subject_count": len(rows), "macro": macro, "per_subject": sorted(rows, key=lambda row: row["subject_id"])}

