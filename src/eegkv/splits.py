"""Outer LOSO fold generation with deterministic subject-disjoint validation."""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import deterministic_validation_groups


def make_loso_folds(rows: list[dict], validation_group_count: int = 4, seed: int = 20260808) -> list[dict]:
    usable = [row for row in rows if row["channel_coverage"] == "complete"]
    groups = sorted({row["split_group"] for row in usable})
    if len(groups) != 23:
        raise ValueError(f"Expected exactly 23 usable participant groups, found {len(groups)}")
    folds: list[dict] = []
    for outer_test in groups:
        val = deterministic_validation_groups(outer_test, groups, validation_group_count, seed)
        train = sorted(set(groups) - {outer_test} - set(val))
        if set(train) & set(val) or outer_test in train or outer_test in val:
            raise AssertionError("Subject leakage while constructing a LOSO fold")
        folds.append({
            "protocol_id": "chbmit_loso_continuous_v1",
            "outer_test_subject": outer_test,
            "validation_subjects": val,
            "training_subjects": train,
            "recordings": {
                "train": [row["recording_id"] for row in usable if row["split_group"] in train],
                "validation": [row["recording_id"] for row in usable if row["split_group"] in val],
                "test": [row["recording_id"] for row in usable if row["split_group"] == outer_test],
            },
        })
    return folds


def write_folds(folds: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        target = output_dir / f"{fold['outer_test_subject']}.json"
        target.write_text(json.dumps(fold, indent=2, sort_keys=True) + "\n", encoding="utf-8")

