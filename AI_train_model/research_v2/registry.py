"""Immutable candidate registry and run provenance for V2 experiments."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .protocol import canonical_json_hash, file_sha256, load_json, save_json


def load_candidate_registry(path: str | Path) -> dict:
    registry = load_json(path)
    candidates = registry.get("candidates", [])
    version = registry.get("version")
    if version in {None, "v2.0.0", "v2.1.0"} and len(candidates) != 8:
        raise ValueError("V2 registry must contain six baselines and two historical references")
    if version == "v2.2.0" and len(candidates) != 1:
        raise ValueError("V2.2-A must contain exactly one predeclared capacity candidate")
    if version == "v2.3.0" and len(candidates) != 1:
        raise ValueError("V2.3 must contain exactly one predeclared hard-negative candidate")
    if version not in {None, "v2.0.0", "v2.1.0", "v2.2.0", "v2.3.0"}:
        raise ValueError(f"Unsupported V2 candidate registry version: {version}")
    seen = set()
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not candidate_id or candidate_id in seen:
            raise ValueError("Candidate IDs must be unique and non-empty")
        seen.add(candidate_id)
        if candidate["parameter_budget"] not in {"classical", "5k_15k", "up_to_25k", "25k_100k"}:
            raise ValueError(f"Unsupported parameter budget for {candidate_id}")
    registry["registry_hash"] = canonical_json_hash({"candidates": candidates, "version": registry.get("version")})
    return registry


def git_commit(project_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_run_provenance(
    output_path: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
    split_path: str | Path,
    checkpoint_path: str | Path | None,
    training_seed: int,
    dataset_sampling_seed: int,
    precision: str,
    registry_path: str | Path | None = None,
    candidate_id: str | None = None,
) -> dict:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": git_commit(project_root),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "split_path": str(split_path),
        "split_sha256": file_sha256(split_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path else None,
        "training_seed": int(training_seed),
        "dataset_sampling_seed": int(dataset_sampling_seed),
        "precision": precision,
        "candidate_registry_path": str(registry_path) if registry_path else None,
        "candidate_registry_sha256": file_sha256(registry_path) if registry_path else None,
        "candidate_id": candidate_id,
    }
    payload["provenance_hash"] = canonical_json_hash(payload)
    save_json(output_path, payload)
    return payload
