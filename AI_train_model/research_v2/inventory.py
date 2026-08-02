"""Non-destructive inventory of legacy experiment artifacts."""

from __future__ import annotations

from pathlib import Path

from .protocol import file_sha256, save_json


INTERESTING_FILES = (
    "best_model.pth",
    "model_spec.json",
    "training_summary.json",
    "validation_window_metrics.json",
    "event_metrics.json",
    "hyperparameters.json",
)


def inventory_legacy_roots(roots: list[str | Path]) -> dict:
    entries = []
    for root_value in roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
            files = [directory / name for name in INTERESTING_FILES if (directory / name).is_file()]
            if not files:
                continue
            entries.append({
                "directory": str(directory),
                "files": [
                    {"name": file.name, "bytes": file.stat().st_size, "sha256": file_sha256(file)}
                    for file in files
                ],
            })
    return {"inventory_version": 1, "entries": entries, "entry_count": len(entries)}


def write_inventory(output_path: str | Path, roots: list[str | Path]) -> dict:
    inventory = inventory_legacy_roots(roots)
    save_json(output_path, inventory)
    return inventory
