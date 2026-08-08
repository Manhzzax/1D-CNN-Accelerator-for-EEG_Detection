"""Command line interface for the clean-room G0--G2 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .events import Event, false_positives_per_day, score_events
from .manifest import build_manifest, read_jsonl, validate_manifest, write_jsonl
from .results import aggregate_subject_results
from .splits import make_loso_folds, write_folds
from .training import train_fp32


def _events(value: str) -> list[Event]:
    return [Event(float(item[0]), float(item[1])) for item in json.loads(Path(value).read_text(encoding="utf-8"))]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eegkv")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--edf-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    folds = commands.add_parser("make-loso-folds")
    folds.add_argument("--manifest", type=Path, required=True)
    folds.add_argument("--output", type=Path, required=True)
    folds.add_argument("--seed", type=int, default=20260808)
    train = commands.add_parser("train-fp32")
    train.add_argument("--train-npz", type=Path, required=True)
    train.add_argument("--validation-npz", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    score = commands.add_parser("score-events")
    score.add_argument("--reference", required=True)
    score.add_argument("--predicted", required=True)
    score.add_argument("--replay-seconds", type=float, required=True)
    score.add_argument("--subject-id", required=True)
    score.add_argument("--output", type=Path, required=True)
    aggregate = commands.add_parser("aggregate-results")
    aggregate.add_argument("--inputs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-manifest":
        rows = build_manifest(args.edf_root); validate_manifest(rows); write_jsonl(rows, args.output)
        print(json.dumps({"recordings": len(rows), "complete": sum(row["channel_coverage"] == "complete" for row in rows)}))
    elif args.command == "make-loso-folds":
        output = make_loso_folds(read_jsonl(args.manifest), seed=args.seed); write_folds(output, args.output); print(json.dumps({"folds": len(output)}))
    elif args.command == "train-fp32":
        print(json.dumps(train_fp32(args.train_npz, args.validation_npz, args.output, args.seed), sort_keys=True))
    elif args.command == "score-events":
        result = score_events(_events(args.reference), _events(args.predicted)); result.update({"subject_id": args.subject_id, "replay_seconds": args.replay_seconds, "false_positives_per_day": false_positives_per_day(result["false_positive_events"], args.replay_seconds)})
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        result = aggregate_subject_results(args.inputs); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
