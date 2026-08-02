"""Small config-first command line interface for the V2 research workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

from .folds import load_recording_manifest, select_feasible_protocol, write_protocol_artifacts
from .inventory import write_inventory
from .preparation import prepare_fold_windows
from .protocol import load_json, validate_protocol_config
from .registry import load_candidate_registry


def _validate(args: argparse.Namespace) -> None:
    validate_protocol_config(load_json(args.protocol))
    registry = load_candidate_registry(args.registry)
    print(f"V2 protocol valid. Candidate registry: {len(registry['candidates'])} entries.")


def _fold_audit(args: argparse.Namespace) -> None:
    config = load_json(args.protocol)
    validate_protocol_config(config)
    rows = load_recording_manifest(args.manifest)
    artifact, selected = select_feasible_protocol(
        rows,
        requested_folds=config["split"]["requested_outer_folds"],
        fallback_folds=config["split"]["fallback_outer_folds"],
    )
    write_protocol_artifacts(args.output, artifact, selected)
    print(
        f"Temporal fold audit: selected {artifact['selected_outer_folds']} folds "
        f"(fallback_used={artifact['fallback_used']}). Output: {Path(args.output)}"
    )


def _inventory(args: argparse.Namespace) -> None:
    result = write_inventory(args.output, args.roots)
    print(f"Legacy inventory: {result['entry_count']} artifact directories. Output: {args.output}")


def _prepare_fold(args: argparse.Namespace) -> None:
    config = load_json(args.protocol)
    validate_protocol_config(config)
    summary = prepare_fold_windows(args.fold_manifest, args.output, config)
    print(f"V2 fold preparation complete: {summary['outputs']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 CHB-MIT protocol utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate frozen V2 config and registry")
    validate.add_argument("--protocol", required=True)
    validate.add_argument("--registry", required=True)
    validate.set_defaults(handler=_validate)

    folds = subparsers.add_parser("fold-audit", help="Generate blocked forward temporal fold manifests")
    folds.add_argument("--protocol", required=True)
    folds.add_argument("--manifest", required=True)
    folds.add_argument("--output", required=True)
    folds.set_defaults(handler=_fold_audit)

    inventory = subparsers.add_parser("inventory", help="Hash legacy artifacts without moving them")
    inventory.add_argument("--roots", required=True, nargs="+")
    inventory.add_argument("--output", required=True)
    inventory.set_defaults(handler=_inventory)

    prepare = subparsers.add_parser("prepare-fold", help="Prepare causal windows for one frozen fold")
    prepare.add_argument("--protocol", required=True)
    prepare.add_argument("--fold-manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(handler=_prepare_fold)

    args = parser.parse_args()
    args.handler(args)
