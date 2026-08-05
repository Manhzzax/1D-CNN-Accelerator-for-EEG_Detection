"""Small config-first command line interface for the V2 research workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

from .folds import load_recording_manifest, select_feasible_protocol, write_protocol_artifacts
from .inventory import write_inventory
from .preparation import prepare_fold_windows, prepare_v21_confirmation_windows, prepare_v21_final_windows
from .protocol import load_json, save_json, validate_protocol_config
from .registry import load_candidate_registry, write_run_provenance
from .v21 import audit_session_timestamps, audit_v21, create_final_freeze, verify_final_freeze, write_v21_artifacts


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


def _v21_audit(args: argparse.Namespace) -> None:
    config = load_json(args.protocol)
    validate_protocol_config(config)
    rows = load_recording_manifest(args.manifest)
    required = {"edf_path", "sample_count", "sampling_rate_hz", "seizure_intervals_json"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"V2.1 manifest requires fields missing from audit: {sorted(missing)}")
    audit = audit_v21(rows, config)
    audit["session_audit"] = audit_session_timestamps(rows, config)
    report = write_v21_artifacts(args.output, audit, config)
    if not audit["valid"]:
        raise RuntimeError("V2.1 duration-based confirmation partitions failed the predeclared feasibility gate")
    print(
        f"V2.1 audit passed: {len(audit['confirmation_folds'])} confirmation folds | "
        f"audit hash: {report['audit_hash']}"
    )


def _v21_prepare_confirmation(args: argparse.Namespace) -> None:
    config = load_json(args.protocol)
    validate_protocol_config(config)
    summary = prepare_v21_confirmation_windows(args.fold_manifest, args.output, config)
    print(f"V2.1 confirmation preparation complete: {summary['outputs']}")


def _v21_freeze_final(args: argparse.Namespace) -> None:
    config = load_json(args.protocol)
    validate_protocol_config(config)
    freeze = create_final_freeze(args.protocol, args.final_manifest, args.decision, args.output)
    print(f"V2.1 final freeze written: {args.output} ({freeze['freeze_hash']})")


def _v21_prepare_final(args: argparse.Namespace) -> None:
    config = load_json(args.protocol)
    validate_protocol_config(config)
    freeze = verify_final_freeze(args.freeze, args.protocol, args.final_manifest)
    marker = Path(args.freeze).with_suffix(".opened.json")
    if marker.exists():
        raise RuntimeError(f"V2.1 final holdout has already been materialized: {marker}")
    if Path(args.output).exists():
        raise RuntimeError(f"V2.1 final output path must be new and empty: {args.output}")
    summary = prepare_v21_final_windows(args.final_manifest, args.output, config)
    save_json(marker, {
        "freeze_hash": freeze["freeze_hash"], "final_manifest": str(args.final_manifest),
        "prepared_output": str(args.output), "status": "final_holdout_materialized_no_science_changes_allowed",
    })
    print(f"V2.1 final preparation authorized by {freeze['freeze_hash']}: {summary['outputs']}")


def _v21_evaluate_confirmation(args: argparse.Namespace) -> None:
    from .v21_evaluation import score_and_evaluate_run

    config = load_json(args.protocol)
    validate_protocol_config(config)
    protocol_label = str(config.get("version", "V2")).upper()
    result = score_and_evaluate_run(
        args.run_dir, args.prepared_dir, args.fold_manifest, config, args.output,
        reuse_existing_scores=args.reuse_existing_scores,
    )
    if result["policy_selection_status"] == "no_feasible_calibration_policy":
        print(f"{protocol_label} temporal confirmation: no calibration policy met the FAR target; temporal evaluation remained sealed")
    else:
        print(
            f"{protocol_label} temporal confirmation: calibration policy {result['selected_calibration_policy']['policy_name']} | "
            f"temporal sensitivity {result['temporal_evaluation']['event_sensitivity']:.4f}"
        )


def _v23_mine_policy_hard_negatives(args: argparse.Namespace) -> None:
    """Materialize only the V2.3 train-derived hard-negative cache."""
    from .v23_hard_negative import build_policy_hard_negative_cache

    config = load_json(args.protocol)
    validate_protocol_config(config)
    registry = load_candidate_registry(args.registry)
    if registry["candidates"][0]["candidate_id"] != "H1_c1_policy_hardneg_57k":
        raise ValueError("V2.3 requires the frozen H1 candidate registry")
    summary = build_policy_hard_negative_cache(
        project_root=args.project_root,
        config=config,
        fold_index=args.fold,
        fold_manifest=args.fold_manifest,
        source_prepared_dir=args.source_prepared_dir,
        output_dir=args.output,
    )
    print(
        f"V2.3 F{args.fold} policy hard-negative cache: "
        f"{summary['positive_windows']} ictal + {summary['source_normal_windows']} source normals + "
        f"{summary['hard_negative_windows']} hard negatives "
        f"(requested {summary['requested_hard_negative_windows']}; reused={summary['cache_reused']})"
    )


def _v24_mine_score_ranked_hard_negatives(args: argparse.Namespace) -> None:
    """Materialize only the V2.4 train-derived score-ranked hard-negative cache."""
    from .v24_score_hard_negative import build_score_ranked_hard_negative_cache

    config = load_json(args.protocol)
    validate_protocol_config(config)
    registry = load_candidate_registry(args.registry)
    if registry["candidates"][0]["candidate_id"] != "H2_c1_score_ranked_hardneg_57k":
        raise ValueError("V2.4 requires the frozen H2 candidate registry")
    summary = build_score_ranked_hard_negative_cache(
        project_root=args.project_root,
        config=config,
        fold_index=args.fold,
        fold_manifest=args.fold_manifest,
        source_prepared_dir=args.source_prepared_dir,
        output_dir=args.output,
        source_score_cache_dir=args.source_score_cache,
    )
    print(
        f"V2.4 F{args.fold} score-ranked hard-negative cache: "
        f"{summary['positive_windows']} ictal + {summary['source_normal_windows']} source normals + "
        f"{summary['hard_negative_windows']} hard negatives "
        f"(requested {summary['requested_hard_negative_windows']}; reused={summary['cache_reused']})"
    )


def _inventory(args: argparse.Namespace) -> None:
    result = write_inventory(args.output, args.roots)
    print(f"Legacy inventory: {result['entry_count']} artifact directories. Output: {args.output}")


def _prepare_fold(args: argparse.Namespace) -> None:
    config = load_json(args.protocol)
    validate_protocol_config(config)
    summary = prepare_fold_windows(args.fold_manifest, args.output, config, include_test=args.include_test)
    print(f"V2 fold preparation complete: {summary['outputs']}")


def _provenance(args: argparse.Namespace) -> None:
    provenance = write_run_provenance(
        args.output,
        project_root=args.project_root,
        config_path=args.protocol,
        split_path=args.fold_manifest,
        checkpoint_path=args.checkpoint,
        training_seed=args.training_seed,
        dataset_sampling_seed=args.dataset_sampling_seed,
        precision=args.precision,
        registry_path=args.registry,
        candidate_id=args.candidate_id,
    )
    print(f"V2 provenance written: {args.output} ({provenance['provenance_hash']})")


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

    v21_folds = subparsers.add_parser("v21-audit", help="Create sealed V2.1 patient-group duration manifests")
    v21_folds.add_argument("--protocol", required=True)
    v21_folds.add_argument("--manifest", required=True)
    v21_folds.add_argument("--output", required=True)
    v21_folds.set_defaults(handler=_v21_audit)

    v21_prepare = subparsers.add_parser("v21-prepare-confirmation", help="Prepare a V2.1 train/calibration/temporal-evaluation cache")
    v21_prepare.add_argument("--protocol", required=True)
    v21_prepare.add_argument("--fold-manifest", required=True)
    v21_prepare.add_argument("--output", required=True)
    v21_prepare.set_defaults(handler=_v21_prepare_confirmation)

    v21_freeze = subparsers.add_parser("v21-freeze-final", help="Freeze the final decision before opening block 6")
    v21_freeze.add_argument("--protocol", required=True)
    v21_freeze.add_argument("--final-manifest", required=True)
    v21_freeze.add_argument("--decision", required=True)
    v21_freeze.add_argument("--output", required=True)
    v21_freeze.set_defaults(handler=_v21_freeze_final)

    v21_final = subparsers.add_parser("v21-prepare-final", help="Prepare block 6 only with a matching final freeze")
    v21_final.add_argument("--protocol", required=True)
    v21_final.add_argument("--final-manifest", required=True)
    v21_final.add_argument("--freeze", required=True)
    v21_final.add_argument("--output", required=True)
    v21_final.set_defaults(handler=_v21_prepare_final)

    v21_eval = subparsers.add_parser("v21-evaluate-confirmation", help="Calibrate on val and evaluate once on temporal_eval")
    v21_eval.add_argument("--protocol", required=True)
    v21_eval.add_argument("--fold-manifest", required=True)
    v21_eval.add_argument("--prepared-dir", required=True)
    v21_eval.add_argument("--run-dir", required=True)
    v21_eval.add_argument("--output", required=True)
    v21_eval.add_argument("--reuse-existing-scores", action="store_true", help="Recover an interrupted evaluation only after recording IDs are verified")
    v21_eval.set_defaults(handler=_v21_evaluate_confirmation)

    v23_mine = subparsers.add_parser("v23-mine-policy-hard-negatives", help="Build a train-only V2.3 policy-aligned hard-negative cache")
    v23_mine.add_argument("--project-root", required=True)
    v23_mine.add_argument("--protocol", required=True)
    v23_mine.add_argument("--registry", required=True)
    v23_mine.add_argument("--fold", required=True, choices=("00", "01", "02"))
    v23_mine.add_argument("--fold-manifest", required=True)
    v23_mine.add_argument("--source-prepared-dir", required=True)
    v23_mine.add_argument("--output", required=True)
    v23_mine.set_defaults(handler=_v23_mine_policy_hard_negatives)

    v24_mine = subparsers.add_parser("v24-mine-score-ranked-hard-negatives", help="Build a train-only V2.4 score-ranked hard-negative cache")
    v24_mine.add_argument("--project-root", required=True)
    v24_mine.add_argument("--protocol", required=True)
    v24_mine.add_argument("--registry", required=True)
    v24_mine.add_argument("--fold", required=True, choices=("00", "01", "02"))
    v24_mine.add_argument("--fold-manifest", required=True)
    v24_mine.add_argument("--source-prepared-dir", required=True)
    v24_mine.add_argument("--output", required=True)
    v24_mine.add_argument("--source-score-cache", help="Optional validated V2.3 train-only source-score cache")
    v24_mine.set_defaults(handler=_v24_mine_score_ranked_hard_negatives)

    inventory = subparsers.add_parser("inventory", help="Hash legacy artifacts without moving them")
    inventory.add_argument("--roots", required=True, nargs="+")
    inventory.add_argument("--output", required=True)
    inventory.set_defaults(handler=_inventory)

    prepare = subparsers.add_parser("prepare-fold", help="Prepare causal windows for one frozen fold")
    prepare.add_argument("--protocol", required=True)
    prepare.add_argument("--fold-manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--include-test", action="store_true", help="Materialize outer-test windows only after model freeze")
    prepare.set_defaults(handler=_prepare_fold)

    provenance = subparsers.add_parser("provenance", help="Write immutable metadata for a completed V2 run")
    provenance.add_argument("--project-root", required=True)
    provenance.add_argument("--protocol", required=True)
    provenance.add_argument("--registry")
    provenance.add_argument("--candidate-id")
    provenance.add_argument("--fold-manifest", required=True)
    provenance.add_argument("--checkpoint")
    provenance.add_argument("--training-seed", required=True, type=int)
    provenance.add_argument("--dataset-sampling-seed", required=True, type=int)
    provenance.add_argument("--precision", required=True, choices=("amp_fp16_train_fp32_evaluate", "fp32"))
    provenance.add_argument("--output", required=True)
    provenance.set_defaults(handler=_provenance)

    args = parser.parse_args()
    args.handler(args)
