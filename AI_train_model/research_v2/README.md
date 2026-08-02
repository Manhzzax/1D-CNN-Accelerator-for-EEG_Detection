# Research V2

This workspace replaces neither `src/` nor legacy runs. It is the immutable
scientific contract for the new CHB-MIT study: shared-model, within-case,
blocked forward temporal evaluation with causal preprocessing and labels.

## Scientific boundaries

- Historical checkpoints, thresholds, and observed test probes are archived
  evidence only. They cannot select a V2 architecture or operating point.
- V2 is an internal longitudinal evaluation, not patient-independent or
  external validation.
- `balanced_accuracy` and `AUROC` are primary window metrics. Accuracy is only
  shown together with its window class distribution. Event sensitivity at
  `FAR <= 0.5/h` is the system endpoint.

## First commands on the server

Run from `AI_train_model` after pulling the repository:

```bash
python -m research_v2 validate --protocol research_v2/configs/protocol_v2.json --registry research_v2/configs/candidate_registry_v2.json
python -m research_v2 inventory --roots results/archive results/reference ../server_results --output research_v2/reports/legacy_inventory.json
python -m research_v2 fold-audit --protocol research_v2/configs/protocol_v2.json --manifest data/chbmit_audit/recording_manifest.csv --output research_v2/manifests/temporal_v2
python -m research_v2 prepare-fold --protocol research_v2/configs/protocol_v2.json --fold-manifest research_v2/manifests/temporal_v2/fold_00_manifest.csv --output research_v2/generated/fold_00
```

The fold audit selects five folds only if every aggregate inner-validation and
outer-test partition contains seizure events. Otherwise it writes the frozen
three-fold fallback; it never moves recordings to manufacture coverage.

`tools/train_fold.sh` trains one V2 candidate without evaluating its outer test
windows. It expects a frozen prepared fold, a unique run ID, and one of the
five declared training seeds. Outer evaluation is a separate post-freeze step.
After training, write `provenance.json` in the run output with the `provenance`
subcommand, then use `bash research_v2/tools/package_run.sh <run_id> <output>`
to commit only the reproducibility artifact.

## Layout

- `configs/`: immutable protocol and candidate registry.
- `manifests/`: generated hashes and forward-fold CSVs; generated on the server.
- `reports/`: inventory, statistical reports, and final tables.
- `literature/`: literature cards and the L1/L2 comparison tables.
- `hardware/`: DPU feasibility evidence and later export contracts.

Generated manifests, prepared EEG data, continuous scores, and model outputs
must not be committed. Commit only configs, source, concise reports, and the
packaged reproducibility artifact for a frozen result.
