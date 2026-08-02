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

`prepare-fold` deliberately materializes only train/validation windows by
default. Use `--include-test` only after the candidate and all inner-fold
choices are frozen; this prevents outer future recordings from entering a
development process even as unlabeled tensors.

`tools/train_fold.sh` trains one V2 candidate without evaluating its outer test
windows. It expects a frozen prepared fold, a unique run ID, and one of the
five declared training seeds. Outer evaluation is a separate post-freeze step.
After training, write `provenance.json` in the run output with the `provenance`
subcommand, then use `bash research_v2/tools/package_run.sh <run_id> <output>`
to commit the small checkpoint and reproducibility artifact.

For the selected M31 configuration, create a new train/validation-only
directory and run `bash research_v2/tools/run_m31_seed_confirmation.sh
research_v2/generated/fold_00_trainval_v2`. It enforces the frozen 50/12/12
epoch contract, runs seeds `7, 42, 123, 314, 2718`, and never opens the outer
test set. Its final line prints a single `package_runs.sh` command, which adds
the five small checkpoints and reproducibility files as one result commit.

`tools/screen_hierarchical_grid.sh` performs the same 4-setting inner-grid
screen for a predeclared hierarchical candidate on seed 7. For example, P2
uses candidate tag `p2_m47` and kernels `47 7 3`. It also refuses a prepared
directory containing outer-test tensors.

## Layout

- `configs/`: immutable protocol and candidate registry.
- `manifests/`: generated hashes and forward-fold CSVs; generated on the server.
- `reports/`: inventory, statistical reports, and final tables.
- `literature/`: literature cards and the L1/L2 comparison tables.
- `hardware/`: DPU feasibility evidence and later export contracts.

Generated manifests, prepared EEG data, continuous scores, and model outputs
must not be committed. Commit only configs, source, concise reports, and the
packaged reproducibility artifact for a frozen result.
