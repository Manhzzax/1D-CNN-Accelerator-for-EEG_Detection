# Research V2

## V2.1 Patient-Group Protocol

V2.1 is the active protocol on branch `research/v2.1-patient-forward`. It
preserves the V2 causal 5 s / 1 s shared-model setting but corrects the
evaluation design: `chb01` and `chb21` form one patient group, recording blocks
are formed by cumulative EEG duration, and block 6 cannot be materialized
until a hashed final decision is frozen. V2.0 folders and results are pilot
archive evidence only.

Confirmation has three forward folds: `F00: train 0, calibrate 1, evaluate 2`,
`F01: train 0-1, calibrate 2, evaluate 3`, and `F02: train 0-2, calibrate 3,
evaluate 4`. Blocks 5 and 6 are excluded from confirmation. Calibration alone
chooses the predeclared threshold/policy grid; temporal evaluation is reported
once with that policy. Results use patient-group cluster bootstrap CIs and a
Poisson FAR interval, so repeated seeds and the two sessions of subject 01/21
are never treated as independent patients.

On the server, first run:

```bash
bash research_v2/tools/prepare_v21_confirmation_caches.sh
bash research_v2/tools/train_v21_candidate.sh research_v2/generated_v21/f00_confirmation 00 B2_deep_matched_1dcnn
```

The first command caches each fold once and validates hashes on every later
call. The second command is restricted to B0, B1, B2, and B4 and never creates
or reads a block-6 tensor. Package a completed candidate using the exact
command printed by the script.

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

For the full study, use `bash research_v2/tools/prepare_all_trainval_folds.sh`
once after the fold audit. It builds the reusable train/validation-only NPZ
cache for every selected outer fold, validates its manifest and protocol hashes
on later invocations, and refuses both partial caches and any outer-test tensor.
Every architecture, optimizer setting, and seed must reuse that fold cache.

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

After a grid selects a configuration by checkpoint validation loss,
`tools/confirm_hierarchical_seeds.sh` confirms it across the five fixed seeds.
For P2, use run prefix `v2_f00_p2_m47_lr3e4_wd5e4`, kernels `47 7 3`, learning
rate `0.0003`, and weight decay `0.0005`.

For baselines B0 through B5, use `tools/screen_baseline_grid.sh` with the
exact candidate ID from `configs/candidate_registry_v2.json`. The script maps
the candidate ID to its frozen implementation, performs the four-setting
inner grid on seed 7, records the registry hash in provenance, and refuses
outer-test tensors.

For any baseline that is competitive in its seed-7 screen, use
`tools/confirm_baseline_seeds.sh` with the selected grid setting before making
an architecture decision. It preserves the same V2 budget and packages the
five fixed seed IDs for a single push.

## Layout

- `configs/`: immutable protocol and candidate registry.
- `manifests/`: generated hashes and forward-fold CSVs; generated on the server.
- `reports/`: inventory, statistical reports, and final tables.
- `literature/`: literature cards and the L1/L2 comparison tables.
- `hardware/`: DPU feasibility evidence and later export contracts.

Generated manifests, prepared EEG data, continuous scores, and model outputs
must not be committed. Commit only configs, source, concise reports, and the
packaged reproducibility artifact for a frozen result.
