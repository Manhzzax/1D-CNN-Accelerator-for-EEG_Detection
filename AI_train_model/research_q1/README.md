# Q1 Patient-Independent EEG Track

This namespace is isolated from legacy experiments. G1 performs only a
read-only CHB-MIT snapshot audit; it does not choose a montage, create a split,
prepare windows, train a model, or run hardware work.

## Layout

- `configs/`: immutable audit policy and documented tolerances.
- `docs/`: G1 recording-boundary and portability contract.
- `src/`: standard-library EDF/header, inventory, annotation and census logic.
- `scripts/`: server entrypoint.
- `tests/`: fixture-based invariants with no dataset dependency.
- `manifests/` and `reports/`: small generated audit artifacts only; they are
  intentionally absent until the server run and are never auto-committed.

## Server run

```bash
cd /home/ubuntu/Manh/1D-CNN-Accelerator-for-EEG_Detection-q1/AI_train_model
export CHBMIT_RAW_DIR=/home/ubuntu/Manh/datasets/CHB-MIT/1.0.0
bash research_q1/scripts/run_g1_audit.sh
```

The command checks all snapshot checksums and fails on any source-manifest,
annotation, interval, duplicate identity, or EDF-header anomaly. It does not
overwrite generated G1 outputs. Use `python research_q1/scripts/run_g1_audit.py
--replace` only after explicitly preserving a prior small audit artifact set.
