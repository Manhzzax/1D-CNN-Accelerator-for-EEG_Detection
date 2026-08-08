# EEG-KV260 Q1 Clean-Room Research

This repository starts a new, patient-independent continuous EEG seizure-
detection study. Its only scientific source of truth is the implementation
specification in `docs/spec/`.

The first implementation milestone is deliberately limited to G0--G2:

1. an audited strict 19-channel CHB-MIT manifest and subject-disjoint LOSO
   folds;
2. continuous, event-level evaluation with validation-only policy selection;
3. a raw FP32 1-D CNN baseline.

No historical experiment, model, checkpoint, split, metric, or hardware
result is part of this repository or may be used as a baseline.

## Quick start

Create an environment and install this project, then set the location of an
independently acquired CHB-MIT corpus:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
$env:EEG_DATA_ROOT = 'D:\datasets\chbmit-1.0.0'
eegkv build-manifest --edf-root $env:EEG_DATA_ROOT --output artifacts/manifest.jsonl
eegkv make-loso-folds --manifest artifacts/manifest.jsonl --output artifacts/folds
```

`build-manifest` fails rather than inventing missing channels. The data and
generated artifacts are intentionally ignored by Git.

