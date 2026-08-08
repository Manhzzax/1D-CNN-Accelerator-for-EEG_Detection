# EEG-KV260 Q1 Clean-Room Research

This repository starts a new, patient-independent continuous EEG seizure-
detection study. Its only scientific source of truth is the implementation
specification in `docs/spec/`.

The active implementation milestone is **G1 only**: a read-only audit and
canonical manifest over the verified CHB-MIT snapshot. Montage selection,
window extraction, patient-held-out training, event scoring, quantization and
hardware evaluation are blocked until G1 artifacts are reviewed.

No historical experiment, model, checkpoint, split, metric, or hardware
result is part of this repository or may be used as a baseline.

## Quick start

The real audit runs only on the server hosting the independently acquired
CHB-MIT corpus:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
$env:CHBMIT_RAW_DIR = '/home/ubuntu/Manh/datasets/CHB-MIT/1.0.0'
eegkv audit-g1 --output-root artifacts/g1
```

`audit-g1` never downloads, copies, resamples or loads EEG waveforms. Small
audit artifacts under `artifacts/g1/` are reviewable Git artifacts; raw EEG,
prepared arrays, checkpoints and all other generated data remain ignored.
