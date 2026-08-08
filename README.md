# EEG-KV260 Q1 Clean-Room Research

This repository starts a new, patient-independent continuous EEG seizure-
detection study. Its only scientific source of truth is the implementation
specification in `docs/spec/`.

The active implementation milestone is **G1A only**: a read-only audit and
canonical manifest contract over a pinned CHB-MIT snapshot. Montage selection,
window extraction, patient-held-out training, event scoring, quantization, and
hardware evaluation are blocked until real G1A artifacts are reviewed.

No historical experiment, model, checkpoint, split, metric, or hardware result
is part of this repository or may be used as a baseline.

## Execution model

Codex/local development never has authority to claim a CHB-MIT audit passed.
The raw corpus is available only to the approved server operator. See
[`docs/EXECUTION_MODEL.md`](docs/EXECUTION_MODEL.md) for the handoff model and
[`docs/spec/ERRATA.md`](docs/spec/ERRATA.md) for snapshot-specific facts.

## Local development — no CHB-MIT data

Use synthetic tests only. They do not establish anything about the real dataset
or SERVER-02.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

## SERVER-02 operator — approved commit only

After the G1A pull request is approved and merged, substitute the approved
commit SHA below. The server must be clean and must checkout that exact SHA;
it does not run an unreviewed branch tip. From
`/home/ubuntu/Manh/repos/eegkv`, use this one audit command:

Prerequisites are server-operator responsibilities: Python 3.11 or newer,
project dependencies already installed, and successful imports of `pyedflib`
and `wfdb`. The audit script does not install packages or modify the server
environment.

```bash
git fetch origin
git status --short
git checkout --detach <APPROVED_COMMIT_SHA>
CHBMIT_RAW_DIR=/home/ubuntu/Manh/datasets/CHB-MIT/1.0.0 \
  bash scripts/run_g1_audit.sh
```

The script first runs G1 synthetic tests, then the read-only JSON preflight,
then the full checksum audit. It never uses checksum-skip mode or overwrites an
existing artifact. `audit-g1` never downloads, copies, resamples, or loads EEG
waveforms. Only manifests, hashes, census, reports, and shareable provenance
may be committed; raw EDF, cache, windows, checkpoints, and private server logs
remain ignored.
