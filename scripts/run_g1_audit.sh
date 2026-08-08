#!/usr/bin/env bash
set -euo pipefail

: "${CHBMIT_RAW_DIR:?CHBMIT_RAW_DIR is required; no fallback path is permitted}"
PYTHONPATH=src python -m unittest discover -s tests -p 'test_g1_*.py' -v
PYTHONPATH=src python -m eegkv audit-g1 --output-root artifacts/g1
