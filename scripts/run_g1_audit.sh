#!/usr/bin/env bash
set -euo pipefail

: "${CHBMIT_RAW_DIR:?CHBMIT_RAW_DIR is required; no fallback path is permitted}"
python -m unittest discover -s tests -p 'test_g1_*.py' -v
python -m eegkv audit-g1 --output-root artifacts/g1
