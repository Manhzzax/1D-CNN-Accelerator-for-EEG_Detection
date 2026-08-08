#!/usr/bin/env bash
set -euo pipefail

: "${CHBMIT_RAW_DIR:?CHBMIT_RAW_DIR is required; no fallback path is permitted}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing G1A audit: repository worktree is dirty." >&2
  exit 1
fi

PYTHONPATH=src python -m unittest discover -s tests -p 'test_g1_*.py' -v
PYTHONPATH=src python -m eegkv preflight-g1
PYTHONPATH=src python -m eegkv audit-g1 --output-root artifacts/g1
