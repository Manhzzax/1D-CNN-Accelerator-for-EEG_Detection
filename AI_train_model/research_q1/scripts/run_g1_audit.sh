#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CHBMIT_RAW_DIR:?CHBMIT_RAW_DIR is required; no fallback path is permitted}"
cd "${ROOT}"
python -m unittest discover -s research_q1/tests -v
G1_TEST_STATUS=passed python research_q1/scripts/run_g1_audit.py
