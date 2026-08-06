#!/usr/bin/env bash
# Lock within-case chronological 60/20/20 split for clean-slate v1.
set -euo pipefail

ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"

export CHBMIT_CONFIG_PATH=config/clean_slate_v1.yaml

if [[ ! -d data/chbmit_audit ]]; then
  echo "Audit artifacts missing. Run: python main.py --mode audit"
  echo "Ensure config data.raw_dir points at the CHB-MIT root first."
  exit 1
fi

if [[ -f data/chbmit_protocol_clean_slate_v1/recording_split_manifest.csv ]]; then
  echo "Protocol already exists: data/chbmit_protocol_clean_slate_v1"
  echo "Refusing to overwrite a locked clean-slate split."
  echo "Remove it explicitly only if you intend to re-lock a NEW protocol version."
  exit 1
fi

python main.py --mode plan

echo "Locked:"
ls -la data/chbmit_protocol_clean_slate_v1/
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("data/chbmit_protocol_clean_slate_v1/split_plan_summary.json").read_text())
print(json.dumps(summary["aggregate"], indent=2))
print("ratios=", summary["split_ratios"])
print("strategy=", summary["strategy"])
PY
