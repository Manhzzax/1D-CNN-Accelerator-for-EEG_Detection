#!/usr/bin/env bash
# Prepare raw 5 s windows under the locked clean-slate split.
set -euo pipefail

ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"

export CHBMIT_CONFIG_PATH=config/clean_slate_v1.yaml

if [[ ! -f data/chbmit_protocol_clean_slate_v1/recording_split_manifest.csv ]]; then
  echo "Missing locked split. Run tools/clean_slate/01_plan_split.sh first."
  exit 1
fi

if [[ -f data/chbmit_prepared_raw_5s_clean_v1/chbmit_train.npz ]]; then
  echo "Prepared set already exists: data/chbmit_prepared_raw_5s_clean_v1"
  echo "Refusing to overwrite. Delete only if intentionally rebuilding v1 prepared data."
  exit 1
fi

python main.py --mode preprocess

echo "Prepared:"
ls -la data/chbmit_prepared_raw_5s_clean_v1/ | head
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("data/chbmit_prepared_raw_5s_clean_v1/preparation_summary.json").read_text())
print("filter_mode=", summary.get("filter_mode"))
print("window_sec=", summary.get("window_sec"))
print("outputs=", json.dumps(summary.get("outputs"), indent=2))
PY
