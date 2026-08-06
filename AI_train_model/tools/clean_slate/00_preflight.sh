#!/usr/bin/env bash
# Clean-slate preflight on the training server.
set -euo pipefail

ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
RAW_DIR="${CHBMIT_RAW_DIR:-/home/ubuntu/Manh/datasets/CHB-MIT/1.0.0}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"

echo "=== git ==="
git status -sb
git branch --show-current
git rev-parse --short HEAD

echo "=== python / cuda ==="
python - <<'PY'
import torch, mne, sklearn, yaml
print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mne=", mne.__version__)
print("sklearn=", sklearn.__version__)
print("yaml_ok=", True)
PY

echo "=== paths ==="
test -d "${RAW_DIR}"
test -f config/clean_slate_v1.yaml
test -d data/chbmit_audit || echo "WARN: data/chbmit_audit missing — run audit before plan"
df -h . "${RAW_DIR}"
nvidia-smi

echo "=== clean_slate config load ==="
CHBMIT_CONFIG_PATH=config/clean_slate_v1.yaml python - <<'PY'
from src.data_loader import load_config
cfg = load_config()
assert cfg["data"]["split_ratios"]["train"] == 0.6
assert cfg["preprocessing"]["window_sec"] == 5
assert cfg["preprocessing"]["filter_mode"] == "causal_iir"
assert cfg["model"]["architecture"] == "hierarchical_separable_1dcnn"
assert cfg["model"]["input_length"] == 1280
print("clean_slate_v1 config OK")
print("protocol=", cfg["data"]["protocol_output_dir"])
print("prepared=", cfg["data"]["prepared_output_dir"])
PY

echo "Preflight complete."
