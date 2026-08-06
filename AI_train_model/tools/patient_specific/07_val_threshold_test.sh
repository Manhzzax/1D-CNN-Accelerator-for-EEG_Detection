#!/usr/bin/env bash
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"
export CHBMIT_CONFIG_PATH=config/patient_specific_a1.yaml
export CHBMIT_BALANCED_TEST_SEED=42
python main.py --mode ps_val_threshold_eval
