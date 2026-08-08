#!/usr/bin/env bash
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"
python -m research_path_a_final audit \
  --window-glob 'outputs/ps_a12_test_ps_a12_chb*_s42/checkpoint_test_evaluation.json' \
  --output research_v2/reports/path_a/ps_a12_exploratory_window_audit.json
