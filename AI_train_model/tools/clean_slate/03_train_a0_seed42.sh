#!/usr/bin/env bash
# Clean-slate A0: hierarchical 31/7/3, seed 42, validation only (test sealed).
set -euo pipefail

ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"

export CHBMIT_CONFIG_PATH=config/clean_slate_v1.yaml
export CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_5s_clean_v1
export CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn
export CHBMIT_TRAIN_SEED=42
export CHBMIT_RUN_ID=clean_a0_r2_31_7_3_s42
export CHBMIT_SKIP_TEST_EVALUATION=1
export CHBMIT_CLASS_BALANCED_BATCHES=1

if [[ ! -f data/chbmit_prepared_raw_5s_clean_v1/chbmit_train.npz ]]; then
  echo "Prepared data missing. Run tools/clean_slate/02_prepare_windows.sh first."
  exit 1
fi

if [[ -d outputs/clean_a0_r2_31_7_3_s42 ]]; then
  echo "Run directory already exists: outputs/clean_a0_r2_31_7_3_s42"
  echo "Refusing to overwrite. Choose a new CHBMIT_RUN_ID if this was intentional."
  exit 1
fi

python main.py --mode train

echo "A0 complete. Inspect:"
echo "  outputs/clean_a0_r2_31_7_3_s42/validation_window_metrics.json"
echo "  outputs/clean_a0_r2_31_7_3_s42/model_spec.json"
echo "  outputs/clean_a0_r2_31_7_3_s42/training_summary.json"
echo "Test remains sealed (SKIP_TEST_EVALUATION=1)."
