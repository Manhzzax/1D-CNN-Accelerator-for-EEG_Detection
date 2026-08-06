#!/usr/bin/env bash
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"
export CHBMIT_CONFIG_PATH=config/patient_specific_a1.yaml
export CHBMIT_BALANCED_TEST_SEED=42
for run_dir in outputs/ps_a12_chb*_s42; do
  [[ -d "${run_dir}" ]] || continue
  src="$(basename "${run_dir}")"
  case_id="${src#ps_a12_}"
  case_id="${case_id%_s42}"
  export CHBMIT_PREPARED_OUTPUT_DIR="chbmit_prepared_ps_a1_v1/${case_id}"
  export CHBMIT_CHECKPOINT_SOURCE_RUN_ID="${src}"
  export CHBMIT_RUN_ID="ps_a12_test_${src}"
  if [[ -f "outputs/${CHBMIT_RUN_ID}/checkpoint_test_evaluation.json" ]]; then
    echo "Skip existing test: ${CHBMIT_RUN_ID}"
    continue
  fi
  echo "=== A1.2 TEST ${case_id} ==="
  python main.py --mode checkpoint_eval
done
echo "Done. Aggregate with tools/patient_specific/14_aggregate_a12.sh"
