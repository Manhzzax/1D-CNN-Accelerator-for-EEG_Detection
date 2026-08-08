#!/usr/bin/env bash
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"
export CHBMIT_CONFIG_PATH=config/patient_specific_a1.yaml
export CHBMIT_PATH_A_REPLAY_ALLOW_EXPOSED_TEST=true
export CHBMIT_PATH_A_REPLAY_USE_AMP=false
for run_dir in outputs/ps_a12_chb*_s42; do
  [[ -d "${run_dir}" ]] || continue
  source_run="$(basename "${run_dir}")"
  case_id="${source_run#ps_a12_}"
  case_id="${case_id%_s42}"
  export CHBMIT_CASE_ID="${case_id}"
  export CHBMIT_CHECKPOINT_SOURCE_RUN_ID="${source_run}"
  export CHBMIT_RUN_ID="ps_a12_event_${source_run}"
  if [[ -f "outputs/${CHBMIT_RUN_ID}/patient_specific_event_replay.json" ]]; then
    echo "Skip existing replay: ${case_id}"
    continue
  fi
  python scripts/run_patient_specific_event_replay.py
done
