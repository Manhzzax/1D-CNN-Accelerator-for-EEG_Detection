#!/usr/bin/env bash
# A1.2: same A1.0 hierarchical 31/7/3 backbone + train-only SupCon (coeff 0.05, T=0.1).
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate chbmit-cnn
cd "${ROOT}"
export CHBMIT_CONFIG_PATH=config/patient_specific_a1.yaml
export CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn
export CHBMIT_TRAIN_SEED=42
export CHBMIT_SKIP_TEST_EVALUATION=1
export CHBMIT_CLASS_BALANCED_BATCHES=1
export CHBMIT_SUPERVISED_CONTRASTIVE=1
export CHBMIT_SUPERVISED_CONTRASTIVE_COEFFICIENT=0.05
export CHBMIT_SUPERVISED_CONTRASTIVE_TEMPERATURE=0.1
PREP_ROOT=data/chbmit_prepared_ps_a1_v1
if [[ ! -d "${PREP_ROOT}" ]]; then
  echo "Missing ${PREP_ROOT}."
  exit 1
fi
for case_dir in "${PREP_ROOT}"/chb*; do
  [[ -d "${case_dir}" ]] || continue
  case_id="$(basename "${case_dir}")"
  [[ -f "${case_dir}/chbmit_train.npz" ]] || continue
  run_id="ps_a12_${case_id}_s42"
  if [[ -d "outputs/${run_id}" ]]; then
    echo "Skip existing train: ${run_id}"
    continue
  fi
  if ! python -c "import json; from pathlib import Path; p=Path('${case_dir}')/'preparation_summary.json';
s=json.loads(p.read_text()) if p.is_file() else None; import sys
o=s['outputs'] if s else None
sys.exit(0 if o and o['train']['positive_windows']>=10 and o['val']['positive_windows']>=5 and o['test']['positive_windows']>=5 else 1)"; then
    echo "Skip thin case windows: ${case_id}"
    continue
  fi
  echo "=== A1.2 SUPCON TRAIN ${case_id} ==="
  export CHBMIT_PREPARED_OUTPUT_DIR="chbmit_prepared_ps_a1_v1/${case_id}"
  export CHBMIT_RUN_ID="${run_id}"
  python main.py --mode train
done
echo "A1.2 SupCon training finished."
