#!/usr/bin/env bash
# Train one inner-selected V2 configuration without exposing its outer test set.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <prepared_train_val_dir> <run_id> <training_seed>" >&2
  exit 2
fi

prepared_dir="$1"
run_id="$2"
seed="$3"
root="$(git rev-parse --show-toplevel)/AI_train_model"

cd "$root"
CHBMIT_WINDOW_SEC=5 \
CHBMIT_FILTER_MODE=causal_iir \
CHBMIT_PREPARED_DIR="$prepared_dir" \
CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn \
CHBMIT_TRAIN_SEED="$seed" \
CHBMIT_RUN_ID="$run_id" \
CHBMIT_SKIP_TEST_EVALUATION=1 \
CHBMIT_TRAIN_EPOCHS=50 \
CHBMIT_EARLY_STOPPING_MIN_EPOCHS=12 \
CHBMIT_EARLY_STOPPING_PATIENCE=12 \
CHBMIT_EARLY_STOPPING_MIN_DELTA=0.001 \
python main.py --mode train
