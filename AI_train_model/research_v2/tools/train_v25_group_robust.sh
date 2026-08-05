#!/usr/bin/env bash
# Run the preregistered five-seed V2.5 patient-group robust candidate on one fold.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(00|01|02)$ ]]; then
  echo "Usage: $0 <fold_index:00|01|02>" >&2
  exit 2
fi

fold="$1"
repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.5-patient-group-robustness" ]] || { echo "Wrong branch: $branch" >&2; exit 1; }
prepared_dir="$root/research_v2/generated_v21/f${fold}_confirmation"
manifest="$root/research_v2/manifests/temporal_v21/confirmation_f${fold}_manifest.csv"
[[ -f "$prepared_dir/chbmit_train.npz" && -f "$prepared_dir/chbmit_val.npz" && -f "$prepared_dir/chbmit_temporal_eval.npz" ]] || {
  echo "Run prepare_v25_group_robust_caches.sh first" >&2; exit 1;
}
[[ ! -e "$prepared_dir/chbmit_test.npz" && ! -e "$prepared_dir/continuous_test_recordings.csv" ]] || {
  echo "Prepared cache contains a sealed-test artifact" >&2; exit 1;
}

cd "$root"
for seed in 7 42 123 314 2718; do
  run="v25_f${fold}_g1_groupdro57k_s${seed}"
  run_dir="$root/outputs/$run"
  [[ ! -e "$run_dir" ]] || {
    echo "Refusing to overwrite an existing V2.5 run directory: $run_dir" >&2
    echo "Inspect and remove only an incomplete directory before re-running this immutable run ID." >&2
    exit 1
  }
  CHBMIT_V2_MODEL_ARCHITECTURE=paper_a_multiscale_residual_1dcnn \
  CHBMIT_TRAIN_LEARNING_RATE=3e-4 \
  CHBMIT_TRAIN_WEIGHT_DECAY=5e-4 \
  CHBMIT_CLASS_BALANCED_BATCHES=false \
  CHBMIT_PATIENT_GROUP_BALANCED_BATCHES=true \
  CHBMIT_GROUP_DRO=true \
  CHBMIT_GROUP_DRO_ETA=0.01 \
  CHBMIT_PAPERA_TEMPORAL_FILTERS_PER_BRANCH=2 \
  CHBMIT_PAPERA_STAGE_ONE_FILTERS=64 \
  CHBMIT_PAPERA_STAGE_TWO_FILTERS=96 \
  CHBMIT_PAPERA_STAGE_THREE_FILTERS=128 \
  CHBMIT_PAPERA_SHORT_KERNEL=15 \
  CHBMIT_PAPERA_LONG_KERNEL=47 \
  CHBMIT_PAPERA_STAGE_ONE_KERNEL=7 \
  CHBMIT_PAPERA_STAGE_TWO_KERNEL=5 \
  CHBMIT_PAPERA_STAGE_THREE_KERNEL=3 \
  CHBMIT_PAPERA_STAGE_TWO_DILATION=2 \
  CHBMIT_PAPERA_STAGE_THREE_DILATION=4 \
  CHBMIT_PAPERA_DROPOUT=0.25 \
  bash research_v2/tools/train_fold.sh "$prepared_dir" "$run" "$seed"
  python - "$run/model_spec.json" <<'PY'
import json
import sys

spec = json.load(open(f"outputs/{sys.argv[1]}", encoding="utf-8"))
if spec.get("architecture") != "paper_a_multiscale_residual_1dcnn" or spec.get("parameter_count") != 57446:
    raise SystemExit(f"Unexpected V2.5 G1 model contract: {spec.get('architecture')} / {spec.get('parameter_count')}")
PY
  python -m research_v2 provenance --project-root "$repo_root" --protocol research_v2/configs/protocol_v2_5.json --registry research_v2/configs/candidate_registry_v2_5.json --candidate-id G1_c1_groupdro_balanced_57k --fold-manifest "$manifest" --checkpoint "outputs/$run/best_model.pth" --training-seed "$seed" --dataset-sampling-seed 20260802 --precision amp_fp16_train_fp32_evaluate --output "outputs/$run/provenance.json"
  python -m research_v2 v21-evaluate-confirmation --protocol research_v2/configs/protocol_v2_5.json --fold-manifest "$manifest" --prepared-dir "$prepared_dir" --run-dir "outputs/$run" --output "outputs/$run/v25_development"
done

printf 'V2.5 G1 fold completed. Package with:\n'
printf 'bash research_v2/tools/package_v25_runs.sh'
for seed in 7 42 123 314 2718; do printf ' v25_f%s_g1_groupdro57k_s%s' "$fold" "$seed"; done
printf '\n'
