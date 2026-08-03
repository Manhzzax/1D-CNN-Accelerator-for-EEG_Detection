#!/usr/bin/env bash
# Confirm one inner-selected V2 baseline setting across all five fixed seeds.
set -euo pipefail

if [[ $# -ne 4 && $# -ne 5 ]]; then
  echo "Usage: $0 <prepared_train_val_dir> <candidate_id> <learning_rate> <weight_decay> [fold_index]" >&2
  exit 2
fi

prepared_dir="$1"
candidate_id="$2"
learning_rate="$3"
weight_decay="$4"
fold_index="${5:-0}"
if [[ ! "$fold_index" =~ ^[0-9]+$ ]]; then
  echo "fold_index must be a non-negative integer" >&2
  exit 2
fi
fold_tag="$(printf 'f%02d' "$fold_index")"
repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || {
  echo "Refusing to run from a detached HEAD" >&2
  exit 1
}
if [[ "$branch" != "research/v2-scientific-reports" ]]; then
  echo "Refusing to run V2 confirmation outside research/v2-scientific-reports" >&2
  exit 1
fi
[[ -f "$prepared_dir/chbmit_train.npz" && -f "$prepared_dir/chbmit_val.npz" ]] || {
  echo "Missing V2 train/validation tensors in $prepared_dir" >&2
  exit 1
}
[[ ! -e "$prepared_dir/chbmit_test.npz" ]] || {
  echo "Refusing a prepared directory containing outer-test tensors" >&2
  exit 1
}

case "$candidate_id" in
  B0_bandpower_linear) tag="b0_bandpower_linear"; architecture="v2_bandpower_linear" ;;
  B1_vanilla_1dcnn) tag="b1_vanilla"; architecture="v2_vanilla_1dcnn" ;;
  B2_deep_matched_1dcnn) tag="b2_deep_matched"; architecture="v2_deep_matched_1dcnn" ;;
  B3_multiscale_inception_1dcnn) tag="b3_multiscale"; architecture="parallel_multikernel_1dcnn" ;;
  B4_dilated_lightseizure_like) tag="b4_dilated"; architecture="dilated_hierarchical_separable_1dcnn" ;;
  B5_eegwavenet_like) tag="b5_residual"; architecture="paper_a_multiscale_residual_1dcnn" ;;
  *) echo "Unknown registered baseline: $candidate_id" >&2; exit 2 ;;
esac

case "$learning_rate:$weight_decay" in
  0.001:0.0001) setting="lr1e3_wd1e4" ;;
  0.0003:0.0001) setting="lr3e4_wd1e4" ;;
  0.001:0.0005) setting="lr1e3_wd5e4" ;;
  0.0003:0.0005) setting="lr3e4_wd5e4" ;;
  *) echo "Learning rate and weight decay must be one frozen V2 grid setting" >&2; exit 2 ;;
esac

cd "$root"
run_ids=()
for seed in 7 42 123 314 2718; do
  run_id="v2_${fold_tag}_${tag}_${setting}_e50_s${seed}"
  run_ids+=("$run_id")
  CHBMIT_V2_MODEL_ARCHITECTURE="$architecture" \
  CHBMIT_TRAIN_LEARNING_RATE="$learning_rate" \
  CHBMIT_TRAIN_WEIGHT_DECAY="$weight_decay" \
  bash research_v2/tools/train_fold.sh "$prepared_dir" "$run_id" "$seed"
  python -m research_v2 provenance \
    --project-root "$repo_root" \
    --protocol research_v2/configs/protocol_v2.json \
    --registry research_v2/configs/candidate_registry_v2.json \
    --candidate-id "$candidate_id" \
    --fold-manifest "research_v2/manifests/temporal_v2/fold_$(printf '%02d' "$fold_index")_manifest.csv" \
    --checkpoint "outputs/$run_id/best_model.pth" \
    --training-seed "$seed" \
    --dataset-sampling-seed 20260802 \
    --precision amp_fp16_train_fp32_evaluate \
    --output "outputs/$run_id/provenance.json"
done

printf 'All seed-confirmation runs completed. Package them with:\n'
printf 'bash research_v2/tools/package_runs.sh'
printf ' %q' "${run_ids[@]}"
printf '\n'
