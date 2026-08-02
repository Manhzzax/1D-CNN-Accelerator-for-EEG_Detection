#!/usr/bin/env bash
# Screen the frozen V2 optimizer grid for one 31/7/3-style hierarchical CNN.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <prepared_train_val_dir> <candidate_tag> <kernel_1> <kernel_2> <kernel_3>" >&2
  exit 2
fi

prepared_dir="$1"
candidate_tag="$2"
kernel_one="$3"
kernel_two="$4"
kernel_three="$5"
repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || {
  echo "Refusing to run from a detached HEAD" >&2
  exit 1
}
if [[ "$branch" != "research/v2-scientific-reports" ]]; then
  echo "Refusing to run V2 screening outside research/v2-scientific-reports" >&2
  exit 1
fi
[[ "$candidate_tag" =~ ^[a-z0-9_]+$ ]] || {
  echo "candidate_tag must contain only lowercase letters, digits, or underscores" >&2
  exit 1
}
[[ -f "$prepared_dir/chbmit_train.npz" && -f "$prepared_dir/chbmit_val.npz" ]] || {
  echo "Missing V2 train/validation tensors in $prepared_dir" >&2
  exit 1
}
[[ ! -e "$prepared_dir/chbmit_test.npz" ]] || {
  echo "Refusing a prepared directory containing outer-test tensors" >&2
  exit 1
}

cd "$root"
run_ids=()
for spec in lr1e3_wd1e4:0.001:0.0001 lr3e4_wd1e4:0.0003:0.0001 lr1e3_wd5e4:0.001:0.0005 lr3e4_wd5e4:0.0003:0.0005; do
  IFS=: read -r tag learning_rate weight_decay <<< "$spec"
  run_id="v2_f00_${candidate_tag}_${tag}_e50_s07"
  run_ids+=("$run_id")
  CHBMIT_TRAIN_LEARNING_RATE="$learning_rate" \
  CHBMIT_TRAIN_WEIGHT_DECAY="$weight_decay" \
  CHBMIT_HIERARCHICAL_TEMPORAL_KERNEL="$kernel_one" \
  CHBMIT_HIERARCHICAL_SECOND_KERNEL="$kernel_two" \
  CHBMIT_HIERARCHICAL_THIRD_KERNEL="$kernel_three" \
  bash research_v2/tools/train_fold.sh "$prepared_dir" "$run_id" 7
  python -m research_v2 provenance \
    --project-root "$repo_root" \
    --protocol research_v2/configs/protocol_v2.json \
    --fold-manifest research_v2/manifests/temporal_v2/fold_00_manifest.csv \
    --checkpoint "outputs/$run_id/best_model.pth" \
    --training-seed 7 \
    --dataset-sampling-seed 20260802 \
    --precision amp_fp16_train_fp32_evaluate \
    --output "outputs/$run_id/provenance.json"
done

printf 'All inner-validation grid runs completed. Package them with:\n'
printf 'bash research_v2/tools/package_runs.sh'
printf ' %q' "${run_ids[@]}"
printf '\n'
