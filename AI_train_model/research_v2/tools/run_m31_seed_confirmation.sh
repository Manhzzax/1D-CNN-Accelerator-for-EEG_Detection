#!/usr/bin/env bash
# Confirm the preselected M31 setting on all five declared V2 training seeds.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <prepared_train_val_dir>" >&2
  exit 2
fi

prepared_dir="$1"
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

cd "$root"
run_ids=()
for seed in 7 42 123 314 2718; do
  run_id="v2_f00_p1_m31_lr1e3_wd5e4_e50_s${seed}"
  run_ids+=("$run_id")
  CHBMIT_TRAIN_LEARNING_RATE=0.001 \
  CHBMIT_TRAIN_WEIGHT_DECAY=0.0005 \
  bash research_v2/tools/train_fold.sh "$prepared_dir" "$run_id" "$seed"
  python -m research_v2 provenance \
    --project-root "$repo_root" \
    --protocol research_v2/configs/protocol_v2.json \
    --fold-manifest research_v2/manifests/temporal_v2/fold_00_manifest.csv \
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
