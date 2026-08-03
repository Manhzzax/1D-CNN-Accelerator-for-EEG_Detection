#!/usr/bin/env bash
# Retrain only the V2.1 predeclared candidate/seed schedule; never opens block 6.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <prepared_confirmation_dir> <fold_index:00|01|02> <B0_bandpower_linear|B1_vanilla_1dcnn|B2_deep_matched_1dcnn|B4_dilated_lightseizure_like>" >&2
  exit 2
fi

prepared_dir="$1"
fold="$2"
candidate="$3"
repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.1-patient-forward" ]] || { echo "Wrong branch: $branch" >&2; exit 1; }
[[ -f "$prepared_dir/chbmit_train.npz" && -f "$prepared_dir/chbmit_val.npz" && -f "$prepared_dir/chbmit_temporal_eval.npz" ]] || { echo "Invalid V2.1 confirmation cache" >&2; exit 1; }
[[ ! -e "$prepared_dir/chbmit_test.npz" ]] || { echo "Confirmation cache must not contain test tensors" >&2; exit 1; }

case "$candidate" in
  B0_bandpower_linear) architecture="v2_bandpower_linear"; lr="1e-3"; wd="1e-4"; seeds=(42); tag="b0_bandpower" ;;
  B1_vanilla_1dcnn) architecture="v2_vanilla_1dcnn"; lr="1e-3"; wd="5e-4"; seeds=(42); tag="b1_vanilla" ;;
  B2_deep_matched_1dcnn) architecture="v2_deep_matched_1dcnn"; lr="3e-4"; wd="5e-4"; seeds=(7 42 123 314 2718); tag="b2_deep" ;;
  B4_dilated_lightseizure_like) architecture="dilated_hierarchical_separable_1dcnn"; lr="1e-3"; wd="5e-4"; seeds=(7 42 123 314 2718); tag="b4_dilated" ;;
  *) echo "Unknown frozen V2.1 candidate: $candidate" >&2; exit 2 ;;
esac

cd "$root"
manifest="research_v2/manifests/temporal_v21/confirmation_f${fold}_manifest.csv"
for seed in "${seeds[@]}"; do
  run="v21_f${fold}_${tag}_s${seed}"
  CHBMIT_V2_MODEL_ARCHITECTURE="$architecture" \
  CHBMIT_TRAIN_LEARNING_RATE="$lr" \
  CHBMIT_TRAIN_WEIGHT_DECAY="$wd" \
  bash research_v2/tools/train_fold.sh "$prepared_dir" "$run" "$seed"
  python -m research_v2 provenance --project-root "$repo_root" --protocol research_v2/configs/protocol_v2_1.json --registry research_v2/configs/candidate_registry_v2.json --candidate-id "$candidate" --fold-manifest "$manifest" --checkpoint "outputs/$run/best_model.pth" --training-seed "$seed" --dataset-sampling-seed 20260802 --precision amp_fp16_train_fp32_evaluate --output "outputs/$run/provenance.json"
  python -m research_v2 v21-evaluate-confirmation --protocol research_v2/configs/protocol_v2_1.json --fold-manifest "$manifest" --prepared-dir "$prepared_dir" --run-dir "outputs/$run" --output "outputs/$run/v21_confirmation"
done

echo "V2.1 candidate completed. Package with: bash research_v2/tools/package_v21_runs.sh ${seeds[@]/#/v21_f${fold}_${tag}_s}"
