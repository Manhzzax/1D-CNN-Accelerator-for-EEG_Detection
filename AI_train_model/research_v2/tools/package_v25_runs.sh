#!/usr/bin/env bash
# Package concise V2.5 artifacts; raw EEG and continuous score streams stay local.
set -euo pipefail

[[ $# -gt 0 ]] || { echo "Usage: $0 <run_id> [run_id ...]" >&2; exit 2; }
repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.5-patient-group-robustness" ]] || { echo "Refusing to package V2.5 outside its branch" >&2; exit 1; }
git_clean() { env -u LD_LIBRARY_PATH -u LD_PRELOAD git -C "$repo_root" "$@"; }

for run in "$@"; do
  source="$root/outputs/$run"
  destination="$root/research_v2/artifacts/$run"
  required=(best_model.pth model_spec.json hyperparameters.json training_summary.json validation_window_metrics.json provenance.json scaler_mean.npy scaler_scale.npy)
  for file in "${required[@]}"; do [[ -f "$source/$file" ]] || { echo "MISSING: $source/$file" >&2; exit 1; }; done
  [[ -f "$source/v25_development/temporal_confirmation.json" ]] || { echo "MISSING V2.5 development evaluation: $run" >&2; exit 1; }
  mkdir -p "$destination"
  cp "$source"/{best_model.pth,model_spec.json,hyperparameters.json,training_summary.json,validation_window_metrics.json,provenance.json,scaler_mean.npy,scaler_scale.npy} "$destination/"
  cp "$source/v25_development"/{temporal_confirmation.json,calibration_policy_sweep.json} "$destination/"
  git -C "$repo_root" add -f -- "AI_train_model/research_v2/artifacts/$run"
done

git -C "$repo_root" diff --cached --quiet && { echo "No new V2.5 artifacts; nothing to commit."; exit 0; }
git -C "$repo_root" commit -m "results(v2.5): add group-robust artifacts"
git_clean fetch origin
git_clean rebase "origin/$branch"
git_clean push origin "HEAD:$branch"
