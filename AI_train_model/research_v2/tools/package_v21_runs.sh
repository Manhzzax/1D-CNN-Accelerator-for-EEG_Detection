#!/usr/bin/env bash
# Package concise V2.1 result artifacts; continuous EDF scores/caches remain local.
set -euo pipefail

[[ $# -gt 0 ]] || { echo "Usage: $0 <run_id> [run_id ...]" >&2; exit 2; }
repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.1-patient-forward" ]] || { echo "Refusing to package V2.1 outside its branch" >&2; exit 1; }

for run in "$@"; do
  source="$root/outputs/$run"
  destination="$root/research_v2/artifacts/$run"
  required=(best_model.pth model_spec.json training_summary.json validation_window_metrics.json provenance.json)
  for file in "${required[@]}"; do [[ -f "$source/$file" ]] || { echo "MISSING: $source/$file" >&2; exit 1; }; done
  [[ -f "$source/v21_confirmation/temporal_confirmation.json" ]] || { echo "MISSING confirmation evaluation: $run" >&2; exit 1; }
  mkdir -p "$destination"
  cp "$source"/{best_model.pth,model_spec.json,training_summary.json,validation_window_metrics.json,provenance.json} "$destination/"
  cp "$source/v21_confirmation"/{temporal_confirmation.json,calibration_policy_sweep.json} "$destination/"
  [[ -f "$source/hyperparameters.json" ]] && cp "$source/hyperparameters.json" "$destination/"
  git -C "$repo_root" add -f -- "AI_train_model/research_v2/artifacts/$run"
done

git -C "$repo_root" diff --cached --quiet && { echo "No new V2.1 artifacts; nothing to commit."; exit 0; }
git -C "$repo_root" commit -m "results(v2.1): add confirmation artifacts"
git -C "$repo_root" fetch origin
git -C "$repo_root" rebase "origin/$branch"
git -C "$repo_root" push origin "HEAD:$branch"
