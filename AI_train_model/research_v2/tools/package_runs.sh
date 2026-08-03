#!/usr/bin/env bash
# Package multiple completed V2 runs and push them as one atomic result commit.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [<run_id> ...]" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || {
  echo "Refusing to package from a detached HEAD" >&2
  exit 1
}
if [[ "$branch" != "research/v2-scientific-reports" ]]; then
  echo "Refusing to push V2 artifacts outside research/v2-scientific-reports" >&2
  exit 1
fi

required=(best_model.pth provenance.json model_spec.json training_summary.json validation_window_metrics.json)
artifact_paths=()
for run_id in "$@"; do
  result_dir="$repo_root/AI_train_model/outputs/$run_id"
  destination="$repo_root/AI_train_model/research_v2/artifacts/$run_id"
  for file in "${required[@]}"; do
    [[ -f "$result_dir/$file" ]] || { echo "MISSING: $result_dir/$file" >&2; exit 1; }
  done
  mkdir -p "$destination"
  cp "$result_dir"/{best_model.pth,provenance.json,model_spec.json,training_summary.json,validation_window_metrics.json} "$destination/"
  [[ -f "$result_dir/hyperparameters.json" ]] && cp "$result_dir/hyperparameters.json" "$destination/"
  [[ -f "$result_dir/event_metrics.json" ]] && cp "$result_dir/event_metrics.json" "$destination/"
  [[ -f "$result_dir/int16_validation.json" ]] && cp "$result_dir/int16_validation.json" "$destination/"
  [[ -f "$result_dir/model_mac_activation.json" ]] && cp "$result_dir/model_mac_activation.json" "$destination/"
  artifact_path="AI_train_model/research_v2/artifacts/$run_id"
  artifact_paths+=("$artifact_path")
  git -C "$repo_root" add -f -- "$artifact_path"
done

git -C "$repo_root" diff --cached --quiet -- "${artifact_paths[@]}" && {
  echo "No new V2 artifact changes to commit; pushing any earlier local result commit" >&2
  git -C "$repo_root" push origin "HEAD:$branch"
  exit 0
}
git -C "$repo_root" commit --only -m "results(v2): add seed confirmation batch" -- "${artifact_paths[@]}"
git -C "$repo_root" push origin "HEAD:$branch"
