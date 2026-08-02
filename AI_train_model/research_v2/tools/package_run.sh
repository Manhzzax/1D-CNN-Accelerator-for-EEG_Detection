#!/usr/bin/env bash
# Package one already-completed V2 result without adding raw EEG or caches.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <run_id> <result_directory>" >&2
  exit 2
fi

run_id="$1"
result_dir="$2"
repo_root="$(git rev-parse --show-toplevel)"
destination="$repo_root/AI_train_model/research_v2/artifacts/$run_id"
required=(provenance.json model_spec.json training_summary.json validation_window_metrics.json)

for file in "${required[@]}"; do
  [[ -f "$result_dir/$file" ]] || { echo "MISSING: $result_dir/$file" >&2; exit 1; }
done

mkdir -p "$destination"
cp "$result_dir"/{provenance.json,model_spec.json,training_summary.json,validation_window_metrics.json} "$destination/"
[[ -f "$result_dir/event_metrics.json" ]] && cp "$result_dir/event_metrics.json" "$destination/"
[[ -f "$result_dir/int16_validation.json" ]] && cp "$result_dir/int16_validation.json" "$destination/"
[[ -f "$result_dir/model_mac_activation.json" ]] && cp "$result_dir/model_mac_activation.json" "$destination/"

git -C "$repo_root" add -f -- "AI_train_model/research_v2/artifacts/$run_id"
git -C "$repo_root" commit -m "results(v2): add $run_id"
git -C "$repo_root" push origin main
