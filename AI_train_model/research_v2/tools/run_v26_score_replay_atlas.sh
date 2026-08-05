#!/usr/bin/env bash
# Replay only score streams from already consumed V2.1 folds. Never trains.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.6-score-replay-diagnostics" ]] || {
  echo "Wrong branch: $branch" >&2
  exit 1
}

rescore_flag=""
artifacts=()
for arg in "$@"; do
  if [[ "$arg" == "--rescore-missing" ]]; then
    rescore_flag="--rescore-missing"
  elif [[ "$arg" == --* ]]; then
    echo "Usage: $0 [--rescore-missing] [artifact_id ...]" >&2
    exit 2
  else
    artifacts+=("$arg")
  fi
done

output="research_v2/reports"
artifact_args=()
if [[ ${#artifacts[@]} -gt 0 ]]; then
  output="research_v2/reports/v26_score_replay_subset"
  for artifact in "${artifacts[@]}"; do
    artifact_args+=(--artifact "$artifact")
  done
fi

cd "$root"
python -m research_v2 v26-audit-score-replays \
  --artifact-config research_v2/configs/protocol_v2_6_diagnostics.json \
  --score-replay-config research_v2/configs/protocol_v2_6_score_replay.json \
  --artifact-root research_v2/artifacts \
  --run-root outputs \
  --manifest-root research_v2/manifests/temporal_v21 \
  --score-cache research_v2/diagnostic_cache_v26 \
  --output "$output" \
  $rescore_flag "${artifact_args[@]}"

printf 'V2.6 score-replay diagnostic complete. Outputs:\n'
printf '  %s/v26_score_replay_atlas.{json,md}\n' "$output"
printf '  %s/v26_score_replay_runs.csv\n' "$output"
