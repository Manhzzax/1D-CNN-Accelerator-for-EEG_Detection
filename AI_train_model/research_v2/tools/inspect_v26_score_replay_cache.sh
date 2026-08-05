#!/usr/bin/env bash
# Verify V2.6 score-pair provenance without loading NPZ arrays or opening EEG.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.6-score-replay-diagnostics" ]] || {
  echo "Wrong branch: $branch" >&2
  exit 1
}

cd "$root"
python -m research_v2 v26-inspect-score-replays \
  --artifact-config research_v2/configs/protocol_v2_6_diagnostics.json \
  --score-replay-config research_v2/configs/protocol_v2_6_score_replay.json \
  --artifact-root research_v2/artifacts \
  --run-root outputs \
  --manifest-root research_v2/manifests/temporal_v21 \
  --output research_v2/reports

printf 'V2.6 score-replay inventory complete. Outputs:\n'
printf '  research_v2/reports/v26_score_replay_inventory.{json,csv}\n'
