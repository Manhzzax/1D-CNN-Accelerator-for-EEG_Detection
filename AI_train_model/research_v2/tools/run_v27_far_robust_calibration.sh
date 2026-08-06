#!/usr/bin/env bash
# Evaluate a simultaneous-UCB calibration rule from existing development artifacts only.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.7-far-robust-calibration" ]] || {
  echo "Wrong branch: $branch" >&2
  exit 1
}

cd "$root"
python -m research_v2 v27-far-robust-calibration \
  --config research_v2/configs/protocol_v2_7_far_robust_calibration.json \
  --artifact-config research_v2/configs/protocol_v2_6_diagnostics.json \
  --artifact-root research_v2/artifacts \
  --run-root outputs \
  --manifest-root research_v2/manifests/temporal_v21 \
  --output research_v2/reports

printf 'V2.7 FAR-robust calibration diagnostic complete. Outputs:\n'
printf '  research_v2/reports/v27_far_robust_calibration.{json,md}\n'
printf '  research_v2/reports/v27_far_robust_calibration_runs.csv\n'
