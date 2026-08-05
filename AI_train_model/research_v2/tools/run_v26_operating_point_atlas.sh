#!/usr/bin/env bash
# Generate an artifact-only V2.6 diagnostic; this script cannot train or score EEG.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.6-operating-point-diagnostics" ]] || {
  echo "Wrong branch: $branch" >&2
  exit 1
}

cd "$root"
python -m research_v2 v26-audit-artifacts \
  --config research_v2/configs/protocol_v2_6_diagnostics.json \
  --artifact-root research_v2/artifacts \
  --output research_v2/reports

printf 'V2.6 artifact-only diagnostic complete. Outputs:\n'
printf '  research_v2/reports/v26_operating_point_atlas.{json,md}\n'
printf '  research_v2/reports/v26_operating_point_runs.csv\n'
