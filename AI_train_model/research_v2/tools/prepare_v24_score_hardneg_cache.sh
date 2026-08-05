#!/usr/bin/env bash
# Build one V2.4 train-only score-ranked hard-negative cache for a development fold.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(00|01|02)$ ]]; then
  echo "Usage: $0 <fold_index:00|01|02>" >&2
  exit 2
fi

fold="$1"
repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.4-score-ranked-hardneg" ]] || { echo "Wrong branch: $branch" >&2; exit 1; }

cd "$root"
protocol="research_v2/configs/protocol_v2_4.json"
registry="research_v2/configs/candidate_registry_v2_4.json"
manifest="research_v2/manifests/temporal_v21/confirmation_f${fold}_manifest.csv"
source_prepared="research_v2/generated_v21/f${fold}_confirmation"
output="research_v2/generated_v24/f${fold}_score_hardneg"

python -m research_v2 validate --protocol "$protocol" --registry "$registry"
[[ -f "$manifest" ]] || { echo "Missing locked V2.1 fold manifest: $manifest" >&2; exit 1; }
[[ -f "$source_prepared/chbmit_train.npz" && -f "$source_prepared/chbmit_val.npz" && -f "$source_prepared/chbmit_temporal_eval.npz" ]] || {
  echo "Missing V2.1 confirmation cache; prepare it first" >&2; exit 1;
}
[[ ! -e "$source_prepared/chbmit_test.npz" && ! -e "$source_prepared/continuous_test_recordings.csv" ]] || {
  echo "Refusing a V2.1 source cache containing sealed-test artifacts" >&2; exit 1;
}
[[ ! -e "$output/chbmit_test.npz" && ! -e "$output/continuous_test_recordings.csv" ]] || {
  echo "Refusing a V2.4 cache containing sealed-test artifacts" >&2; exit 1;
}

python -m research_v2 v24-mine-score-ranked-hard-negatives \
  --project-root "$repo_root" --protocol "$protocol" --registry "$registry" --fold "$fold" \
  --fold-manifest "$manifest" --source-prepared-dir "$source_prepared" --output "$output"

echo "V2.4 F${fold} cache is ready. Re-running this command verifies/reuses it without re-scoring train EEG."
