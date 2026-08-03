#!/usr/bin/env bash
# Build or verify reusable train/validation-only tensors for every frozen V2 fold.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [prepared_root]" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
prepared_root="${1:-$root/research_v2/generated}"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || {
  echo "Refusing to prepare V2 data from a detached HEAD" >&2
  exit 1
}
if [[ "$branch" != "research/v2-scientific-reports" ]]; then
  echo "Refusing V2 preparation outside research/v2-scientific-reports" >&2
  exit 1
fi

cd "$root"
protocol="research_v2/configs/protocol_v2.json"
manifest_root="research_v2/manifests/temporal_v2"
python -m research_v2 fold-audit \
  --protocol "$protocol" \
  --manifest data/chbmit_audit/recording_manifest.csv \
  --output "$manifest_root"

fold_count="$(python - <<'PY'
import json
with open('research_v2/manifests/temporal_v2/temporal_fold_feasibility.json', encoding='utf-8') as source:
    print(json.load(source)['selected_outer_folds'])
PY
)"

for ((fold=0; fold<fold_count; fold++)); do
  fold_name="$(printf 'fold_%02d' "$fold")"
  manifest="$manifest_root/${fold_name}_manifest.csv"
  output="$prepared_root/${fold_name}_trainval_v2"
  if [[ -e "$output/chbmit_test.npz" ]]; then
    echo "Refusing cached outer-test tensor: $output/chbmit_test.npz" >&2
    exit 1
  fi
  if [[ -f "$output/chbmit_train.npz" && -f "$output/chbmit_val.npz" && -f "$output/preparation_summary.json" ]]; then
    python - "$output" "$manifest" "$protocol" <<'PY'
import json
import sys
from research_v2.protocol import canonical_json_hash, file_sha256, load_json

output, manifest, protocol = sys.argv[1:]
with open(f'{output}/preparation_summary.json', encoding='utf-8') as source:
    summary = json.load(source)
if summary.get('included_splits') != ['train', 'val']:
    raise SystemExit(f'Invalid cached split set: {output}')
if summary.get('fold_manifest_sha256') != file_sha256(manifest):
    raise SystemExit(f'Cached manifest hash differs: {output}')
if summary.get('config_hash') != canonical_json_hash(load_json(protocol)):
    raise SystemExit(f'Cached protocol hash differs: {output}')
PY
    echo "Reusing verified V2 train/validation cache: $output"
    continue
  fi
  if [[ -e "$output" ]]; then
    echo "Refusing incomplete V2 cache directory: $output" >&2
    exit 1
  fi
  python -m research_v2 prepare-fold \
    --protocol "$protocol" \
    --fold-manifest "$manifest" \
    --output "$output"
done

echo "V2 train/validation caches are ready under: $prepared_root"
