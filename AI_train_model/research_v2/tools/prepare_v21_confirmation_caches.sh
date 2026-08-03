#!/usr/bin/env bash
# Build immutable V2.1 caches once. They contain no sealed final-test tensors.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [prepared_root]" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
prepared_root="${1:-$root/research_v2/generated_v21}"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.1-patient-forward" ]] || {
  echo "Refusing V2.1 preparation outside research/v2.1-patient-forward" >&2
  exit 1
}

cd "$root"
protocol="research_v2/configs/protocol_v2_1.json"
manifest_root="research_v2/manifests/temporal_v21"
python -m research_v2 v21-audit --protocol "$protocol" --manifest data/chbmit_audit/recording_manifest.csv --output "$manifest_root"

for fold in 00 01 02; do
  manifest="$manifest_root/confirmation_f${fold}_manifest.csv"
  output="$prepared_root/f${fold}_confirmation"
  if [[ -e "$output/chbmit_test.npz" || -e "$output/continuous_test_recordings.csv" ]]; then
    echo "Refusing sealed-test tensor or manifest in confirmation cache: $output" >&2
    exit 1
  fi
  if [[ -f "$output/chbmit_train.npz" && -f "$output/chbmit_val.npz" && -f "$output/chbmit_temporal_eval.npz" && -f "$output/preparation_summary.json" ]]; then
    python - "$output" "$manifest" "$protocol" <<'PY'
import json, sys
from research_v2.protocol import canonical_json_hash, file_sha256, load_json
output, manifest, protocol = sys.argv[1:]
with open(f"{output}/preparation_summary.json", encoding="utf-8") as source:
    summary = json.load(source)
if summary.get("included_splits") != ["train", "val", "temporal_eval"]:
    raise SystemExit(f"Invalid V2.1 cache split set: {output}")
if summary.get("fold_manifest_sha256") != file_sha256(manifest):
    raise SystemExit(f"Cached manifest hash differs: {output}")
if summary.get("config_hash") != canonical_json_hash(load_json(protocol)):
    raise SystemExit(f"Cached protocol hash differs: {output}")
PY
    echo "Reusing verified V2.1 cache: $output"
    continue
  fi
  [[ ! -e "$output" ]] || { echo "Refusing incomplete V2.1 cache: $output" >&2; exit 1; }
  python -m research_v2 v21-prepare-confirmation --protocol "$protocol" --fold-manifest "$manifest" --output "$output"
done

echo "V2.1 confirmation caches ready under: $prepared_root"
