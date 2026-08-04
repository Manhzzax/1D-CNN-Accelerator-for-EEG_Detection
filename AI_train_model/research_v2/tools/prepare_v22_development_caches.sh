#!/usr/bin/env bash
# Verify read-only V2.1 confirmation caches before V2.2-A capacity experiments.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
root="$repo_root/AI_train_model"
branch="$(git -C "$repo_root" symbolic-ref --quiet --short HEAD)" || exit 1
[[ "$branch" == "research/v2.2-far-robustness" ]] || { echo "Wrong branch: $branch" >&2; exit 1; }

cd "$root"
python -m research_v2 validate --protocol research_v2/configs/protocol_v2_2.json --registry research_v2/configs/candidate_registry_v2_2.json

for fold in 00 01 02; do
  cache="research_v2/generated_v21/f${fold}_confirmation"
  for file in chbmit_train.npz chbmit_val.npz chbmit_temporal_eval.npz preparation_summary.json; do
    [[ -f "$cache/$file" ]] || { echo "Missing V2.1 cache: $cache/$file" >&2; exit 1; }
  done
  [[ ! -e "$cache/chbmit_test.npz" && ! -e "$cache/continuous_test_recordings.csv" ]] || {
    echo "Refusing cache with a sealed-test artifact: $cache" >&2; exit 1;
  }
  python - "$cache/preparation_summary.json" "$fold" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"train", "val", "temporal_eval"}
if summary.get("protocol") != "research_v2_1_confirmation":
    raise SystemExit("Cache is not a V2.1 confirmation cache")
if int(summary.get("window_samples", 0)) != 1280:
    raise SystemExit("Cache does not contain five-second windows")
if set(summary.get("outputs", {})) != expected:
    raise SystemExit("Cache splits do not match the V2.2 development contract")
print(f"F{sys.argv[2]} cache verified: {summary['fold_manifest_sha256']}")
PY
done

echo "V2.2-A cache preflight passed. Reusing read-only V2.1 causal windows; blocks 5 and 6 remain absent."
