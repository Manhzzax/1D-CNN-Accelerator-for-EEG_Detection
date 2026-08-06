#!/usr/bin/env bash
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
cd "${ROOT}"
python - <<'PY'
import json
from pathlib import Path
weak = ["chb16", "chb18", "chb24", "chb13", "chb07"]
print("case\ttrain_ict\tval_ict\ttest_ict\tthr05_bal\tsen\tauroc")
for case in weak:
    prep = Path(f"data/chbmit_prepared_ps_a1_v1/{case}/preparation_summary.json")
    test = Path(f"outputs/ps_a1_test_ps_a1_{case}_s42/checkpoint_test_evaluation.json")
    if not prep.is_file():
        print(f"{case}\tMISSING_PREP")
        continue
    o = json.loads(prep.read_text())["outputs"]
    line = f"{case}\t{o['train']['positive_windows']}\t{o['val']['positive_windows']}\t{o['test']['positive_windows']}"
    if test.is_file():
        t = json.loads(test.read_text())
        m = t["balanced_test_diagnostic"]["metrics"]
        line += f"\t{100*m['balanced_accuracy']:.2f}\t{100*m['sensitivity']:.2f}\t{100*m['auroc']:.2f}"
    print(line)
PY
