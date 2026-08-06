#!/usr/bin/env bash
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
cd "${ROOT}"
python - <<'PY'
import json, statistics
from pathlib import Path
rows = []
for path in sorted(Path("outputs").glob("ps_a1_test_ps_a1_chb*_s42/checkpoint_test_evaluation.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    case = data["source_run_id"].replace("ps_a1_", "").replace("_s42", "")
    bal = data["balanced_test_diagnostic"]["metrics"]["balanced_accuracy"]
    raw = data["test_prevalence_metrics"]["accuracy"]
    sen = data["balanced_test_diagnostic"]["metrics"]["sensitivity"]
    auroc = data["balanced_test_diagnostic"]["metrics"]["auroc"]
    rows.append((case, bal, raw, sen, auroc))
    print(f"{case}: balanced={100*bal:.3f}% raw={100*raw:.3f}% sen={100*sen:.3f}% auroc={100*auroc:.3f}%")
if not rows:
    raise SystemExit("No patient-specific test results found.")
bals = [r[1] for r in rows]
mean = statistics.fmean(bals)
sd = statistics.stdev(bals) if len(bals) > 1 else 0.0
print(f"N={len(bals)} mean_balanced={100*mean:.3f}% sd={100*sd:.3f}% target=95.000% pass={mean>=0.95}")
out = {
    "n_cases": len(bals),
    "mean_test_balanced_accuracy": mean,
    "sd_test_balanced_accuracy": sd,
    "primary_success_threshold": 0.95,
    "passed": mean >= 0.95,
    "per_case": [
        {"case_id": c, "test_balanced_accuracy": b, "test_raw_accuracy": r, "test_sensitivity": s, "test_auroc": a}
        for c, b, r, s, a in rows
    ],
}
Path("outputs/ps_a1_cohort_test_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("Wrote outputs/ps_a1_cohort_test_summary.json")
PY
