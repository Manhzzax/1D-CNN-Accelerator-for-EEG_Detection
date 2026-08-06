#!/usr/bin/env bash
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
cd "${ROOT}"
python - <<'PY'
import json, statistics
from pathlib import Path
rows = []
for path in sorted(Path("outputs").glob("ps_a1_valthr_test_ps_a1_chb*_s42/val_threshold_test_evaluation.json")):
    d = json.loads(path.read_text(encoding="utf-8"))
    case = d["case_id"]
    thr = d["threshold_selection"]["threshold"]
    bal = d["balanced_test_diagnostic"]["metrics"]["balanced_accuracy"]
    bal05 = d["balanced_test_at_0_5"]["balanced_accuracy"]
    sen = d["balanced_test_diagnostic"]["metrics"]["sensitivity"]
    n_pos = d["test_window_counts"]["positive"]
    rows.append((case, thr, bal, bal05, sen, n_pos))
    print(f"{case}: thr={thr:.2f} bal@thr={100*bal:.3f}% bal@0.5={100*bal05:.3f}% delta={100*(bal-bal05):+.2f}pp test_ictal={n_pos}")
if not rows:
    raise SystemExit("No val-threshold evaluations found. Run 07_val_threshold_test.sh first.")
bals = [r[2] for r in rows]
bals05 = [r[3] for r in rows]
mean = statistics.fmean(bals)
sd = statistics.stdev(bals) if len(bals) > 1 else 0.0
mean05 = statistics.fmean(bals05)
# Secondary: cases with at least 20 ictal test windows (predeclared diagnostic)
core = [r[2] for r in rows if r[5] >= 20]
mean_core = statistics.fmean(core) if core else float("nan")
sd_core = statistics.stdev(core) if len(core) > 1 else 0.0
print(f"N={len(bals)} mean@val_thr={100*mean:.3f}% sd={100*sd:.3f}% | mean@0.5={100*mean05:.3f}% | target=95 pass={mean>=0.95}")
print(f"Secondary N(test_ictal>=20)={len(core)} mean={100*mean_core:.3f}% sd={100*sd_core:.3f}%")
out = {
    "n_cases": len(bals),
    "mean_test_balanced_accuracy_val_threshold": mean,
    "sd_test_balanced_accuracy_val_threshold": sd,
    "mean_test_balanced_accuracy_threshold_0_5": mean05,
    "primary_success_threshold": 0.95,
    "passed": mean >= 0.95,
    "secondary_min_test_ictal_windows": 20,
    "secondary_n_cases": len(core),
    "secondary_mean_test_balanced_accuracy": mean_core,
    "secondary_sd_test_balanced_accuracy": sd_core,
    "per_case": [
        {
            "case_id": c,
            "selected_threshold": thr,
            "test_balanced_accuracy": bal,
            "test_balanced_accuracy_at_0_5": bal05,
            "test_sensitivity": sen,
            "test_ictal_windows": npos,
        }
        for c, thr, bal, bal05, sen, npos in rows
    ],
}
Path("outputs/ps_a1_valthr_cohort_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("Wrote outputs/ps_a1_valthr_cohort_summary.json")
PY
