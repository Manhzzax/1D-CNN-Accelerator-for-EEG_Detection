#!/usr/bin/env bash
set -euo pipefail
ROOT="${CHBMIT_REPO_ROOT:-$HOME/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model}"
cd "${ROOT}"
python - <<'PY'
import json, statistics
from pathlib import Path
rows = []
for path in sorted(Path("outputs").glob("ps_a12_test_ps_a12_chb*_s42/checkpoint_test_evaluation.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    case = data["source_run_id"].replace("ps_a12_", "").replace("_s42", "")
    bal = data["balanced_test_diagnostic"]["metrics"]["balanced_accuracy"]
    raw = data["test_prevalence_metrics"]["accuracy"]
    sen = data["balanced_test_diagnostic"]["metrics"]["sensitivity"]
    auroc = data["balanced_test_diagnostic"]["metrics"]["auroc"]
    rows.append((case, bal, raw, sen, auroc))
    print(f"{case}: balanced={100*bal:.3f}% raw={100*raw:.3f}% sen={100*sen:.3f}% auroc={100*auroc:.3f}%")
if not rows:
    raise SystemExit("No A1.2 test results found.")
bals = [r[1] for r in rows]
mean = statistics.fmean(bals)
sd = statistics.stdev(bals) if len(bals) > 1 else 0.0
a10 = 0.9071853608224304
a11 = 0.88120
print(f"N={len(bals)} mean_balanced={100*mean:.3f}% sd={100*sd:.3f}%")
print(f"vs A1.0={100*a10:.3f}% delta={100*(mean-a10):+.3f}pp | vs A1.1={100*a11:.3f}% | target=95 pass={mean>=0.95}")
out = {
    "protocol": "path_a_a1_2_hierarchical_supcon_0p05_t0p1",
    "n_cases": len(bals),
    "mean_test_balanced_accuracy": mean,
    "sd_test_balanced_accuracy": sd,
    "a1_0_mean_test_balanced_accuracy": a10,
    "a1_1_mean_test_balanced_accuracy": a11,
    "delta_vs_a1_0_pp": 100.0 * (mean - a10),
    "primary_success_threshold": 0.95,
    "passed": mean >= 0.95,
    "promoted_over_a1_0": mean > a10,
    "per_case": [
        {"case_id": c, "test_balanced_accuracy": b, "test_raw_accuracy": r, "test_sensitivity": s, "test_auroc": a}
        for c, b, r, s, a in rows
    ],
}
Path("outputs/ps_a12_cohort_test_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
rep = Path("research_v2/reports/path_a")
rep.mkdir(parents=True, exist_ok=True)
(rep / "ps_a12_cohort_test_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print("Wrote outputs/ps_a12_cohort_test_summary.json and research_v2/reports/path_a/ps_a12_cohort_test_summary.json")
PY
