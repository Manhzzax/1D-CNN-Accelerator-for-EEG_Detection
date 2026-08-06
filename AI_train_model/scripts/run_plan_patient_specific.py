import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.chbmit_patient_specific import create_patient_specific_split_plans
from src.data_loader import load_config


def main():
    config = load_config()
    audit_dir = os.path.join(project_dir, "data", config["data"].get("audit_output_dir", "chbmit_audit"))
    protocol_root = os.environ.get(
        "CHBMIT_PS_PROTOCOL_ROOT",
        config["data"].get("patient_specific_protocol_root", "chbmit_protocol_ps_a1_v1"),
    )
    output_dir = os.path.join(project_dir, "data", protocol_root)
    ratios = config["data"]["split_ratios"]
    split_ratios = [ratios["train"], ratios["val"], ratios["test"]]

    print("=" * 60)
    print("PLANNING PATIENT-SPECIFIC CHRONOLOGICAL SPLITS (PATH A)")
    print("=" * 60)
    print(f"Output root: {output_dir}")
    print(f"Ratios: train={split_ratios[0]:.2f}, val={split_ratios[1]:.2f}, test={split_ratios[2]:.2f}")

    if os.path.isdir(output_dir) and os.path.isfile(os.path.join(output_dir, "cohort_summary.json")):
        raise SystemExit(
            f"Protocol root already exists: {output_dir}. "
            "Refusing to overwrite a locked patient-specific cohort."
        )

    cohort = create_patient_specific_split_plans(audit_dir, output_dir, split_ratios)
    print(f"Eligible cases: {len(cohort['eligible_cases'])}")
    print(f"Skipped cases: {len(cohort['skipped_cases'])}")
    for case_id in cohort["eligible_cases"]:
        case = cohort["cases"][case_id]
        print(
            f"  {case_id}: train {case['train']['recordings']}r/{case['train']['seizures']}sz | "
            f"val {case['val']['recordings']}r/{case['val']['seizures']}sz | "
            f"test {case['test']['recordings']}r/{case['test']['seizures']}sz"
        )
    if cohort["skipped_cases"]:
        print("Skipped:")
        for item in cohort["skipped_cases"]:
            print(f"  {item['case_id']}: {item['reason']}")


if __name__ == "__main__":
    main()
