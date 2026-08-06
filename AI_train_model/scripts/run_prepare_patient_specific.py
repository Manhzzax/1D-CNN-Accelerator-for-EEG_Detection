import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.chbmit_preparation import prepare_chbmit_windows
from src.data_loader import load_config
from src.feature_representation import get_feature_spec


def main():
    config = load_config()
    protocol_root_name = os.environ.get(
        "CHBMIT_PS_PROTOCOL_ROOT",
        config["data"].get("patient_specific_protocol_root", "chbmit_protocol_ps_a1_v1"),
    )
    prepared_root_name = os.environ.get(
        "CHBMIT_PS_PREPARED_ROOT",
        config["data"].get("patient_specific_prepared_root", "chbmit_prepared_ps_a1_v1"),
    )
    protocol_root = os.path.join(project_dir, "data", protocol_root_name)
    prepared_root = os.path.join(project_dir, "data", prepared_root_name)
    cohort_path = os.path.join(protocol_root, "cohort_summary.json")
    if not os.path.isfile(cohort_path):
        raise SystemExit(f"Missing cohort summary: {cohort_path}. Run plan_patient_specific first.")

    with open(cohort_path, "r", encoding="utf-8") as input_file:
        cohort = json.load(input_file)

    only_case = os.environ.get("CHBMIT_PS_CASE_ID", "").strip()
    cases = [only_case] if only_case else list(cohort["eligible_cases"])
    if only_case and only_case not in cohort["eligible_cases"]:
        raise SystemExit(f"Case {only_case} is not in eligible_cases")

    feature_spec = get_feature_spec(config["preprocessing"])
    seed = int(config["data"]["seed"])
    os.makedirs(prepared_root, exist_ok=True)

    print("=" * 60)
    print("PREPARING PATIENT-SPECIFIC WINDOWS (PATH A)")
    print("=" * 60)
    summaries = {}
    for case_id in cases:
        protocol_dir = os.path.join(protocol_root, case_id)
        output_dir = os.path.join(prepared_root, case_id)
        if os.path.isfile(os.path.join(output_dir, "chbmit_train.npz")):
            print(f"Skip existing prepared case: {case_id}")
            continue
        print(f"Preparing {case_id} ...")
        summary = prepare_chbmit_windows(
            protocol_dir=protocol_dir,
            output_dir=output_dir,
            preprocessing=config["preprocessing"],
            seed=seed,
            feature_spec=feature_spec,
        )
        summaries[case_id] = summary["outputs"]
        print(
            f"  {case_id}: train {summary['outputs']['train']['positive_windows']} ictal | "
            f"val {summary['outputs']['val']['positive_windows']} ictal | "
            f"test {summary['outputs']['test']['positive_windows']} ictal"
        )

    index_path = os.path.join(prepared_root, "prepare_index.json")
    with open(index_path, "w", encoding="utf-8") as output_file:
        json.dump({"cases": summaries, "protocol_root": protocol_root_name}, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
