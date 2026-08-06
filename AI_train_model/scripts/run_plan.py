import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.chbmit_split import create_chronological_split_plan
from src.data_loader import get_protocol_output_dir_name, load_config


def main():
    config = load_config()
    audit_dir_name = config["data"].get("audit_output_dir", "chbmit_audit")
    protocol_dir_name = get_protocol_output_dir_name(config)
    audit_dir = os.path.join(project_dir, "data", audit_dir_name)
    output_dir = os.path.join(project_dir, "data", protocol_dir_name)
    ratios_config = config["data"]["split_ratios"]
    split_ratios = [ratios_config["train"], ratios_config["val"], ratios_config["test"]]

    print("=" * 60)
    print("PLANNING CHB-MIT CASE-WISE CHRONOLOGICAL SPLIT")
    print("=" * 60)
    print(
        f"Protocol dir: {protocol_dir_name} | ratios: "
        f"train={split_ratios[0]:.2f}, val={split_ratios[1]:.2f}, test={split_ratios[2]:.2f}"
    )
    plan, manifest_path = create_chronological_split_plan(audit_dir, output_dir, split_ratios)
    aggregate = plan["aggregate"]
    print(f"Split manifest: {manifest_path}")
    print(
        f"Train: {aggregate['train']['recordings']} recordings, {aggregate['train']['seizures']} seizures | "
        f"Val: {aggregate['val']['recordings']} recordings, {aggregate['val']['seizures']} seizures | "
        f"Test: {aggregate['test']['recordings']} recordings, {aggregate['test']['seizures']} seizures"
    )
    print(f"Cases without complete event coverage: {plan['cases_without_full_event_coverage']}")


if __name__ == "__main__":
    main()
