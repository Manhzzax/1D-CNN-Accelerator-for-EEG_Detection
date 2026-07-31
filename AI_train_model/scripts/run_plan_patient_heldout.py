"""Create the patient-group-disjoint CHB-MIT protocol manifest."""

import os
import re
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.chbmit_patient_split import create_patient_heldout_split_plan
from src.data_loader import load_config


def main():
    config = load_config()
    data_config = config["data"]
    audit_dir = os.path.join(project_dir, "data", data_config.get("audit_output_dir", "chbmit_audit"))
    output_name = os.environ.get(
        "CHBMIT_PATIENT_HELDOUT_PROTOCOL_OUTPUT_DIR",
        data_config.get("patient_heldout_protocol_output_dir", "chbmit_protocol_patient_holdout_v1"),
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        raise ValueError(
            "CHBMIT_PATIENT_HELDOUT_PROTOCOL_OUTPUT_DIR must contain only letters, digits, underscores, or hyphens"
        )
    ratios_config = data_config["patient_heldout_split_ratios"]
    ratios = [ratios_config[split_name] for split_name in ("train", "val", "test")]
    output_dir = os.path.join(project_dir, "data", output_name)

    print("=" * 60)
    print("PLANNING CHB-MIT PATIENT-GROUP-DISJOINT HOLDOUT SPLIT")
    print("=" * 60)
    plan, manifest_path = create_patient_heldout_split_plan(
        audit_dir, output_dir, ratios, data_config["seed"]
    )
    aggregate = plan["aggregate"]
    print(f"Split manifest: {manifest_path}")
    for split_name in ("train", "val", "test"):
        details = aggregate[split_name]
        print(
            f"{split_name}: {details['patient_groups']} patient groups | {details['cases']} cases | "
            f"{details['recordings']} recordings | {details['seizures']} seizures"
        )


if __name__ == "__main__":
    main()
