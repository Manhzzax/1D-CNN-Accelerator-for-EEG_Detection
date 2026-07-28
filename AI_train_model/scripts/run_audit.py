import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.chbmit_audit import run_chbmit_audit
from src.data_loader import load_config


def main():
    config = load_config()
    raw_dataset_dir = config["data"]["raw_dir"]
    audit_dir_name = config["data"].get("audit_output_dir", "chbmit_audit")
    output_dir = os.path.join(project_dir, "data", audit_dir_name)

    print("=" * 60)
    print("RUNNING CHB-MIT EDF AND ANNOTATION AUDIT")
    print("=" * 60)
    success = run_chbmit_audit(raw_dataset_dir, output_dir)
    if not success:
        raise SystemExit("Audit failed. Review audit_summary.json before preprocessing.")


if __name__ == "__main__":
    main()
