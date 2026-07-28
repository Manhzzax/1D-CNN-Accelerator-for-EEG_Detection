import os
import sys

# Add project root directory to sys.path to enable src imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.chbmit_preparation import prepare_chbmit_windows
from src.data_loader import load_config

def main():
    config = load_config()
    protocol_dir = os.path.join(project_dir, "data", config["data"]["protocol_output_dir"])
    output_dir = os.path.join(project_dir, "data", config["data"]["prepared_output_dir"])
    
    print("=" * 60)
    print("RUNNING LOCKED-SPLIT CHB-MIT PREPROCESSING")
    print("=" * 60)
    
    summary = prepare_chbmit_windows(
        protocol_dir=protocol_dir,
        output_dir=output_dir,
        preprocessing=config["preprocessing"],
        seed=config["data"]["seed"],
    )
    print("\nPreprocessing completed successfully.")
    for split_name, details in summary["outputs"].items():
        print(
            f"  {split_name}: {details['positive_windows']} ictal + "
            f"{details['normal_windows']} interictal windows"
        )
    print("=" * 60)

if __name__ == "__main__":
    main()
