import os
import sys

# Add project root directory to sys.path to enable src imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.data_loader import load_config
from src.preprocess_chbmit import run_chbmit_preprocessing

def main():
    config = load_config()
    raw_dataset_dir = config['data']['raw_dir']
    
    # Save output to data/ folder inside AI_train_model
    output_path = os.path.join(project_dir, "data", config['data']['preprocessed_filename'])
    
    sample_rate = config['model']['input_length']  # 256 Hz
    
    print("=" * 60)
    print("RUNNING CHB-MIT EDF PREPROCESSING PIPELINE")
    print("=" * 60)
    
    success = run_chbmit_preprocessing(
        raw_dataset_dir=raw_dataset_dir,
        output_path=output_path,
        sample_rate=sample_rate,
        window_sec=1
    )
    
    if success:
        print("\nPreprocessing step completed successfully!")
    else:
        print("\nPreprocessing step failed or was skipped. Please verify data paths.")
    print("=" * 60)

if __name__ == "__main__":
    main()
