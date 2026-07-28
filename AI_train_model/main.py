import argparse
import sys
import os

# Add current folder to path to enable package and script imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from scripts import run_audit, run_preprocess, run_eda, run_train, run_quantize

def main():
    parser = argparse.ArgumentParser(description="EEG Seizure Detection 1D-CNN Accelerator Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["audit", "preprocess", "eda", "train", "quantize", "all"],
        default="all",
        help="Pipeline phase to run: 'audit' (EDF metadata), 'preprocess' (EDF slicing), 'eda' (Analysis), 'train' (Training), 'quantize' (Quantize & export), or 'all' (Run all)."
    )
    
    args = parser.parse_args()
    
    if args.mode == "audit":
        run_audit.main()
    elif args.mode == "preprocess":
        run_preprocess.main()
    elif args.mode == "eda":
        run_eda.main()
    elif args.mode == "train":
        run_train.main()
    elif args.mode == "quantize":
        run_quantize.main()
    elif args.mode == "all":
        print("=" * 60)
        print("STARTING FULL EEG SEIZURE ACCELERATOR PIPELINE")
        print("=" * 60)
        run_preprocess.main()
        run_eda.main()
        run_train.main()
        run_quantize.main()
        print("=" * 60)
        print("FULL PIPELINE EXECUTED SUCCESSFULLY!")
        print("All outputs are saved to AI_train_model/outputs/")
        print("=" * 60)

if __name__ == "__main__":
    main()
