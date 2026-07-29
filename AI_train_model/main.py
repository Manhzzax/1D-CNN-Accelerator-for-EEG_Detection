import argparse
import sys
import os

# Add current folder to path to enable package and script imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from scripts import (
    run_audit,
    run_event_diagnostics,
    run_event_eval,
    run_hard_negative_mining,
    run_plan,
    run_preprocess,
    run_temporal_score_tcn,
    run_eda,
    run_train,
    run_quantize,
)

def main():
    parser = argparse.ArgumentParser(description="EEG Seizure Detection 1D-CNN Accelerator Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["audit", "plan", "preprocess", "mine_hard_negatives", "temporal_score_tcn", "eda", "train", "event_eval", "event_diagnostics", "quantize", "all"],
        default="all",
        help="Pipeline phase to run: 'audit' (EDF metadata), 'plan' (grouped split), 'preprocess' (EDF slicing), 'eda' (Analysis), 'train' (Training), 'event_eval' (continuous metrics), 'quantize' (Quantize & export), or 'all' (Run all)."
    )
    
    args = parser.parse_args()
    
    if args.mode == "audit":
        run_audit.main()
    elif args.mode == "plan":
        run_plan.main()
    elif args.mode == "preprocess":
        run_preprocess.main()
    elif args.mode == "mine_hard_negatives":
        run_hard_negative_mining.main()
    elif args.mode == "temporal_score_tcn":
        run_temporal_score_tcn.main()
    elif args.mode == "eda":
        run_eda.main()
    elif args.mode == "train":
        run_train.main()
    elif args.mode == "event_eval":
        run_event_eval.main()
    elif args.mode == "event_diagnostics":
        run_event_diagnostics.main()
    elif args.mode == "quantize":
        run_quantize.main()
    elif args.mode == "all":
        raise SystemExit(
            "The legacy full pipeline is disabled. Run audit and plan first; "
            "the leakage-safe preprocessing stage is being introduced separately."
        )

if __name__ == "__main__":
    main()
