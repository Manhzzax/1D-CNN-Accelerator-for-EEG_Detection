"""Generate descriptive false-alarm diagnostics from saved continuous score arrays."""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from src.event_diagnostics import analyze_event_run
from src.data_loader import load_config
from src.utils import get_outputs_dir


def main():
    run_id = os.environ.get("CHBMIT_ANALYSIS_RUN_ID", "")
    split_name = os.environ.get("CHBMIT_ANALYSIS_SPLIT", "val")
    if split_name not in {"val", "test"}:
        raise ValueError("CHBMIT_ANALYSIS_SPLIT must be val or test")
    run_output_dir = get_outputs_dir(run_id)
    config = load_config()
    summary = analyze_event_run(
        run_output_dir,
        split_name,
        config["preprocessing"],
        config["evaluation"],
    )
    aggregate = summary["aggregate"]
    print("=" * 60)
    print(f"EVENT DIAGNOSTICS: {split_name}")
    print("=" * 60)
    print(
        f"Recordings: {aggregate['recordings']} | Events: {aggregate['detected_events']}/"
        f"{aggregate['total_events']} | FAR/h: {aggregate['false_alarms_per_hour']:.4f}"
    )
    print(f"Summary: {run_output_dir}/event_diagnostics_{split_name}_summary.json")
