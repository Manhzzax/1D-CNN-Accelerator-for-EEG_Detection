"""Command line entry point for the Path A final-evaluation audit."""

import argparse
from pathlib import Path

from .aggregate import write_audit


def _paths(pattern):
    return sorted(Path().glob(pattern))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Path A final-evaluation audit")
    parser.add_argument("audit", nargs="?", default="audit", choices=["audit"])
    parser.add_argument("--window-glob", required=True, help="Glob for checkpoint_test_evaluation.json files")
    parser.add_argument("--event-glob", default="", help="Optional glob for patient_specific_event_replay.json files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    windows = _paths(args.window_glob)
    events = _paths(args.event_glob) if args.event_glob else []
    result = write_audit(windows, events, args.output, args.bootstrap_replicates, args.bootstrap_seed)
    print(
        "Path A audit written: "
        f"cases={result['window']['case_count']} groups={result['window']['patient_group_count']} "
        f"final_eligible={result['window']['final_claim_eligible']} output={args.output}"
    )


if __name__ == "__main__":
    main()
