"""Validation-only hyperparameter sweep for the locked CHB-MIT detection protocol."""

import csv
import json
import os
import re
import subprocess
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
outputs_root = os.path.join(project_dir, "outputs")


BASELINE_MIXED_TRIALS = [
    {"name": "a_lr1e3_wd1e4_balanced", "learning_rate": 1e-3, "weight_decay": 1e-4, "balanced": True},
    {"name": "b_lr3e4_wd1e4_balanced", "learning_rate": 3e-4, "weight_decay": 1e-4, "balanced": True},
    {"name": "c_lr3e4_wd5e4_balanced", "learning_rate": 3e-4, "weight_decay": 5e-4, "balanced": True},
    {"name": "d_lr1e3_wd5e4_balanced", "learning_rate": 1e-3, "weight_decay": 5e-4, "balanced": True},
    {"name": "e_lr3e4_wd1e4_nobalance", "learning_rate": 3e-4, "weight_decay": 1e-4, "balanced": False},
    {"name": "f_lr1e3_wd1e4_nobalance", "learning_rate": 1e-3, "weight_decay": 1e-4, "balanced": False},
]

SEPARABLE_RAW_TRIALS = [
    {"name": "a_lr1e3_wd1e4_balanced", "learning_rate": 1e-3, "weight_decay": 1e-4, "balanced": True},
    {"name": "b_lr3e4_wd1e4_balanced", "learning_rate": 3e-4, "weight_decay": 1e-4, "balanced": True},
    {"name": "c_lr3e4_wd5e4_balanced", "learning_rate": 3e-4, "weight_decay": 5e-4, "balanced": True},
    {"name": "d_lr1e3_wd5e4_balanced", "learning_rate": 1e-3, "weight_decay": 5e-4, "balanced": True},
    {"name": "e_lr3e4_wd1e4_nobalance", "learning_rate": 3e-4, "weight_decay": 1e-4, "balanced": False},
    {"name": "f_lr1e3_wd1e4_nobalance", "learning_rate": 1e-3, "weight_decay": 1e-4, "balanced": False},
]

SEPARABLE_RAW_REFINE_TRIALS = [
    {
        "name": "a_reference_lr1e3_wd1e4",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "balanced": False,
        "separable_overrides": {},
    },
    {
        "name": "b_lr5e4_wd1e4",
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "balanced": False,
        "separable_overrides": {},
    },
    {
        "name": "c_lr1e3_wd3e4",
        "learning_rate": 1e-3,
        "weight_decay": 3e-4,
        "balanced": False,
        "separable_overrides": {},
    },
    {
        "name": "d_dropout10",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "balanced": False,
        "separable_overrides": {"CHBMIT_SEPARABLE_DROPOUT": 0.10},
    },
    {
        "name": "e_spatial48",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "balanced": False,
        "separable_overrides": {"CHBMIT_SEPARABLE_SPATIAL_FILTERS": 48},
    },
    {
        "name": "f_temporal3",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "balanced": False,
        "separable_overrides": {"CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL": 3},
    },
]

SWEEP_FAMILIES = {
    "baseline_mixed": {
        "architecture": "baseline_1dcnn",
        "default_prepared_output_dir": "chbmit_prepared_mixed_hardneg_v1",
        "trials": BASELINE_MIXED_TRIALS,
    },
    "separable_raw": {
        "architecture": "separable_1dcnn",
        "default_prepared_output_dir": "chbmit_prepared_v1",
        "trials": SEPARABLE_RAW_TRIALS,
    },
    "separable_raw_refine": {
        "architecture": "separable_1dcnn",
        "default_prepared_output_dir": "chbmit_prepared_v1",
        "trials": SEPARABLE_RAW_REFINE_TRIALS,
    },
}


def _sweep_id():
    value = os.environ.get("CHBMIT_SWEEP_ID", "run_09_hparam")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("CHBMIT_SWEEP_ID must contain only letters, digits, underscores, or hyphens")
    return value


def _run(command, environment):
    subprocess.run(command, cwd=project_dir, env=environment, check=True)


def _load_result(run_id):
    path = os.path.join(outputs_root, run_id, "event_metrics.json")
    with open(path, "r", encoding="utf-8") as input_file:
        summary = json.load(input_file)
    with open(os.path.join(outputs_root, run_id, "validation_window_metrics.json"), "r", encoding="utf-8") as input_file:
        window = json.load(input_file)
    selected = summary["threshold_selection"]
    return {
        "run_id": run_id,
        "event_sensitivity": selected["event_sensitivity"],
        "detected_events": selected["detected_events"],
        "total_events": selected["total_events"],
        "false_alarms_per_hour": selected["false_alarms_per_hour"],
        "false_alarms": selected["false_alarms"],
        "median_detection_delay_sec": selected["median_detection_delay_sec"],
        "threshold": selected["threshold"],
        "policy_name": selected["policy_name"],
        "target_far_met": summary["target_false_alarms_per_hour_met_on_validation"],
        "validation_window_accuracy": window["accuracy"],
        "validation_balanced_accuracy": window["balanced_accuracy"],
        "validation_ictal_f1": window["f1"],
        "validation_auroc": window["auroc"],
    }


def _rank_key(row):
    delay = row["median_detection_delay_sec"]
    delay = float("inf") if delay is None else delay
    return (
        row["target_far_met"],
        row["event_sensitivity"],
        -row["false_alarms_per_hour"],
        -delay,
        row["validation_auroc"],
        row["validation_ictal_f1"],
        row["validation_window_accuracy"],
    )


def main():
    sweep_id = _sweep_id()
    family_name = os.environ.get("CHBMIT_SWEEP_FAMILY", "baseline_mixed")
    if family_name not in SWEEP_FAMILIES:
        raise ValueError(f"CHBMIT_SWEEP_FAMILY must be one of {sorted(SWEEP_FAMILIES)}")
    family = SWEEP_FAMILIES[family_name]
    prepared_output_dir = os.environ.get("CHBMIT_PREPARED_OUTPUT_DIR", family["default_prepared_output_dir"])
    resume = os.environ.get("CHBMIT_SWEEP_RESUME", "0") == "1"
    results = []
    for trial in family["trials"]:
        run_id = f"{sweep_id}_{trial['name']}"
        result_path = os.path.join(outputs_root, run_id, "event_metrics.json")
        if resume and os.path.isfile(result_path):
            print(f"Reusing completed validation trial: {run_id}")
        else:
            environment = os.environ.copy()
            environment.update({
                "CHBMIT_RUN_ID": run_id,
                "CHBMIT_MODEL_RUN_ID": run_id,
                "CHBMIT_MODEL_ARCHITECTURE": family["architecture"],
                "CHBMIT_PREPARED_OUTPUT_DIR": prepared_output_dir,
                "CHBMIT_TRAIN_LEARNING_RATE": str(trial["learning_rate"]),
                "CHBMIT_TRAIN_WEIGHT_DECAY": str(trial["weight_decay"]),
                "CHBMIT_CLASS_BALANCED_BATCHES": str(trial["balanced"]).lower(),
                "CHBMIT_SKIP_TEST_EVALUATION": "1",
                "CHBMIT_EVENT_EVAL_SPLITS": "val",
            })
            for environment_name, value in trial.get("separable_overrides", {}).items():
                environment[environment_name] = str(value)
            print("=" * 60)
            print(f"VALIDATION-ONLY TRIAL: {run_id}")
            print("=" * 60)
            _run([sys.executable, "main.py", "--mode", "train"], environment)
            _run([sys.executable, "main.py", "--mode", "event_eval"], environment)
        result = _load_result(run_id)
        result.update(trial)
        result["separable_overrides"] = json.dumps(trial.get("separable_overrides", {}), sort_keys=True)
        result["architecture"] = family["architecture"]
        result["prepared_output_dir"] = prepared_output_dir
        result["internal_clinical_screen_pass_validation"] = bool(
            result["event_sensitivity"] >= 0.90
            and result["false_alarms_per_hour"] <= 0.50
            and result["median_detection_delay_sec"] is not None
            and result["median_detection_delay_sec"] <= 10.0
        )
        results.append(result)

    results.sort(key=_rank_key, reverse=True)
    sweep_dir = os.path.join(outputs_root, sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    with open(os.path.join(sweep_dir, "validation_leaderboard.json"), "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    with open(os.path.join(sweep_dir, "validation_leaderboard.csv"), "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    winner = results[0]
    print("=" * 60)
    print(
        "Validation winner: "
        f"{winner['run_id']} | sensitivity={winner['event_sensitivity']:.4f} | "
        f"FAR/h={winner['false_alarms_per_hour']:.4f} | "
        f"delay={winner['median_detection_delay_sec']} | "
        f"validation_accuracy={winner['validation_window_accuracy']:.4f}"
    )
    print(f"Leaderboard: {os.path.join(sweep_dir, 'validation_leaderboard.csv')}")
    print(f"Sweep family: {family_name} | architecture: {family['architecture']}")
    print("No test recording was scored by this sweep.")


if __name__ == "__main__":
    main()
