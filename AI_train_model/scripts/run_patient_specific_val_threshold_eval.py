"""Path A next step: val-selected threshold, then one sealed test evaluation.

For each trained patient-specific run:
1. Score validation windows and choose threshold maximizing balanced accuracy.
2. Apply that frozen threshold to the sealed test set once (prevalence + balanced).
3. Never re-select the threshold using test labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from scripts.run_checkpoint_eval import _balanced_test_indices
from scripts.run_train import _score_window_loader, _window_metrics
from src.data_loader import get_train_val_test_datasets, load_config
from src.model import build_model_from_run
from src.operating_point import select_threshold_max_balanced_accuracy
from src.utils import get_outputs_dir


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false/1/0")


def _discover_case_runs():
    outputs = os.path.join(project_dir, "outputs")
    pattern = re.compile(r"^ps_a1_(chb\d+)_s42$")
    runs = []
    if not os.path.isdir(outputs):
        return runs
    for name in sorted(os.listdir(outputs)):
        match = pattern.match(name)
        if not match:
            continue
        checkpoint = os.path.join(outputs, name, "best_model.pth")
        if os.path.isfile(checkpoint):
            runs.append((match.group(1), name))
    return runs


def _evaluate_one(case_id, source_run_id, config):
    source_dir = get_outputs_dir(source_run_id)
    checkpoint_path = os.path.join(source_dir, "best_model.pth")
    eval_run_id = f"ps_a1_valthr_test_{source_run_id}"
    output_dir = get_outputs_dir(eval_run_id)
    if os.path.abspath(output_dir) == os.path.abspath(source_dir):
        raise ValueError("Evaluation run id must differ from the source run id")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "val_threshold_test_evaluation.json")
    if os.path.isfile(result_path) and not _env_bool("CHBMIT_PS_OVERWRITE_EVAL", False):
        print(f"Skip existing: {result_path}")
        return result_path

    # Point dataset loading side-effects at this evaluation artifact directory.
    os.environ["CHBMIT_RUN_ID"] = eval_run_id
    os.environ["CHBMIT_PREPARED_OUTPUT_DIR"] = f"chbmit_prepared_ps_a1_v1/{case_id}"
    train_ds, val_ds, test_ds = get_train_val_test_datasets(include_test=True)
    batch_size = int(os.environ.get("CHBMIT_EVAL_BATCH_SIZE", config["training"]["batch_size"]))
    num_workers = int(config["training"].get("num_workers", 4))
    pin_memory = bool(config["training"].get("pin_memory", True))

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = _env_bool("CHBMIT_EVAL_USE_AMP", False) and device.type == "cuda"
    model = build_model_from_run(source_dir).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    val_prob, val_y = _score_window_loader(model, val_loader, device, use_amp)
    selection = select_threshold_max_balanced_accuracy(val_y, val_prob)
    threshold = selection["threshold"]

    test_prob, test_y = _score_window_loader(model, test_loader, device, use_amp)
    balanced_seed = int(os.environ.get("CHBMIT_BALANCED_TEST_SEED", "42"))
    balanced_indices = _balanced_test_indices(test_y, balanced_seed)

    # Balanced validation diagnostic under the selected threshold (not for selection of architecture).
    if len(np.unique(val_y)) >= 2 and int(np.sum(val_y == 1)) > 0:
        val_bal_idx = _balanced_test_indices(val_y, balanced_seed)
        val_balanced_metrics = _window_metrics(val_y[val_bal_idx], val_prob[val_bal_idx], threshold=threshold)
    else:
        val_balanced_metrics = None

    result = {
        "evaluation_kind": "patient_specific_val_selected_threshold_then_sealed_test",
        "selection_rule": (
            "Threshold maximizes validation balanced accuracy on the full validation windows "
            "using grid 0.05:0.01:0.95. Test is evaluated once and never used for selection."
        ),
        "case_id": case_id,
        "source_run_id": source_run_id,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "threshold_selection": selection,
        "validation_full_metrics_at_selected_threshold": _window_metrics(val_y, val_prob, threshold=threshold),
        "validation_balanced_metrics_at_selected_threshold": val_balanced_metrics,
        "test_prevalence_metrics_at_selected_threshold": _window_metrics(test_y, test_prob, threshold=threshold),
        "balanced_test_diagnostic": {
            "seed": balanced_seed,
            "positive_windows": int(np.sum(test_y[balanced_indices] == 1)),
            "negative_windows": int(np.sum(test_y[balanced_indices] == 0)),
            "metrics": _window_metrics(
                test_y[balanced_indices], test_prob[balanced_indices], threshold=threshold
            ),
        },
        "test_prevalence_metrics_at_0_5": _window_metrics(test_y, test_prob, threshold=0.5),
        "balanced_test_at_0_5": _window_metrics(
            test_y[balanced_indices], test_prob[balanced_indices], threshold=0.5
        ),
        "test_window_counts": {
            "total": int(len(test_y)),
            "positive": int(np.sum(test_y == 1)),
            "negative": int(np.sum(test_y == 0)),
        },
        "val_window_counts": {
            "total": int(len(val_y)),
            "positive": int(np.sum(val_y == 1)),
            "negative": int(np.sum(val_y == 0)),
        },
    }
    with open(result_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    bal = result["balanced_test_diagnostic"]["metrics"]
    print(
        f"{case_id}: thr={threshold:.2f} val_bal_sel={100*selection['validation_balanced_accuracy']:.2f}% "
        f"test_bal={100*bal['balanced_accuracy']:.2f}% (was@0.5 {100*result['balanced_test_at_0_5']['balanced_accuracy']:.2f}%)"
    )
    return result_path


def main():
    os.environ.setdefault("CHBMIT_CONFIG_PATH", "config/patient_specific_a1.yaml")
    config = load_config()
    torch.set_num_threads(int(config["training"].get("num_threads", 4)))
    only_case = os.environ.get("CHBMIT_PS_CASE_ID", "").strip()
    runs = _discover_case_runs()
    if only_case:
        runs = [item for item in runs if item[0] == only_case]
        if not runs:
            raise SystemExit(f"No trained run for case {only_case}")

    print("=" * 60)
    print("PATH A: VAL-SELECTED THRESHOLD → SEALED TEST")
    print("=" * 60)
    paths = []
    for case_id, source_run_id in runs:
        paths.append(_evaluate_one(case_id, source_run_id, config))
    print(f"Completed {len(paths)} case evaluations.")


if __name__ == "__main__":
    main()
