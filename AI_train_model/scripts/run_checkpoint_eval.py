"""Evaluate a stored checkpoint without retraining it.

This runner is intended for explicitly labelled exploratory test probes. It
never writes into the source model directory and defaults to FP32 inference so
AMP training can be checked independently.
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

from scripts.run_train import _score_window_loader, _window_metrics
from src.data_loader import get_train_val_test_datasets, load_config
from src.model import build_model_from_run
from src.utils import get_outputs_dir


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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_test_indices(targets, seed):
    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    if not len(positive) or len(negative) < len(positive):
        raise ValueError("Cannot construct a balanced test diagnostic from this split")
    generator = np.random.default_rng(seed)
    selected_negative = generator.choice(negative, size=len(positive), replace=False)
    return np.sort(np.concatenate([positive, selected_negative]))


def main():
    source_run_id = os.environ.get("CHBMIT_CHECKPOINT_SOURCE_RUN_ID", "")
    if not source_run_id:
        raise ValueError("CHBMIT_CHECKPOINT_SOURCE_RUN_ID is required")

    source_dir = get_outputs_dir(source_run_id)
    checkpoint_path = os.path.join(source_dir, "best_model.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    output_dir = get_outputs_dir()
    if os.path.abspath(output_dir) == os.path.abspath(source_dir):
        raise ValueError("CHBMIT_RUN_ID must be a new evaluation-artifact directory")
    os.makedirs(output_dir, exist_ok=True)

    config = load_config()
    torch.set_num_threads(int(config["training"].get("num_threads", 4)))
    _, _, test_dataset = get_train_val_test_datasets()
    batch_size = int(os.environ.get("CHBMIT_EVAL_BATCH_SIZE", config["training"]["batch_size"]))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = _env_bool("CHBMIT_EVAL_USE_AMP", False) and device.type == "cuda"
    model = build_model_from_run(source_dir).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    probabilities, targets = _score_window_loader(model, test_loader, device, use_amp)

    balanced_seed = int(os.environ.get("CHBMIT_BALANCED_TEST_SEED", "42"))
    balanced_indices = _balanced_test_indices(targets, balanced_seed)
    result = {
        "evaluation_kind": "exploratory_test_probe",
        "selection_warning": (
            "This test evaluation is exploratory. Do not use it to select a new architecture, "
            "hyperparameter, or threshold."
        ),
        "source_run_id": source_run_id,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "inference": {
            "device": str(device),
            "use_amp": use_amp,
            "precision_label": "AMP" if use_amp else "FP32",
            "model_parameter_dtype": str(next(model.parameters()).dtype),
        },
        "test_prevalence_metrics": _window_metrics(targets, probabilities),
        "balanced_test_diagnostic": {
            "seed": balanced_seed,
            "positive_windows": int(np.sum(targets[balanced_indices] == 1)),
            "negative_windows": int(np.sum(targets[balanced_indices] == 0)),
            "metrics": _window_metrics(targets[balanced_indices], probabilities[balanced_indices]),
        },
        "test_window_counts": {
            "total": int(len(targets)),
            "positive": int(np.sum(targets == 1)),
            "negative": int(np.sum(targets == 0)),
        },
    }
    result_path = os.path.join(output_dir, "checkpoint_test_evaluation.json")
    with open(result_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    balanced = result["balanced_test_diagnostic"]["metrics"]
    prevalence = result["test_prevalence_metrics"]
    print("=" * 60)
    print("CHECKPOINT TEST EVALUATION")
    print("=" * 60)
    print(f"Source run: {source_run_id} | inference: {result['inference']['precision_label']}")
    print(f"Prevalence test accuracy: {100.0 * prevalence['accuracy']:.3f}%")
    print(f"Balanced test accuracy: {100.0 * balanced['accuracy']:.3f}%")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
