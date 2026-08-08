"""Deterministic FP32 training from fold-local prepared arrays."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .models import build_reference_model


def train_fp32(train_npz: Path, validation_npz: Path, output_dir: Path, seed: int, epochs: int = 50, batch_size: int = 64) -> dict:
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("numpy and torch are required; install project dependencies") from error
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train = np.load(train_npz); validation = np.load(validation_npz)
    x_train, y_train = train["x"].astype("float32"), train["y"].astype("int64")
    x_val, y_val = validation["x"].astype("float32"), validation["y"].astype("int64")
    if x_train.ndim != 3 or x_train.shape[1:] != (19, 1024):
        raise ValueError("Training data must have shape [N,19,1024]")
    counts = np.bincount(y_train, minlength=2)
    if not counts.all():
        raise ValueError("Both classes are required in training data")
    weights = torch.as_tensor([1.0 / counts[label] for label in y_train], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_reference_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_loss, best_epoch, patience = float("inf"), 0, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(features.to(device)), labels.to(device)).backward()
            optimizer.step()
        model.eval(); total_loss = 0.0; total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                total_loss += float(loss_fn(model(features.to(device)), labels.to(device))) * len(labels)
                total += len(labels)
        value = total_loss / total
        if value < best_loss - 0.001:
            best_loss, best_epoch, patience = value, epoch, 0
            torch.save({"state_dict": model.state_dict(), "seed": seed, "best_epoch": epoch}, checkpoint)
        else:
            patience += 1
            if epoch >= 12 and patience >= 12:
                break
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {"seed": seed, "best_epoch": best_epoch, "validation_cross_entropy": best_loss, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash}
    (output_dir / "training_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

