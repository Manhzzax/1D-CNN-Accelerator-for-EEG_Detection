"""Causal temporal context model over a frozen 1-second CNN score stream."""

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class TemporalScoreTCN(nn.Module):
    """Small causal TCN that classifies the latest score using prior score context."""

    def __init__(self, context_windows, hidden_channels, dropout):
        super().__init__()
        self.context_windows = context_windows
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x):
        x = functional.pad(x, (2, 0))
        x = functional.relu(self.conv1(x))
        x = functional.pad(x, (4, 0))
        x = functional.relu(self.conv2(x))
        x = self.dropout(x[:, :, -1])
        return self.classifier(x)


def _score_to_logit(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    clipped = np.clip(probabilities, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def score_context_features(probabilities, context_windows):
    """Return a causal feature row for every score without crossing recordings."""
    if context_windows < 1:
        raise ValueError("context_windows must be positive")
    logits = _score_to_logit(probabilities)
    padded = np.pad(logits, (context_windows - 1, 0), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, context_windows)
    return np.ascontiguousarray(windows[:, None, :], dtype=np.float32)


def score_labels(scores, preprocessing):
    """Create full-ictal labels and guard-validity flags for continuous scores."""
    sample_rate = preprocessing["sample_rate_hz"]
    window_samples = int(preprocessing["window_sec"] * sample_rate)
    guard_samples = int(preprocessing["interictal_guard_sec"] * sample_rate)
    labels = []
    valid = []
    for record_index, record in enumerate(scores["records"]):
        offset_start = int(scores["record_offsets"][record_index])
        offset_end = int(scores["record_offsets"][record_index + 1])
        starts = scores["start_samples"][offset_start:offset_end]
        ends = starts + window_samples
        record_ictal = np.zeros(len(starts), dtype=bool)
        record_guarded = np.zeros(len(starts), dtype=bool)
        for seizure_start, seizure_end in record["seizure_intervals"]:
            record_ictal |= (starts >= seizure_start) & (ends <= seizure_end)
            guarded_start = max(0, seizure_start - guard_samples)
            guarded_end = min(record["sample_count"], seizure_end + guard_samples)
            record_guarded |= (ends > guarded_start) & (starts < guarded_end)
        labels.append(record_ictal.astype(np.int64))
        valid.append(record_ictal | ~record_guarded)
    return np.concatenate(labels), np.concatenate(valid)


def build_context_dataset(scores, preprocessing, context_windows, random_ratio, hard_ratio, seed):
    """Sample all ictal score contexts plus random and hard normal score contexts."""
    if random_ratio < 0 or hard_ratio < 0:
        raise ValueError("Context normal sampling ratios cannot be negative")
    feature_blocks = []
    for record_index in range(len(scores["records"])):
        start = int(scores["record_offsets"][record_index])
        end = int(scores["record_offsets"][record_index + 1])
        probabilities = scores["probabilities"][start:end]
        feature_blocks.append(score_context_features(probabilities, context_windows))
    x = np.concatenate(feature_blocks)
    y, valid = score_labels(scores, preprocessing)
    positive_indices = np.flatnonzero(y == 1)
    normal_indices = np.flatnonzero((y == 0) & valid)
    if not len(positive_indices) or not len(normal_indices):
        raise ValueError("Temporal context training requires ictal and non-seizure score windows")

    random_state = np.random.default_rng(seed)
    random_count = min(int(round(len(positive_indices) * random_ratio)), len(normal_indices))
    random_indices = random_state.choice(normal_indices, size=random_count, replace=False)
    remaining_mask = (y == 0) & valid
    remaining_mask[random_indices] = False
    remaining_normals = np.flatnonzero(remaining_mask)
    hard_count = min(int(round(len(positive_indices) * hard_ratio)), len(remaining_normals))
    context_priority = np.max(x[:, 0, :], axis=1)
    hard_order = (
        np.argpartition(context_priority[remaining_normals], len(remaining_normals) - hard_count)[-hard_count:]
        if hard_count else []
    )
    hard_indices = remaining_normals[hard_order] if hard_count else np.empty(0, dtype=np.int64)
    selected = np.concatenate((positive_indices, random_indices, hard_indices))
    selected = random_state.permutation(selected)
    summary = {
        "positive_sequences": int(len(positive_indices)),
        "random_normal_sequences": int(len(random_indices)),
        "hard_normal_sequences": int(len(hard_indices)),
        "selected_sequences": int(len(selected)),
        "guard_excluded_normal_sequences": int(((y == 0) & ~valid).sum()),
        "context_windows": context_windows,
    }
    return x[selected], y[selected], summary


def adjust_scores_with_tcn(model, device, scores, context_windows, batch_size, use_amp):
    """Replace base CNN probabilities with causal TCN probabilities per recording."""
    model.eval()
    adjusted = []
    with torch.no_grad():
        for record_index in range(len(scores["records"])):
            start = int(scores["record_offsets"][record_index])
            end = int(scores["record_offsets"][record_index + 1])
            features = score_context_features(scores["probabilities"][start:end], context_windows)
            probabilities = []
            for batch_start in range(0, len(features), batch_size):
                inputs = torch.from_numpy(features[batch_start:batch_start + batch_size]).to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(inputs)
                else:
                    logits = model(inputs)
                probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            adjusted.append(np.concatenate(probabilities))
    return {
        **scores,
        "probabilities": np.concatenate(adjusted),
    }


@dataclass
class TemporalTrainingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    stopped_early: bool


def train_tcn(model, device, train_x, train_y, val_x, val_y, options, output_path, use_amp):
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    counts = torch.bincount(torch.from_numpy(train_y), minlength=2).to(torch.float64)
    sample_weights = (1.0 / counts[torch.from_numpy(train_y)]).to(torch.double)
    train_loader = DataLoader(
        train_dataset,
        batch_size=options["batch_size"],
        sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=options["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    validation_counts = torch.bincount(torch.from_numpy(val_y), minlength=2).to(torch.float32)
    validation_weights = len(val_y) / (2.0 * validation_counts)
    validation_criterion = nn.CrossEntropyLoss(weight=validation_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    early = options["early_stopping"]
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    stopped_early = False

    for epoch in range(options["epochs"]):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    loss = criterion(model(inputs), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(inputs), targets)
                loss.backward()
                optimizer.step()

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        loss = validation_criterion(model(inputs), targets)
                else:
                    loss = validation_criterion(model(inputs), targets)
                validation_loss += loss.item() * len(inputs)
        validation_loss /= len(val_dataset)
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Temporal TCN validation loss is not finite")
        print(f"TCN Epoch [{epoch + 1:2d}/{options['epochs']}] | Val Loss: {validation_loss:.4f}")
        if validation_loss < best_loss - early["min_delta"]:
            best_loss = validation_loss
            best_epoch = epoch + 1
            no_improvement = 0
            torch.save(model.state_dict(), output_path)
        else:
            no_improvement += 1
        if epoch + 1 >= early["min_epochs"] and no_improvement >= early["patience"]:
            stopped_early = True
            print(f"TCN early stopping at epoch {epoch + 1}")
            break
    if best_epoch == 0:
        raise RuntimeError("Temporal TCN did not produce a valid checkpoint")
    return TemporalTrainingResult(best_epoch, best_loss, epoch + 1, stopped_early)


def write_json(path, content):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(content, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
