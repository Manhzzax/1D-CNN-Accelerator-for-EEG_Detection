"""Mine train-only interictal windows from persistent alarm-like score contexts."""

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from .event_evaluation import load_scores, score_continuous_recordings
from .hard_negative_mining import (
    _copy_fixed_splits,
    _load_source_train_windows,
    _load_train_rows,
    _normal_starts,
    _recording_data,
)


def _load_or_score_train_stream(
    cache_path,
    rows,
    model,
    device,
    preprocessing,
    batch_size,
    use_amp,
    scaler_mean,
    scaler_std,
):
    """Reuse a verified source-model train score stream when it is available."""
    if cache_path:
        path = Path(cache_path)
        if path.is_file() and path.with_suffix(".records.json").is_file():
            scores = load_scores(path)
            expected_ids = [row["recording_id"] for row in rows]
            cached_ids = [record["recording_id"] for record in scores["records"]]
            if cached_ids != expected_ids:
                raise ValueError("Cached train score recordings do not match the locked training manifest")
            print(f"Reusing cached source train scores: {path}")
            return scores, str(path)
    print("No reusable train score cache found; scoring locked train recordings...")
    return (
        score_continuous_recordings(
            model,
            device,
            rows,
            preprocessing,
            batch_size,
            use_amp,
            scaler_mean,
            scaler_std,
        ),
        None,
    )


def _rolling_sum(values, window_size):
    return np.convolve(values.astype(np.int16), np.ones(window_size, dtype=np.int16), mode="full")[:len(values)]


def _select_recording_candidates(
    starts,
    probabilities,
    normal_starts,
    source_normal_keys,
    recording_id,
    threshold,
    decision_windows,
    min_hits,
    min_separation_windows,
):
    """Select separated windows from fully interictal persistent score contexts."""
    normal_mask = np.isin(starts, normal_starts, assume_unique=True)
    threshold_hits = probabilities >= threshold
    hit_counts = _rolling_sum(threshold_hits, decision_windows)
    normal_context_counts = _rolling_sum(normal_mask, decision_windows)
    source_mask = np.fromiter(
        ((recording_id, int(start)) in source_normal_keys for start in starts),
        dtype=bool,
        count=len(starts),
    )
    eligible = (
        normal_mask
        & (normal_context_counts == decision_windows)
        & (hit_counts >= min_hits)
        & ~source_mask
    )
    eligible_indices = np.flatnonzero(eligible)
    if not len(eligible_indices):
        return [], 0, 0

    # Prefer stronger persistence, then stronger latest CNN score.  The local
    # exclusion keeps a long artifact episode from filling the whole dataset.
    order = eligible_indices[np.lexsort((starts[eligible_indices], -probabilities[eligible_indices], -hit_counts[eligible_indices]))]
    blocked = np.zeros(len(starts), dtype=bool)
    selected = []
    for index in order:
        if blocked[index]:
            continue
        selected.append((int(starts[index]), int(hit_counts[index]), float(probabilities[index])))
        lower = max(0, index - min_separation_windows + 1)
        upper = min(len(starts), index + min_separation_windows)
        blocked[lower:upper] = True

    episode_count = int(np.count_nonzero(eligible & ~np.r_[False, eligible[:-1]]))
    return selected, int(len(eligible_indices)), episode_count


def mine_temporal_hard_negative_windows(
    protocol_dir,
    source_prepared_dir,
    output_dir,
    source_score_cache,
    model,
    device,
    preprocessing,
    batch_size,
    use_amp,
    scaler_mean,
    scaler_std,
    hard_negative_to_seizure_ratio,
    threshold,
    decision_windows,
    min_hits,
    min_separation_sec,
    seed,
    source_model_path,
):
    """Build a mixed dataset with separated, persistent train-only hard negatives."""
    if hard_negative_to_seizure_ratio <= 0:
        raise ValueError("hard_negative_to_seizure_ratio must be positive")
    if not 1 <= min_hits <= decision_windows:
        raise ValueError("min_hits must be between 1 and decision_windows")
    source_dir = Path(source_prepared_dir)
    target_dir = Path(output_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"Temporal hard-negative output already exists and is non-empty: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    source_x, source_y, source_records, source_starts, channels = _load_source_train_windows(source_dir)
    positive_count = int(source_y.sum())
    source_normal = source_y == 0
    source_normal_keys = {
        (str(recording_id), int(start))
        for recording_id, start in zip(source_records[source_normal], source_starts[source_normal])
    }
    rows = _load_train_rows(protocol_dir)
    scores, reused_cache = _load_or_score_train_stream(
        source_score_cache,
        rows,
        model,
        device,
        preprocessing,
        batch_size,
        use_amp,
        scaler_mean,
        scaler_std,
    )
    target_count = int(round(positive_count * hard_negative_to_seizure_ratio))
    sample_rate = preprocessing["sample_rate_hz"]
    stride_samples = int(preprocessing["stride_sec"] * sample_rate)
    min_separation_windows = max(1, int(round(min_separation_sec * sample_rate / stride_samples)))

    candidates = []
    eligible_count = 0
    episode_count = 0
    for row_index, (row, record) in enumerate(zip(rows, scores["records"])):
        offset_start = int(scores["record_offsets"][row_index])
        offset_end = int(scores["record_offsets"][row_index + 1])
        starts = scores["start_samples"][offset_start:offset_end]
        probabilities = scores["probabilities"][offset_start:offset_end]
        normal_starts = _normal_starts(row, int(record["sample_count"]), preprocessing)
        selected, record_eligible, record_episodes = _select_recording_candidates(
            starts,
            probabilities,
            normal_starts,
            source_normal_keys,
            row["recording_id"],
            threshold,
            decision_windows,
            min_hits,
            min_separation_windows,
        )
        eligible_count += record_eligible
        episode_count += record_episodes
        candidates.extend((hits, score, row_index, start) for start, hits, score in selected)

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    if len(candidates) < target_count:
        raise RuntimeError(
            f"Only {len(candidates)} separated persistent hard negatives are available; "
            f"need {target_count}. Lower min_hits, lower min_separation_sec, or lower the ratio."
        )
    selected_by_row = defaultdict(list)
    for hits, score, row_index, start in candidates[:target_count]:
        selected_by_row[row_index].append((start, hits, score))

    normal_signals = []
    normal_records = []
    normal_starts = []
    normal_scores = []
    normal_hits = []
    for row_index, selected in sorted(selected_by_row.items()):
        row = rows[row_index]
        data = _recording_data(row, preprocessing)
        window_samples = int(preprocessing["window_sec"] * sample_rate)
        for start, hits, score in sorted(selected):
            normal_signals.append(data[:, start:start + window_samples].copy())
            normal_records.append(row["recording_id"])
            normal_starts.append(start)
            normal_scores.append(score)
            normal_hits.append(hits)
        print(f"  Extracted temporal hard negatives: {len(normal_signals)}/{target_count}")

    x = np.concatenate((source_x, np.stack(normal_signals, axis=0)), axis=0)
    y = np.concatenate((source_y, np.zeros(len(normal_signals), dtype=np.int64)))
    recording_ids = np.concatenate((source_records, np.asarray(normal_records)))
    starts = np.concatenate((source_starts, np.asarray(normal_starts, dtype=np.int64)))
    order = np.random.default_rng(seed).permutation(len(y))
    np.savez_compressed(
        target_dir / "chbmit_train.npz",
        X=x[order], y=y[order], recording_id=recording_ids[order],
        start_sample=starts[order], channels=channels, split="train",
    )
    _copy_fixed_splits(source_dir, target_dir)

    summary = {
        "strategy": "persistent_train_only_score_contexts",
        "source_prepared_dir": str(source_dir),
        "source_model_path": str(source_model_path),
        "source_model_sha256": hashlib.sha256(Path(source_model_path).read_bytes()).hexdigest(),
        "source_score_cache": reused_cache,
        "positive_windows": positive_count,
        "source_normal_windows": int(source_normal.sum()),
        "hard_negative_windows": int(len(normal_signals)),
        "total_normal_windows": int((y == 0).sum()),
        "total_normal_to_seizure_ratio": float((y == 0).sum() / positive_count),
        "hard_negative_to_seizure_ratio": hard_negative_to_seizure_ratio,
        "threshold": threshold,
        "decision_window_windows": decision_windows,
        "min_hits_in_context": min_hits,
        "min_separation_sec": min_separation_sec,
        "persistent_candidate_windows": eligible_count,
        "persistent_candidate_episodes": episode_count,
        "separated_candidate_windows": int(len(candidates)),
        "hard_negative_score_min": float(min(normal_scores)),
        "hard_negative_score_max": float(max(normal_scores)),
        "hard_negative_hit_count_min": int(min(normal_hits)),
        "hard_negative_hit_count_max": int(max(normal_hits)),
        "channels": channels.astype(str).tolist(),
    }
    with (target_dir / "temporal_hard_negative_mining_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return summary
