"""V2.3 train-only, policy-aligned hard-negative cache construction."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from src.chbmit_preparation import extract_canonical_bipolar_data, filter_eeg
from src.event_evaluation import generate_alarms, load_scores, save_scores, score_continuous_recordings

from .protocol import canonical_json_hash, causal_window_index, file_sha256, save_json


def _v23_preprocessing(config: dict) -> dict:
    return {
        "sample_rate_hz": int(config["dataset"]["sample_rate_hz"]),
        "window_sec": float(config["preprocessing"]["window_sec"]),
        "stride_sec": float(config["preprocessing"]["stride_sec"]),
        "bandpass_low_hz": float(config["preprocessing"]["bandpass_hz"][0]),
        "bandpass_high_hz": float(config["preprocessing"]["bandpass_hz"][1]),
        "notch_hz": float(config["preprocessing"]["notch_hz"]),
        "filter_mode": config["preprocessing"]["filter_mode"],
        "interictal_guard_sec": float(config["preprocessing"]["interictal_guard_sec"]),
    }


def _load_manifest_rows(path: str | Path, split: str) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} rows in {path}")
    return rows


def _load_source_train_windows(prepared_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(prepared_dir / "chbmit_train.npz", allow_pickle=False) as source:
        required = {"X", "y", "recording_id", "start_sample", "channels"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"Source train cache is missing fields: {sorted(missing)}")
        x = np.asarray(source["X"], dtype=np.float32)
        y = np.asarray(source["y"], dtype=np.int64)
        recording_ids = np.asarray(source["recording_id"]).astype(str)
        starts = np.asarray(source["start_sample"], dtype=np.int64)
        channels = np.asarray(source["channels"]).astype(str)
    if x.ndim != 3 or len(x) != len(y) or len(y) != len(recording_ids) or len(y) != len(starts):
        raise ValueError("Source V2.1 train cache has inconsistent tensor metadata")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("Source V2.1 train cache must contain ictal and normal samples")
    return x, y, recording_ids, starts, channels


def _hash_checked_source(source_dir: Path, source: dict) -> tuple[Path, np.ndarray, np.ndarray]:
    required = ("best_model.pth", "model_spec.json", "scaler_mean.npy", "scaler_scale.npy", "temporal_confirmation.json")
    for name in required:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(f"Missing frozen V2.2 source artifact: {source_dir / name}")
    expected_hashes = {
        "best_model.pth": source["checkpoint_sha256"],
        "scaler_mean.npy": source["scaler_mean_sha256"],
        "scaler_scale.npy": source["scaler_scale_sha256"],
    }
    for name, expected in expected_hashes.items():
        observed = file_sha256(source_dir / name)
        if observed != expected:
            raise ValueError(f"Frozen source hash differs for {name}: {observed} != {expected}")
    with (source_dir / "model_spec.json").open("r", encoding="utf-8") as input_file:
        model_spec = json.load(input_file)
    if model_spec.get("architecture") != "paper_a_multiscale_residual_1dcnn" or model_spec.get("parameter_count") != 57446:
        raise ValueError("V2.3 requires the exact 57,446-parameter C1 source model")
    mean = np.load(source_dir / "scaler_mean.npy").astype(np.float32)
    scale = np.load(source_dir / "scaler_scale.npy").astype(np.float32)
    if mean.shape != (17,) or scale.shape != (17,) or np.any(scale <= 0):
        raise ValueError("Frozen source scaler is invalid")
    return source_dir / "best_model.pth", mean, scale


def _verify_source_policy(source_dir: Path, expected: dict) -> dict:
    """Use only the calibration-selected policy persisted by V2.2."""
    with (source_dir / "temporal_confirmation.json").open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if payload.get("policy_selection_status") != "feasible_calibration_policy_selected":
        raise ValueError("V2.3 source run has no feasible calibration policy")
    selected = payload.get("selected_calibration_policy")
    if not isinstance(selected, dict):
        raise ValueError("V2.3 source run lacks a selected calibration policy")
    fields = ("policy_name", "positive_windows", "decision_window_windows", "threshold")
    for field in fields:
        if field not in selected or field not in expected:
            raise ValueError(f"V2.3 source policy lacks {field}")
        if isinstance(expected[field], float):
            if not np.isclose(float(selected[field]), float(expected[field]), rtol=0.0, atol=1e-12):
                raise ValueError(f"V2.3 source policy mismatch for {field}")
        elif selected[field] != expected[field]:
            raise ValueError(f"V2.3 source policy mismatch for {field}")
    return {field: selected[field] for field in fields}


def _verify_v21_cache(prepared_dir: Path, manifest_path: str | Path) -> None:
    required = ("chbmit_train.npz", "chbmit_val.npz", "chbmit_temporal_eval.npz", "feature_representation.json", "preparation_summary.json")
    for name in required:
        if not (prepared_dir / name).is_file():
            raise FileNotFoundError(f"Missing V2.1 confirmation cache file: {prepared_dir / name}")
    if (prepared_dir / "chbmit_test.npz").exists() or (prepared_dir / "continuous_test_recordings.csv").exists():
        raise ValueError("V2.3 refuses a source cache containing a sealed-test artifact")
    with (prepared_dir / "preparation_summary.json").open("r", encoding="utf-8") as input_file:
        summary = json.load(input_file)
    if summary.get("protocol") != "research_v2_1_confirmation" or int(summary.get("window_samples", 0)) != 1280:
        raise ValueError("V2.3 requires a verified V2.1 five-second confirmation cache")
    if summary.get("fold_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("V2.1 source cache does not match the locked fold manifest")


def _normal_start_set(row: dict, sample_count: int, preprocessing: dict) -> set[int]:
    sample_rate = int(preprocessing["sample_rate_hz"])
    intervals_seconds = json.loads(row["seizure_intervals_json"])
    intervals = [(round(float(start) * sample_rate), round(float(end) * sample_rate)) for start, end in intervals_seconds]
    _, normal, _ = causal_window_index(
        int(sample_count), intervals,
        int(preprocessing["window_sec"] * sample_rate),
        int(preprocessing["stride_sec"] * sample_rate),
        int(preprocessing["interictal_guard_sec"] * sample_rate),
    )
    return set(map(int, normal))


def select_policy_aligned_candidates(
    starts: np.ndarray,
    probabilities: np.ndarray,
    clean_starts: set[int],
    source_normal_keys: set[tuple[str, int]],
    recording_id: str,
    patient_group: str,
    window_samples: int,
    threshold: float,
    positive_windows: int,
    decision_window_windows: int,
    refractory_samples: int,
) -> tuple[list[dict], dict]:
    """Select one hard hit from each fully clean source false-alarm context."""
    endpoints = np.asarray(starts, dtype=np.int64) + int(window_samples)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(endpoints) != len(probabilities):
        raise ValueError("Score timestamps and probabilities differ in length")
    alarms = generate_alarms(
        endpoints, probabilities, threshold, refractory_samples, positive_windows, decision_window_windows,
    )
    hits = probabilities >= threshold
    candidates = []
    clean_alarm_contexts = skipped_source_sampled = 0
    for alarm in alarms:
        index = int(np.searchsorted(endpoints, alarm))
        if index >= len(endpoints) or int(endpoints[index]) != int(alarm):
            raise ValueError("Source policy alarm does not map to a decision endpoint")
        context_start = index - decision_window_windows + 1
        if context_start < 0:
            continue
        context = range(context_start, index + 1)
        if not all(int(starts[position]) in clean_starts for position in context):
            continue
        clean_alarm_contexts += 1
        available = [
            position for position in context
            if hits[position] and (recording_id, int(starts[position])) not in source_normal_keys
        ]
        if not available:
            skipped_source_sampled += 1
            continue
        chosen = max(available, key=lambda position: (float(probabilities[position]), -int(starts[position])))
        candidates.append({
            "recording_id": recording_id,
            "patient_group": patient_group,
            "start_sample": int(starts[chosen]),
            "source_score": float(probabilities[chosen]),
            "source_alarm_sample": int(alarm),
            "context_hits": int(hits[context_start:index + 1].sum()),
        })
    return candidates, {
        "source_policy_alarm_count": len(alarms),
        "clean_false_alarm_contexts": clean_alarm_contexts,
        "skipped_contexts_all_hits_already_sampled": skipped_source_sampled,
    }


def _separate_by_recording(candidates: list[dict], min_separation_samples: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["recording_id"]].append(candidate)
    selected = []
    for recording_id in sorted(grouped):
        accepted: list[int] = []
        for candidate in sorted(grouped[recording_id], key=lambda item: (-item["source_score"], item["start_sample"])):
            if all(abs(candidate["start_sample"] - previous) >= min_separation_samples for previous in accepted):
                accepted.append(candidate["start_sample"])
                selected.append(candidate)
    return selected


def _round_robin_patient_groups(candidates: list[dict], target_count: int) -> list[dict]:
    """Prevent a few noisy source patient groups from filling the hard set."""
    by_group: dict[str, deque[dict]] = {}
    for group in sorted({candidate["patient_group"] for candidate in candidates}):
        ordered = sorted(
            (candidate for candidate in candidates if candidate["patient_group"] == group),
            key=lambda item: (-item["source_score"], item["recording_id"], item["start_sample"]),
        )
        by_group[group] = deque(ordered)
    selected = []
    while len(selected) < target_count and any(by_group.values()):
        for group in sorted(by_group):
            if by_group[group] and len(selected) < target_count:
                selected.append(by_group[group].popleft())
    return selected


def _recording_data(row: dict, preprocessing: dict) -> np.ndarray:
    import mne

    raw = mne.io.read_raw_edf(row["edf_path"], preload=True, verbose="ERROR")
    try:
        if int(round(raw.info["sfreq"])) != int(preprocessing["sample_rate_hz"]):
            raise ValueError(f"Unexpected sample rate in {row['recording_id']}")
        data = extract_canonical_bipolar_data(raw)
    finally:
        raw.close()
    return filter_eeg(
        data, preprocessing["sample_rate_hz"], preprocessing["bandpass_low_hz"],
        preprocessing["bandpass_high_hz"], preprocessing["notch_hz"], preprocessing["filter_mode"],
    )


def _validate_existing_output(output_dir: Path, config: dict, manifest_path: str | Path) -> dict | None:
    summary_path = output_dir / "policy_hard_negative_mining_summary.json"
    if not output_dir.exists():
        return None
    if not summary_path.is_file():
        raise RuntimeError(f"Refusing incomplete V2.3 hard-negative cache: {output_dir}")
    with summary_path.open("r", encoding="utf-8") as input_file:
        summary = json.load(input_file)
    expected = {
        "protocol": "research_v2_3_policy_aligned_hard_negative",
        "protocol_hash": canonical_json_hash(config),
        "fold_manifest_sha256": file_sha256(manifest_path),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Existing V2.3 cache contract differs: {output_dir}")
    source = config["hard_negative_mining"]["source_runs"].get(str(summary.get("fold_index")))
    if source is None:
        raise RuntimeError(f"Existing V2.3 cache has an unknown fold: {output_dir}")
    if summary.get("source_checkpoint_sha256") != source["checkpoint_sha256"]:
        raise RuntimeError(f"Existing V2.3 cache used a different source checkpoint: {output_dir}")
    if summary.get("source_calibration_policy") != source["calibration_policy"]:
        raise RuntimeError(f"Existing V2.3 cache used a different source calibration policy: {output_dir}")
    frozen = summary.get("frozen_scaler", {})
    if frozen.get("source_mean_sha256") != source["scaler_mean_sha256"] or frozen.get("source_scale_sha256") != source["scaler_scale_sha256"]:
        raise RuntimeError(f"Existing V2.3 cache used a different source scaler: {output_dir}")
    required = ("chbmit_train.npz", "chbmit_val.npz", "chbmit_temporal_eval.npz", "feature_representation.json", "frozen_train_scaler.npz", "source_train_scores.npz", "source_train_scores.records.json")
    if any(not (output_dir / name).is_file() for name in required):
        raise RuntimeError(f"Existing V2.3 cache is incomplete: {output_dir}")
    if summary.get("source_train_scores_sha256") != file_sha256(output_dir / "source_train_scores.npz"):
        raise RuntimeError(f"Existing V2.3 cache source scores differ: {output_dir}")
    if summary.get("source_train_score_records_sha256") != file_sha256(output_dir / "source_train_scores.records.json"):
        raise RuntimeError(f"Existing V2.3 cache source score metadata differs: {output_dir}")
    if (output_dir / "chbmit_test.npz").exists() or (output_dir / "continuous_test_recordings.csv").exists():
        raise RuntimeError(f"Existing V2.3 cache contains a sealed-test artifact: {output_dir}")
    return summary


def build_policy_hard_negative_cache(
    *,
    project_root: str | Path,
    config: dict,
    fold_index: str,
    fold_manifest: str | Path,
    source_prepared_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Write a derived V2.3 train cache without reading temporal-evaluation EEG."""
    project_root = Path(project_root)
    source_prepared_dir, output_dir = Path(source_prepared_dir), Path(output_dir)
    existing = _validate_existing_output(output_dir, config, fold_manifest)
    if existing is not None:
        return {**existing, "cache_reused": True}
    _verify_v21_cache(source_prepared_dir, fold_manifest)
    mining = config["hard_negative_mining"]
    source_contract = mining["source_runs"][fold_index]
    source_dir = project_root / "AI_train_model" / mining["source_artifact_root"] / source_contract["artifact_dir"]
    checkpoint_path, mean, scale = _hash_checked_source(source_dir, source_contract)
    policy = _verify_source_policy(source_dir, source_contract["calibration_policy"])
    preprocessing = _v23_preprocessing(config)
    train_rows = _load_manifest_rows(fold_manifest, "train")
    source_x, source_y, source_records, source_starts, channels = _load_source_train_windows(source_prepared_dir)
    source_mean = source_x.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    source_scale = np.maximum(source_x.std(axis=(0, 2), dtype=np.float64).astype(np.float32), np.finfo(np.float32).eps)
    if not np.allclose(source_mean, mean, rtol=1e-6, atol=1e-6) or not np.allclose(source_scale, scale, rtol=1e-6, atol=1e-6):
        raise ValueError("Frozen V2.2 scaler does not match the V2.1 source train cache")

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        from src.model import build_model_from_run

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model_from_run(source_dir).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        scores = score_continuous_recordings(
            model, device, train_rows, preprocessing, batch_size=128, use_amp=False,
            scaler_mean=mean, scaler_std=scale, normalization_mode="train_channel_zscore",
        )
        expected_recordings = [row["recording_id"] for row in train_rows]
        observed_recordings = [record["recording_id"] for record in scores["records"]]
        if observed_recordings != expected_recordings:
            raise ValueError("Source train scores do not match the locked V2.3 training manifest")
        score_path = output_dir / "source_train_scores.npz"
        save_scores(score_path, scores)

        source_normal_keys = {
            (recording_id, int(start))
            for recording_id, start, label in zip(source_records, source_starts, source_y)
            if label == 0
        }
        window_samples = int(preprocessing["window_sec"] * preprocessing["sample_rate_hz"])
        refractory_samples = int(config["evaluation"]["refractory_sec"] * preprocessing["sample_rate_hz"])
        raw_candidates, aggregate = [], defaultdict(int)
        for row_index, (row, record) in enumerate(zip(train_rows, scores["records"])):
            offset_start = int(scores["record_offsets"][row_index])
            offset_end = int(scores["record_offsets"][row_index + 1])
            candidates, counts = select_policy_aligned_candidates(
                scores["start_samples"][offset_start:offset_end], scores["probabilities"][offset_start:offset_end],
                _normal_start_set(row, int(record["sample_count"]), preprocessing), source_normal_keys,
                row["recording_id"], row["patient_group"], window_samples, float(policy["threshold"]),
                int(policy["positive_windows"]), int(policy["decision_window_windows"]), refractory_samples,
            )
            raw_candidates.extend(candidates)
            for key, value in counts.items():
                aggregate[key] += int(value)
        separated = _separate_by_recording(
            raw_candidates, int(round(float(mining["minimum_separation_sec"]) * preprocessing["sample_rate_hz"])),
        )
        positive_count = int(source_y.sum())
        requested = int(round(positive_count * float(mining["hard_negative_to_positive_ratio"])))
        selected = _round_robin_patient_groups(separated, requested)
        if not selected:
            raise RuntimeError("V2.3 found zero eligible policy-aligned train-only hard negatives")

        rows_by_id = {row["recording_id"]: row for row in train_rows}
        extracted = []
        for recording_id in sorted({candidate["recording_id"] for candidate in selected}):
            data = _recording_data(rows_by_id[recording_id], preprocessing)
            for candidate in sorted(
                (item for item in selected if item["recording_id"] == recording_id),
                key=lambda item: item["start_sample"],
            ):
                start = candidate["start_sample"]
                extracted.append((data[:, start:start + window_samples].copy(), candidate))
        if len(extracted) != len(selected):
            raise RuntimeError("Hard-negative extraction count does not match selected contexts")

        hard_x = np.stack([item[0] for item in extracted]).astype(np.float32, copy=False)
        hard_records = np.asarray([item[1]["recording_id"] for item in extracted]).astype(str)
        hard_starts = np.asarray([item[1]["start_sample"] for item in extracted], dtype=np.int64)
        x = np.concatenate((source_x, hard_x), axis=0)
        y = np.concatenate((source_y, np.zeros(len(hard_x), dtype=np.int64)))
        records = np.concatenate((source_records, hard_records))
        starts = np.concatenate((source_starts, hard_starts))
        weights = np.concatenate((
            np.ones(len(source_y), dtype=np.float32),
            np.full(len(hard_x), float(mining["sampling_multiplier"]), dtype=np.float32),
        ))
        order = np.random.default_rng(int(config["training"]["dataset_sampling_seed"])).permutation(len(y))
        np.savez_compressed(
            output_dir / "chbmit_train.npz", X=x[order], y=y[order], recording_id=records[order],
            start_sample=starts[order], channels=channels, sampling_weight=weights[order], split="train",
        )
        for name in ("chbmit_val.npz", "chbmit_temporal_eval.npz", "feature_representation.json"):
            shutil.copy2(source_prepared_dir / name, output_dir / name)
        np.savez_compressed(output_dir / "frozen_train_scaler.npz", mean=mean, scale=scale)
        selected_groups = defaultdict(int)
        for candidate in selected:
            selected_groups[candidate["patient_group"]] += 1
        summary = {
            "protocol": "research_v2_3_policy_aligned_hard_negative",
            "protocol_hash": canonical_json_hash(config),
            "fold_index": fold_index,
            "fold_manifest": str(fold_manifest),
            "fold_manifest_sha256": file_sha256(fold_manifest),
            "source_prepared_dir": str(source_prepared_dir),
            "source_preparation_sha256": file_sha256(source_prepared_dir / "preparation_summary.json"),
            "source_artifact_dir": str(source_dir),
            "source_checkpoint_sha256": file_sha256(checkpoint_path),
            "source_train_scores_sha256": file_sha256(score_path),
            "source_train_score_records_sha256": file_sha256(score_path.with_suffix(".records.json")),
            "source_calibration_policy": policy,
            "frozen_scaler": {"mean_sha256": file_sha256(output_dir / "frozen_train_scaler.npz"), "source_mean_sha256": source_contract["scaler_mean_sha256"], "source_scale_sha256": source_contract["scaler_scale_sha256"]},
            "positive_windows": positive_count,
            "source_normal_windows": int((source_y == 0).sum()),
            "hard_negative_windows": int(len(hard_x)),
            "requested_hard_negative_windows": requested,
            "candidate_limited": len(hard_x) < requested,
            "hard_negative_to_positive_ratio": float(len(hard_x) / positive_count),
            "sampling_multiplier": float(mining["sampling_multiplier"]),
            "hard_negative_normal_draw_fraction": float((len(hard_x) * mining["sampling_multiplier"]) / ((source_y == 0).sum() + len(hard_x) * mining["sampling_multiplier"])),
            "raw_policy_aligned_candidates": int(len(raw_candidates)),
            "separated_policy_aligned_candidates": int(len(separated)),
            "source_alarm_diagnostics": dict(aggregate),
            "selected_patient_group_counts": dict(sorted(selected_groups.items())),
            "clean_context_rule": mining["clean_context_rule"],
            "candidate_rule": mining["candidate_rule"],
            "selection_rule": mining["selection_rule"],
            "cache_reused": False,
        }
        save_json(output_dir / "preparation_summary.json", {
            "protocol": summary["protocol"], "fold_manifest": summary["fold_manifest"],
            "fold_manifest_sha256": summary["fold_manifest_sha256"], "config_hash": summary["protocol_hash"],
            "window_samples": window_samples, "included_splits": ["train", "val", "temporal_eval"],
            "source_v21_cache": str(source_prepared_dir), "hard_negative_summary": str(output_dir / "policy_hard_negative_mining_summary.json"),
        })
        save_json(output_dir / "policy_hard_negative_mining_summary.json", summary)
        return summary
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
