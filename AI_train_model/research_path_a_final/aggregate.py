"""Patient-group-aware aggregation for Path A evaluation artifacts.

The legacy A1.2 result is deliberately kept as an exploratory result.  This
module makes that status machine-readable and prevents the two CHB-MIT
sessions chb01/chb21 from being counted as independent patients.
"""

import json
import math
import random
import statistics
from pathlib import Path

from src.chbmit_patient_split import patient_group_for_case


EXPLORATORY_KIND = "exploratory_test_probe"


def _case_id_from_source_run(source_run_id):
    prefix = "ps_a12_"
    suffix = "_s42"
    if not source_run_id.startswith(prefix) or not source_run_id.endswith(suffix):
        raise ValueError(f"Unsupported Path A1.2 source run id: {source_run_id}")
    return source_run_id[len(prefix):-len(suffix)]


def _rate_interval(contributions, numerator_key, denominator_key, replicates, seed):
    """Cluster bootstrap for a pooled rate, sampling patient groups once."""
    if not contributions:
        return {"estimable": False, "reason": "no_patient_groups"}
    numerators = [float(row[numerator_key]) for row in contributions]
    denominators = [float(row[denominator_key]) for row in contributions]
    if any(value < 0.0 for value in numerators) or any(value <= 0.0 for value in denominators):
        raise ValueError("Bootstrap contributions must have non-negative numerators and positive denominators")
    estimate = float(sum(numerators) / sum(denominators))
    rng = random.Random(seed)
    sampled = []
    for _ in range(replicates):
        indices = [rng.randrange(len(contributions)) for _ in contributions]
        sampled.append(sum(numerators[index] for index in indices) / sum(denominators[index] for index in indices))
    sampled.sort()
    def quantile(probability):
        position = (len(sampled) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return sampled[lower]
        return sampled[lower] + (sampled[upper] - sampled[lower]) * (position - lower)
    return {
        "estimable": True,
        "estimate": estimate,
        "lower_95": float(quantile(0.025)),
        "upper_95": float(quantile(0.975)),
        "replicates": int(replicates),
        "resampling_unit": "patient_group",
    }


def _window_contribution(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("evaluation_kind") != EXPLORATORY_KIND:
        raise ValueError(f"{path} is not an explicitly labelled exploratory Path A test probe")
    source_run_id = payload["source_run_id"]
    case_id = _case_id_from_source_run(source_run_id)
    diagnostic = payload["balanced_test_diagnostic"]
    counts = diagnostic["positive_windows"], diagnostic["negative_windows"]
    positive, negative = (int(value) for value in counts)
    metrics = diagnostic["metrics"]
    sensitivity = float(metrics["sensitivity"])
    balanced_accuracy = float(metrics["balanced_accuracy"])
    specificity = 2.0 * balanced_accuracy - sensitivity
    if positive <= 0 or negative <= 0 or not 0.0 <= specificity <= 1.0:
        raise ValueError(f"Invalid balanced metrics in {path}")
    full_counts = payload["test_window_counts"]
    full_positive, full_negative = int(full_counts["positive"]), int(full_counts["negative"])
    full_metrics = payload["test_prevalence_metrics"]
    full_sensitivity = float(full_metrics["sensitivity"])
    full_balanced = float(full_metrics["balanced_accuracy"])
    full_specificity = 2.0 * full_balanced - full_sensitivity
    if full_positive <= 0 or full_negative <= 0 or not 0.0 <= full_specificity <= 1.0:
        raise ValueError(f"Invalid full-prevalence metrics in {path}")
    return {
        "case_id": case_id,
        "patient_group": patient_group_for_case(case_id),
        "source_run_id": source_run_id,
        "path": str(path),
        "true_positive_windows": int(round(sensitivity * positive)),
        "positive_windows": positive,
        "true_negative_windows": int(round(specificity * negative)),
        "negative_windows": negative,
        "balanced_accuracy": balanced_accuracy,
        "full_true_positive_windows": int(round(full_sensitivity * full_positive)),
        "full_positive_windows": full_positive,
        "full_true_negative_windows": int(round(full_specificity * full_negative)),
        "full_negative_windows": full_negative,
        "full_accuracy": float(full_metrics["accuracy"]),
        "full_balanced_accuracy": full_balanced,
    }


def _merge_window_contributions(rows):
    grouped = {}
    for row in rows:
        target = grouped.setdefault(row["patient_group"], {
            "patient_group": row["patient_group"], "case_ids": [],
            "true_positive_windows": 0, "positive_windows": 0,
            "true_negative_windows": 0, "negative_windows": 0,
            "full_true_positive_windows": 0, "full_positive_windows": 0,
            "full_true_negative_windows": 0, "full_negative_windows": 0,
        })
        target["case_ids"].append(row["case_id"])
        for key in (
            "true_positive_windows", "positive_windows", "true_negative_windows", "negative_windows",
            "full_true_positive_windows", "full_positive_windows",
            "full_true_negative_windows", "full_negative_windows",
        ):
            target[key] += row[key]
    merged = []
    for row in grouped.values():
        sensitivity = row["true_positive_windows"] / row["positive_windows"]
        specificity = row["true_negative_windows"] / row["negative_windows"]
        row["balanced_accuracy"] = 0.5 * (sensitivity + specificity)
        full_sensitivity = row["full_true_positive_windows"] / row["full_positive_windows"]
        full_specificity = row["full_true_negative_windows"] / row["full_negative_windows"]
        row["full_balanced_accuracy"] = 0.5 * (full_sensitivity + full_specificity)
        row["full_accuracy"] = (
            row["full_true_positive_windows"] + row["full_true_negative_windows"]
        ) / (row["full_positive_windows"] + row["full_negative_windows"])
        row["case_ids"] = sorted(row["case_ids"])
        merged.append(row)
    return sorted(merged, key=lambda item: item["patient_group"])


def aggregate_window_artifacts(paths, replicates=10000, seed=20260808):
    rows = [_window_contribution(path) for path in paths]
    if not rows:
        raise ValueError("No Path A checkpoint_test_evaluation.json artifacts matched")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Only one checkpoint test evaluation is allowed per Path A case")
    groups = _merge_window_contributions(rows)
    pooled_sensitivity = _rate_interval(groups, "true_positive_windows", "positive_windows", replicates, seed)
    pooled_specificity = _rate_interval(groups, "true_negative_windows", "negative_windows", replicates, seed + 1)
    pooled_balanced = 0.5 * (pooled_sensitivity["estimate"] + pooled_specificity["estimate"])
    full_sensitivity = _rate_interval(groups, "full_true_positive_windows", "full_positive_windows", replicates, seed + 2)
    full_specificity = _rate_interval(groups, "full_true_negative_windows", "full_negative_windows", replicates, seed + 3)
    full_balanced = 0.5 * (full_sensitivity["estimate"] + full_specificity["estimate"])
    full_true_positive = sum(row["full_true_positive_windows"] for row in groups)
    full_true_negative = sum(row["full_true_negative_windows"] for row in groups)
    full_window_count = sum(row["full_positive_windows"] + row["full_negative_windows"] for row in groups)
    case_mean = float(statistics.fmean(row["balanced_accuracy"] for row in rows))
    group_mean = float(statistics.fmean(row["balanced_accuracy"] for row in groups))
    return {
        "evaluation_scope": "exploratory_reproducibility_only",
        "final_claim_eligible": False,
        "final_claim_blockers": [
            "A1.2 was promoted after comparing exploratory test probes from the same cohort.",
            "Only training seed 42 is present; across-seed variation is not available.",
            "Continuous event-level replay is a required companion measurement.",
        ],
        "case_count": len(rows),
        "patient_group_count": len(groups),
        "chb01_chb21_rule": "merged into subject_01_21 before uncertainty estimation",
        "legacy_unweighted_case_mean_balanced_accuracy": case_mean,
        "unweighted_patient_group_mean_balanced_accuracy": group_mean,
        "pooled_patient_group_window_metrics": {
            "balanced_diagnostic": {
                "balanced_accuracy": pooled_balanced,
                "sensitivity": pooled_sensitivity,
                "specificity": pooled_specificity,
            },
            "full_prevalence": {
                "accuracy": (full_true_positive + full_true_negative) / full_window_count,
                "balanced_accuracy": full_balanced,
                "sensitivity": full_sensitivity,
                "specificity": full_specificity,
                "positive_windows": sum(row["full_positive_windows"] for row in groups),
                "negative_windows": sum(row["full_negative_windows"] for row in groups),
            },
        },
        "per_case": rows,
        "per_patient_group": groups,
    }


def _event_contribution(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("evaluation_kind") != "exploratory_test_replay":
        raise ValueError(f"{path} is not an explicitly labelled exploratory Path A event replay")
    case_id = str(payload["case_id"])
    result = payload["test_event_metrics"]
    return {
        "case_id": case_id,
        "patient_group": patient_group_for_case(case_id),
        "detected_events": int(result["detected_events"]),
        "total_events": int(result["total_events"]),
        "false_alarms": int(result["false_alarms"]),
        "interictal_hours": float(result["interictal_hours"]),
        "detection_delays_sec": [float(value) for value in result.get("detection_delays_sec", [])],
        "policy": {
            key: result[key] for key in ("threshold", "positive_windows", "decision_window_windows", "policy_name")
        },
    }


def aggregate_event_artifacts(paths, replicates=10000, seed=20260808):
    rows = [_event_contribution(path) for path in paths]
    if not rows:
        return None
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Only one event replay is allowed per Path A case")
    grouped = {}
    for row in rows:
        target = grouped.setdefault(row["patient_group"], {
            "patient_group": row["patient_group"], "case_ids": [], "detected_events": 0,
            "total_events": 0, "false_alarms": 0, "interictal_hours": 0.0, "detection_delays_sec": [],
        })
        target["case_ids"].append(row["case_id"])
        for key in ("detected_events", "total_events", "false_alarms", "interictal_hours"):
            target[key] += row[key]
        target["detection_delays_sec"].extend(row["detection_delays_sec"])
    groups = sorted(grouped.values(), key=lambda item: item["patient_group"])
    for row in groups:
        row["case_ids"] = sorted(row["case_ids"])
    sensitivity = _rate_interval(groups, "detected_events", "total_events", replicates, seed)
    far_per_hour = _rate_interval(groups, "false_alarms", "interictal_hours", replicates, seed + 1)
    delays = [delay for row in groups for delay in row["detection_delays_sec"]]
    policies = {json.dumps(row["policy"], sort_keys=True) for row in rows}
    return {
        "evaluation_scope": "exploratory_reproducibility_only",
        "patient_group_count": len(groups),
        "policy_consistent_across_cases": len(policies) == 1,
        "per_case_policy_count": len(policies),
        "pooled_event_sensitivity": sensitivity,
        "pooled_false_alarms_per_hour": far_per_hour,
        "detected_events": sum(row["detected_events"] for row in groups),
        "total_events": sum(row["total_events"] for row in groups),
        "false_alarms": sum(row["false_alarms"] for row in groups),
        "interictal_hours": sum(row["interictal_hours"] for row in groups),
        "median_detection_delay_sec": float(statistics.median(delays)) if delays else None,
        "mean_detection_delay_sec": float(statistics.fmean(delays)) if delays else None,
        "per_patient_group": groups,
    }


def write_audit(window_paths, event_paths, output_path, replicates=10000, seed=20260808):
    result = {
        "protocol_id": "path_a_final_evaluation_v1",
        "window": aggregate_window_artifacts(window_paths, replicates=replicates, seed=seed),
        "event": aggregate_event_artifacts(event_paths, replicates=replicates, seed=seed),
        "final_result_requirements": [
            "A model, all hyperparameters, window threshold, event policy, and post-processing must be frozen before final testing.",
            "Use an external cohort or a newly declared, previously unobserved outer test protocol.",
            "Report seed variation and temporal/patient-group variation separately.",
            "Report full-prevalence window metrics and continuous EEG event metrics with patient-group confidence intervals.",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
