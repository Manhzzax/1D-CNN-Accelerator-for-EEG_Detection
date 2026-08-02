"""Paired statistical evidence used by the V2 promotion gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapInterval:
    mean_difference: float
    lower: float
    upper: float
    replicates: int


def paired_bootstrap_interval(reference, candidate, replicates: int = 10_000, seed: int = 20260802) -> BootstrapInterval:
    """Return a percentile CI for paired metric differences.

    Inputs must be matched observations, normally a common `(outer_fold,
    training_seed)` pair.  The bootstrap does not convert repeated seeds into
    independent patient cohorts; fold and seed summaries remain separate in
    the final report.
    """
    import numpy as np

    baseline = np.asarray(reference, dtype=np.float64)
    proposed = np.asarray(candidate, dtype=np.float64)
    if baseline.ndim != 1 or proposed.ndim != 1 or len(baseline) != len(proposed) or len(baseline) < 2:
        raise ValueError("Paired bootstrap needs at least two equally sized one-dimensional samples")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    differences = proposed - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    sampled = differences[indices].mean(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    return BootstrapInterval(float(differences.mean()), float(lower), float(upper), replicates)


def promotion_decision(
    balanced_accuracy: BootstrapInterval,
    auroc: BootstrapInterval,
    event_sensitivity_difference: float,
    pareto_nondominated: bool,
    parameter_count: int,
) -> dict:
    """Implement the same promotion gate for proposed models and baselines."""
    if parameter_count < 1:
        raise ValueError("parameter_count must be positive")
    needs_gate = parameter_count > 25_000
    statistical_gain = balanced_accuracy.lower > 0.0 and auroc.lower > 0.0
    event_non_regression = event_sensitivity_difference >= 0.0
    allowed = (not needs_gate) or (statistical_gain and event_non_regression and pareto_nondominated)
    return {
        "parameter_count": int(parameter_count),
        "requires_large_model_gate": needs_gate,
        "balanced_accuracy_ci": balanced_accuracy.__dict__,
        "auroc_ci": auroc.__dict__,
        "event_sensitivity_difference": float(event_sensitivity_difference),
        "pareto_nondominated": bool(pareto_nondominated),
        "promoted": bool(allowed),
        "reason": (
            "within_25k_budget"
            if not needs_gate
            else "large_model_gate_passed"
            if allowed
            else "large_model_gate_failed"
        ),
    }
