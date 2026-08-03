"""Paired statistical evidence used by the V2 promotion gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapInterval:
    mean_difference: float
    lower: float
    upper: float
    replicates: int


@dataclass(frozen=True)
class RateInterval:
    estimate: float
    lower: float
    upper: float
    numerator: int
    denominator: float


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


def patient_group_cluster_bootstrap(
    contributions: dict[str, tuple[int, int]], replicates: int = 10_000, seed: int = 20260802,
) -> BootstrapInterval:
    """Bootstrap a rate by patient group, never by sessions or windows."""
    import numpy as np

    if len(contributions) < 2:
        raise ValueError("Patient-group bootstrap requires at least two independent groups")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    values = np.asarray(list(contributions.values()), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or np.any(values < 0) or np.any(values[:, 1] <= 0):
        raise ValueError("Contributions must be non-negative (numerator, denominator) pairs")
    estimate = values[:, 0].sum() / values[:, 1].sum() if values[:, 1].sum() else 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    sampled = values[indices].sum(axis=1)
    rates = np.divide(sampled[:, 0], sampled[:, 1], out=np.zeros(replicates), where=sampled[:, 1] > 0)
    lower, upper = np.quantile(rates, [0.025, 0.975])
    return BootstrapInterval(float(estimate), float(lower), float(upper), replicates)


def poisson_exact_far_interval(false_alarms: int, nonictal_hours: float, confidence: float = 0.95) -> RateInterval:
    """Garwood exact Poisson interval for FAR/h from all available replay hours."""
    from scipy.stats import chi2

    if false_alarms < 0 or nonictal_hours <= 0.0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid Poisson FAR interval inputs")
    alpha = 1.0 - confidence
    lower_count = 0.0 if false_alarms == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * false_alarms)
    upper_count = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (false_alarms + 1))
    return RateInterval(
        estimate=float(false_alarms / nonictal_hours), lower=float(lower_count / nonictal_hours),
        upper=float(upper_count / nonictal_hours), numerator=int(false_alarms), denominator=float(nonictal_hours),
    )
