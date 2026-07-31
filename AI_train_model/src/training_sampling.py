"""Training-only sampling strategies for source patient-group robustness."""

import numpy as np


def patient_group_balanced_weights(labels, patient_group_labels, importance):
    """Equalize observed `(class, patient group)` strata without duplicating data.

    Within each stratum, ``importance`` preserves any existing hard-negative
    weighting. The sum of weights is one for every non-empty stratum, so the
    weighted sampler draws every ictal/non-ictal patient group equally often.
    """
    labels = np.asarray(labels, dtype=np.int64)
    patient_group_labels = np.asarray(patient_group_labels, dtype=np.int64)
    importance = np.asarray(importance, dtype=np.float64)
    if labels.ndim != 1 or patient_group_labels.ndim != 1 or importance.ndim != 1:
        raise ValueError("Labels, patient groups, and importance must be one-dimensional")
    if not (len(labels) == len(patient_group_labels) == len(importance)):
        raise ValueError("Labels, patient groups, and importance must have equal length")
    if len(labels) == 0 or np.any(labels < 0) or np.any(patient_group_labels < 0):
        raise ValueError("Labels and patient groups must be non-empty non-negative integers")
    if not np.all(np.isfinite(importance)) or np.any(importance <= 0):
        raise ValueError("Importance weights must be finite and positive")

    group_count = int(patient_group_labels.max()) + 1
    stratum_labels = labels * group_count + patient_group_labels
    weights = np.empty(len(labels), dtype=np.float64)
    strata = []
    for stratum in np.unique(stratum_labels):
        mask = stratum_labels == stratum
        total_importance = float(importance[mask].sum())
        weights[mask] = importance[mask] / total_importance
        strata.append({
            "class_index": int(stratum // group_count),
            "patient_group_index": int(stratum % group_count),
            "samples": int(mask.sum()),
            "importance_sum": total_importance,
        })
    return weights, strata
