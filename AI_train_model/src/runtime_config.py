"""Explicit, reproducible runtime overrides for controlled experiments."""

import math
import os


def apply_runtime_overrides(config):
    """Apply the optional window-duration ablation to a loaded YAML config."""
    value = os.environ.get("CHBMIT_WINDOW_SEC")
    if value is None:
        return config
    window_sec = float(value)
    if not math.isfinite(window_sec) or window_sec <= 0:
        raise ValueError("CHBMIT_WINDOW_SEC must be a positive finite number")
    preprocessing = config["preprocessing"]
    sample_rate = float(preprocessing["sample_rate_hz"])
    input_length = window_sec * sample_rate
    if not math.isclose(input_length, round(input_length), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("CHBMIT_WINDOW_SEC must produce an integer number of samples")
    if window_sec < float(preprocessing["stride_sec"]):
        raise ValueError("CHBMIT_WINDOW_SEC must be at least CHBMIT stride_sec")
    preprocessing["window_sec"] = window_sec
    config["model"]["input_length"] = int(round(input_length))
    return config
