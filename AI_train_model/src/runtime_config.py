"""Explicit, reproducible runtime overrides for controlled experiments."""

import math
import os


def apply_runtime_overrides(config):
    """Apply explicit controlled-experiment overrides to a loaded YAML config."""
    filter_mode = os.environ.get("CHBMIT_FILTER_MODE")
    if filter_mode is not None:
        if filter_mode not in {"zero_phase", "causal_iir"}:
            raise ValueError("CHBMIT_FILTER_MODE must be zero_phase or causal_iir")
        config["preprocessing"]["filter_mode"] = filter_mode
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
