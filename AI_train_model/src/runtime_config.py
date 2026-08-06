"""Explicit, reproducible runtime overrides for controlled experiments."""

import math
import os


def _apply_split_ratio_overrides(config):
    """Optional CHBMIT_SPLIT_RATIOS=train,val,test (e.g. 0.6,0.2,0.2)."""
    raw = os.environ.get("CHBMIT_SPLIT_RATIOS")
    if raw is None:
        return
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("CHBMIT_SPLIT_RATIOS must be train,val,test (three comma-separated values)")
    ratios = [float(part) for part in parts]
    if any(not math.isfinite(ratio) or ratio <= 0.0 for ratio in ratios):
        raise ValueError("CHBMIT_SPLIT_RATIOS values must be positive finite numbers")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("CHBMIT_SPLIT_RATIOS must sum to 1.0")
    config["data"]["split_ratios"] = {
        "train": ratios[0],
        "val": ratios[1],
        "test": ratios[2],
    }


def apply_runtime_overrides(config):
    """Apply explicit controlled-experiment overrides to a loaded YAML config."""
    _apply_split_ratio_overrides(config)
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
