"""Normalization operators shared by sampled-window and continuous inference."""

import numpy as np


def window_channel_zscore(windows):
    """Standardize each channel using only samples in its current input window.

    A window is classified only after its final sample arrives, so this
    normalization is causal for the fixed-window detector. It does not fit any
    statistic on validation/test recordings or use labels.
    """
    data = np.asarray(windows, dtype=np.float32)
    single_window = data.ndim == 2
    if single_window:
        data = data[None, ...]
    if data.ndim != 3:
        raise ValueError(f"Expected (N, C, T) windows, got {data.shape}")
    mean = data.mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
    std = data.std(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.finfo(np.float32).eps)
    normalized = (data - mean) / std
    return normalized[0] if single_window else normalized
