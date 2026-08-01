"""Small training-only perturbations for normalized multichannel EEG windows."""

import torch


def mild_eeg_augmentation(inputs, gain_delta=0.1, noise_std=0.02):
    """Apply shared amplitude scaling plus low-amplitude Gaussian jitter.

    One gain is sampled per window and applied to every channel and time point,
    preserving the montage's relative spatial amplitude pattern. Gaussian
    jitter is measured in the prepared train-channel-z-score domain. This
    function must only be called on training batches.
    """
    if inputs.ndim != 3:
        raise ValueError("EEG augmentation inputs must have shape [batch, channels, time]")
    if gain_delta < 0.0 or gain_delta >= 1.0:
        raise ValueError("EEG augmentation gain_delta must be in [0, 1)")
    if noise_std < 0.0:
        raise ValueError("EEG augmentation noise_std must be non-negative")

    gain = inputs.new_empty((inputs.shape[0], 1, 1)).uniform_(
        1.0 - gain_delta, 1.0 + gain_delta
    )
    augmented = inputs * gain
    if noise_std > 0.0:
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented
