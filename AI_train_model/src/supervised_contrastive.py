"""Training-only supervised contrastive objective for EEG window embeddings."""

import torch
import torch.nn.functional as functional


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Return a stable same-class contrastive loss for one labeled batch.

    The classifier embedding is L2-normalized. Every other sample of the same
    class is a positive; all samples from the other class are negatives. An
    anchor without another same-class sample is omitted rather than producing a
    NaN. The caller owns the coefficient used to combine this loss with CE.
    """
    if features.ndim != 2:
        raise ValueError("Supervised contrastive features must have shape [batch, features]")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("Labels must have shape [batch] and match features")
    if features.shape[0] < 2:
        raise ValueError("Supervised contrastive loss requires at least two samples")
    if temperature <= 0.0:
        raise ValueError("Supervised contrastive temperature must be positive")

    # Keep similarity/log-sum-exp in FP32 even when the model is trained with AMP.
    with torch.amp.autocast(device_type=features.device.type, enabled=False):
        normalized = functional.normalize(features.float(), p=2, dim=1, eps=1e-12)
        similarities = normalized @ normalized.T / float(temperature)
        diagonal = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        positives = labels[:, None].eq(labels[None, :]) & ~diagonal
        valid_anchors = positives.any(dim=1)
        if not torch.any(valid_anchors):
            return features.sum() * 0.0

        logits = similarities.masked_fill(diagonal, float("-inf"))
        log_denominator = torch.logsumexp(logits, dim=1, keepdim=True)
        log_probabilities = similarities - log_denominator
        positive_log_probability = (
            (log_probabilities * positives).sum(dim=1)
            / positives.sum(dim=1).clamp_min(1)
        )
        return -positive_log_probability[valid_anchors].mean()
