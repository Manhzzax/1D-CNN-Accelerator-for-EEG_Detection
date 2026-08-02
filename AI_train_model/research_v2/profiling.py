"""Static parameter, MAC, and peak-activation profiling for CNN Pareto analysis."""

from __future__ import annotations


def profile_model(model, input_shape: tuple[int, int, int]) -> dict:
    """Profile Conv1d/Linear MACs and output activations from a dry forward pass."""
    import torch
    import torch.nn as nn

    if len(input_shape) != 3 or any(int(value) < 1 for value in input_shape):
        raise ValueError("input_shape must be (batch, channels, samples) with positive values")
    macs = 0
    peak_activation_bytes = 0
    layer_rows = []

    def hook(module, inputs, output):
        nonlocal macs, peak_activation_bytes
        if not isinstance(output, torch.Tensor):
            return
        activation_bytes = output.numel() * output.element_size()
        peak_activation_bytes = max(peak_activation_bytes, activation_bytes)
        layer_macs = 0
        if isinstance(module, nn.Conv1d):
            batch, channels, length = output.shape
            kernel = module.kernel_size[0]
            layer_macs = batch * channels * length * (module.in_channels // module.groups) * kernel
        elif isinstance(module, nn.Linear):
            layer_macs = output.numel() * module.in_features
        else:
            return
        macs += int(layer_macs)
        layer_rows.append({
            "layer": module.__class__.__name__,
            "output_shape": list(output.shape),
            "macs": int(layer_macs),
            "activation_bytes_fp32": int(activation_bytes),
        })

    hooks = [module.register_forward_hook(hook) for module in model.modules() if isinstance(module, (nn.Conv1d, nn.Linear))]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(input_shape, dtype=torch.float32))
    finally:
        for hook_handle in hooks:
            hook_handle.remove()
        model.train(was_training)

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "input_shape": list(input_shape),
        "parameter_count": int(parameter_count),
        "parameter_bytes_fp32": int(parameter_count * 4),
        "parameter_bytes_int16": int(parameter_count * 2),
        "macs_per_window": int(macs),
        "peak_activation_bytes_fp32": int(peak_activation_bytes),
        "peak_activation_bytes_int16": int(peak_activation_bytes // 2),
        "layers": layer_rows,
    }
