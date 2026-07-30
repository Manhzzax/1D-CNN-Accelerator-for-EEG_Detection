"""Export a folded, fixed-point SeparableEEG1DCNN package for FPGA integration."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .model import SeparableEEG1DCNN, build_model_from_run
from .utils import fold_batchnorm, project_dir


INT16_LIMIT = 32767
INT32_LIMIT = 2147483647


def _environment_int(name, default, minimum=0):
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _resolve_directory(name, default_relative):
    value = os.environ.get(name)
    path = Path(value) if value else Path(project_dir) / default_relative
    return path if path.is_absolute() else Path(project_dir) / path


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symmetric_scale(tensor):
    maximum = float(torch.max(torch.abs(tensor)).item())
    return maximum / INT16_LIMIT if maximum > 0.0 else 1.0


def _quantize_int16(tensor, scale):
    values = torch.round(tensor.to(torch.float64) / scale)
    return torch.clamp(values, -INT16_LIMIT, INT16_LIMIT).to(torch.int64)


def _quantize_bias_int32(tensor, accumulator_scale, name):
    values = torch.round(tensor.to(torch.float64) / accumulator_scale).to(torch.int64)
    maximum = int(torch.max(torch.abs(values)).item())
    if maximum > INT32_LIMIT:
        raise OverflowError(f"{name} needs {maximum}, outside signed int32 range")
    return values


def _requantize(accumulator, accumulator_scale, output_scale, relu=False):
    values = torch.round(accumulator.to(torch.float64) * (accumulator_scale / output_scale))
    values = torch.clamp(values, -INT16_LIMIT, INT16_LIMIT).to(torch.int64)
    return torch.clamp(values, min=0) if relu else values


def _integer_conv1d(inputs, weights, bias, padding, groups):
    output = functional.conv1d(
        inputs.to(torch.float64), weights.to(torch.float64), None,
        stride=1, padding=padding, dilation=1, groups=groups,
    )
    output = torch.round(output).to(torch.int64)
    return output + bias.reshape(1, -1, 1)


def _integer_average_pool4(inputs):
    if inputs.shape[-1] % 4:
        raise ValueError(f"AveragePool1d(4) requires divisible length, got {inputs.shape[-1]}")
    batches, channels, length = inputs.shape
    reshaped = inputs.reshape(batches, channels, length // 4, 4)
    # Inputs are non-negative at each pooling site; this is round-to-nearest.
    return (reshaped.sum(dim=-1) + 2) // 4


def _integer_global_average(inputs):
    length = inputs.shape[-1]
    if length < 1:
        raise ValueError("Global average pooling received an empty sequence")
    return (inputs.sum(dim=-1) + length // 2) // length


def _fuse_separable_model(model):
    temporal_weight, temporal_bias = fold_batchnorm(model.temporal_depthwise, model.temporal_bn)
    spatial_weight, spatial_bias = fold_batchnorm(model.spatial_pointwise, model.spatial_bn)
    refine_pointwise_weight, refine_pointwise_bias = fold_batchnorm(
        model.refine_pointwise, model.refine_bn
    )
    return {
        "temporal_depthwise_weight": temporal_weight.detach().cpu().contiguous(),
        "temporal_depthwise_bias": temporal_bias.detach().cpu().contiguous(),
        "spatial_pointwise_weight": spatial_weight.detach().cpu().contiguous(),
        "spatial_pointwise_bias": spatial_bias.detach().cpu().contiguous(),
        "refine_depthwise_weight": model.refine_depthwise.weight.detach().cpu().contiguous(),
        "refine_pointwise_weight": refine_pointwise_weight.detach().cpu().contiguous(),
        "refine_pointwise_bias": refine_pointwise_bias.detach().cpu().contiguous(),
        "classifier_weight": model.classifier.weight.detach().cpu().contiguous(),
        "classifier_bias": model.classifier.bias.detach().cpu().contiguous(),
    }


def _float_forward(inputs, weights):
    temporal = functional.relu(functional.conv1d(
        inputs, weights["temporal_depthwise_weight"], weights["temporal_depthwise_bias"],
        padding=weights["temporal_depthwise_weight"].shape[-1] // 2,
        groups=inputs.shape[1],
    ))
    spatial = functional.relu(functional.conv1d(
        temporal, weights["spatial_pointwise_weight"], weights["spatial_pointwise_bias"]
    ))
    pooled_spatial = functional.avg_pool1d(spatial, kernel_size=4, stride=4)
    refine_depthwise = functional.conv1d(
        pooled_spatial, weights["refine_depthwise_weight"], None,
        padding=weights["refine_depthwise_weight"].shape[-1] // 2,
        groups=pooled_spatial.shape[1],
    )
    refine = functional.relu(functional.conv1d(
        refine_depthwise, weights["refine_pointwise_weight"], weights["refine_pointwise_bias"]
    ))
    pooled_refine = functional.avg_pool1d(refine, kernel_size=4, stride=4)
    global_features = functional.adaptive_avg_pool1d(pooled_refine, 1).squeeze(-1)
    logits = functional.linear(global_features, weights["classifier_weight"], weights["classifier_bias"])
    return {
        "temporal_relu": temporal,
        "spatial_relu": spatial,
        "refine_depthwise": refine_depthwise,
        "refine_relu": refine,
        "global_features": global_features,
        "logits": logits,
    }


def _calibrate_activation_scales(train_x, mean, std, weights, window_count, batch_size):
    indices = np.linspace(0, len(train_x) - 1, min(window_count, len(train_x)), dtype=np.int64)
    maxima = {
        "input": 0.0,
        "temporal_relu": 0.0,
        "spatial_relu": 0.0,
        "refine_depthwise": 0.0,
        "refine_relu": 0.0,
    }
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            raw = train_x[indices[start:start + batch_size]]
            inputs = torch.from_numpy((raw - mean[None, :, None]) / std[None, :, None])
            outputs = _float_forward(inputs, weights)
            maxima["input"] = max(maxima["input"], float(torch.max(torch.abs(inputs)).item()))
            for name in ("temporal_relu", "spatial_relu", "refine_depthwise", "refine_relu"):
                maxima[name] = max(maxima[name], float(torch.max(torch.abs(outputs[name])).item()))
    return {
        name: maximum / INT16_LIMIT if maximum > 0.0 else 1.0
        for name, maximum in maxima.items()
    }, maxima, len(indices)


def _build_quantized_tensors(weights, activation_scales):
    quantized = {}
    tensor_scales = {}
    for name in (
        "temporal_depthwise_weight", "spatial_pointwise_weight", "refine_depthwise_weight",
        "refine_pointwise_weight", "classifier_weight",
    ):
        tensor_scales[name] = _symmetric_scale(weights[name])
        quantized[name] = _quantize_int16(weights[name], tensor_scales[name])

    quantized["temporal_depthwise_bias"] = _quantize_bias_int32(
        weights["temporal_depthwise_bias"],
        activation_scales["input"] * tensor_scales["temporal_depthwise_weight"],
        "temporal_depthwise_bias",
    )
    quantized["spatial_pointwise_bias"] = _quantize_bias_int32(
        weights["spatial_pointwise_bias"],
        activation_scales["temporal_relu"] * tensor_scales["spatial_pointwise_weight"],
        "spatial_pointwise_bias",
    )
    quantized["refine_pointwise_bias"] = _quantize_bias_int32(
        weights["refine_pointwise_bias"],
        activation_scales["refine_depthwise"] * tensor_scales["refine_pointwise_weight"],
        "refine_pointwise_bias",
    )
    quantized["classifier_bias"] = _quantize_bias_int32(
        weights["classifier_bias"],
        activation_scales["refine_relu"] * tensor_scales["classifier_weight"],
        "classifier_bias",
    )
    return quantized, tensor_scales


def _integer_forward(inputs, quantized, activation_scales, tensor_scales):
    input_q = _quantize_int16(inputs, activation_scales["input"])
    temporal_acc = _integer_conv1d(
        input_q, quantized["temporal_depthwise_weight"], quantized["temporal_depthwise_bias"],
        padding=quantized["temporal_depthwise_weight"].shape[-1] // 2, groups=inputs.shape[1],
    )
    temporal_q = _requantize(
        temporal_acc,
        activation_scales["input"] * tensor_scales["temporal_depthwise_weight"],
        activation_scales["temporal_relu"], relu=True,
    )
    spatial_acc = _integer_conv1d(
        temporal_q, quantized["spatial_pointwise_weight"], quantized["spatial_pointwise_bias"],
        padding=0, groups=1,
    )
    spatial_q = _requantize(
        spatial_acc,
        activation_scales["temporal_relu"] * tensor_scales["spatial_pointwise_weight"],
        activation_scales["spatial_relu"], relu=True,
    )
    pooled_spatial_q = _integer_average_pool4(spatial_q)
    refine_depthwise_acc = _integer_conv1d(
        pooled_spatial_q, quantized["refine_depthwise_weight"], torch.zeros(
            quantized["refine_depthwise_weight"].shape[0], dtype=torch.int64
        ),
        padding=quantized["refine_depthwise_weight"].shape[-1] // 2,
        groups=pooled_spatial_q.shape[1],
    )
    refine_depthwise_q = _requantize(
        refine_depthwise_acc,
        activation_scales["spatial_relu"] * tensor_scales["refine_depthwise_weight"],
        activation_scales["refine_depthwise"], relu=False,
    )
    refine_acc = _integer_conv1d(
        refine_depthwise_q, quantized["refine_pointwise_weight"], quantized["refine_pointwise_bias"],
        padding=0, groups=1,
    )
    refine_q = _requantize(
        refine_acc,
        activation_scales["refine_depthwise"] * tensor_scales["refine_pointwise_weight"],
        activation_scales["refine_relu"], relu=True,
    )
    pooled_refine_q = _integer_average_pool4(refine_q)
    global_q = _integer_global_average(pooled_refine_q)
    logits_acc = torch.round(functional.linear(
        global_q.to(torch.float64), quantized["classifier_weight"].to(torch.float64), None
    )).to(torch.int64) + quantized["classifier_bias"].reshape(1, -1)
    logits_scale = activation_scales["refine_relu"] * tensor_scales["classifier_weight"]
    return logits_acc, logits_scale, {
        "temporal_depthwise": temporal_acc,
        "spatial_pointwise": spatial_acc,
        "refine_depthwise": refine_depthwise_acc,
        "refine_pointwise": refine_acc,
        "classifier": logits_acc,
    }


def _window_metrics(logits, labels):
    predictions = np.argmax(logits, axis=1)
    labels = labels.astype(np.int64)
    true_positive = int(np.logical_and(predictions == 1, labels == 1).sum())
    false_positive = int(np.logical_and(predictions == 1, labels == 0).sum())
    false_negative = int(np.logical_and(predictions == 0, labels == 1).sum())
    true_negative = int(np.logical_and(predictions == 0, labels == 0).sum())
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def _write_tensor(output_dir, name, tensor, dtype, scale=None):
    dtype_map = {"int16": np.dtype("<i2"), "int32": np.dtype("<i4")}
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported export dtype: {dtype}")
    path = Path(output_dir) / "tensors" / f"{name}.{dtype}.bin"
    array = tensor.detach().cpu().numpy().astype(dtype_map[dtype], copy=False)
    array.tofile(path)
    metadata = {
        "name": name,
        "path": str(path.relative_to(output_dir)).replace("\\", "/"),
        "dtype": dtype,
        "endianness": "little",
        "shape": list(array.shape),
        "elements": int(array.size),
        "bytes": int(array.nbytes),
        "layout": "row_major_flattened",
    }
    if scale is not None:
        metadata["scale"] = float(scale)
    return metadata


def _load_split(path, expected_channels, expected_length):
    with np.load(path, allow_pickle=False) as split:
        x = np.asarray(split["X"], dtype=np.float32)
        y = np.asarray(split["y"], dtype=np.int64)
        channels = np.asarray(split["channels"]).astype(str)
    if x.ndim != 3 or x.shape[1:] != (expected_channels, expected_length):
        raise ValueError(f"Unexpected prepared data shape {x.shape} in {path}")
    return x, y, channels


def export_separable_reference():
    """Create a complete fixed-point package from a selected separable training run."""
    source_run_id = os.environ.get("CHBMIT_FPGA_SOURCE_RUN_ID", "run_21_raw_2s_temporal3")
    source_dir = Path(project_dir) / "outputs" / source_run_id
    checkpoint_path = source_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint_path}")

    prepared_name = os.environ.get("CHBMIT_FPGA_PREPARED_OUTPUT_DIR", "chbmit_prepared_raw_2s_v1")
    prepared_dir = Path(project_dir) / "data" / prepared_name
    train_path = prepared_dir / "chbmit_train.npz"
    val_path = prepared_dir / "chbmit_val.npz"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("FPGA export requires the selected run's prepared train and validation NPZ files")

    output_dir = _resolve_directory("CHBMIT_FPGA_OUTPUT_DIR", "fpga/reference_run_21_int16")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"FPGA export output already exists: {output_dir}")
    (output_dir / "tensors").mkdir(parents=True, exist_ok=True)
    (output_dir / "test_vectors").mkdir(parents=True, exist_ok=True)

    model = build_model_from_run(str(source_dir)).cpu().eval()
    if not isinstance(model, SeparableEEG1DCNN):
        raise NotImplementedError("FPGA reference exporter currently supports separable_1dcnn only")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    weights = _fuse_separable_model(model)

    expected_channels = model.temporal_depthwise.in_channels
    model_spec = _load_json(source_dir / "model_spec.json")
    expected_length = int(model_spec["model_config"]["input_length"])
    train_x, _, train_channels = _load_split(train_path, expected_channels, expected_length)
    val_x, val_y, val_channels = _load_split(val_path, expected_channels, expected_length)
    if not np.array_equal(train_channels, val_channels):
        raise ValueError("Prepared train and validation channel orders differ")

    normalization_spec = _load_json(source_dir / "normalization_spec.json")
    if normalization_spec.get("mode") != "train_channel_zscore":
        raise ValueError("FPGA reference exporter requires train_channel_zscore normalization")
    mean_path = source_dir / "scaler_mean.npy"
    scale_path = source_dir / "scaler_scale.npy"
    if not mean_path.is_file() or not scale_path.is_file():
        raise FileNotFoundError("Missing scaler_mean.npy or scaler_scale.npy in selected run output")
    mean = np.asarray(np.load(mean_path), dtype=np.float32)
    std = np.asarray(np.load(scale_path), dtype=np.float32)
    if mean.shape != (expected_channels,) or std.shape != (expected_channels,) or np.any(std <= 0):
        raise ValueError("Selected normalization constants are invalid")

    calibration_windows = _environment_int("CHBMIT_FPGA_CALIBRATION_WINDOWS", 1024, 1)
    validation_limit = _environment_int("CHBMIT_FPGA_VALIDATION_WINDOWS", 0, 0)
    batch_size = _environment_int("CHBMIT_FPGA_BATCH_SIZE", 64, 1)
    activation_scales, activation_maxima, calibration_count = _calibrate_activation_scales(
        train_x, mean, std, weights, calibration_windows, batch_size
    )
    quantized, tensor_scales = _build_quantized_tensors(weights, activation_scales)

    validation_indices = np.arange(len(val_x), dtype=np.int64)
    if validation_limit:
        validation_indices = validation_indices[:min(validation_limit, len(validation_indices))]
    float_logits = []
    quant_logits = []
    direct_logits = []
    labels = []
    accumulator_maxima = {name: 0 for name in (
        "temporal_depthwise", "spatial_pointwise", "refine_depthwise", "refine_pointwise", "classifier"
    )}
    with torch.inference_mode():
        for start in range(0, len(validation_indices), batch_size):
            indices = validation_indices[start:start + batch_size]
            raw = val_x[indices]
            inputs = torch.from_numpy((raw - mean[None, :, None]) / std[None, :, None])
            direct_logits.append(model(inputs).cpu().numpy())
            fused_logits = _float_forward(inputs, weights)["logits"]
            integer_logits, logits_scale, accumulators = _integer_forward(
                inputs, quantized, activation_scales, tensor_scales
            )
            float_logits.append(fused_logits.cpu().numpy())
            quant_logits.append((integer_logits.to(torch.float64) * logits_scale).cpu().numpy())
            labels.append(val_y[indices])
            for name, accumulator in accumulators.items():
                accumulator_maxima[name] = max(
                    accumulator_maxima[name], int(torch.max(torch.abs(accumulator)).item())
                )

    direct_logits = np.concatenate(direct_logits)
    float_logits = np.concatenate(float_logits)
    quant_logits = np.concatenate(quant_logits)
    labels = np.concatenate(labels)
    fused_delta = float(np.max(np.abs(direct_logits - float_logits)))
    if fused_delta > 1e-4:
        raise RuntimeError(f"BatchNorm folding verification failed: max logit delta {fused_delta}")

    tensors = [
        _write_tensor(output_dir, "temporal_depthwise_weight", quantized["temporal_depthwise_weight"], "int16", tensor_scales["temporal_depthwise_weight"]),
        _write_tensor(output_dir, "temporal_depthwise_bias", quantized["temporal_depthwise_bias"], "int32", activation_scales["input"] * tensor_scales["temporal_depthwise_weight"]),
        _write_tensor(output_dir, "spatial_pointwise_weight", quantized["spatial_pointwise_weight"], "int16", tensor_scales["spatial_pointwise_weight"]),
        _write_tensor(output_dir, "spatial_pointwise_bias", quantized["spatial_pointwise_bias"], "int32", activation_scales["temporal_relu"] * tensor_scales["spatial_pointwise_weight"]),
        _write_tensor(output_dir, "refine_depthwise_weight", quantized["refine_depthwise_weight"], "int16", tensor_scales["refine_depthwise_weight"]),
        _write_tensor(output_dir, "refine_pointwise_weight", quantized["refine_pointwise_weight"], "int16", tensor_scales["refine_pointwise_weight"]),
        _write_tensor(output_dir, "refine_pointwise_bias", quantized["refine_pointwise_bias"], "int32", activation_scales["refine_depthwise"] * tensor_scales["refine_pointwise_weight"]),
        _write_tensor(output_dir, "classifier_weight", quantized["classifier_weight"], "int16", tensor_scales["classifier_weight"]),
        _write_tensor(output_dir, "classifier_bias", quantized["classifier_bias"], "int32", activation_scales["refine_relu"] * tensor_scales["classifier_weight"]),
    ]

    test_index = _environment_int("CHBMIT_FPGA_TEST_VECTOR_INDEX", 0, 0)
    if test_index >= len(val_x):
        raise IndexError(f"CHBMIT_FPGA_TEST_VECTOR_INDEX must be < {len(val_x)}")
    test_input = torch.from_numpy(((val_x[test_index] - mean[:, None]) / std[:, None])[None, ...])
    test_logits_acc, test_logits_scale, _ = _integer_forward(
        test_input, quantized, activation_scales, tensor_scales
    )
    test_input_q = _quantize_int16(test_input, activation_scales["input"])[0]
    test_input_q.cpu().numpy().astype("<i2", copy=False).tofile(output_dir / "test_vectors" / "input_i16.bin")
    np.savetxt(output_dir / "test_vectors" / "expected_logits_i64.txt", test_logits_acc[0].cpu().numpy(), fmt="%d")
    np.savetxt(
        output_dir / "test_vectors" / "expected_logits_fp64.txt",
        (test_logits_acc.to(torch.float64) * test_logits_scale)[0].cpu().numpy(), fmt="%.12g",
    )

    preparation_summary_path = prepared_dir / "preparation_summary.json"
    preparation_summary = _load_json(preparation_summary_path) if preparation_summary_path.is_file() else {}
    normalization_payload = {
        "mode": normalization_spec["mode"],
        "scope": normalization_spec.get("scope", "train_only"),
        "channels": train_channels.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "accelerator_input": "filtered_raw_eeg_then_train_channel_zscore",
    }
    _save_json(output_dir / "normalization.json", normalization_payload)
    _save_json(output_dir / "test_vectors" / "manifest.json", {
        "input_path": "input_i16.bin",
        "input_shape": [expected_channels, expected_length],
        "input_dtype": "int16",
        "input_scale": activation_scales["input"],
        "expected_logits_path": "expected_logits_i64.txt",
        "expected_logits_scale": test_logits_scale,
        "expected_class": int(torch.argmax(test_logits_acc, dim=1).item()),
        "source_validation_index": test_index,
        "source_label": int(val_y[test_index]),
    })

    model_manifest = {
        "format_version": 1,
        "source_run_id": source_run_id,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "architecture": "separable_1dcnn",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "input": {
            "shape": [1, expected_channels, expected_length],
            "dtype": "int16",
            "scale": activation_scales["input"],
            "channel_order": train_channels.tolist(),
        },
        "fixed_point": {
            "weight_dtype": "int16",
            "bias_dtype": "int32",
            "accumulator_dtype": "int64_reference",
            "rounding": "nearest; nonnegative average pools use floor((sum + divisor/2) / divisor)",
            "zero_point": 0,
        },
        "activation_scales": activation_scales,
        "layers": [
            {"name": "temporal_depthwise", "op": "conv1d", "groups": expected_channels, "kernel": int(model.temporal_depthwise.kernel_size[0]), "padding": int(model.temporal_depthwise.padding[0]), "activation": "relu", "weight": "temporal_depthwise_weight", "bias": "temporal_depthwise_bias", "output_scale": activation_scales["temporal_relu"]},
            {"name": "spatial_pointwise", "op": "conv1d", "groups": 1, "kernel": 1, "padding": 0, "activation": "relu", "weight": "spatial_pointwise_weight", "bias": "spatial_pointwise_bias", "output_scale": activation_scales["spatial_relu"]},
            {"name": "pool_1", "op": "average_pool1d", "kernel": 4, "stride": 4},
            {"name": "refine_depthwise", "op": "conv1d", "groups": int(model.refine_depthwise.groups), "kernel": int(model.refine_depthwise.kernel_size[0]), "padding": int(model.refine_depthwise.padding[0]), "activation": "none", "weight": "refine_depthwise_weight", "bias": None, "output_scale": activation_scales["refine_depthwise"]},
            {"name": "refine_pointwise", "op": "conv1d", "groups": 1, "kernel": 1, "padding": 0, "activation": "relu", "weight": "refine_pointwise_weight", "bias": "refine_pointwise_bias", "output_scale": activation_scales["refine_relu"]},
            {"name": "pool_2", "op": "average_pool1d", "kernel": 4, "stride": 4},
            {"name": "global_average", "op": "adaptive_average_pool1d", "output_length": 1},
            {"name": "classifier", "op": "linear", "weight": "classifier_weight", "bias": "classifier_bias", "logit_scale": activation_scales["refine_relu"] * tensor_scales["classifier_weight"]},
        ],
        "tensors": tensors,
        "preprocessing_contract": {
            "sample_rate_hz": preparation_summary.get("sample_rate_hz"),
            "bandpass_hz": preparation_summary.get("bandpass_hz"),
            "notch_hz": preparation_summary.get("notch_hz"),
            "warning": "The current corpus preparation uses offline zero-phase filtering. This package starts at normalized window input; causal FPGA filtering remains a separate validation task.",
        },
    }
    _save_json(output_dir / "model_manifest.json", model_manifest)

    agreement = float(np.mean(np.argmax(float_logits, axis=1) == np.argmax(quant_logits, axis=1)))
    report = {
        "source_run_id": source_run_id,
        "calibration": {
            "split": "train",
            "windows": calibration_count,
            "activation_max_abs": activation_maxima,
            "activation_scales": activation_scales,
        },
        "batchnorm_fold_max_logit_abs_delta": fused_delta,
        "validation": {
            "split": "val",
            "windows": int(len(labels)),
            "float_fused": _window_metrics(float_logits, labels),
            "int16_emulated": _window_metrics(quant_logits, labels),
            "prediction_agreement": agreement,
            "max_logit_abs_error": float(np.max(np.abs(float_logits - quant_logits))),
        },
        "observed_accumulator_max_abs": accumulator_maxima,
        "deployment_status": "integer reference package exported; FPGA synthesis and hardware-in-the-loop validation pending",
    }
    _save_json(output_dir / "quantization_report.json", report)
    print(f"FPGA export package: {output_dir}")
    print(f"Validation prediction agreement: {agreement:.4f}")
    print(f"Quantized validation metrics: {json.dumps(report['validation']['int16_emulated'], sort_keys=True)}")
    return output_dir
