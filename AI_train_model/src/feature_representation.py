"""Deterministic EEG feature representations shared by preparation and inference."""

import json
import os
from pathlib import Path

import numpy as np


FEATURE_SPEC_FILENAME = "feature_representation.json"


def get_feature_spec(preprocessing):
    """Resolve a representation without changing the raw-data baseline by default."""
    input_length = int(round(
        float(preprocessing["sample_rate_hz"]) * float(preprocessing["window_sec"])
    ))
    if input_length < 1:
        raise ValueError("Feature representation requires a positive input length")
    name = os.environ.get(
        "CHBMIT_FEATURE_REPRESENTATION", preprocessing.get("feature_representation", "raw")
    )
    if name == "raw":
        return {"name": "raw", "input_shape": [17, input_length]}
    if name == "dwt_db4_l3":
        options = preprocessing.get("dwt", {})
        return {
            "name": name,
            "wavelet": str(options.get("wavelet", "db4")),
            "level": int(options.get("level", 3)),
            "mode": str(options.get("mode", "periodization")),
            "input_shape": [17, input_length],
            "coefficient_order": "cA3,cD3,cD2,cD1",
        }
    raise ValueError(f"Unsupported CHBMIT_FEATURE_REPRESENTATION: {name}")


def load_feature_spec(directory):
    """Load a persisted representation contract, falling back for legacy raw artifacts."""
    path = Path(directory) / FEATURE_SPEC_FILENAME
    if not path.is_file():
        return {"name": "raw", "input_shape": [17, 256], "source": "legacy_default"}
    with path.open("r", encoding="utf-8") as input_file:
        spec = json.load(input_file)
    if spec.get("name") not in {"raw", "dwt_db4_l3"}:
        raise ValueError(f"Unsupported persisted feature representation: {spec.get('name')}")
    return spec


def save_feature_spec(directory, spec):
    path = Path(directory) / FEATURE_SPEC_FILENAME
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(spec, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def transform_windows(windows, spec, batch_size=512):
    """Transform `(N, channels, samples)` windows while preserving its shape.

    `periodization` preserves the input length when the DWT coefficients are
    concatenated, including the two-second (512-sample) protocol.
    """
    data = np.asarray(windows, dtype=np.float32)
    single_window = data.ndim == 2
    if single_window:
        data = data[None, ...]
    if data.ndim != 3:
        raise ValueError(f"Expected windows with shape (N, C, T), got {data.shape}")
    if spec["name"] == "raw":
        return data[0] if single_window else data
    if spec["name"] != "dwt_db4_l3":
        raise ValueError(f"Unsupported feature representation: {spec['name']}")

    _, channels, sample_count = data.shape
    level = int(spec["level"])
    if sample_count % (2 ** level) != 0:
        raise ValueError(f"DWT level {level} requires a sample count divisible by {2 ** level}")
    try:
        import pywt
    except ImportError as error:
        raise ImportError("DWT representation requires PyWavelets. Run pip install -r requirements.txt.") from error

    transformed = np.empty_like(data, dtype=np.float32)
    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch = data[batch_start:batch_end]
        flattened = batch.reshape(-1, sample_count)
        coefficients = pywt.wavedec(
            flattened,
            wavelet=spec["wavelet"],
            mode=spec["mode"],
            level=level,
            axis=-1,
        )
        concatenated = np.concatenate(coefficients, axis=-1)
        if concatenated.shape[-1] != sample_count:
            raise ValueError(
                f"DWT coefficient length {concatenated.shape[-1]} does not match input length {sample_count}"
            )
        transformed[batch_start:batch_end] = concatenated.reshape(batch.shape)
    return transformed[0] if single_window else transformed
