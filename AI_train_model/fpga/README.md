# FPGA Reference Package

Run `python main.py --mode export_fpga` on the training server after selecting
a separable-model checkpoint. The default source is
`outputs/run_21_raw_2s_temporal3/` and the default package is written to
`fpga/reference_run_21_int16/`.

The exporter folds BatchNorm, performs symmetric per-tensor INT16 weight and
activation quantization, writes INT32 biases, and emits binary tensors in
little-endian row-major order. `model_manifest.json` is the hardware contract;
it specifies tensor shape, scale, layer order, group count, padding, pooling,
and channel order. `normalization.json` contains the exact train-only z-score
constants. `test_vectors/` contains one quantized input and expected integer
logits for RTL/HLS verification.

The current package starts at filtered and normalized EEG windows. The corpus
preprocessing still uses offline zero-phase filters, so causal FPGA filtering
and hardware-in-the-loop event evaluation remain separate work.

## Verified EpiSepNet-5K Package

The committed package at `reference_run_21_int16/` was generated from
EpiSepNet-5K evidence run `run_21_raw_2s_temporal3`, whose checkpoint SHA-256 is
`c55b430b00deb5b67b92fb36f7208e89d629fb5f6c8f5e2b41180f69cb3b91f8`.

| Check | Result |
|---|---:|
| Folded-float versus PyTorch maximum logit delta | `3.81e-06` |
| Validation FP32 window accuracy | `90.0718%` |
| Validation INT16-emulated window accuracy | `90.0462%` |
| Accuracy change after INT16 export | `-0.0257` percentage points |
| Validation prediction agreement | `99.9743%` |
| Validation sensitivity before and after export | `90.7645%` |

The package contains all 5,013 model parameters. Tensor byte counts, shapes,
and scales are verified against `model_manifest.json`.

## HLS/RTL Data Path

The accelerator input is one normalized window with shape `[17, 512]`, stored
as signed INT16. The hardware must preserve the channel order in
`normalization.json` and use its train-only mean and standard deviation before
input quantization.

```text
INT16 input [17,512]
  -> depthwise Conv1D: 17 groups, 3 filters/channel, kernel 31, pad 15
  -> ReLU
  -> pointwise Conv1D: 51 to 32 channels, kernel 1
  -> ReLU -> average pool 4
  -> depthwise Conv1D: 32 groups, kernel 15, pad 7
  -> pointwise Conv1D: 32 to 32 channels, kernel 1
  -> ReLU -> average pool 4
  -> global average pool
  -> linear: 32 to 2 logits
```

Weights are signed INT16 and biases are signed INT32. Use the per-tensor
scales in `model_manifest.json`; all zero points are zero. A universal signed
`ap_int<48>` accumulator is required for the first KV260 implementation.
Observed validation maxima require 33 bits for temporal depthwise, spatial
pointwise, and refine depthwise layers, 32 bits for refine pointwise, and 34
bits for the classifier. A 32-bit accumulator can overflow.

## First Hardware Test

Use `test_vectors/input_i16.bin` as one `[17,512]` input in row-major order.
The expected quantized result is in `expected_logits_i64.txt`; multiply each
integer logit by `expected_logits_scale` from `test_vectors/manifest.json` to
recover the floating-point logit. Compare logits, not the dataset label: this
particular validation input has label `1` while the selected model predicts
class `0`.

The initial SoC boundary is *after* EEG filtering and z-score normalization.
Before a final deployment claim, implement causal filtering, verify the exact
fixed-point data path in HLS/RTL, and measure KV260 resource use, latency,
throughput, power, and continuous event metrics.
