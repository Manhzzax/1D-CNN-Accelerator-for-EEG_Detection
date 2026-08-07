# Arithmetic Contract: EpiSepNet-5K H0

## Authority and scope

The frozen package at `../../reference_run_21_int16/` is authoritative. Its
`model_manifest.json`, binary tensor byte order, and golden vector supersede
any value copied into a document. The accelerator input starts after filtering
and train-only z-score normalisation; it never receives raw EDF samples in
version H0.

## Graph and tensor schedule

| Step | Input -> output | Arithmetic | Scale / post-operation |
|---|---|---|---|
| 0 | `[17][512]` | signed INT16 | input scale `0.0007052223512213722` |
| 1 | `17 -> 51`, k31, groups17, pad15 | INT16 x INT16 + INT32 bias | requantise, saturate to `[-32767,32767]`, ReLU |
| 2 | `51 -> 32`, k1 | INT16 x INT16 + INT32 bias | requantise, saturate, ReLU |
| 3 | `[32][512] -> [32][128]` | average pool k4/s4 | non-negative round-to-nearest |
| 4 | `32 -> 32`, k15, groups32, pad7 | INT16 x INT16 | requantise and saturate; **no** ReLU |
| 5 | `32 -> 32`, k1 | INT16 x INT16 + INT32 bias | requantise, saturate, ReLU |
| 6 | `[32][128] -> [32][32]` | average pool k4/s4 | non-negative round-to-nearest |
| 7 | `[32][32] -> [32]` | global average | non-negative round-to-nearest |
| 8 | `32 -> 2` | INT16 x INT16 + INT32 bias | retain signed INT64 logits |

All tensor bytes are little-endian and row-major. Depthwise output channel
`o` maps to source channel `floor(o / 3)` in step 1. Padding values are
integer zero, not the zero-point of a different quantiser, because every
zero-point is zero.

## M1a: frozen-emulator reference

M1a must reproduce the exporter before optimisation:

```text
acc = sum(int16_activation * int16_weight) + int32_bias
q   = round_to_nearest_even(acc * (accumulator_scale / output_scale))
q   = saturate(q, -32767, 32767)
q   = max(q, 0)                 # only where the manifest specifies ReLU
```

Average pool is exactly `floor((sum + 2) / 4)`; the final global average is
`floor((sum + 16) / 32)`. `round_to_nearest_even` follows PyTorch's
`torch.round` on the finite FP64 calculation. The C golden implementation
sets `FE_TONEAREST` and uses `std::nearbyint`; do not replace it with C++ casts
or a truncating right shift.

The supplied vector must return the two signed logits in
`test_vectors/expected_logits_i64.txt`. Its source label is `1`, but its frozen
model prediction is class `0`; correctness is logit equality, not label
correctness.

## Width and saturation checklist

| Quantity | Requirement |
|---|---|
| Stored input, weights, layer activations | signed 16 bit; legal range `[-32767,32767]` |
| Stored folded biases | signed 32 bit in the accumulator scale |
| Convolution and classifier accumulation | signed `ap_int<48>` for H0 v1; host M1a uses `int64_t` |
| Observed maxima in exported validation replay | temporal `2,257,439,519`; spatial `2,228,274,584`; refine-DW `2,328,674,082`; refine-PW `1,545,656,677`; classifier `4,337,920,357` |
| M1b multiplier product | minimum signed 80 bits for `ap_int<48> * signed Q31 multiplier`; use `ap_int<81>` before rounding/shift |
| Requantisation output | apply rounding, then saturation, then ReLU; never wrap on overflow |
| BatchNorm | already folded; hardware has no BatchNorm state or epsilon |

`int32_t` is insufficient even for the observed classifier accumulator.
Observed maxima are a validation-replay check, not a proof of all possible
input bounds; H0 uses 48-bit accumulators as the conservative implementation
contract.

## M1b: integer-only requantisation

The exported emulator currently uses an FP64 ratio. An FPGA result requires a
separate frozen M1b contract for every requantisation site:

```text
ratio = accumulator_scale / output_scale
q31_multiplier = round(ratio * 2^31)
product = ap_int<48>(acc) * ap_int<32>(q31_multiplier)
q = round_to_nearest_even(product / 2^31)
```

The header generator records the ratios and Q31 candidates; it does **not**
claim M1b equivalence. Before synthesis, a test must replay all 3,898 frozen
validation inputs and report M1a-to-M1b logit and prediction agreement. If
Q31 changes any result, evaluate larger fixed multipliers/shifts or document
the measured loss. Floating-point division, `float`, and `double` are
forbidden inside M1b HLS/RTL.

## Golden-vector gates

1. Hash the package and regenerate headers from its manifest.
2. M1a C test: exact pair of INT64 logits on the committed vector.
3. M1a replay: all frozen validation vectors must match the Python emulator.
4. M1b replay: record logit error, agreement, and accuracy relative to M1a.
5. C/RTL co-simulation: run the same self-checking vector suite.
6. Board HIL: compare exactly the same input bytes and integer logits.

A class-only comparison is insufficient: equal argmax can hide an arithmetic
or saturation error.
