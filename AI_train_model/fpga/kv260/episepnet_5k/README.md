# EpiSepNet-5K KV260 Implementation Track

This directory is the **only** implementation track for the frozen hardware
reference `EpiSepNet-5K`. It does not use the active patient-specific model,
the five-second clean-slate model, or any future accuracy candidate.

## Frozen identity

| Field | Frozen value |
|---|---|
| Source evidence | `outputs/run_21_raw_2s_temporal3/` |
| Graph | `separable_1dcnn`, temporal/refinement kernels `31/15` |
| Input boundary | One already-filtered and train-only-z-scored `int16[17][512]` window |
| Sample rate/window | 256 Hz / 2 s |
| Trainable parameters | 5,013 (includes BatchNorm parameters before folding) |
| Deployed tensor package | `../../reference_run_21_int16/` |
| Deployed coefficient bytes | 10,030 B: 4,781 INT16 weights and 117 INT32 biases |
| Numeric reference | BatchNorm-folded, symmetric per-tensor INT16; zero point 0 |

`../../reference_run_21_int16/model_manifest.json` is the authoritative
machine-readable interface. No source code is allowed to silently change a
shape, channel order, scale, padding rule, pooling rule, or checkpoint hash.

## Evidence status

Completed:

- BatchNorm folding is verified against PyTorch (maximum logit delta
  `3.814697e-06`).
- An INT16 emulator was evaluated on 3,898 validation windows: 90.0462%
  balanced-window accuracy and 99.9743% prediction agreement with folded
  FP32.
- One quantised input and its expected pair of INT64 logits are committed.
- The M1a host C golden implementation reproduces that committed logit pair.

Not completed:

- M1a deterministic validation-vector replay;
- multiplier/shift-only requantisation and its validation agreement;
- HLS C synthesis and C/RTL co-simulation;
- AXI/DMA integration, post-route implementation, and KV260 hardware-in-loop
  measurement.

Therefore this directory supports an *offline normalised-window accelerator*
claim only. The source corpus preparation uses zero-phase filtering. A causal
filter implementation and end-to-end streaming event validation are separate
work and must not be implied by a successful core benchmark.

## Directory map

```text
arithmetic_contract.md       Exact M1a/M1b arithmetic obligations
measurement_contract.md      PPA, latency, power, and energy definitions
implementation_plan.md       Ordered gates W1--W8 and server/board commands
literature_architecture_decision.md  Literature-grounded accelerator choice
tools/generate_hls_headers.py  Regenerates constants from the frozen package
hls/include/                Generated model constants; do not hand edit
hls/src/                    Host C golden source (M1a)
hls/tb/                     Self-checking golden-vector testbench
```

## First reproducibility command

Run this from `AI_train_model` after checking out
`research/kv260-episepnet-5k`:

```bash
python fpga/kv260/episepnet_5k/tools/generate_hls_headers.py && g++ -std=c++17 -O2 -Wall -Wextra -Werror -Ifpga/kv260/episepnet_5k/hls/include fpga/kv260/episepnet_5k/hls/src/episepset_5k_golden.cpp fpga/kv260/episepnet_5k/hls/tb/episepset_5k_golden_tb.cpp -o /tmp/episepset_5k_golden && /tmp/episepset_5k_golden fpga/reference_run_21_int16/test_vectors
```

Expected result: the two returned INT64 logits exactly equal
`test_vectors/expected_logits_i64.txt`. This is M1a only; it is not HLS,
RTL, FPGA, or board evidence.

See [arithmetic_contract.md](arithmetic_contract.md) and
[measurement_contract.md](measurement_contract.md) before changing a pragma
or adding an AXI interface.

The chosen accelerator architecture and the evidence that supports it are in
[literature_architecture_decision.md](literature_architecture_decision.md).
