# Hardware-Aware EEG Accelerator: Research Verification

## Decision

The selected direction is valid: deploy the frozen EpiSepNet-5K detector as a
custom KV260 accelerator, and make the contribution the measured
accuracy-efficiency trade-off. It is not a clinical state-of-the-art claim.

The model is a suitable first accelerator target because it has 5,013 trained
parameters, a static `17 x 512` INT16 input, only Conv1D/pooling/ReLU/linear
operators, and a locked INT16 software reference. Its two temporal depthwise
layers and two pointwise layers expose the relevant FPGA trade-off between DSP
parallelism, BRAM-resident weights, and streaming buffer storage.

## Research Basis

| Evidence | Verified lesson for this project |
|---|---|
| [Bahr et al., 2021](https://doi.org/10.3390/bios11070203) | A 10,162-parameter CHB-MIT CNN was deployed in fixed point on a GAP8 processor. It reports patient-specific performance and measured 35 ms per one-second EEG input plus 4.9 uJ/inference. This supports reporting deployed accuracy, latency, and energy together. |
| [Li et al., 2022](https://doi.org/10.1109/TBCAS.2022.3185584) | A 10,778-parameter parallel CNN was co-designed with a memristive accelerator. Its CHB-MIT result is seizure prediction, so it is not an accuracy comparator; it is direct evidence that model architecture and hardware dataflow must be co-designed. |
| [Gu et al., 2026](https://doi.org/10.1038/s41598-026-44536-y) | A 61,218-parameter PSD-CNN reports leave-one-subject-out CHB-MIT results, 85.84% accuracy, 1.9 ms/sample, and 0.33 FA/h. Its explicit PSD frontend means neural parameter count alone is not total system cost. Our paper must declare its input boundary. |
| [Ahlawat et al., 2026](https://arxiv.org/abs/2607.16296) | INT8, channel pruning, and sparsity can lower model storage and estimated energy, but this is a preprint and CPU-oriented evidence. It justifies a later precision/channel ablation, not an FPGA claim. |
| [AMD Vitis HLS C/RTL co-simulation](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Automatically-Verifying-the-RTL) | A self-checking C testbench must compare generated RTL outputs against known-good vectors. C simulation, synthesis, and RTL co-simulation are separate evidence stages. |
| [AMD Vitis HLS latency guidance](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Analyzing-RTL-Simulations) | HLS estimates and co-simulation latency are different. The paper must distinguish cycle-level core results from measured host-plus-DMA board latency. |

## Why Custom Vitis HLS Is The First Flow

FINN is valuable for networks trained as quantized QNNs through Brevitas/QONNX,
but its documented flow begins by training a QNN and translating to FINN-ONNX.
EpiSepNet-5K instead has an already locked, layer-scaled INT16 package with
custom depthwise Conv1D dimensions. A direct Vitis HLS implementation preserves
the scientific control point: same tensors, same arithmetic contract, and
explicit dataflow choices. FINN can be evaluated later only as a separate
tool-flow baseline; it must not silently change the network or quantization.

## Non-Negotiable Arithmetic Stages

The current package is a correct **software INT16 reference**, but its
requantization uses floating-point scale ratios during emulation. Therefore the
hardware path has two mandatory stages.

1. **M1a: exact C reference.** Implement the package arithmetic exactly as
   exported, including signed accumulation, round-to-nearest behavior, ReLU,
   and rounded average pooling. The supplied input must reproduce the two
   expected integer logits exactly.
2. **M1b: integer-requantization contract.** Convert each real scale ratio to
   documented signed integer multiplier and shift constants. Re-run the full
   validation-window agreement test against M1a. Only this multiplier/shift
   implementation is allowed to become HLS hardware.

This separation prevents an invalid claim that a floating-point scaling model
is an integer FPGA accelerator. Any agreement or accuracy change introduced by
M1b is a deployment result and must be reported.

## First Implementation Package

Create these source-controlled files after confirming the installed Vitis
version and target part:

```text
fpga/kv260/hls/
  include/episepset_int16_config.h       Fixed dimensions and multiplier/shift contract
  generated/episepset_int16_weights.h    Generated C arrays from the frozen tensor package
  src/episepset_int16.cpp                HLS top function, no file I/O
  tb/episepset_int16_tb.cpp              Self-checking known-vector testbench
  hls_config.cfg                         Exact target part, clock, interface settings
  README.md                              Build commands and result archive rules
```

The HLS top function accepts exactly one `[17][512]` signed-INT16 input and
returns two signed logits. File reading belongs only in the testbench; weights
must be compile-time constants or initialized ROMs in synthesizable code.

## Controlled Hardware Experiments

Use the same checkpoint, integer-requantization constants, test vector, clock
constraint, and board for every implementation.

| ID | Design change | Purpose |
|---|---|---|
| H0 | Straightforward scalar C implementation | Functional and resource baseline only |
| H1 | Pipeline temporal Conv1D loops; BRAM-resident weights | Establish an II/latency improvement |
| H2 | Controlled pointwise-channel unroll factor | Measure DSP/BRAM versus latency trade-off |
| H3 | Optional dataflow between layers with explicit buffers | Test whether streaming improves end-to-end kernel latency |

For each successful design archive post-route resources, achieved frequency,
kernel-only latency, host-plus-DMA latency, throughput, and power/energy. Do
not compare an HLS estimate from H0 against on-board timing from H2.

## Evidence Gates

| Gate | Required result | Claim enabled |
|---|---|---|
| G0 | Frozen tensor manifest, SHA-256, and expected logits | Reproducible software deployment package |
| G1 | M1a C test returns exact expected logits | Arithmetic implementation is correct |
| G2 | M1b matches M1a at the declared validation tolerance | Integer-only hardware arithmetic is validated |
| G3 | C/RTL co-simulation passes the same self-checking testbench | Generated RTL is functionally verified |
| G4 | KV260 implementation meets timing and has post-route resource report | FPGA implementation claim |
| G5 | At least 1,000 measured batch-1 inferences, kernel-only and host-plus-DMA rows, measured power method | Latency, throughput, power, and energy claim |
| G6 | FPGA scores the frozen validation set or a declared representative subset | Hardware-in-the-loop accuracy/agreement claim |

The patient-held-out experiment remains a biomedical validity appendix. It
cannot be replaced by a favorable within-case number, but it also must not
delay G1--G6.

## Immediate Start

1. Confirm the KV260 build-host operating system, Vitis/Vivado release, and
   exact K26 target part/platform.
2. Generate the immutable HLS weight headers and the integer multiplier/shift
   contract from `fpga/reference_run_21_int16/`.
3. Build M1a as a host C++ self-check before opening Vivado.
4. Run C synthesis and C/RTL co-simulation for H0.
5. Only after G3, integrate AXI DMA and measure on the board.

No new CNN training run is required for steps 1--5.
