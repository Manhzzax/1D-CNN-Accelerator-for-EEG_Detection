# H0 Gap Analysis and TBioCAS Claim Control

## Current evidence

| Gate | Status | Evidence | Permitted statement |
|---|---|---|---|
| G0 package freeze | complete | Manifest SHA `ca5270a...8d662b40`, checkpoint SHA in manifest, tensor binaries, normalisation and one vector | A reproducible exported INT16 package exists. |
| G1a M1a golden vector | complete | `episepset_5k_golden` returns `179692407, -122067309` exactly | The host C model reproduces the committed vector. |
| G1b M1a vector replay | missing | One vector is not a validation-set arithmetic proof | No full C-reference agreement claim. |
| G2 M1b multiplier/shift | missing | Q31 candidates are generated but not replayed | No integer-only datapath claim. |
| G3 HLS C/RTL co-simulation | missing | No HLS project or RTL log | No RTL correctness claim. |
| G4 post-route implementation | missing | No target part, clock constraint, routed bitstream, or reports | No FPGA resource/frequency claim. |
| G5 KV260 timing/power | missing | No AXI/DMA host application or board logs | No latency, throughput, power, or energy claim. |
| G6 HIL validation | missing | Emulator-only validation set evidence | No FPGA accuracy/agreement claim. |

## Frozen AI evidence: cite narrowly

The following are support for the accelerator input contract, not a general
seizure-detection leaderboard:

- 5,013 trainable parameters in the original BatchNorm-containing graph;
- 10,030 B in the exported coefficient package;
- 90.0718% balanced **validation-window** FP32 accuracy on 3,898 windows;
- 90.0462% INT16 emulator accuracy, -0.0257 percentage points versus folded
  FP32, and 99.9743% agreement;
- 90.7645% validation-window sensitivity in both paths.

The H0 package uses an offline zero-phase 0.5--45 Hz bandpass and 60 Hz notch
before the train-only channel z-score. It does not prove causal streaming. Its
historical event operating point (about 79% validation event sensitivity and
about 0.47 false alarms/hour) is contextual only and must be reported with the
exact validation policy rather than promoted to a board result.

Do **not** cite the Path-A 5 s, patient-specific 24-model A1.2 accuracy as H0
hardware accuracy: it is a different model family, input duration, and
evaluation protocol. It can only appear in a clearly separated future-design
trade-off discussion.

## Accelerator architecture decision

The recommended v1 architecture is one HLS compute core with static ROM
weights and ping-pong/scratch activation buffers. Expose a simple batch-1
memory-mapped input/output interface plus AXI4-Lite control first. This creates
a defensible core and DMA baseline before a more aggressive AXI-stream dataflow
variant. The input is small (17,408 B/window) and weights are 10,030 B, so
coefficient bandwidth is not the principal reason to add external DDR traffic.

Design experiments are limited to the same H0 numeric graph:

1. `H0_scalar`: no intentional pointwise unroll; functional/resource baseline.
2. `H1_pipeline`: pipeline loop structure while preserving all arithmetic.
3. `H2_unroll_u`: sweep one predeclared pointwise unroll factor; compare
   post-route latency/resource/energy at a common achieved clock.
4. `H3_dataflow` only if H1/H2 shows internal buffering limits throughput.

The model, scales, padding, and input boundary are fixed across H0--H3. A
different 5 s model is a later H1 design point, not a parameter of this sweep.

## High-priority risks and mitigations

| Risk | Why it matters | Required mitigation |
|---|---|---|
| Offline zero-phase preprocessing | It uses future samples and cannot support a streaming/real-time claim. | Call H0 an offline normalised-window core; later add a separately validated causal-IIR path. |
| Emulator requantises with FP64 ratios | An HLS core using floating division would not establish integer hardware arithmetic. | Freeze M1b multiplier/shift constants and compare all outputs to M1a before HLS. |
| INT32 overflow | Classifier validation accumulator reaches 4,337,920,357. | Use `ap_int<48>` accumulator and at least `ap_int<81>` multiplier product; record saturation. |
| One golden vector | It can miss layout, padding and rare saturation errors. | Replay a deterministic frozen validation set and compare logits, not only labels. |
| Validation-centric biomedical metrics | They are not patient-independent or a final clinical test. | Label them validation-window evidence; present event metrics and all limits separately. |
| Model-family mismatch | Mixing 2 s H0 with 5 s Path A makes the comparison scientifically invalid. | Separate H0 and optional later H1 tables/claims. |
| Board power ambiguity | Tool telemetry, rail power and wall power are different quantities. | Use the primary 12-V input method or identify telemetry-derived measurement explicitly. |

## TBioCAS framing

TBioCAS requires a demonstrated synergy between circuits/systems and
medicine/biology. The central contribution must therefore be *the bit-verified
KV260 implementation and measured accuracy--latency--energy/resource trade-off
for a frozen EEG model*, not a 90% accuracy number. IEEE's scope and
submission guide also make a circuits/systems contribution and up-to-date
related literature essential.

Allowed title direction after G6:

> Bit-Verified INT16 EpiSepNet-5K Acceleration for Multichannel EEG Windows on
> an AMD Kria KV260

Avoid `real-time`, `clinical`, `patient-independent`, `end-to-end EEG`, and
`state of the art` until their separate evidence conditions are satisfied.

## Primary external sources

- AMD Vitis HLS documents C simulation and C/RTL co-simulation as distinct
  verification stages and supports self-checking testbenches:
  https://docs.amd.com/r/2024.2-English/ug1399-vitis-hls/Automatically-Verifying-the-RTL
- AMD Vitis HLS documents `m_axi` and `s_axilite` interfaces for HLS kernels:
  https://docs.amd.com/r/en-US/ug1399-vitis-hls/Interface
- AMD's KV260 product brief documents the K26 platform capacities used only as
  capacity context:
  https://www.amd.com/content/dam/xilinx/publications/product-briefs/xilinx-kv260-product-brief.pdf
- XRT documents the `xbutil examine --report electrical` capability; sensor
  availability is platform-dependent:
  https://xilinx.github.io/XRT/2024.1/html/xbutil.html
- TBioCAS scope explicitly requires circuits/systems and biomedical synergy:
  https://ieee-cas.org/publication/TBioCAS/tbiocas-manuscript-submission-guide
