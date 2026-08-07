# Ordered Implementation Plan: W1--W8

## Non-goals for H0 v1

- No Path-A 95% mean accuracy ladder, multiscale model, four-layer model, or
  patient-specific 24-model FPGA deployment.
- No Image and Vision Computing framing or cross-paper accuracy leaderboard.
- No change to the frozen run-21 checkpoint, `31/15` graph, package scales,
  or input shape to improve silicon results.
- No clinical, streaming, or end-to-end EEG claim while the boundary remains
  after zero-phase filtering and z-score normalisation.

## W0 - Board and build-host preflight

**Deliverable:** one captured preflight directory from the x86 build host and
one from the KV260 target. The two machines have different roles: Vitis HLS
and Vivado normally run on the x86 build host; XRT, the FPGA image, DMA, and
power measurement run on the KV260 target.

Run on each machine from its checkout of this hardware branch:

```bash
cd AI_train_model && bash fpga/kv260/episepnet_5k/tools/kv260_preflight.sh fpga/kv260/episepnet_5k/runs/preflight_$(hostname -s)
```

The script captures architecture, OS, relevant Xilinx/XRT tools, device nodes,
XRT inspection output, and the H0 package SHA ledger without collecting
credentials or arbitrary environment variables.

**Gate:** the build host has a recorded Vitis HLS/Vivado build path, the
target has a recorded `aarch64` runtime plus XRT/device status, and the H0
package ledger is identical on both checkouts. A missing HLS installation on
the target is expected; a missing target runtime or missing build-host toolchain
is a blocker to record, not something to hide.

## W1 - Freeze and inspect the package

**Deliverable:** SHA-256 ledger for manifest, normalisation, tensors, test
input, and expected logits.

```bash
cd AI_train_model && sha256sum fpga/reference_run_21_int16/model_manifest.json fpga/reference_run_21_int16/normalization.json fpga/reference_run_21_int16/tensors/* fpga/reference_run_21_int16/test_vectors/* | tee fpga/kv260/episepnet_5k/package_sha256.txt
```

**Gate:** the checkpoint SHA in the manifest is
`c55b430b00deb5b67b92fb36f7208e89d629fb5f6c8f5e2b41180f69cb3b91f8`.

## W2 - M1a C golden model

**Deliverable:** generated C headers and a self-checking testbench.

```bash
cd AI_train_model && python fpga/kv260/episepnet_5k/tools/generate_hls_headers.py && g++ -std=c++17 -O2 -Wall -Wextra -Werror -Ifpga/kv260/episepnet_5k/hls/include fpga/kv260/episepnet_5k/hls/src/episepset_5k_golden.cpp fpga/kv260/episepnet_5k/hls/tb/episepset_5k_golden_tb.cpp -o /tmp/episepset_5k_golden && /tmp/episepset_5k_golden fpga/reference_run_21_int16/test_vectors
```

**Gate:** exact two-logit equality. Then add a deterministic validation-vector
set and prove every M1a output equals the Python emulator.

## W3 - M1b fixed multiplier/shift

**Deliverable:** checked-in multiplier/shift constants and a full M1a/M1b
replay report. Implement multiplication in `ap_int`, not floating point.

**Gate:** predeclare and report any logit or prediction difference. If it is
not acceptable, change only the numeric representation and repeat the replay;
do not retune the CNN.

## W4 - HLS core

**Deliverable:** HLS top using compile-time ROM weights, no file I/O;
self-checking C simulation and C/RTL co-simulation logs. Target part and tool
release are recorded in `hls_config.cfg` after the board is inspected.

**Design sweep:** scalar baseline H0, pipelined temporal loops H1, and one
pointwise unroll-factor sweep H2. Keep clock and all arithmetic constants
fixed while comparing them.

**Gate:** C/RTL logit equality and synthesis without unintended DSP/BRAM
resource overrun.

## W5 - AXI/DMA integration

**Deliverable:** Vivado block design or Vitis kernel/platform, host control
application, buffer layout, and command log.

**Gate:** one input DMA, one output DMA, and exact golden logits on the board.
Measure `kernel_only`, `dma_kernel`, and `host_dma_kernel` separately.

## W6 - Post-route PPA

**Deliverable:** per-design implementation reports and rows in
`../measurement_template.csv` with `power_w` and `energy_mj_per_window` empty
until the board measurement is complete.

**Gate:** achieved frequency, timing closure, and utilisation from the same
bitstream used on the board.

## W7 - On-board measurement

**Deliverable:** raw power logs, latency-cycle logs, host timing logs, thermal
state, 1,000+ batch-1 measurements, and reproducible analysis script.

**Gate:** PPA, median/p95 latency, throughput, power method, and energy/window
are all reported with the boundary definitions in `measurement_contract.md`.

## W8 - Hardware-in-loop validation and paper freeze

**Deliverable:** FP32/folded-float/M1a/M1b/RTL/KV260 agreement table and a
frozen 3,898-window validation or declared representative-subset replay.

**Gate:** write the TBioCAS Results/Discussion/Abstract from tracked data only.
The allowed model-quality wording remains: 90.0718% balanced validation-window
accuracy in FP32 and 90.0462% in the INT16 emulator. These are not a test,
patient-independent, event-level, or final-board accuracy result.
