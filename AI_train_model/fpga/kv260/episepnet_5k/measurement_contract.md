# KV260 Measurement Contract: EpiSepNet-5K H0

Only post-route and on-board numbers may populate
`../measurement_template.csv`. HLS estimates belong in a separate synthesis
report and must be labelled as estimates.

## Implementations and frozen interface

Every measured variant consumes one batch-1 `[17][512]` signed INT16 window
(17,408 input bytes) and produces two signed INT64 logits (16 output bytes).
Weights are fixed read-only constants. The interface must provide an AXI4-Lite
control plane and a DMA-visible input/output boundary. A Vitis kernel can use
`m_axi` for input/output buffers and `s_axilite` for control; a Vivado-IP flow
may use AXI DMA with AXI4-Stream. The selected flow, platform, tool versions,
and exact interface pragmas must be archived with every implementation.

## Frequency and PPA

Report the requested clock constraint and achieved post-route timing result;
never report the requested clock as the achieved frequency. Archive
`report_utilization`, `report_timing_summary`, and `report_power`, and record:

- LUT, LUTRAM, FF, BRAM18K, URAM, DSP48E2 counts **and** percentages;
- device/part, Vivado/Vitis/Vitis HLS/XRT version, board image and bitstream
  SHA-256;
- clock target, achieved frequency, WNS/TNS, implementation strategy;
- on-chip coefficient storage, activation buffers, and DDR/DMA buffer bytes.

The KV260 product brief identifies the starter kit as a K26 Zynq UltraScale+
MPSoC platform with 4 GB DDR4, 144 BRAM blocks, 64 URAM blocks and about 1.2K
DSP slices. Those figures are capacity context only; the paper reports actual
post-route utilisation, not a percentage inferred from parameter count.

## Latency definitions

Each implementation reports at least 1,000 batch-1 invocations after warm-up.
Use a monotonic clock or PL cycle counter and report median, p95, min, max,
number of warm-ups, and number of measured windows.

| Row | Start | Stop | Includes |
|---|---|---|---|
| `kernel_only` | PL `ap_start` acceptance / first core cycle | PL `ap_done` / final core cycle | accelerator compute and its internal buffers only |
| `dma_kernel` | MM2S command acceptance | S2MM completion for the two logits | DMA transfer plus PL compute; excludes host scheduling |
| `host_dma_kernel` | immediately before host buffer synchronisation / DMA submission | after output buffer completion and host visibility | host driver/XRT overhead, DMA and PL compute |

`kernel_only` may be calculated from measured cycles divided by achieved clock.
`host_dma_kernel` must not be called kernel latency. Sustained throughput is
the number of complete batch-1 windows divided by the elapsed timed interval;
also report real-time factor as `throughput_windows_per_s * 2.0` EEG seconds
per wall-clock second.

## Power and energy

Primary method: measure the KV260 12-V board input with a calibrated inline
power analyser or data logger during an idle interval and a continuous
accelerator interval. Record instrument model, calibration/accuracy,
sample-rate, observation duration, ambient/board temperature, board image,
and workload. Let `P_dynamic = mean(P_active) - mean(P_idle)` and calculate:

```text
dynamic_energy_per_window_mJ = 1000 * P_dynamic_W / sustained_windows_per_s
total_board_energy_per_window_mJ = 1000 * mean(P_active)_W / sustained_windows_per_s
```

Report both only if both powers are measured. Do not derive energy by
multiplying a one-off power snapshot by a median latency. On-board telemetry
or XRT electrical sensors may be archived as a secondary cross-check; they are
not silently interchangeable with input-power measurement. If only telemetry
is available, label the result as telemetry-derived board power and document
the sensor and sampling interval.

## Correctness during measurement

Before timing a bitstream, verify the committed golden input gives exact
integer logits. During a throughput run, sample at least 100 input windows
from the frozen validation bytes and record exact-logit agreement. FPGA
accuracy is reported only after scoring a declared frozen validation subset or
all 3,898 validation inputs through the same input boundary.

## Result-table language

The paper may state *"post-route KV260 implementation"* after timing closure
and *"measured on KV260"* only after the complete on-board protocol above.
It may state *"end-to-end EEG"* only after causal filtering and z-score
normalisation are part of the tested deployed path. The current H0 package is
an offline normalised-window accelerator reference.
