# Literature-Grounded Architecture Decision

## Decision

Implement EpiSepNet-5K as a **custom, partially-unrolled, layer-fused INT16
dataflow accelerator** in the KV260 programmable logic (PL), controlled by the
ARM processing system (PS). The core uses dedicated depthwise-convolution
(DWC) and pointwise-convolution (PWC) processing engines, on-chip
ROM/BRAM-resident weights, and DMA only at the input/output boundary.

This is the selected primary direction for H0. It is not a claim that this
choice has the highest clock frequency before synthesis. The selection must be
confirmed by post-route PPA and board measurements against the baseline and
unroll variants specified below.

```text
ARM PS control
  | AXI4-Lite: start, buffer addresses, status, cycle counters
  v
AXI DMA / m_axi input (one int16[17][512] window)
  v
input BRAM -> temporal DWC (k=31) -> 51-to-32 PWC -> avg-pool(4)
                                                    |
                                               32 x 128 BRAM
                                                    v
                  refinement DWC (k=15) -> 32-to-32 PWC -> avg-pool(4)
                                                    v
                                        global average -> FC(32-to-2)
                                                    v
                                  AXI DMA / m_axi output (two int64 logits)
```

The temporal DWC and both PWC stages use the frozen per-layer scale,
requantisation, saturation, padding, and pooling rules in
[`arithmetic_contract.md`](arithmetic_contract.md). The PS supplies an
already-prepared 2 s window; this core does not yet include causal EEG
preprocessing.

## Why this is the right first architecture

### EpiSepNet-5K does not need a fully mapped CNN

The frozen graph has exactly 1,837,632 MACs per 2 s window:

| Operator | MACs/window |
|---|---:|
| Temporal DWC: 51 channels x 512 samples x k31 | 809,472 |
| First PWC: 32 outputs x 512 samples x 51 inputs | 835,584 |
| Refinement DWC: 32 channels x 128 samples x k15 | 61,440 |
| Refinement PWC: 32 outputs x 128 samples x 32 inputs | 131,072 |
| Classifier | 64 |
| **Total** | **1,837,632** |

Real-time processing of non-overlapping 2 s windows therefore requires only
0.919 MMAC/s. It is a low-throughput problem, so full spatial mapping is not
needed to meet the clinical input rate. In particular, fully unrolling just one
output time-step of the first PWC requires 32 x 51 = 1,632 products, already
larger than the KV260's approximately 1.2K DSP-slice capacity before DWC,
requantisation, and other logic are accounted for. The KV260 capacity is from
the [AMD KV260 product brief](https://www.amd.com/content/dam/xilinx/publications/product-briefs/xilinx-kv260-product-brief.pdf).

A full-map design could be examined only after the partial designs fail a
predeclared latency target. It is not the default because it spends DSP/BRAM
capacity without a real-time necessity and complicates a fair energy result.

### Layer fusion is more valuable than generic compute parallelism

Cai et al. deploy a quantised EEGNet on ARM+FPGA and explicitly use layer
fusion plus separate DWC and PWC engines. Their paper explains that without
inter-layer parallelism, feature maps are written to and read from memory
between layers; their DWC engine uses shift-register style input reuse. That is
the closest structural precedent for this depthwise-separable H0 graph.
However, their task is SSVEP BCI, not CHB-MIT seizure detection, so it informs
architecture rather than accuracy or PPA comparison.

For H0, fusion avoids materialising the 51 x 512 temporal map and 32 x 512
spatial map. The necessary activation storage is bounded mainly by:

| Buffer | Minimum payload |
|---|---:|
| Input window, 17 x 512 INT16 | 17,408 B |
| First pooled output, 32 x 128 INT16 | 8,192 B |
| Refinement DWC context, 32 x 15 INT16 | 960 B |

The table is a payload estimate, not a post-synthesis BRAM claim. Banking,
ping-pong buffering, FIFOs, and alignment change the final BRAM use. The point
is that a fused schedule can keep the graph on chip without a DDR feature-map
round trip.

### INT16 correctness and measured energy matter more than a simulated PPA

Bahr et al. show a CHB-MIT CNN deployed on a low-power RISC-V processor. They
report the implementation loss as well as deployment time and power, rather
than treating software accuracy as hardware evidence. This directly supports
our required sequence: frozen numeric contract, bit-exact replay, RTL
co-simulation, then measured latency/power/energy on the KV260.

Li et al. in *IEEE TBioCAS* demonstrate the value of parallel convolution
execution and QAT in a memristive-CNN system, but their hardware results are
crossbar simulation/layout estimates. H0 has no independent parallel CNN
branches, so it would be incorrect to copy their branch-parallel architecture
or compare its simulated area/power directly with KV260 measurements. The
transferable lesson is to expose a controlled parallelism sweep and retain the
numeric quantisation contract.

## Design-space variants

All variants below implement the *same H0 graph and integer contract*. A
variant that changes a scale, rounding mode, or model checkpoint is a new
numeric experiment and cannot be included in the H0 PPA comparison.

| ID | Architecture | Purpose | Decision rule |
|---|---|---|---|
| V0 | Sequential 48-bit exact core; feature-map buffers | Correctness and low-risk HLS/RTL baseline | Must pass all golden vectors and establish a PPA floor. |
| V1 | Fused DWC-to-PWC-to-pool schedule; line buffers | Primary core | Must match V0 logits exactly under M1a and match the M1b acceptance contract. |
| V2-U1/U2/U4/U8 | V1 plus PWC output-channel unroll factor U | Measured throughput-energy-resource sweep | Select a Pareto point using post-route PPA and measured energy, not assumed Fmax. |
| V3 | Fully mapped/dataflow extreme | Contingency only | Open only if V2 cannot satisfy the predeclared practical latency requirement. |

V0 is not a publication endpoint. It makes a functional failure diagnosable.
V1 is the expected publication architecture. V2 produces the evidence needed
to claim a hardware-aware choice rather than an arbitrary pragma setting.

## Explicitly rejected as the primary implementation

| Approach | Decision | Reason |
|---|---|---|
| Generic DPU or FINN flow | Do not use for H0 v1 | The frozen package uses signed INT16, layer-specific scales, INT32 biases, INT64 reference logits, and custom M1a rounding. Adapting it to a generic low-bit DPU changes the arithmetic contract before proving equivalence. |
| Full unroll of all operators | Do not use initially | The workload has a 2 s deadline and the first PWC is DSP-expensive when fully spatially unrolled. |
| DDR-resident weights or intermediate maps | Do not use initially | The 10,030 B coefficient package fits on chip; recurrent off-chip traffic makes energy and latency less attributable to the core. |
| Preprocessing in PL in v1 | Deferred | H0 begins after zero-phase-filtered, train-only-z-scored input. A causal filter needs a separate clinical and numeric validation protocol. |
| Patient-specific Path A model | Out of scope | H0 is the frozen shared EpiSepNet-5K reference, not the accuracy project. |

## Interface and verification decision

Use an HLS IP with an AXI4-Lite control plane and AXI4 memory-mapped data
ports for the first board integration. This gives the ARM PS explicit control
of input/output addresses, status, and measurement counters while allowing a
single window transfer through DMA or an HLS `m_axi` interface. AMD documents
`s_axilite`, `m_axi`, and `axis` as supported HLS interface modes and notes
that `ap_ctrl_none` can hinder C/RTL co-simulation. Therefore H0 should use a
handshaked control interface for the verification phase.

Vitis HLS is the preferred construction tool because it supports C simulation,
C/RTL co-simulation with the same testbench, controlled loop pipeline/unroll
directives, and export to RTL IP. Tool availability on the actual KV260 build
host remains a gate; this document does not assume a particular Vivado/Vitis
release is installed.

## Evidence hierarchy for the paper

1. **Numeric correctness:** M1a all-vector exact replay, then M1b agreement
   against the frozen INT16 emulator. One committed vector is insufficient.
2. **RTL correctness:** C/RTL co-simulation using the same vector set.
3. **Implementation evidence:** post-route timing and LUT/LUTRAM/FF/BRAM/URAM/
   DSP utilisation for V0, V1, and selected V2 variants.
4. **Board evidence:** KV260 kernel-only, DMA+kernel, and host+DMA+kernel
   latency; calibrated board-input active-minus-idle power; energy/window.
5. **Clinical boundary:** report H0 validation-window INT16 agreement and
   the already known event-level limitation. Do not call the board core an
   end-to-end streaming seizure detector while preprocessing remains offline.

This hierarchy matches the TBioCAS requirement for a real circuits/systems and
biomedical synergy: the contribution must be a measured biomedical hardware
system, not an accuracy comparison with an unimplemented CNN.

## Literature used as design evidence

| Source | What is used here | What is not transferred |
|---|---|---|
| [Bahr et al., *Biosensors*, 2021](https://pubmed.ncbi.nlm.nih.gov/34201480/) | Fixed-point embedded seizure CNN; implementation-level sensitivity, time, power, and energy reporting | Their patient-specific model, processor, 1 s boundary, and metrics are not an H0 PPA baseline. |
| [Li et al., *IEEE TBioCAS*, 2022](https://researchonline.jcu.edu.au/76604/) | Quantisation-aware co-design and controlled convolution parallelism | RRAM crossbar simulation/layout estimates and their branch-parallel CNN are not KV260 measurements. |
| [Cai et al., *Biomedical Engineering Letters*, 2024](https://pubmed.ncbi.nlm.nih.gov/39781056/) | ARM+FPGA partition, quantisation, DWC/PWC-specific PEs, layer fusion, and on-chip data reuse | SSVEP task, model, FPGA, accuracy, latency, and power cannot be directly compared. |
| [AMD Vitis HLS documentation](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html) | C-to-RTL workflow, C/RTL co-simulation, design-space directives, and RTL-IP export | Vendor capability is not evidence of H0 timing or resource use. |
| [AMD HLS interface reference](https://docs.amd.com/r/2025.1-English/ug1399-vitis-hls/pragma-HLS-interface) | AXI4-Lite control and AXI data/stream interface contract | Actual AXI bandwidth/latency must be measured on the board. |

## Next executable gate

Before writing HLS pragmas, log into the KV260 and record the installed
toolchain, board image, XRT availability, and a reproducible clock/bitstream
build path. Then execute W1 in
[`implementation_plan.md`](implementation_plan.md): deterministic M1a replay
over validation vectors. Only a passing M1a replay may progress to the
multiplier/shift M1b design.
