# Research Argument Map

This map converts the reference corpus into claims that are appropriate for a
hardware-aware biomedical AI paper. It also defines what evidence we still need
from KV260.

## Claim 1: A compact raw-EEG CNN is a valid accelerator target

**Argument.** Raw multichannel temporal convolutions avoid an obligatory DWT,
STFT, CSP, or hand-engineered-feature hardware frontend. Depthwise and
pointwise convolutions make channel mixing explicit and give direct control over
DSP, BRAM, and input-buffer costs.

**Sources.** `M01` EEGNet and `M02` Deep ConvNets establish compact EEG CNN
operators; `A04` and `A05` are feature-heavy DWT/recurrent alternatives;
`A13` is a compact depthwise-separable CHB-MIT comparator; `H02` demonstrates
raw low-channel hardware-oriented seizure monitoring.

**Paper-safe wording.** The EpiSepNet-5K architecture is selected for a
controlled accuracy-efficiency experiment, not because it is presumed to
outperform every feature-engineered or recurrent detector.

**KV260 evidence required.** MAC schedule, on-chip storage placement, actual
post-route resources, and latency for the declared `17 x 512` INT16 boundary.

## Claim 2: Window accuracy alone is insufficient for seizure monitoring

**Argument.** Long continuous EEG is strongly imbalanced. A detector can have
high sampled-window accuracy while missing seizures or raising clinically
unusable false alarms. Event sensitivity, FAR per interictal hour, and onset
delay therefore address a different and necessary decision layer.

**Sources.** `D03` explicitly frames continuous data, cross-subject variation,
and event detection as overlooked CHB-MIT issues. `A02`, `H02`, and `H03`
report event-oriented outcomes. `M05` motivates reporting accuracy alongside
system metrics rather than as a substitute for them.

**Paper-safe wording.** The `90.07%` number is balanced window accuracy under
the locked within-case validation protocol. It is neither clinical event
sensitivity nor patient-independent accuracy.

**KV260 evidence required.** Hardware-in-the-loop score agreement and the
unchanged event-policy result using causal timestamps.

## Claim 3: Quantization is an arithmetic contract, not merely a smaller file

**Argument.** A valid fixed-point accelerator must define input/weight/activation
types, accumulator width, rounding, saturation, and requantization. Floating
scale ratios in a software emulator are not a fully integer implementation.

**Sources.** `M03` supplies the integer-only inference principle. `H01`,
`H03`, and `H04` demonstrate that deployment results must be measured on a
specific physical target.

**Paper-safe wording.** The current INT16 package is a frozen software
reference. Its FP32/INT16 agreement is deployment-readiness evidence; it is not
an FPGA performance claim until integer multiplier/shift arithmetic and HLS
co-simulation pass.

**KV260 evidence required.** Exact C reference (M1a), integer multiplier/shift
reference (M1b), C/RTL agreement, then board-level agreement on the frozen
validation set or a declared representative subset.

## Claim 4: The contribution is a measured trade-off, not an unmatched accuracy claim

**Argument.** Published seizure studies use incompatible label definitions,
channels, windows, balance schemes, subjects, and splits. High numbers from
random-window or patient-specific experiments cannot establish superiority over
a within-case chronological protocol, and prediction is a separate task.

**Sources.** `D03`, `D04`, `A01`, `A02`, `A04`, `A08`, `A15`, and `A17` show
different protocol/task choices. `H01` and `H03` establish a TBioCAS-style
model-hardware co-design context.

**Paper-safe wording.** We compare deployment scale and measurement practice
only when task/protocol differences are disclosed. We do not claim a universal
CHB-MIT rank.

**KV260 evidence required.** A table with the same model, precision, clock,
measurement boundary, iterations, resources, latency, power, energy, and
accuracy agreement for H0--H3.

## Claim 5: TBioCAS fit

**Argument.** The closest papers in the target journal couple seizure
algorithms with an actual platform and report a system-oriented trade-off:
memristive CNN co-design (`H01`), low-FAR MCU deployment (`H02`), and FPGA SNN
deployment (`H03`). Broader TBioCAS reviews (`T01`--`T04`) emphasize the same
co-design and embedded-system framing.

**Paper-safe wording.** This work targets an FPGA evidence chain from frozen
EEG tensors through quantized arithmetic, RTL verification, and KV260
measurement. It does not claim medical-device validation, clinical safety, or
an implantable ASIC.

## Excluded claims until new evidence exists

- Clinical readiness or a universal acceptable FAR threshold.
- Patient-independent superiority.
- FPGA speed, energy, or resource superiority before G4--G6 evidence exists.
- Comparison against a paper whose original protocol or parameter count has
  not been verified from its primary source.
