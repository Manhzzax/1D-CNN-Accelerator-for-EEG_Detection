# Critical Appraisal: EpiSepNet-R2-5K Five-Second Candidate

## Scope

The current candidate is the raw `hierarchical_separable_1dcnn` with temporal
kernels `31/7/3`, 17 canonical bipolar channels, a 5 s (`17 x 1280`) input,
train-only channel z-score, and 4,917 trainable parameters. Runs `60`, `61`,
and `62` use one locked within-case chronological train/validation split with
seeds `42`, `7`, and `123`.

The development result is **93.081% +/- 1.096% balanced validation-window
accuracy**, 92.714% +/- 1.127% sensitivity, 98.154% +/- 0.425% AUROC, and
93.055% +/- 1.101% F1. It is not yet a clinical, patient-independent,
real-time, or FPGA result.

## What Is Genuinely Strong

1. **Weight compactness is real.** The model has 4,917 FP32 weights, or
   19,668 B before metadata. INT16 weights would occupy 9,834 B before scales,
   biases, and packaging. This is 3.7x smaller than the 18,024-parameter
   LMPSeizNet and 12.5x smaller than the 61,218-parameter PSD-LW-DCN in the
   literature context.
2. **The design has an EEG-specific mechanism.** Grouped temporal convolution
   learns three temporal filters per channel; the following `1x1` convolution
   learns cross-channel mixtures. The first projector has
   `17 x 3 x 31 + 51 x 32 = 3,213` weights, compared with
   `17 x 32 x 31 = 16,864` for a direct standard `17 -> 32` convolution with
   the same temporal kernel.
3. **The 5 s direction is replicated.** Matched seed accuracies move from
   91.175%, 89.815%, and 89.610% at 2 s to 92.830%, 92.132%, and 94.280% at
   5 s. All three paired directions are positive.
4. **Known waveform-level train/validation duplication is avoided.** The split
   is planned at recording level before window extraction. Train-only z-score
   does not fit normalisation statistics on validation EEG.

Depthwise and separable convolution is a credible compact-EEG choice: EEGNet
introduced it to retain temporal and spatial feature learning with fewer
parameters. [Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c)

## Threats to Validity

| Severity | Issue | Consequence | Required control |
|---|---|---|---|
| Critical | Within-case, not patient-held-out | All 23 patient groups occur in development training and validation. Later recordings share patient physiology, montage, artifacts, and seizure morphology. | Claim within-case development only. Freeze the model, then perform one patient-group-disjoint causal test. |
| Critical | Balanced sampled-window accuracy | Validation has exactly 1,862 ictal and 1,862 normal windows. Real continuous EEG is overwhelmingly interictal, so 50% prevalence accuracy does not predict FAR/h. | Do not call 93.081% clinical accuracy. Report event sensitivity, FAR/h, and delay. |
| High | Different 5 s population | Positives are fully inside seizures. The 5 s validation artifact has 1,862 ictal windows versus 1,949 at 2 s. | Report a new context protocol, not a direct same-example 2 s gain. Stratify by seizure duration. |
| High | Overlap reduces sample independence | A 5 s window at 1 s stride shares 80% waveform with its neighbour. 3,724 windows are not 3,724 independent observations. | Bootstrap by recording/patient, never by overlapping windows. |
| High | Offline filter risk | The repository default is `zero_phase`, which uses future samples. The saved run artifacts do not independently retain the preparation summary that proves the effective filter mode. | Treat it as offline unless the artifact proves causality. Reproduce frozen candidate with causal IIR before streaming/KV260 claims. |
| High | Validation-selection optimism | Many topology, optimiser, kernel, and duration screens inspected one validation cohort. | Preserve the ablation ledger and use a final never-observed patient-group test after freezing. |
| Medium | Three seeds are not three cohorts | Seeds measure initialisation and minibatch randomness only; `n=3` gives imprecise uncertainty. | Repeat frozen winner and baseline at five seeds; separately report patient/recording uncertainty. |
| Medium | Weights do not equal accelerator cost | Weights omit input/activation memory, preprocessing, DMA, and dataflow overhead. | Report MACs, activation memory, INT16 agreement, latency, PPA, and energy on KV260. |
| Medium | No R2 5 s continuous evaluation | Isolated full-ictal windows can score well while causing unusable alarms on interictal EEG. | Select policy only on continuous validation and lock it before a final test. |
| Medium | CHB-MIT cohort and labels | Audit establishes 686 EDF and 198 primary intervals but does not create external or clinician-re-reviewed validation. | State annotation precedence and do not imply clinical generalisation. |

The need to report segment and continuous event evidence together is visible in
the CHB-MIT study of Chung et al., which reports segment metrics alongside
event sensitivity, FAR/h, and latency. [Chung et al.,
2024](https://doi.org/10.3389/fneur.2024.1389731) A cross-subject continuous
CHB-MIT study reports substantially lower event sensitivity when the task is
made more realistic. [Ali et al., 2024](https://doi.org/10.1098/rsos.230601)

## Does 4,917 Parameters Mean FPGA-Ready?

**No. It is promising, not proof.** The 5 s graph has approximately 4,519,744
convolution/classifier MACs per inference window. At a 1 s stride, a naive
full recomputation is about 4.52 MMAC/s. The simple non-streamed INT16 tensors
are:

| Tensor | Shape | INT16 memory |
|---|---:|---:|
| Input window | `17 x 1280` | 43,520 B (42.5 KiB) |
| First temporal activation | `51 x 1280` | 130,560 B (127.5 KiB) |
| First spatial activation | `32 x 1280` | 81,920 B (80.0 KiB) |
| Weights only | 4,917 values | 9,834 B (9.6 KiB) |

Streaming or tiling can avoid holding whole activations in BRAM, but that is a
dataflow hypothesis until HLS and board measurement. A 5 s input raises input
buffering and first-layer work 2.5x relative to the frozen 2 s model while the
weight count remains unchanged. Parameters must therefore not stand in for
latency, BRAM, power, or energy.

## Why a Small Model Can Work Here

1. **Appropriate inductive bias:** channel-wise temporal filters followed by
   spatial mixing match multichannel EEG structure.
2. **Simplified development task:** complete ictal windows are compared with
   interictal windows at least 30 s from seizures, excluding onset/offset and
   peri-ictal ambiguity.
3. **More temporal evidence:** 5 s supplies more rhythmic morphology without
   additional weights, but also incurs a 5 s decision-availability delay.
4. **Capacity regularisation:** 4,917 weights, dropout 0.25, train-only
   normalisation, and global average pooling can reduce memorisation. Failed
   width/depth screens support this locally but do not prove global optimality.

A compact model can still underfit new patients, artifacts, rare morphologies,
missing channels, and onset transitions. Global average pooling may also lose
critical timing information. High within-case window accuracy cannot prove
that the network learned general seizure features rather than patient-specific
shortcuts.

## Allowed Claim and Research Gates

Allowed now:

> Under a locked within-case chronological CHB-MIT validation protocol with
> balanced fully ictal/interictal 5 s windows, raw EpiSepNet-R2-5K achieved
> 93.081% +/- 1.096% validation accuracy across three training seeds using
> 4,917 parameters.

Not allowed now: `95% achieved`, state of the art, patient-independent,
clinical, low-FAR, real-time, low-power, or KV260 deployment claims.

Before a headline paper result:

1. Meet the professor-mandated internal gate: a validation-loss-selected
   seed-42 checkpoint >=95.0%, then a mean >=95.0% across seeds 42/7/123.
   Maximum epoch accuracy or a maximum seed does not qualify.
2. Complete the prespecified dilated 5 s screen; if it wins, repeat it at
   seeds 42/7/123 rather than promoting one seed.
3. Freeze the winner and R2 baseline, then run five seeds and report every run.
4. Measure the frozen candidate with causal-IIR preprocessing.
5. Run continuous validation, lock threshold/policy at the FAR target, and
   then evaluate once on a patient-group-disjoint test.
6. Export INT16 and report bit agreement. Measure KV260 clock, latency,
   throughput, BRAM/DSP/LUT/FF, board power, energy/window, and buffers.

TRIPOD+AI requires transparent reporting of data, development, evaluation, and
uncertainty for clinical prediction-model studies. [TRIPOD+AI,
2024](https://www.bmj.com/content/385/bmj-2023-078378)
