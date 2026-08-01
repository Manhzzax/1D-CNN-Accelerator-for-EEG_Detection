# Two-Track Research Plan: Frozen Accelerator Reference and Accuracy Development

## Decision

The work now proceeds in two isolated tracks. They share the audited CHB-MIT
corpus and source code, but they must never share a mutable checkpoint, selected
threshold, or test-based design decision.

| Track | Name | Objective | Status |
|---|---|---|---|
| A | `Deployment Reference` | Turn the existing EpiSepNet-5K INT16 package into a verified KV260 accelerator. | Frozen now |
| B | `Accuracy Research` | Develop a new hardware-feasible detector toward 95% balanced validation-window accuracy. | Experimental |

The 90.0718% result remains valid only for Track A's declared within-case
validation protocol. Track B does not replace it until it passes every
promotion gate below.

## Track A: Deployment Reference

### Immutable identity

| Item | Value |
|---|---|
| Model | `EpiSepNet-5K` |
| Source run | `run_21_raw_2s_temporal3` |
| Input boundary | 17 bipolar channels x 512 samples INT16 |
| Architecture | separable 1D-CNN with temporal kernels 31 and 15 |
| Trainable parameters | 5,013 |
| Validation balanced-window accuracy | 90.0718% |
| Validation sensitivity | 90.7645% |
| INT16 tensor package | `fpga/reference_run_21_int16/` |
| Tensor package size | 10,030 B |

No Track A action may modify this checkpoint, model manifest, normalisation
contract, expected logits, or the reported metrics. New HLS implementations are
implementation variants `H0` through `H3`, not new AI experiments.

### Track A next work

1. Implement the exact host C++ reference for the frozen INT16 tensors.
2. Replace floating requantisation ratios with documented integer
   multiplier-and-shift constants and check agreement.
3. Complete HLS C synthesis and C/RTL co-simulation.
4. Integrate AXI and measure KV260 resources, timing, latency, power, and
   energy.

Track A is sufficient for the core hardware-aware TBioCAS paper even if Track
B does not reach its accuracy target.

## Track B: Accuracy Research

### Primary target

The target is **mean balanced validation-window accuracy >= 95.0% over three
independent seeds** on the existing locked within-case chronological validation
split. It is not a test result, patient-independent result, or event metric.

For every candidate also report sensitivity, F1, AUROC, average precision,
parameter count, MAC estimate, and continuous validation event sensitivity,
FAR/h, and median delay. A model with 95% sampled-window accuracy but poor
continuous behavior is not a clinical improvement.

### Constraints for the first development cycle

- Preserve the 17-channel, 2-second, 256 Hz raw EEG input and the locked
  training/validation split. This makes the improvement interpretable against
  EpiSepNet-5K.
- Select architecture, normalisation, threshold, and alarm policy using train
  and validation only. The historical test set remains unavailable for
  selection.
- Keep the inference graph limited to Conv1D, pooling, ReLU, and linear layers
  for the first cycle. No LSTM, Transformer, GAN, or DWT frontend is permitted
  in a hardware-promotion candidate.
- Set a provisional limit of 25,000 trainable parameters and an INT16 export
  path. Larger models may be reported as AI-only ablations but cannot replace
  the KV260 target without a new feasibility decision.
- Use causal-deployable preprocessing as a separate ablation. Do not claim an
  offline zero-phase result is an end-to-end real-time result.

### Ordered experiments

| Stage | Change from frozen model | Purpose | Advance only when |
|---|---|---|---|
| B0 | Re-run frozen architecture for seeds 7 and 123 | Establish seed variance of the reference. | Metrics and data hashes reproduce. |
| B1 | Professor hierarchy: `31/7`, then `31/7/3` Lite and Full | Isolate depth and short-scale refinement while retaining the reference receptive field. | Improvement repeats on 3 seeds. |
| B2 | Width/depth sweep on the B1 winner: temporal filters 3, 4 and spatial filters 32, 48 | Locate capacity frontier under 25K parameters. | Gain is material after variance and complexity are reported. |
| B3 | Compact multiscale 15/31 depthwise branch | Test a topology change only if hierarchical refinement does not win. | Mean accuracy and event Pareto improve. |
| B4 | Causal IIR filter and train-only normalisation ablation | Measure deployable-preprocessing cost explicitly. | Accuracy loss is quantified and acceptable. |
| B5 | INT16 export and three-seed confirmation of selected candidate | Validate hardware feasibility before HLS work. | No unacceptable FP32/INT16 agreement loss. |

Run only one stage at a time. A rejected stage is retained as a negative
ablation; it is not silently overwritten.

### Promotion gate: Track B to a new accelerator reference

A Track B candidate receives a new model identity, for example
`EpiSepNet-MK`, and may become a second accelerator target only if all of the
following are true:

1. Mean validation balanced-window accuracy is at least 95.0% over three
   seeds, with each seed and standard deviation reported.
2. Validation sensitivity and F1 do not decrease materially from the frozen
   reference without an explicit justification.
3. Its continuous validation event sensitivity/FAR/h point is reported using a
   threshold selected only on validation.
4. The graph exports to integer arithmetic and has at most 25,000 parameters.
5. Estimated storage, MACs, and input-buffer requirements are recorded before
   commissioning a second HLS implementation.

Until then, its correct label is **accuracy research candidate**, not
"best model" and not a replacement for EpiSepNet-5K.

## Artifact and naming rules

```text
outputs/
  run_21_raw_2s_temporal3/          Immutable historical source run
  run_40_seed07_reference/          Track B0
  run_41_seed123_reference/         Track B0
  run_42_temporal_kernel_47_s42/    Track B1 example
  run_...                            One change per run

fpga/
  reference_run_21_int16/           Track A immutable tensor package
  candidate_<model_name>_int16/     Created only after the promotion gate
```

Each Track B run must save `model_spec.json`, `training_summary.json`,
`validation_window_metrics.json`, feature/preprocessing metadata, data-split
summary, and its source commit. A run is pushed to `server_results/` only after
training and validation have completed; a test report is never required for
model selection.

## Interpretation

Track A answers: **Can a frozen 5K-parameter EEG CNN be implemented and
measured rigorously on KV260?**

Track B answers: **Can a still-deployable Conv1D model achieve a stronger AI
result without sacrificing reproducibility or a credible FPGA path?**

These are complementary. Track A supplies the publishable hardware evidence;
Track B can strengthen the model-quality contribution, but must not delay
hardware verification.

### Accuracy-Only Campaign

Until an accuracy candidate is selected, Track B uses **balanced validation
window accuracy** as its primary screen metric. Event FAR, causal preprocessing,
INT16 export, and test evaluation are intentionally deferred. They are not
discarded; they are promotion gates after the accuracy search.

The target is a three-seed mean validation accuracy of at least 95%, not an
isolated high seed. A single seed-42 run is used only to screen a controlled
change. A candidate advances only when its best validation accuracy is at
least 0.5 points above the R2 seed-42 reference (91.175%), then it must be
confirmed with seeds 7 and 123.

The ordered search is:

1. R2 capacity axes independently: temporal filters per channel `3 -> 4`,
   then spatial width `32 -> 48`.
2. Combine the two only if an individual axis passes the seed-42 screen.
3. On the best capacity, sweep the first temporal kernel `15`, `31`, `47`,
   and `63` with the `7/3` refinement fixed.
4. Test the parameter-matched residual `31/7/3` hierarchy. It adds only
   identity additions, so it isolates information preservation from capacity.
5. Test one separate multiscale `15+31` architecture after the hierarchy
   family; it is not mixed into a kernel result because it changes topology.
6. Tune optimizer/regularization only for the best topology: AdamW versus
   Adam, weight decay, dropout, then learning-rate/schedule. Patience is not a
   primary accuracy factor unless a learning-rate-reduced run reaches a later
   lower validation-loss basin.

Every candidate remains validation-only and uses `chbmit_prepared_raw_2s_v2`.
No historical test recording may be inspected for architecture selection.

#### Capacity Screen Result

On seed 42, R2 Lite `m3/f32` reaches 91.175% accuracy with 4,917 parameters.
Increasing temporal filters to `m4/f32` gives 90.328% with 6,022 parameters;
increasing spatial width to `m3/f48` gives 90.046% with 7,301 parameters.
Neither reaches the 91.675% promotion threshold. Both have higher train than
validation accuracy at their selected checkpoint, so the next controlled axis
is temporal kernel length, not the combined `m4/f48` model.

#### Temporal-Kernel Screen Result

With the R2 Lite `7/3`, `m3/f32` topology fixed, the first temporal kernel
shows a non-monotonic accuracy response on seed 42:

| First kernel | Local receptive field | Parameters | Accuracy | AUROC | F1 | Sensitivity |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 336 ms | 4,101 | 89.277% | 95.774% | 89.012% | 86.865% |
| 31 | 398 ms | 4,917 | **91.175%** | **96.645%** | **91.102%** | **90.354%** |
| 47 | 461 ms | 5,733 | 90.457% | 96.385% | 90.287% | 88.712% |
| 63 | 523 ms | 6,549 | 89.302% | 95.509% | 89.104% | 87.481% |

Kernel 47 has a slightly lower cross-entropy validation loss than kernel 31,
but lower fixed-threshold accuracy, F1, and sensitivity. Since this campaign
optimizes accuracy, kernel 31 remains the hierarchy reference. The result
rejects a monotonic "longer temporal context is better" hypothesis. The next
architecture hypothesis should add complementary short/medium paths through a
compact multiscale or residual topology, rather than increasing one kernel or
flat channel capacity.

#### Parameter-Matched Residual Screen Result

The residual `31/7/3` topology preserves the R2 Lite convolution graph and
adds identity residual additions before the second and third pooling operations.
It has the same 4,917 trainable parameters, so this is a topology test rather
than a capacity increase. On seed 42 it was selected at epoch 15 by the exact
minimum validation loss and produced the following result:

| Model | Parameters | Accuracy | AUROC | F1 | Sensitivity |
|---|---:|---:|---:|---:|---:|
| R2 Lite `31/7/3` | 4,917 | **91.175%** | **96.645%** | **91.102%** | **90.354%** |
| Residual `31/7/3` | 4,917 | 89.302% | 95.937% | 88.989% | 86.455% |
| Residual minus R2 Lite | 0 | -1.873 pp | -0.708 pp | -2.113 pp | -3.899 pp |

The residual path is rejected without a multi-seed confirmation: it is well
below the seed-42 promotion threshold and degrades every primary discriminative
metric. This is useful negative evidence that skip additions alone do not
improve this compact raw-EEG hierarchy.

#### Compact Multiscale Screen Result

The compact depthwise multiscale `15+31` model was then screened on the same
seed-42 raw `17x512` validation protocol. It has 3,636 parameters, so its two
parallel temporal paths do not provide an accuracy gain merely by increasing
model capacity. Checkpoint epoch 28 is the exact minimum validation-loss epoch.

| Model | Parameters | Accuracy | AUROC | F1 | Sensitivity |
|---|---:|---:|---:|---:|---:|
| R2 Lite `31/7/3` | 4,917 | **91.175%** | **96.645%** | **91.102%** | **90.354%** |
| Multiscale `15+31` | 3,636 | 87.891% | 95.003% | 87.916% | 88.096% |
| Multiscale minus R2 Lite | -1,281 | -3.284 pp | -1.642 pp | -3.186 pp | -2.258 pp |

This topology is rejected without additional seeds. Its late low-loss checkpoint
does not rescue its accuracy, F1, or sensitivity. With both topology changes
rejected, R2 Lite remains the architecture reference for the accuracy campaign.
The next isolated screen is **AdamW with the same learning rate and weight
decay** as R2 Lite. This tests decoupled regularisation without changing
architecture, data, batch composition, schedule, or early-stopping rule.

Increasing early-stopping patience is not the next experiment. In the R2 Lite
seed-42 run, the scheduler reduced the learning rate only after epoch 26; its
following validation losses (0.2540 and 0.2886) did not improve the selected
0.2474 loss at epoch 23. Therefore a longer tail would not have recovered a
better selected checkpoint in that run.

#### AdamW Optimizer Screen Result

R2 Lite was re-trained with AdamW at the identical learning rate (`1e-3`) and
weight decay (`1e-4`); data, seed, architecture, batch sampling, scheduler, and
checkpoint rule were unchanged. The AdamW checkpoint at epoch 28 is the exact
minimum validation-loss epoch.

| Optimizer | Accuracy | AUROC | F1 | Sensitivity | Precision |
|---|---:|---:|---:|---:|---:|
| Adam (R2 Lite) | **91.175%** | **96.645%** | **91.102%** | **90.354%** | 91.862% |
| AdamW, same decay | 90.713% | 96.482% | 90.474% | 88.199% | **92.869%** |
| AdamW minus Adam | -0.462 pp | -0.163 pp | -0.628 pp | -2.155 pp | +1.006 pp |

AdamW is rejected for the accuracy objective: its modest precision gain comes
with lower accuracy and substantially lower sensitivity. Its selected
train-to-validation accuracy gap (4.038 pp) is also larger than Adam's
3.604 pp. The next experiment is the predeclared **5-second context ablation**
of the unchanged Adam R2 Lite model. It requires a new prepared dataset and is
reported separately because its full-ictal eligibility and window population
differ from the 2-second protocol.

#### Five-Second Context Result

The unchanged Adam R2 Lite `31/7/3` was trained on a separate raw `17x1280`
dataset with seeds 42/7/123. It retains 4,917 parameters and obtains balanced
validation accuracy 92.830%, 92.132%, and 94.280%, respectively: **93.081% +/-
1.096%**. The matched 2-second R2 mean is 90.200% +/- 0.850%; the paired
accuracy direction is +2.881% +/- 1.585%. The 5-second population contains
fewer fully-ictal windows, so this is evidence that context helps, not a direct
replacement of the 2-second benchmark. It is the current accuracy candidate and
still falls below the internal 95% mean target.

The next controlled candidate retains this exact 5-second protocol and adds two
residual depthwise `k=3` temporal convolutions at dilations 4 and 8 after the
second pooling operation. This `dilated_hierarchical_separable_1dcnn` increases
the local receptive field from about 398 ms to about 1.9 s, adds only 320
parameters (5,237 total), and keeps a Conv1D-only FPGA path. It is not the
previous rejected residual topology: the new blocks specifically add dilated
temporal context after downsampling.

The exact 31/7/3 rationale, cost calculation, and server commands are in
[`kernel_architecture_research_protocol.md`](kernel_architecture_research_protocol.md).
