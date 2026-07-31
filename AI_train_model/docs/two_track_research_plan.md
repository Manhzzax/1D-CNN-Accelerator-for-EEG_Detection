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

The exact 31/7/3 rationale, cost calculation, and server commands are in
[`kernel_architecture_research_protocol.md`](kernel_architecture_research_protocol.md).
