# V2.2-A Capacity Study

## Purpose

V2.1 completed its predeclared comparison and found no candidate that kept the
calibration-selected operating point at FAR <= 0.5/h across all three
forward-time folds. V2.2-A tests one hardware-bounded explanation: a 5K
parameter classifier may not have enough multiscale temporal capacity to keep
interictal artifact scores separated from ictal scores in continuous replay.

This is a **development-only** study. Its inputs are the already-consumed V2.1
F00--F02 partitions. It is not an independent confirmation and it must never
be reported as a final test result. Blocks 5 and 6 remain unopened.

## Frozen Candidate

`C1_multiscale_residual_57k` is a raw, causal 1D CNN with 57,446 trainable
parameters. It uses depthwise 15- and 47-sample input branches, pointwise
mixing, residual depthwise-separable 7/5/3 refinement, and dilations 1/2/4.
It contains only Conv1D, depthwise Conv1D, pointwise Conv1D, BatchNorm, ReLU,
AvgPool, residual add, global average pool, and a linear classifier.

The candidate is below the 100K-parameter project limit and uses the same
`[1, 17, 1280]` raw-EEG interface as V2.1. The exact model contract, learning
rate (3e-4), weight decay (5e-4), and five seeds are in
`configs/candidate_registry_v2_2.json`.

## Why This Is the Next Controlled Test

- V2.1 B2/B4 show that sampled window quality can be near 90% balanced
  accuracy while a calibration-selected policy has unstable future FAR.
- A larger multiscale residual CNN is a controlled capacity change, not a
  change of task, label definition, causal filter, montage, window duration,
  or alarm rule.
- The model remains a hardware-relevant raw 1D CNN. Its operations are within
  the fixed INT16 accelerator interface, although FPGA synthesis is prohibited
  until a later final protocol freezes a candidate.

This rationale is consistent with the event-level and continuous-replay
requirements captured in local evidence cards `D03` (Ali et al.), `A02`
(Chung et al.), the compact EEG-CNN design rationale in `M01` (EEGNet), and
the hardware measurement boundary in `H01`--`H05`.

## Execution

1. Run `tools/prepare_v22_development_caches.sh`. It only validates and reuses
   the V2.1 causal caches; it never reads or writes a test cache.
2. Run `tools/train_v22_capacity_candidate.sh 00`, then `01`, then `02`.
   Each command trains all five predeclared seeds.
3. Package each completed fold with `tools/package_v22_runs.sh`.
4. Summarize seed variation inside each fold and fold variation separately.
   Do not pool fold-by-seed runs as independent patients.

## Decision Boundary

The V2.2-A results may support a later written freeze decision only if the
candidate is clinically consistent across development folds. They cannot
authorize final training, quantization calibration, hardware export, block-5
access, or block-6 access. A separate final protocol must be authored and
approved after the V2.2 decision is recorded.

Policy-aligned hard-negative mining is deliberately excluded from V2.2-A.
It changes the training distribution and requires its own source-model,
train-only score-cache, sampling-weight, and threshold-context contract. It
may be proposed only as a later protocol amendment; it must not be added after
seeing a V2.2-A result.
