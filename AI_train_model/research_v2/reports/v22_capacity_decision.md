# V2.2-A Capacity Decision

## Decision

`C1_multiscale_residual_57k` is **not promoted**. The candidate did not keep
the predeclared operating target of temporal FAR <= 0.5/h across the three
development folds and five fixed seeds. This is a development conclusion,
not a final validation result. Blocks 5 and 6 remain sealed.

## Audited Contract

- Candidate: raw causal multiscale residual depthwise-separable 1D CNN.
- Parameters: 57,446 for every run, versus 5,622 for the V2.1 B2 compact
  reference.
- Frozen optimizer: Adam, learning rate 3e-4, weight decay 5e-4.
- Seeds: 7, 42, 123, 314, and 2718 in every fold.
- Data: read-only V2.1 causal 17-channel, 5-second/1-second confirmation
  caches. No block-5 or block-6 tensor, score stream, or prediction was
  created.
- Every result artifact has a matching checkpoint, protocol hash, registry
  hash, model specification, and training-only normalization tensors.

## V2.2-A Results

Values are mean +/- sample standard deviation across the five training seeds
*within the stated fold*. They are not pooled as 15 independent observations.

| Development fold | Balanced accuracy (%) | AUROC (%) | Temporal event sensitivity (%) | Temporal FAR/h | Seeds at FAR <= 0.5/h |
| --- | ---: | ---: | ---: | ---: | ---: |
| F00: train 0, calibrate 1, evaluate 2 | 83.57 +/- 1.11 | 90.40 +/- 0.78 | 56.52 +/- 11.91 | 2.648 +/- 0.723 | 0 / 5 |
| F01: train 0-1, calibrate 2, evaluate 3 | 88.40 +/- 2.22 | 94.17 +/- 1.50 | 64.71 +/- 9.07 | 0.279 +/- 0.172 | 5 / 5 |
| F02: train 0-2, calibrate 3, evaluate 4 | 86.89 +/- 1.46 | 93.59 +/- 0.81 | 74.78 +/- 10.83 | 0.900 +/- 0.108 | 0 / 5 |

The calibration procedure itself was followed: all 15 runs selected a policy
whose calibration FAR was <= 0.5/h. Transfer to the next temporal block held
only in F01. Thus the failure is temporal operating-point transfer, not a
failure to search the declared calibration grid.

## Interpretation Against V2.1 References

The V2.1 B2 compact CNN used the same folds and seed set. Its mean temporal
FAR/h was 1.486, 0.802, and 0.466 for F00, F01, and F02 respectively; C1 was
2.648, 0.279, and 0.900. C1 improves FAR only in F01. Its balanced accuracy
and AUROC do not show a consistent advantage over B2 or B4 either.

Increasing capacity by about 10.2 times therefore does not resolve the
continuous false-alarm problem under this protocol. F01 is encouraging but
cannot override F00 and F02. There is no basis to select C1 for final
training, INT16 calibration, or FPGA synthesis.

## Consequences

1. Retain C1 as a preregistered capacity ablation and report all three folds.
2. Do not tune C1's width, kernels, loss, threshold grid, or temporal policy
   after these results.
3. Do not materialize blocks 5 or 6.
4. Any next intervention must be a new written protocol amendment. The most
   plausible next hypothesis is policy-aligned, train-only hard-negative
   training, but it requires a fixed source model, train-only score cache,
   separation rule, sampling weights, and an unchanged calibration policy.
   It must not reuse temporal-evaluation recordings for mining or selection.
