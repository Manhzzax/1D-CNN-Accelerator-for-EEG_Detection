# V2.4 Score-Ranked Hard-Negative Decision

## Decision

`H2_c1_score_ranked_hardneg_57k` is **not promoted**. The predeclared
score-ranked, clean-interictal hard-negative intervention did not transfer the
calibration-selected `FAR <= 0.5/h` operating target consistently across all
three forward development folds and five fixed seeds. This is a development
decision, not final validation. Blocks 5 and 6 remain sealed.

## Audited Contract

- Inference graph: the unchanged C1 raw causal multiscale residual
  depthwise-separable 1D CNN, with 57,446 parameters.
- Intervention: add exactly 0.10 unique clean, train-only, score-ranked
  interictal hard negatives per ictal window; retain the source scaler and
  use a sampling multiplier of three.
- Candidate selection: C1 seed-42 train scores only; source-sampled normals
  excluded; 30-second seizure guard and within-record separation; patient-group
  round-robin selection. All three folds reached their required quota.
- Optimizer and replay: Adam, learning rate `3e-4`, weight decay `5e-4`,
  50/12/12 epoch rule, fixed seeds 7/42/123/314/2718, and the unchanged
  calibration threshold/policy grid.
- Scope: F00--F02 only. No block-5/block-6 tensor, score stream, prediction,
  quantization calibration, or FPGA measurement was created.

## H2 Results

Values are mean +/- sample standard deviation across five training seeds
*within the stated fold*. The folds are temporal development evidence and
are not pooled as 15 independent observations. Window measures use the
explicitly balanced validation windows; the clinical endpoint is the next
temporal block replay after calibration on the preceding block.

| Development fold | Balanced accuracy (%) | AUROC (%) | Temporal event sensitivity (%) | Temporal FAR/h | Median delay (s) | Seeds at FAR <= 0.5/h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F00: train 0, calibrate 1, evaluate 2 | 82.36 +/- 1.72 | 89.79 +/- 0.84 | 74.78 +/- 4.76 | 0.837 +/- 0.686 | 15.8 +/- 4.3 | 2 / 5 |
| F01: train 0-1, calibrate 2, evaluate 3 | 85.39 +/- 3.36 | 92.62 +/- 2.68 | 67.65 +/- 10.19 | 0.484 +/- 0.407 | 16.2 +/- 4.5 | 3 / 5 |
| F02: train 0-2, calibrate 3, evaluate 4 | 86.87 +/- 1.27 | 93.20 +/- 0.88 | 76.52 +/- 6.59 | 0.757 +/- 0.371 | 11.9 +/- 3.1 | 1 / 5 |

Every seed selected a feasible policy on its calibration partition; the mean
selected calibration FAR/h was 0.434, 0.447, and 0.388 in F00, F01, and F02,
respectively. The failed criterion is therefore forward temporal transfer,
not a failure to obey the frozen calibration rule.

## Direct Comparison With C1

H2 and C1 have the same inference graph, parameter count, folds, seed set,
optimizer, preprocessing, calibration grid, and temporal evaluator. The
following are H2 minus C1 using fold-level five-seed means. Negative FAR is
an improvement.

| Fold | Delta balanced accuracy (points) | Delta AUROC (points) | Delta event sensitivity (points) | Delta FAR/h | FAR-passing seeds: C1 -> H2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F00 | -1.21 | -0.61 | +18.26 | -1.812 | 0 / 5 -> 2 / 5 |
| F01 | -3.01 | -1.55 | +2.94 | +0.205 | 5 / 5 -> 3 / 5 |
| F02 | -0.02 | -0.39 | +1.74 | -0.143 | 0 / 5 -> 1 / 5 |

H2 therefore exposes a real trade-off: it raises temporal sensitivity in all
three folds and reduces mean FAR in F00 and F02, but it lowers the balanced
window measures in every fold and loses the already acceptable C1 F01
operating-point transfer. It does not satisfy a fold-consistent clinical
operating point.

## Consequences

1. Retain H2 as a preregistered, fully reported training-sampling ablation.
2. Do not tune H2's score source, quota, multiplier, separation, loss,
   optimizer, threshold grid, policy grid, or seeds after observing these
   results.
3. Do not select H2 for final training, INT16 calibration, tensor export, or
   KV260 synthesis.
4. Keep blocks 5 and 6 sealed. Any later intervention requires a new written
   protocol amendment and must not be justified by selectively following the
   favorable F00/F02 changes above.
