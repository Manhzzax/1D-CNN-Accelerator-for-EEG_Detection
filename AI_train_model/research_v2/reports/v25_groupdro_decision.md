# V2.5 Patient-Group GroupDRO Decision

## Decision

`G1_c1_groupdro_balanced_57k` is **not promoted**. The predeclared
source-patient-group balanced sampler and GroupDRO objective did not achieve
the required forward temporal operating point consistently. All three
development folds and all five fixed seeds were completed before this decision.
This is a development conclusion, not final validation. Blocks 5 and 6 remain
sealed.

## Audited Contract

- Inference graph: unchanged C1 raw causal multiscale residual
  depthwise-separable 1D CNN, with 57,446 parameters and input `[1, 17, 1280]`.
- Training-only intervention: equal observed `(class, source patient group)`
  sampling strata plus source-patient GroupDRO with exponentiated-gradient
  `eta=0.01`. Patient-group metadata was never an inference input.
- Frozen training: Adam, learning rate `3e-4`, weight decay `5e-4`, dropout
  `0.25`, 50/12/12 training rule, and seeds `7, 42, 123, 314, 2718`.
- Frozen data/evaluation: read-only V2.1 causal 17-channel, 5 s/1 s caches;
  calibration-only threshold/policy selection at FAR <= 0.5/h; one next-block
  temporal replay; F00--F02 only.
- Scope boundary: no block-5/block-6 tensor, score stream, prediction,
  quantization calibration, tensor export, or FPGA measurement was created.

## G1 Results

Values are mean +/- sample standard deviation across five training seeds
*within the stated fold*. The folds are temporal development evidence and are
not pooled as 15 independent observations. Window measures use explicitly
balanced validation windows; the system endpoint is one replay on the next
temporal block after calibration on its predecessor.

| Development fold | Balanced accuracy (%) | AUROC (%) | Temporal event sensitivity (%) | Temporal FAR/h | Median delay (s) | Seeds at FAR <= 0.5/h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F00: train 0, calibrate 1, evaluate 2 | 84.20 +/- 2.01 | 91.34 +/- 1.16 | 61.74 +/- 9.91 | 2.311 +/- 0.832 | 20.2 +/- 7.5 | 0 / 5 |
| F01: train 0-1, calibrate 2, evaluate 3 | 86.57 +/- 2.45 | 93.12 +/- 1.63 | 57.06 +/- 12.23 | 0.429 +/- 0.141 | 26.4 +/- 3.4 | 3 / 5 |
| F02: train 0-2, calibrate 3, evaluate 4 | 84.98 +/- 1.51 | 91.84 +/- 1.60 | 73.04 +/- 4.76 | 1.003 +/- 0.127 | 15.8 +/- 2.9 | 0 / 5 |

Every run selected a feasible calibration policy. Mean calibration FAR/h was
0.484 in F00, 0.468 in F01, and 0.473 in F02. The failure is consequently
forward operating-point transfer, rather than an inability to satisfy the
frozen calibration rule.

## Direct Comparison With C1

G1 and C1 use the same inference graph, parameter count, folds, seeds,
optimizer, preprocessing, calibration grid, and temporal evaluator. The table
shows G1 minus C1 using fold-level five-seed means. Negative FAR is an
improvement.

| Fold | Delta balanced accuracy (points) | Delta AUROC (points) | Delta event sensitivity (points) | Delta FAR/h | FAR-passing seeds: C1 -> G1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F00 | +0.63 | +0.94 | +5.22 | -0.337 | 0 / 5 -> 0 / 5 |
| F01 | -1.83 | -1.05 | -7.65 | +0.150 | 5 / 5 -> 3 / 5 |
| F02 | -1.91 | -1.75 | -1.74 | +0.103 | 0 / 5 -> 0 / 5 |

G1 slightly improves the F00 window metrics and sensitivity, but neither G1
nor C1 transfers the target to F00/F02. More importantly, G1 degrades the only
C1 fold with stable FAR transfer: F01 loses two FAR-passing seeds and 7.65
percentage points of mean event sensitivity. This is not a fold-consistent
improvement.

## Predeclared Gate Check

1. **At least 4/5 temporal FAR passes in every fold:** failed: F00 `0/5`, F01
   `3/5`, F02 `0/5`.
2. **Event sensitivity no worse than C1 by 5 points in each fold:** failed in
   F01 (`-7.65` points); passed in F00 and F02.
3. **Balanced accuracy and AUROC no worse than C1 by 2 points:** passed in all
   folds, although F02 balanced accuracy is close to the limit (`-1.91`
   points).
4. **No final GroupDRO weight above 0.50 in more than one seed per fold:**
   passed. The maximum final source-group weight across all 15 runs was 0.118.

## Consequences

1. Retain G1 as a fully reported, training-only patient-group robustness
   ablation.
2. Do not tune the GroupDRO eta, sampler, group definition, architecture,
   optimizer, regularization, seed list, calibration grid, or temporal policy
   after observing these results.
3. Do not select G1 for final training, INT16 calibration, tensor export, or
   KV260 synthesis.
4. Keep blocks 5 and 6 sealed. Any next intervention requires a separate
   written protocol amendment and must not be justified by the favorable F00
   window metrics alone.
