# V2.5 Patient-Group Robust Training Plan

## Question

Can a source-patient-group robust training objective improve forward temporal
operating-point stability without changing the deployed C1 raw 1D-CNN or its
KV260 tensor interface?

V2.5 addresses a failure that remains after V2.2 capacity scaling and V2.4
score-ranked hard negatives: calibration FAR is feasible in every run, but its
next-block transfer is unstable and false alarms concentrate in different
patient groups across folds. It is a development-only ablation, not a final
test, patient-independent result, or hardware result.

## Frozen Candidate G1

- **Inference graph:** unchanged C1 multiscale residual depthwise-separable
  raw 1D-CNN, 57,446 parameters, NCT input `[1, 17, 1280]`.
- **Training sampler:** equal probability across every observed `(class,
  source patient group)` stratum. Sampling is with replacement and preserves
  the original per-window importance weights.
- **Objective:** source-patient GroupDRO with exponentiated-gradient update
  `eta=0.01`. Patient-group identifiers are training metadata only; they are
  neither network input nor an exported tensor.
- **Regularization:** retain C1 dropout `0.25`, Adam learning rate `3e-4`,
  weight decay `5e-4`, and 50/12/12 early-stopping contract. This is required
  because GroupDRO alone can overfit its worst source group.
- **No other change:** causal preprocessing, labels, train z-score, window
  sampling population, calibration grid, alarm policies, refractory interval,
  and the five training seeds are unchanged.

## Data and Evaluation Boundary

V2.5 reuses only the existing read-only V2.1 F00--F02 cache. The three folds
remain `train -> calibration -> temporal evaluation` as `0 -> 1 -> 2`,
`0-1 -> 2 -> 3`, and `0-2 -> 3 -> 4`. Blocks 5 and 6 must not be prepared,
scored, or inspected.

For each seed, select the threshold and temporal vote policy using calibration
recordings only at observed FAR <= 0.5/h. Replay this exact policy once on the
next temporal block. Continuous replay reports micro event sensitivity, FAR/h,
delay, patient-group cluster bootstrap intervals, and each patient group's
detected events, total events, false alarms, and interictal hours.

## Execution

1. Run `prepare_v25_group_robust_caches.sh` once to verify the existing cache
   hashes, split set, five-second shape, and absence of a test artifact.
2. Run `train_v25_group_robust.sh 00`, then `01`, then `02`. Each invocation
   runs exactly seeds `7, 42, 123, 314, 2718` and refuses existing run IDs.
3. Package only the checkpoint and reproducibility artifacts using the exact
   command printed by the train script. Raw EEG, prepared caches, and score
   streams remain ignored locally.
4. Report seed variation within each fold and temporal-fold variation
   separately. Do not treat 15 fold-by-seed runs as independent patients.

## Decision Rule

G1 is not promotable unless all of the following are met without changing this
protocol:

1. At least four of five seeds satisfy temporal FAR <= 0.5/h in **each** of
   F00, F01, and F02.
2. Each fold's mean event sensitivity is no more than five percentage points
   below the corresponding C1 mean.
3. Each fold's balanced accuracy and AUROC are no more than two percentage
   points below C1's corresponding fold mean.
4. The final GroupDRO weights are reported and no one source group holds more
   than 0.50 weight in more than one seed per fold. This is a stability gate,
   not a training intervention.

Passing these development gates still does not authorize block-5/block-6
access, INT16 activation calibration, tensor export, or FPGA synthesis. A
separate final protocol is required.
