# V2.1 Execution Plan

## Status and boundary

V2.1 starts on `research/v2.1-patient-forward`. All V2.0 outputs are pilot
archive evidence. No model, threshold, post-processing policy, or hardware
claim may be transferred as a V2.1 result without being retrained under this
protocol.

The claim is a shared-model, within-patient forward-chaining evaluation. It is
not patient-independent validation and it is not an external clinical trial.

## Unit of independence and chronology

CHB-MIT contains 24 case IDs but V2.1 reports 23 patient groups. `chb01` and
`chb21` are the two ordered sessions of `subject_01_21`; they are not two
independent observations. EDF header timestamps audit session order, followed
by the recording order inside each session.

Each patient group is partitioned into seven contiguous, duration-balanced
recording blocks. EDF recordings are never fragmented. The audit exports, for
every partition: recordings, total EEG hours, non-ictal replay hours, seizure
events, contributing patient groups, and the per-patient seizure distribution.

## Confirmation stage

| Fold | Training blocks | Calibration block | Temporal evaluation block |
| --- | --- | --- | --- |
| F00 | 0 | 1 | 2 |
| F01 | 0-1 | 2 | 3 |
| F02 | 0-2 | 3 | 4 |

Blocks 5 and 6 have no role in model selection. The gate requires the union
of calibration and temporal evaluation to contain at least 20 seizures, five
seizure-contributing patient groups, and 24 non-ictal replay hours. The audit
fails rather than silently changing a split.

For each fold, the classifier is trained on train only. The scaler is fit on
train only. Threshold and temporal policy are selected only on calibration
from the predeclared 0.850-0.999 grid and eight vote policies, subject to
observed calibration FAR <= 0.5/h. The selected policy is applied once to the
temporal evaluation block. It is not re-selected there.

B0 and B1 are fixed sanity baselines with seed 42. B2 and B4 are retrained on
the five predeclared seeds 7, 42, 123, 314, and 2718. Within each temporal fold
report mean and standard deviation across seeds. Report fold trends separately.
Never pool fold-by-seed outcomes as independent patients. Event confidence
intervals resample patient groups; FAR also receives an exact Poisson interval.

## Final stage

Only after an explicit written decision freezes candidate, architecture,
learning rate, weight decay, all five seeds, policy grid, INT16 policy, and
hardware interface may the final command materialize block 6. The freeze stores
the protocol, final-manifest, and decision hashes. A marker prevents the same
freeze from materializing the sealed partition twice.

Final training uses blocks 0-4; block 5 calibrates early stopping and policy;
block 6 is evaluated once in one fixed five-seed batch. Seed 42 is the
predeclared representative export seed, not the best sealed-test seed.

## Hardware boundary

The frozen interface is NCT `[1, 17, 1280]`, symmetric INT16 weights and
activations, and INT32 bias/accumulators. Activation scales are calibrated on
final train only and checked on block 5. The current contract is not an FPGA
synthesis claim. KV260 claims require a bit-accurate reference, exported
tensors, DPU/HLS operator coverage, and on-board latency/power measurement.
