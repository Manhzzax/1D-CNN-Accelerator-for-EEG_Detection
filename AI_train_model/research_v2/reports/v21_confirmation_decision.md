# V2.1 Confirmation Decision

## Scope

This document closes the predeclared V2.1 confirmation stage on branch
`research/v2.1-patient-forward`. The scope is a shared-model, within-patient
forward-chaining study on CHB-MIT, not patient-independent or external
validation.

The primary operating point is event sensitivity at a calibration-selected
false-alarm rate (FAR) no greater than 0.5 per hour. Threshold and temporal
vote policy are selected on the calibration block only, then applied once to
the later temporal block. Window balanced accuracy and AUROC are secondary
metrics, measured on explicitly balanced windows.

## Protocol Integrity

- 23 patient groups are used; `chb01` and `chb21` are one ordered patient
  group (`subject_01_21`).
- Confirmation folds are F00 (0 -> 1 -> 2), F01 (0-1 -> 2 -> 3), and F02
  (0-2 -> 3 -> 4), expressed as train -> calibration -> temporal evaluation.
- The V2.1 split audit hash is
  `201291428b412c8e8dcdd019e6e84a4a6150d37e9a02ca29e5ef59dcf50a9799`.
- Blocks 5 and 6 remain unmaterialized. No final validation or sealed-test
  prediction was opened during confirmation.

## Candidate Evidence

| Candidate | Parameters | Temporal F00 | Temporal F01 | Temporal F02 | Decision |
| --- | ---: | --- | --- | --- | --- |
| B0 causal bandpower-linear | 172 | 0.0% SEN, 0.0/h FAR | 0.0% SEN, 0.007/h FAR | 34.8% SEN, 0.735/h FAR | Reject as a weak reference baseline. |
| B1 vanilla 1D-CNN | 6,338 | 56.5% SEN, 1.706/h FAR | No calibration policy met 0.5/h; later block not opened | 69.6% SEN, 0.296/h FAR | Reject: not stable across temporal folds. |
| B2 deep matched 1D-CNN | 5,622 | 51.3 +/- 13.5% SEN, 1.486 +/- 0.378/h FAR | 67.1 +/- 11.1% SEN, 0.802 +/- 0.374/h FAR | 75.7 +/- 6.6% SEN, 0.466 +/- 0.162/h FAR | Do not promote: temporal FAR target is not maintained across folds/seeds. |
| B4 dilated hierarchical 1D-CNN | 5,237 | 51.3 +/- 12.1% SEN, 2.871 +/- 0.775/h FAR | 71.8 +/- 5.3% SEN, 0.699 +/- 0.220/h FAR | 75.7 +/- 5.8% SEN, 0.975 +/- 0.249/h FAR | Do not promote: temporal FAR target is not maintained in any fold. |

Values for B2 and B4 are mean +/- sample standard deviation across the five
predeclared seeds (7, 42, 123, 314, 2718), reported independently per temporal
fold. Fold-by-seed runs are not treated as independent patients.

## Interpretation

No predeclared clinical candidate satisfies the V2.1 selection rule of
clinical event sensitivity with temporal-fold consistency at FAR <= 0.5/h.
High balanced-window accuracy is insufficient evidence for promotion: the
clinical FAR changes materially when a calibration-selected policy is replayed
on later recordings.

B2 is the strongest of the tested candidates on F02, while B4 obtains the
highest F01 sensitivity. Neither pattern is sufficient to select an
architecture because both are inconsistent across the three forward-time
folds. Selecting one favourable seed, fold, threshold, or policy would be
post-hoc selection and is prohibited.

## Decision and Next Boundary

1. No V2.1 candidate is frozen for final training.
2. Blocks 5 and 6 remain sealed; do not run final training, final calibration,
   quantization calibration, or sealed-test inference.
3. Preserve all confirmation artifacts as negative and comparative evidence.
4. Any new architecture, loss, sampling strategy, calibration strategy, or
   temporal policy must be declared in a new protocol/registry revision before
   another model-selection experiment begins.
5. The existing INT16/KV260 interface remains a contract only; no FPGA
   performance or synthesis claim is supported until a future candidate is
   frozen and validated under the final protocol.
