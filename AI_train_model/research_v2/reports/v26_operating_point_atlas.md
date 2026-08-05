# V2.6 Operating-Point Atlas

## Scope

This is an artifact-only diagnostic over already consumed V2.1 F00--F02
development replays. It neither trains a model nor selects a threshold or policy.
Blocks 5 and 6 remain sealed.

## Calibration-To-Temporal Transfer

Values are means +/- sample standard deviations across the five fixed seeds
within a fold. Fold-by-seed values are not treated as independent patients.

| Candidate | Fold | Balanced accuracy (%) | AUROC (%) | Event SEN (%) | Calibration FAR/h | Temporal FAR/h | Temporal FAR passes | Mean temporal-calibration FAR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | F00 | 83.57 +/- 1.11 | 90.40 +/- 0.78 | 56.52 +/- 11.91 | 0.478 +/- 0.010 | 2.648 +/- 0.723 | 0/5 at <= 0.5/h | +2.170 |
| C1 | F01 | 88.40 +/- 2.22 | 94.17 +/- 1.50 | 64.71 +/- 9.07 | 0.477 +/- 0.033 | 0.279 +/- 0.172 | 5/5 at <= 0.5/h | -0.197 |
| C1 | F02 | 86.89 +/- 1.46 | 93.59 +/- 0.81 | 74.78 +/- 10.83 | 0.440 +/- 0.038 | 0.900 +/- 0.108 | 0/5 at <= 0.5/h | +0.459 |
| H2 | F00 | 82.36 +/- 1.72 | 89.79 +/- 0.84 | 74.78 +/- 4.76 | 0.434 +/- 0.028 | 0.837 +/- 0.686 | 2/5 at <= 0.5/h | +0.402 |
| H2 | F01 | 85.39 +/- 3.36 | 92.62 +/- 2.68 | 67.65 +/- 10.19 | 0.447 +/- 0.046 | 0.484 +/- 0.407 | 3/5 at <= 0.5/h | +0.037 |
| H2 | F02 | 86.87 +/- 1.27 | 93.20 +/- 0.88 | 76.52 +/- 6.59 | 0.388 +/- 0.071 | 0.757 +/- 0.371 | 1/5 at <= 0.5/h | +0.369 |
| G1 | F00 | 84.20 +/- 2.01 | 91.34 +/- 1.16 | 61.74 +/- 9.91 | 0.484 +/- 0.008 | 2.311 +/- 0.832 | 0/5 at <= 0.5/h | +1.827 |
| G1 | F01 | 86.57 +/- 2.45 | 93.12 +/- 1.63 | 57.06 +/- 12.23 | 0.468 +/- 0.016 | 0.429 +/- 0.141 | 3/5 at <= 0.5/h | -0.040 |
| G1 | F02 | 84.98 +/- 1.51 | 91.84 +/- 1.60 | 73.04 +/- 4.76 | 0.473 +/- 0.020 | 1.003 +/- 0.127 | 0/5 at <= 0.5/h | +0.531 |

## False-Alarm Concentration

Top groups aggregate false alarms and replay hours across the five seeds in
one fold. This is a diagnostic aggregation, not a patient-level confidence interval.

| Candidate | Fold | Top patient group | Top-group FAR/h | Share of false alarms | HHI |
| --- | --- | --- | ---: | ---: | ---: |
| C1 | F00 | subject_04 | 9.813 | 0.630 | 0.421 |
| C1 | F01 | subject_08 | 3.600 | 0.283 | 0.137 |
| C1 | F02 | subject_05 | 6.333 | 0.304 | 0.156 |
| H2 | F00 | subject_05 | 6.126 | 0.309 | 0.199 |
| H2 | F01 | subject_05 | 2.983 | 0.269 | 0.146 |
| H2 | F02 | subject_05 | 8.133 | 0.465 | 0.254 |
| G1 | F00 | subject_04 | 8.554 | 0.629 | 0.420 |
| G1 | F01 | subject_05 | 2.983 | 0.304 | 0.170 |
| G1 | F02 | subject_02 | 6.680 | 0.240 | 0.159 |

## Interpretation Boundary

The committed artifacts establish whether a calibration-feasible operating point
transferred to the next block and whether false alarms are concentrated. They do
not contain full temporal score trajectories, so they cannot distinguish a bad
calibration policy from a representation failure or assign EEG artifact labels.
A future score-replay audit, if approved, must be diagnostic-only, use only the
already consumed F00--F02 recordings, and must not select a threshold, policy,
or new candidate.

## Integrity

- Diagnostic config SHA-256: `446eb1a135cd6c652a1297c1a3a77e534b731b7d23dfd8a1a28b0cfdb73a33f6`
- Artifact records checked: `45`
- No raw EEG, prepared cache, continuous score stream, test result, or hardware
  artifact is created by this command.
