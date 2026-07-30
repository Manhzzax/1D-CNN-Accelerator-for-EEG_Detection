# Benchmark Definition And Comparability Rules

## Current Task

The current project is **seizure detection**: classify continuous CHB-MIT EEG as ictal or non-seizure, then generate an alarm when a temporal decision policy is satisfied. The primary endpoint is an event that overlaps an annotated ictal interval.

This is not seizure prediction. A prediction study classifies pre-ictal versus interictal EEG and must define a seizure prediction horizon and seizure occurrence period. Ictal EEG is normally excluded from the classifier training target.

## Why Table Accuracy Is Not A Direct Benchmark

The frequently cited table from Zhang et al., *Epilepsy Seizure Prediction on EEG Using Common Spatial Pattern and Convolutional Neural Network*, IEEE Journal of Biomedical and Health Informatics 24(2), 2020, is a **patient-specific seizure prediction** result. It uses 5-second trials, pre-ictal/interictal labels, augmentation of pre-ictal trials, leave-one-seizure-out validation within each patient, and a five-trial Kalman smoothing rule. Its reported total sensitivity is 0.92, FPR/h is 0.12, and accuracy is 0.90.

Those values cannot be compared numerically with the current detector because all of the following differ:

- clinical target: warning before onset versus detecting ongoing ictal EEG;
- labels: pre-ictal/interictal versus ictal/non-seizure;
- split: patient-specific leave-one-seizure-out versus locked within-case chronological recordings;
- window duration and post-processing;
- class sampling and therefore the denominator of accuracy.

The paper is useful as a methodological reference for per-patient reporting, FPR/h, temporal alarm smoothing, CSP/time-frequency features, and hardware-conscious shallow CNNs. It is not an accuracy target for the current detection task.

## Metric Rules For This Project

### Secondary window metrics

For the fixed sampled test NPZ only:

`accuracy = (TP + TN) / (TP + TN + FP + FN)`

`precision`, `recall/sensitivity`, F1, AUROC, and average precision are computed from 1-second windows. These are diagnostic metrics only. The sampled test distribution is roughly 91% non-seizure windows, so a classifier that always predicts non-seizure would obtain about 90.9% accuracy and zero event sensitivity.

### Primary continuous event metrics

For all recordings in a split, after applying a threshold, temporal confirmation policy, and refractory period:

`event sensitivity = detected annotated seizure events / total annotated seizure events`

`FAR/h = false alarms not overlapping an annotated seizure / total interictal monitoring hours`

`detection delay = alarm time - seizure onset`, reported for detected events.

The aggregate is micro-averaged: total detected events divided by total events, and total false alarms divided by total interictal hours. Per-recording and per-case tables are also retained to expose concentrated failures; their unweighted mean must be explicitly called macro-average if reported.

## Internal Benchmark Frontier

All values below use the locked 17-channel within-case chronological protocol and are exploratory because the test set has been observed during ablation development.

| Role | Run | Event result and split | FAR/h | Median delay |
|---|---|---:|---:|---:|
| Current causal screening reference | `run_21_raw_2s_temporal3` validation only | 23/29 = 79.31% | 0.467 | 17.0 s |
| High-sensitivity reference | `run_01` | 60/62 = 96.77% | 41.26 | 11.0 s |
| Low-FAR reference | `run_03_mixed_hardneg` | 36/62 = 58.06% | 0.341 | 13.5 s |
| Higher-sensitivity sub-0.5 FAR point | `run_04_score_tcn` | 40/62 = 64.52% | 0.422 | 14.0 s |

`run_21` is the current compact-model reference because it is the first
validation-only causal point above 79% event sensitivity while remaining below
0.5 FAR/h. A future candidate should exceed 79.31% sensitivity while retaining
FAR <= 0.5/h and no worse than the 17 s median delay. The historical test
results remain evidence of earlier trade-offs only; they were observed during
ablation development and are not selection metrics. These are internal
research targets, not clinical standards or a claim of superiority over papers
with a different task/protocol.

## Publication Benchmark Required Later

The final paper needs a prespecified patient-held-out or leave-one-patient-out evaluation, an untouched final test cohort, repeated seeds, and confidence intervals. It should compare only against studies with the same clinical task and make every protocol difference explicit.
