# CHB-MIT Seizure Detection Benchmark Specification

## Decision

This project will be benchmarked as **continuous EEG seizure detection**, not seizure prediction.

The official primary benchmark is event-level detection on continuous held-out EEG:

| Metric | Definition | Minimum paper target | Strong target |
|---|---|---:|---:|
| Event sensitivity | Detected annotated seizures / all annotated seizures | >= 90% | >= 95% |
| FAR/h | False seizure alarms / interictal monitoring hours | <= 0.50/h | <= 0.20/h |
| Median detection delay | Alarm time minus annotated seizure onset, detected events only | <= 10 s | <= 8 s |
| Validation window accuracy (1:1 sampled set) | Correct window predictions / validation windows | >= 90% | >= 95% |

All primary metrics and the validation accuracy gate must be reported together. A run does not pass merely because it has high sensitivity or high window accuracy. The strong target is the target for the final FPGA-aware model; the minimum target is the gate for a credible controlled ablation.

## Why These Targets

The thresholds are deliberately conservative relative to directly relevant CHB-MIT continuous-detection studies:

| Reference | Task and protocol | Sensitivity | FAR/h | Delay | How it is used |
|---|---|---:|---:|---:|---|
| Shoeb and Guttag, ICML 2010 | Patient-specific detector; 916 h continuous EEG, 24 patients, 173 test seizures | 96% | 2 false alarms/day (median), about 0.083/h | 3 s median | Historical full-cohort continuous-detection reference |
| Chung et al., Frontiers in Neurology 2024 | Patient-specific CHB-MIT detector; 13 selected cases; public annotations, single channel | 97.69% +/- 6.96% | 0.16 +/- 0.26/h | 8.0 +/- 9.4 s | Closest published low-channel/device-oriented reference |
| Chung et al., 2024 | Patient-specific CHB-MIT detector; 13 selected cases; 18 channels with reviewed annotations | 100% | 0.30 +/- 0.47/h | 2.1 +/- 6.7 s | Upper contextual reference, not a directly comparable target |

The minimum target (90%, 0.50/h, 10 s) is below the reported results above, leaving margin for a stricter protocol and a compact FPGA-feasible architecture. The strong target (95%, 0.20/h, 8 s) approximately matches the public-annotation single-channel result without claiming direct superiority.

Sources:

- Shoeb, A. and Guttag, J. *Application of Machine Learning To Epileptic Seizure Detection*, ICML 2010. https://physionet.org/physiobank/database/chbmit/shoeb-icml-2010.pdf
- Chung, Y. G. et al. *Single-channel seizure detection with clinical confirmation of seizure locations using CHB-MIT dataset*, Frontiers in Neurology, 2024. https://doi.org/10.3389/fneur.2024.1389731

## Required Evaluation Protocol

The final paper benchmark must use a **patient-held-out protocol**, preferably leave-one-patient-out (LOPO) over all eligible CHB-MIT cases. A patient used for test must contribute no training, normalization fitting, threshold tuning, hard-negative mining, or architecture selection samples.

For each held-out patient:

1. Fit preprocessing statistics, model parameters, and mining only on the training patients.
2. Use a disjoint validation-patient subset to choose threshold, temporal policy, and early-stopping epoch.
3. Run one continuous inference pass over all held-out recordings.
4. Count an event as detected if the confirmed alarm overlaps its annotated ictal interval. Count every alarm outside an ictal interval as a false alarm, applying one prespecified refractory period.

Report micro-average totals over all held-out EEG, plus per-patient values and a 95% confidence interval obtained by patient-level bootstrap. Repeat the complete training for at least three fixed seeds. The input montage, sample rate, filters, normalization, window/stride, temporal policy, refractory time, and total interictal hours must be fixed before final testing.

## Secondary Metrics

Window metrics are supportive only: AUROC, average precision, balanced accuracy, sensitivity, specificity, precision, and F1. Do not use raw accuracy as a decision criterion. In the current sampled test split, approximately 91% of windows are non-seizure, so an all-non-seizure classifier would appear to have about 90.9% accuracy while detecting no seizure events.

For a controlled comparison, report:

| Secondary metric | Acceptance value |
|---|---:|
| Balanced accuracy | >= 0.90 |
| AUROC | >= 0.95 |
| Ictal-window F1 | >= 0.85 |

These are diagnostic gates, not substitutes for the primary event metrics. They must be measured on the same untouched patient-held-out test predictions.

## FPGA Reporting Gate

The final selected model must also be measured after deployment-oriented INT8 quantization on KV260. Report event metrics before and after quantization, parameter count, model size, LUT, FF, BRAM, DSP, clock frequency, throughput, end-to-end latency, and power. The INT8 implementation is accepted only if it loses no more than 2 percentage points of event sensitivity and increases FAR/h by no more than 0.05/h against its FP32 reference at the locked policy.

## What Is Not A Benchmark For This Project

Zhang et al., *Epilepsy Seizure Prediction on EEG Using Common Spatial Pattern and Convolutional Neural Network*, reports sensitivity 0.92, FPR/h 0.12, and accuracy 0.90 for **seizure prediction**. It uses preictal/interictal labels, five-second trials, patient-specific leave-one-seizure-out validation, and Kalman smoothing. It is not comparable to ictal/non-seizure detection and its 0.90 accuracy must not be used as this project's target.

Source: https://doi.org/10.1109/JBHI.2019.2933046

## Status Of Existing Runs

The current runs use a locked within-case chronological split and the test set has already informed exploratory development. They are internal ablations, not official benchmark results. The current best low-FAR exploratory points are:

| Run | Event sensitivity | FAR/h | Median delay | Status |
|---|---:|---:|---:|---|
| `run_03_mixed_hardneg` | 58.06% | 0.341 | 13.5 s | Lowest FAR internal point |
| `run_04_score_tcn` | 64.52% | 0.422 | 14.0 s | Highest sensitivity below 0.5/h internally |

Neither meets the minimum publication target. Further architecture tuning must use validation only; the next substantive work is to implement and lock the patient-held-out protocol before a final model claim.
