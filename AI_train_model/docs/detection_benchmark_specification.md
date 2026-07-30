# CHB-MIT Seizure Detection Benchmark Specification

## Decision

This project will be benchmarked as **continuous EEG seizure detection**, not seizure prediction.

The primary outcome is event-level detection on continuous held-out EEG. The
numbers below are an **internal clinical screening gate** for ablations; they
are not presented as a universal paper benchmark:

| Metric | Definition | Internal screening target | Aspirational target |
|---|---|---:|---:|
| Event sensitivity | Detected annotated seizures / all annotated seizures | >= 90% | >= 95% |
| FAR/h | False seizure alarms / interictal monitoring hours | <= 0.50/h | <= 0.20/h |
| Median detection delay | Alarm time minus annotated seizure onset, detected events only | <= 10 s | <= 8 s |
| Validation window accuracy (1:1 sampled set) | Correct window predictions / validation windows | >= 90% | >= 95% |

All primary metrics and validation accuracy must be reported together. A run
does not pass merely because it has high sensitivity or high window accuracy.
The paper-specific comparison contract, including which published accuracy can
be compared, is in `paper_benchmark_comparison.md`.

## Why These Targets

The thresholds are deliberately conservative relative to directly relevant CHB-MIT continuous-detection studies:

| Reference | Task and protocol | Sensitivity | FAR/h | Delay | How it is used |
|---|---|---:|---:|---:|---|
| Shoeb and Guttag, ICML 2010 | Patient-specific detector; 916 h continuous EEG, 24 patients, 173 test seizures | 96% | 2 false alarms/day (median), about 0.083/h | 3 s median | Historical full-cohort continuous-detection reference |
| Chung et al., Frontiers in Neurology 2024 | Patient-specific CHB-MIT detector; 13 selected cases; public annotations, single channel | 97.69% +/- 6.96% | 0.16 +/- 0.26/h | 8.0 +/- 9.4 s | Closest published low-channel/device-oriented reference |
| Chung et al., 2024 | Patient-specific CHB-MIT detector; 13 selected cases; 18 channels with reviewed annotations | 100% | 0.30 +/- 0.47/h | 2.1 +/- 6.7 s | Upper contextual reference, not a directly comparable target |

The screening target (90%, 0.50/h, 10 s) is below the reported results above,
leaving margin for a compact FPGA-feasible architecture. The aspirational target
(95%, 0.20/h, 8 s) approximates the public-annotation single-channel result;
it must not be described as a direct match unless the patient-specific 13-case,
channel-selection protocol is reproduced.

Sources:

- Shoeb, A. and Guttag, J. *Application of Machine Learning To Epileptic Seizure Detection*, ICML 2010. https://physionet.org/physiobank/database/chbmit/shoeb-icml-2010.pdf
- Chung, Y. G. et al. *Single-channel seizure detection with clinical confirmation of seizure locations using CHB-MIT dataset*, Frontiers in Neurology, 2024. https://doi.org/10.3389/fneur.2024.1389731

## Required Final Evaluation Protocols

The final paper needs two explicitly separated tracks:

1. **Patient-specific continuous detection.** This is the external-comparator
   track for Shoeb and Chung. It must use chronology-safe train/validation/test
   partitions per case, public annotations, a fixed channel policy, and no use
   of a future recording for fitting, mining or threshold selection.
2. **Patient-held-out generalization.** Prefer leave-one-patient-out (LOPO)
   over eligible CHB-MIT cases. A patient used for test must contribute no
   training, normalization fitting, threshold tuning, hard-negative mining, or
   architecture selection samples.

For each held-out patient in the generalization track:

1. Fit preprocessing statistics, model parameters, and mining only on the training patients.
2. Use a disjoint validation-patient subset to choose threshold, temporal policy, and early-stopping epoch.
3. Run one continuous inference pass over all held-out recordings.
4. Timestamp an alarm at the **end** of its input window, because the model
   cannot produce that score until the full window has arrived. Count an event
   as detected only if the confirmed causal alarm timestamp is within its
   annotated ictal interval. Count every other alarm as a false alarm, applying
   one prespecified refractory period.

Report micro-average totals over all held-out EEG, plus per-patient values and a 95% confidence interval obtained by patient-level bootstrap. Repeat the complete training for at least three fixed seeds. The input montage, sample rate, filters, normalization, window/stride, temporal policy, refractory time, and total interictal hours must be fixed before final testing.

The current Butterworth/notch preprocessing is zero-phase offline filtering and
is acceptable only for exploratory score selection. Before an FPGA real-time
claim, replace it with a causal/stateful filter and remeasure the full event
protocol; the window-end alarm convention remains required in both modes.

## Secondary Metrics

Window metrics are supportive only: AUROC, average precision, balanced accuracy, sensitivity, specificity, precision, and F1. Do not use raw accuracy as a decision criterion. In the current sampled test split, approximately 91% of windows are non-seizure, so an all-non-seizure classifier would appear to have about 90.9% accuracy while detecting no seizure events.

For a controlled comparison, always report:

| Secondary metric | Diagnostic reference |
|---|---:|
| Balanced accuracy | >= 0.90 |
| AUROC | >= 0.95 |
| Ictal-window F1 | >= 0.85 |

These values are internal diagnostics, not paper-derived acceptance thresholds
and not substitutes for the primary event metrics. They must be measured on the
same untouched patient-held-out test predictions.

## FPGA Reporting Gate

The final selected model must also be measured after deployment-oriented INT8 quantization on KV260. Report event metrics before and after quantization, parameter count, model size, LUT, FF, BRAM, DSP, clock frequency, throughput, end-to-end latency, and power. The INT8 implementation is accepted only if it loses no more than 2 percentage points of event sensitivity and increases FAR/h by no more than 0.05/h against its FP32 reference at the locked policy.

## What Is Not A Benchmark For This Project

Zhang et al., *Epilepsy Seizure Prediction on EEG Using Common Spatial Pattern and Convolutional Neural Network*, reports sensitivity 0.92, FPR/h 0.12, and accuracy 0.90 for **seizure prediction**. It uses preictal/interictal labels, five-second trials, patient-specific leave-one-seizure-out validation, and Kalman smoothing. It is not comparable to ictal/non-seizure detection and its 0.90 accuracy must not be used as this project's target.

Source: https://doi.org/10.1109/JBHI.2019.2933046

## Status Of Existing Runs

The current runs use a locked within-case chronological split and the test set
has already informed exploratory development. They are internal ablations, not
official benchmark results. The current best validation-only screening point is:

| Run | Event sensitivity | FAR/h | Median delay | Status |
|---|---:|---:|---:|---|
| `run_10_separable_hparam_f_lr1e3_wd1e4_nobalance` | 72.41% (21/29) | 0.455 | 15 s | Best current validation-only raw separable screening point |

It does not meet the internal clinical screening gate. Further architecture
tuning must use validation only; after the architecture is frozen, implement
and lock both final protocols before any model claim.
