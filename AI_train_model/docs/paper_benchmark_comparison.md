# CHB-MIT Paper Benchmark and Comparison Contract

## Decision

The project is **continuous ictal seizure detection**, not seizure prediction.
The external comparison target is a set of paper-specific references, not one
number assembled from incompatible studies. Segment/window accuracy is always
reported, but it is never substituted for event sensitivity, false alarms per
hour (FAR/h), or detection delay.

## Verified External References

| Reference | Evaluation setting | Segment accuracy | Event sensitivity | FAR/h | Delay | Use in this project |
|---|---|---:|---:|---:|---:|---|
| Shoeb and Guttag, ICML 2010 | Patient-specific; >=2 training seizures/patient; 24 patients; 916 h continuous test EEG; 173 test seizures | Not reported | 96% | 2/day = 0.083/h median | 3 s median | Historical continuous-detection reference |
| Chung et al., 2024, public annotations | Patient-specific; 13 selected CHB-MIT cases; one clinically selected channel; k-fold segment evaluation and continuous event evaluation | 94.93% +/- 8.35% | 97.69% +/- 6.96% | 0.16 +/- 0.26 | 8.0 +/- 9.4 s | Primary device-oriented paper comparator |
| Chung et al., 2024, reviewed annotations | Same 13 selected cases, but clinician re-annotation and clinical channel selection | 98.18% +/- 1.83% | 99.62% +/- 1.39% | 0.22 +/- 0.34 | 3.3 +/- 5.5 s | Context only: labels differ from public CHB-MIT labels |
| Kashefi Amiri et al., 2025 | DWT-concatenated 1D CNN-LSTM classification on CHB-MIT | 96.94% +/- 1.22% | Not reported as continuous event sensitivity | Not reported | Not reported | Classification and DWT/LSTM complexity comparator only |
| Cao et al., 2025 | DWT feature fusion, SVM-RFE and CNN-Bi-LSTM classification on all 23 CHB-MIT cases | 98.43% | Not reported as continuous event sensitivity | Not reported | Not reported | Classification upper context; not FPGA-like |

Sources:

- Shoeb and Guttag, *Application of Machine Learning to Epileptic Seizure Detection*, ICML 2010: https://physionet.org/physiobank/database/chbmit/shoeb-icml-2010.pdf
- Chung et al., *Single-channel seizure detection with clinical confirmation of seizure locations using CHB-MIT dataset*, Frontiers in Neurology 2024: https://doi.org/10.3389/fneur.2024.1389731
- Kashefi Amiri et al., *Epileptic seizure detection from electroencephalogram signals based on 1D CNN-LSTM deep learning model using discrete wavelet transform*, Scientific Reports 2025: https://doi.org/10.1038/s41598-025-18479-9
- Cao et al., *A hybrid CNN-Bi-LSTM model with feature fusion for accurate epilepsy seizure detection*, BMC Medical Informatics and Decision Making 2025: https://doi.org/10.1186/s12911-024-02845-0

## What Accuracy Means Here

`accuracy = correctly classified labelled windows / all labelled windows`.
It depends on window length, overlap, normal/ictal ratio, preprocessing and
split. The current validation set is deliberately sampled 1:1, so its accuracy
is interpretable as balanced window accuracy. It is **not** numerically
equivalent to the paper accuracies above:

- Chung reports a mean of patient-specific k-fold accuracies for 13 selected,
  single-channel cases.
- Kashefi reports classification accuracy for a DWT + CNN-LSTM protocol.
- Cao reports classification metrics after DWT feature extraction and
  feature-selection.
- Shoeb reports continuous event metrics but no segment accuracy.

The current exploratory test split contains about 91% non-seizure windows;
therefore its raw accuracy must never be compared to a paper headline or used
as a clinical selection objective.

## Current Position (Validation Only)

The strongest raw separable 1D-CNN configuration so far is
`run_10_separable_hparam_f_lr1e3_wd1e4_nobalance`:

| Item | Value |
|---|---:|
| Architecture | Raw 17-channel separable 1D-CNN |
| Parameters | 3,908 |
| 1:1 validation window accuracy | 86.93% |
| Validation AUROC | 0.9426 |
| Validation ictal F1 | 0.8701 |
| Continuous validation event sensitivity | 21/29 = 72.41% |
| Continuous validation FAR/h | 0.4552 |
| Median detection delay | 15 s |

Against Chung's public-annotation **patient-specific** single-channel result,
this exploratory shared-model validation point is lower by 8.00 percentage
points in segment accuracy and 25.28 percentage points in event sensitivity;
its FAR/h is higher by 0.2952 and its median delay is 7 s longer. This is a
gap analysis, not a direct head-to-head claim, because the protocols differ.

## Protocol Required Before a Paper Claim

Two separate final tracks are required:

1. **Patient-specific paper-comparator track.** Use public CHB-MIT labels,
   chronology-safe per-case train/validation/test partitions, a fixed channel
   policy, and full continuous event evaluation. It is the only track that can
   be compared closely with Shoeb and Chung.
2. **Patient-held-out generalization track.** Use leave-one-patient-out or a
   disjoint patient test partition. No held-out patient data may influence
   normalization, hard-negative mining, early stopping, threshold or temporal
   policy. This is the stronger generalization claim, but patient-specific
   papers are contextual rather than numerical targets.

Current chronological shared-model runs are architecture screening only. They
must not be reported as final test performance. The immediate work is to
improve validation score quality under the same screening protocol, then freeze
the architecture before implementing the two final protocols.
