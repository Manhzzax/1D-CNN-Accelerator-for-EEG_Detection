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

## Current Reference: `run_21_raw_2s_temporal3`

The current model reference is a raw 17-channel separable 1D-CNN trained with
2-second windows at 256 Hz and a 1-second stride. It uses train-only channel
z-score normalization, 5,013 trainable parameters, and a causal alarm time at
the end of each 2-second input window. It is a **validation-only architecture
screening result** under the locked within-case chronological split; no test
inference was used to select it.

| Current-model item | Value |
|---|---:|
| Validation window accuracy, 1:1 sampled windows | **90.0718%** |
| Validation window sensitivity | 90.7645% |
| Validation window F1 | 90.1401% |
| Validation AUROC / average precision | 96.5802% / 96.9764% |
| Causal validation event sensitivity | **23/29 = 79.31%** |
| Causal FAR/h | **0.4671/h** |
| Causal median detection delay | 17 s |
| Training parameters / folded deployment values | 5,013 / 4,898 |
| FP32 checkpoint / folded INT16 tensor package | 28,130 B / 10,030 B |
| INT16-emulated validation accuracy | 90.0462% |
| FP32-to-INT16 accuracy change | -0.0257 percentage points |

The reproducible result summary is
[`results/reference/run_21_raw_2s_temporal3/validation_summary.json`](../results/reference/run_21_raw_2s_temporal3/validation_summary.json).
The fixed-point package and its report are under
[`fpga/reference_run_21_int16/`](../fpga/reference_run_21_int16/).

## Accuracy Benchmark: Classification Papers

This table answers the narrow question, "how far is the current 90.07% window
accuracy from published CHB-MIT classification headlines?" It is a directional
gap analysis only, because their splits, channel counts, labels, windows, and
class ratios differ.

| Study | Task and protocol | Reported accuracy | Difference from current 90.07% | Hardware-size evidence |
|---|---|---:|---:|---|
| **Current `run_21`** | 17-channel ictal/non-seizure; 2 s; locked within-case chronological validation; 1:1 sampled windows | **90.07%** | 0.00 pp | 5,013 trainable parameters; 10,030 B INT16 tensors |
| Chung et al. 2024, public labels | Patient-specific; 13 selected cases; one clinical channel; k-fold segment evaluation | 94.93% +/- 8.35% | -4.86 pp | Not reported in a directly comparable form |
| Kashefi Amiri et al. 2025 | DWT + 1D CNN-LSTM; 10-fold classification | 96.94% +/- 1.22% | -6.87 pp | Reports complexity around 1.67e6; not a parameter count |
| Alharthi et al. 2022 | Selected-channel 1D-CNN + Bi-LSTM + attention; integrated clinical/CHB-MIT data | up to 96.87% | -6.80 pp | Not reported in a directly comparable form |
| Cao et al. 2025 | DWT feature fusion + CNN-Bi-LSTM classification; all 23 CHB-MIT cases | 98.43% | -8.36 pp | Paper describes high complexity; no comparable compact parameter count |

The published numbers are not a valid claim that the current model is behind by
exactly the listed margins. They do establish the present classification gap:
the compact model has crossed 90% under the locked protocol, while published
heavier or patient-specific classifiers report approximately 95-98% under
different protocols.

## Clinical Benchmark: Continuous Event Detection

This table is more important than accuracy for a seizure detector. It includes
only papers that report a continuous event sensitivity plus false-alarm rate.

| Study | Evaluation setting | Event sensitivity | FAR/h | Delay | Comparison status |
|---|---|---:|---:|---:|---|
| **Current `run_21`** | Shared 17-channel model; locked within-case chronological validation; public CHB-MIT labels; causal window-end alarm | **79.31%** | **0.4671** | 17 s median | Current screening reference; not final test result |
| Shoeb and Guttag 2010 | Patient-specific; 24 cases; 916 h continuous test EEG; 173 test seizures | 96% | 0.083 median | 3 s median | Historical continuous-detection comparator |
| Chung et al. 2024, public labels | Patient-specific; 13 selected cases; single clinical channel | 97.69% +/- 6.96% | 0.16 +/- 0.26 | 8.0 +/- 9.4 s | Primary device-oriented comparator |
| Chung et al. 2024, reviewed labels | Patient-specific; clinician-reviewed annotations | 99.62% +/- 1.39% | 0.22 +/- 0.34 | 3.3 +/- 5.5 s | Context only; labels are different |

At the internal screening constraint of FAR <= 0.5/h, `run_21` reaches the
constraint but remains below the external event-sensitivity references. Its
gap to Chung's public-label mean is 18.38 percentage points in sensitivity and
its FAR/h is 0.3071 higher. Delay is not subtracted numerically because the
current value is a median while Chung reports mean +/- standard deviation.

## Hardware Benchmark Position

`run_21` is currently the only model in this repository with both a selected
window-classification result and a verified fixed-point package. BatchNorm
folding changes logits by at most `3.81e-06`; INT16 emulation preserves
validation sensitivity and has 99.9743% prediction agreement with the folded
FP32 model. This supports starting KV260 HLS/RTL work, but it is not yet an
FPGA performance claim. The remaining hardware benchmark is post-synthesis
resource use (LUT, FF, BRAM, DSP), clock, throughput, latency, power, and
continuous event metrics after fixed-point deployment.

## Benchmark Decision

For the next model iterations, retain `run_21` as the **accuracy and compact
hardware reference**. The next candidate must be compared on validation only
and should improve the causal event frontier: event sensitivity above 79.31%
while retaining FAR <= 0.5/h and not increasing the 17 s median delay. A final
paper comparison requires the two protocol tracks below; current results are
not evidence of superiority over any published paper.

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
