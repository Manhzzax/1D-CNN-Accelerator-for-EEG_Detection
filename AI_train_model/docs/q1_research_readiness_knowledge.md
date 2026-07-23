# Q1 Research Readiness Knowledge Base

Created: 2026-07-23

This document defines the stronger research foundation needed before turning the current CHB-MIT 1D-CNN training pipeline into a Q1-journal-level study. It extends `chbmit_paper_knowledge_base.md` with the additional papers downloaded into `docs/papers_chbmit/`.

---

## 1. Current Research Position

Project topic:

> Hardware-aware 1D-CNN accelerator for EEG seizure detection on CHB-MIT, with fixed-point quantization and deployable weights for FPGA/ASIC.

Current repo task:

- Seizure detection: ictal vs non-ictal.
- Dataset target: CHB-MIT EDF files and `chbXX-summary.txt` seizure annotations.
- Input shape: `23 x 256`, meaning 23 bipolar EEG channels and 1 second at 256 Hz.
- Model family: compact Conv1D + pooling + fully connected layers.
- Hardware path: BatchNorm folding, Q16 dynamic quantization, flat weight export.

For Q1-level work, the novelty should not be "we trained a CNN on CHB-MIT". That is already saturated. The defensible novelty should combine:

- leakage-safe CHB-MIT evaluation;
- compact hardware-aware 1D-CNN architecture;
- channel/band reduction;
- fixed-point or quantization-aware training;
- event-level detection metrics;
- accelerator resource/latency/energy analysis.

---

## 2. Research Pillars Now Covered

### 2.1 Dataset and original CHB-MIT foundation

Use these to justify dataset provenance:

- Goldberger et al. 2000 PhysioNet resource paper: `10_goldberger_2000_physionet_resource.pdf`
- Shoeb and Guttag 2010 ICML seizure detection paper from PhysioNet: `12_shoeb_guttag_2010_ml_seizure_detection_icml.pdf`
- CHB-MIT PhysioNet dataset page: https://physionet.org/content/chbmit/1.0.0/

What this gives the project:

- a standard citation path for PhysioNet;
- a CHB-MIT-specific seizure detection baseline citation;
- a defensible explanation for EDF files, seizure annotation files, and summary files.

Action for repo:

- Add CHB-MIT citation metadata to the research docs.
- In preprocessing reports, log case IDs, file IDs, channel count, seizure count, and total ictal/non-ictal duration.

### 2.2 Evaluation validity and data leakage control

Use these to justify stricter evaluation:

- Ali et al. 2024, CHB-MIT overlooked perspectives: `13_ali_2024_chbmit_overlooked_perspectives.pdf`
- Ghosh et al. 2026 patient-independent multi-domain feature paper: blocked, but metadata found from Springer.
- Lee et al. 2022 real-time seizure detection: `28_lee_2022_real_time_seizure_detection_eeg.pdf`

What this gives the project:

- random window split is not enough for a clinical claim;
- event-level seizure detection matters more than segment accuracy alone;
- subject-wise and leave-one-subject-out evaluation should be added;
- imbalanced continuous EEG should be reported, not only artificially balanced windows.

Action for repo:

- Keep random window split only as smoke-test baseline.
- Add `split.mode` options:
  - `random_window`,
  - `subject_wise`,
  - `leave_one_subject_out`,
  - `leave_one_seizure_out`.
- Add event metrics:
  - event sensitivity,
  - false alarms per hour,
  - detection latency,
  - seizure coverage threshold.

### 2.3 EEG-specific CNN architecture design

Use these to justify Conv1D architecture choices:

- EEGNet compact CNN: `16_lawhern_2018_eegnet_compact_cnn.pdf`
- Schirrmeister et al. deep CNN EEG decoding: `17_schirrmeister_2017_deep_cnn_eeg_decoding.pdf`
- Current CHB-MIT CNN/DWT/LSTM papers: P05, P06, P08 from `papers_chbmit/`

What this gives the project:

- temporal convolution can act as a learnable filter bank;
- spatial convolution or channel-mixing layers can replace handcrafted spatial features;
- depthwise and separable convolutions are a principled way to reduce parameters;
- deeper LSTM/DenseNet/attention models are useful comparators but are not the first hardware target.

Action for repo:

- Add at least three model variants:
  1. `baseline_1dcnn`: current Conv1D baseline.
  2. `separable_1dcnn`: depthwise temporal Conv1D + pointwise channel mixing.
  3. `dwt_1dcnn`: DWT or band-feature frontend + compact Conv1D classifier.

### 2.4 Channel and band reduction

Use these to justify smaller input paths:

- Chung et al. 2024 single-channel CHB-MIT detection: `02_chung_2024_single_channel_chbmit_detection.pdf`
- Frontiers 2026 channel/frequency selection: `30_frontiers_2026_channel_frequency_selection_chbmit.pdf`
- Ahlawat 2026 channel pruning and sparsity preprint: `19_ahlawat_2026_int8_quant_channel_pruning_snn.pdf`

What this gives the project:

- reducing channels can be a scientific contribution and a hardware contribution at the same time;
- channel selection should be evaluated, not assumed;
- frequency-band selection can reduce signal path and compute before Conv1D.

Action for repo:

- Add configurable channel subsets:
  - full 23-channel baseline,
  - common 18-channel subset,
  - selected 4-channel subset,
  - single-channel patient-specific experiment.
- Report MACs, parameter count, activation memory, and accuracy/FAR tradeoff for each.

### 2.5 Quantization and hardware deployment

Use these to justify fixed-point inference:

- Jacob et al. 2018 integer-only quantization: `18_jacob_2018_integer_only_quantization.pdf`
- Li et al. 2022 parallel memristive CNN: `09_li_2022_parallel_memristive_cnn_detection_prediction.pdf`
- Alhammadi et al. 2022 1D-CNN FPGA accelerator: `25_alhammadi_2022_1dcnn_fpga_accelerator_hls.pdf`
- Ahlawat 2026 INT8/channel pruning/SNN: `19_ahlawat_2026_int8_quant_channel_pruning_snn.pdf`
- RISC-V CNN coprocessor paper: blocked, but metadata found from IEEE/ResearchGate.

What this gives the project:

- integer-only inference is established and citeable;
- QAT should be considered if post-training Q16 loses accuracy;
- channel pruning and structured sparsity are natural next steps after Q16;
- Conv1D accelerator design should report not only accuracy but resource, latency, and memory.

Action for repo:

- Add model profiling:
  - parameters,
  - MACs per inference,
  - activation memory,
  - weight memory at FP32/Q16/INT8,
  - estimated cycles for each Conv/FC layer.
- Add quantization modes:
  - current PTQ Q16 dynamic scale,
  - static per-layer Q format,
  - optional QAT.

### 2.6 Wearable and clinical deployment context

Use these to motivate constraints:

- EarSD wearable seizure detection: `21_eard_2024_earsd_lightweight_wearable_seizure_detection.pdf`
- Frontiers 2026 AI in wearable seizure detection: `26_biondi_2026_ai_wearable_seizure_detection_devices.pdf`
- Frontiers 2024 review: `14_zhang_2024_review_eeg_processing_deep_learning.pdf`

What this gives the project:

- target use case is continuous monitoring, not one-off offline classification;
- false alarms, comfort, channel count, and power matter;
- a low-compute model is justified even if a larger LSTM/Transformer scores slightly higher.

Action for repo:

- Evaluate using continuous timeline reconstruction, not only balanced test windows.
- Add practical reporting: inference latency, storage, and expected compute.

---

## 3. Minimum Q1-Ready Experiment Stack

The project should aim to produce a table like this:

| Model | Input | Preprocess | Split | Quantization | Metrics | Hardware metrics |
|---|---|---|---|---|---|---|
| Baseline 1D-CNN | 23 x 256 | Raw + z-score | Random window + subject-wise | FP32/Q16 | Acc, Sen, Spec, F1 | Params, MACs, weight KB |
| Separable 1D-CNN | 23 x 256 | Raw + z-score | Subject-wise | FP32/Q16 | Acc, Sen, Spec, F1, FAR/h | Params, MACs, activation KB |
| Channel-reduced 1D-CNN | 18/4/1 channels | Raw or selected bands | Subject-wise | FP32/Q16/INT8 | Event Sen, FAR/h, latency | Input buffer, MAC reduction |
| DWT compact CNN | DWT features | DWT + z-score | Subject-wise | FP32/Q16 | Event metrics | DWT cost + CNN cost |

Claims that should be avoided:

- Do not claim clinical-grade detection from random balanced-window accuracy.
- Do not compare detection accuracy directly against prediction papers.
- Do not deploy GAN/LSTM/DenseNet as the first accelerator target unless resource results justify it.

Claims that can become strong:

- A compact Conv1D model can preserve useful CHB-MIT detection performance under leakage-safe split.
- Channel reduction and fixed-point quantization can cut compute/memory while maintaining event-level performance.
- A hardware-aware architecture can be derived from EEG-specific CNN principles and validated against Q16/INT8 constraints.

---

## 4. Immediate Next Research Tasks

1. Extract exact layer tables from the deployable model papers:
   - P09 Li 2022 memristive CNN.
   - P16 EEGNet.
   - P17 Schirrmeister EEG CNN.
   - P25 1D-CNN FPGA accelerator.
   - P05 DWT + 1D CNN-LSTM.

2. Add experiment configs:
   - `task: detection`
   - `window_sec`
   - `overlap`
   - `subjects`
   - `split.mode`
   - `channels.mode`
   - `quantization.mode`

3. Add metrics and profiling:
   - event-level detection metrics;
   - MAC/parameter counter;
   - per-layer tensor shape report;
   - Q-format report for weights and activations.

4. Build the first architecture proposal:
   - raw compact 1D-CNN baseline;
   - depthwise-separable 1D-CNN;
   - channel-reduced version;
   - optional DWT frontend variant.

---

## 5. Corpus Status

The local PDF corpus currently has 22 verified PDF files:

`AI_train_model/docs/papers_chbmit/`

The detailed manifest is:

`AI_train_model/docs/papers_chbmit/README.md`
