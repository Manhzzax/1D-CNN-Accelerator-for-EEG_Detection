# Research Direction and Architecture Strategy

Created: 2026-07-24

This document consolidates the current PDF corpus into a concrete research direction for the project:

> Hardware-aware 1D-CNN accelerator for EEG seizure detection on CHB-MIT.

The goal is not only to train a high-accuracy model, but to build a defensible research path suitable for a Q1-journal target: correct data protocol, meaningful detection metrics, compact model design, quantization, and hardware-cost reporting.

---

## 1. Short Answer: Correct Direction

The correct primary direction is:

1. Use **CHB-MIT seizure detection** as the main task, not seizure prediction yet.
2. Treat the current local CSV as a **sanity-check dataset only**, not as the main research dataset.
3. Build a leakage-safe CHB-MIT pipeline before optimizing model accuracy.
4. Use the current compact 1D-CNN as a baseline.
5. Add paper-supported variants:
   - EEGNet-inspired separable/depthwise 1D-CNN;
   - parallel multi-kernel 1D-CNN inspired by memristive CNN work;
   - optional DWT + compact 1D-CNN frontend.
6. Evaluate with subject-aware and event-level metrics.
7. Quantize and profile the final compact model for FPGA/ASIC.

Do not make the first journal claim from random balanced-window accuracy. That protocol is too weak.

---

## 2. Current Data Reality

### 2.1 Data expected by the main repo pipeline

The codebase is written for CHB-MIT EDF data:

- raw EDF folder: configured by `data.raw_dir`;
- annotation source: `chbXX-summary.txt`;
- target preprocessed artifact: `AI_train_model/data/chbmit_preprocessed.npz`;
- input tensor: `(N, 23, 256)`;
- labels: `0 = non-seizure`, `1 = seizure`.

Current CHB-MIT state in this workspace:

- CHB-MIT raw folder was not found at `D:/Research/chb-mit-scalp-eeg-database-1.0.0/`.
- `AI_train_model/data/chbmit_preprocessed.npz` is not present.

Therefore, **the main CHB-MIT training pipeline is not ready to run locally yet**. It is ready as code, but not as data.

### 2.2 Data currently present locally

Local file:

`AI_train_model/data/Epileptic_Seizure_Recognition.csv`

Observed structure:

- shape: `11500 x 180`;
- signal columns: `X1..X178`;
- label column: `y`;
- labels: `1, 2, 3, 4, 5`, each with 2300 samples.

This is not CHB-MIT format:

- it is not EDF;
- it is not 23-channel scalp EEG;
- it has 178 samples per record;
- it has five label classes, often collapsed in public examples into binary seizure vs non-seizure by using `y == 1`.

Research decision:

- Use this CSV only for a quick code sanity test or a separate appendix baseline.
- Do not mix it with CHB-MIT claims.
- Do not design the accelerator around this CSV if the project title remains CHB-MIT EEG seizure detection.

---

## 3. What The Papers Tell Us

### 3.1 Dataset and labeling

Use:

- `10_goldberger_2000_physionet_resource.pdf`
- `12_shoeb_guttag_2010_ml_seizure_detection_icml.pdf`
- PhysioNet CHB-MIT page.

Implication:

- CHB-MIT is the correct dataset citation path.
- Labels must come from seizure onset/end annotations, not from arbitrary balanced CSV rows.
- Detection means ictal vs non-ictal.
- Prediction means preictal vs interictal and needs a separate label builder.

For this project, the first publishable track should be **detection**, because the current repo already implements ictal/non-ictal labeling from CHB-MIT summaries.

### 3.2 Evaluation protocol matters more than headline accuracy

Use:

- `13_ali_2024_chbmit_overlooked_perspectives.pdf`
- `28_lee_2022_real_time_seizure_detection_eeg.pdf`
- `02_chung_2024_single_channel_chbmit_detection.pdf`

Implication:

- Random segment split can leak subject/file-specific patterns.
- Balanced window accuracy can look impressive but does not reflect continuous clinical use.
- Continuous seizure detection should report event-level sensitivity, false alarms per hour, and detection latency.

Required protocol levels:

| Level | Purpose | Publishability |
|---|---|---|
| Random balanced-window split | Smoke test only | Weak |
| Subject-wise split | Tests cross-subject generalization | Stronger |
| Leave-one-subject-out | Most defensible generalization test | Strong |
| Patient-specific split | Useful for personalized wearable detector | Strong if stated clearly |
| Continuous event-level evaluation | Clinically meaningful detection | Required for Q1 direction |

### 3.3 EEG-specific CNN design principles

Use:

- `16_lawhern_2018_eegnet_compact_cnn.pdf`
- `17_schirrmeister_2017_deep_cnn_eeg_decoding.pdf`
- `05_kashefi_2025_dwt_1dcnn_lstm_detection.pdf`

Implication:

- Early temporal convolution acts like a learnable filter bank.
- A spatial/channel-mixing operation is needed because seizure signatures are not equally expressed across all electrodes.
- Depthwise/separable convolution is a principled way to reduce compute and parameters.
- DWT helps CHB-MIT because it exposes time-frequency features, but it adds frontend complexity.
- LSTM/Bi-LSTM/attention can improve temporal modeling, but they are not first-choice blocks for a small accelerator.

### 3.4 Hardware and quantization

Use:

- `09_li_2022_parallel_memristive_cnn_detection_prediction.pdf`
- `18_jacob_2018_integer_only_quantization.pdf`
- `19_ahlawat_2026_int8_quant_channel_pruning_snn.pdf`
- `25_alhammadi_2022_1dcnn_fpga_accelerator_hls.pdf`

Implication:

- Conv layers dominate inference cost; architecture should be chosen with MAC reuse and buffering in mind.
- Integer-only inference needs explicit scale/zero-point or fixed-point policy.
- Post-training quantization can work, but small models are more vulnerable to quantization loss; QAT should be an available second step.
- Channel pruning and structured sparsity are natural follow-ups after the baseline Q16 path.

---

## 4. Current Baseline Model

Current file:

`AI_train_model/src/model.py`

Input:

`(batch, 23, 256)`

Layer profile:

| Layer | Shape transition | Main parameters |
|---|---|---|
| Conv1d | `23 x 256 -> 16 x 256` | kernel 5, stride 1, padding 2 |
| BatchNorm + ReLU + MaxPool | `16 x 256 -> 16 x 128` | pool 2 |
| Conv1d | `16 x 128 -> 32 x 128` | kernel 5, stride 1, padding 2 |
| BatchNorm + ReLU + MaxPool | `32 x 128 -> 32 x 64` | pool 2 |
| FC1 | `2048 -> 32` | dense bottleneck |
| FC2 | `32 -> 16` | dense |
| FC3 | `16 -> 2` | logits |

Measured static profile:

- parameters: `70,674`;
- estimated MACs per 1-second inference: about `864,800`;
- largest parameter block: `fc1`, because it consumes flattened `32 x 64 = 2048` features.

Assessment:

- Good as a first baseline.
- Small enough for early hardware export.
- But it is not yet a strong research architecture because:
  - it uses ordinary random split in the loader;
  - it lacks event-level evaluation;
  - it flattens before FC, increasing dense parameters;
  - it does not test channel reduction, separable conv, DWT, or QAT.

---

## 5. Recommended Model Family

### 5.1 Model A: Baseline 1D-CNN

Purpose:

- Establish a reproducible baseline from the current repo.
- Keep it unchanged initially so future improvements are attributable.

Use when:

- validating CHB-MIT preprocessing;
- checking train/quantize/export pipeline;
- comparing random split vs subject-wise split.

Layer plan:

| Stage | Layer |
|---|---|
| Input | `23 x 256` |
| Temporal/channel conv | `Conv1d(23, 16, kernel=5)` |
| Downsample | `MaxPool1d(2)` |
| Feature conv | `Conv1d(16, 32, kernel=5)` |
| Downsample | `MaxPool1d(2)` |
| Classifier | `Flatten -> FC(2048,32) -> FC(32,16) -> FC(16,2)` |

Keep this as `baseline_1dcnn`.

### 5.2 Model B: EEGNet-Inspired Separable 1D-CNN

Paper basis:

- EEGNet uses temporal filtering, depthwise spatial filtering, separable convolution, average pooling, and dropout.
- Schirrmeister supports the idea of learning from raw EEG through temporal and spatial convolution.

Purpose:

- Reduce parameters and MACs while keeping EEG-specific inductive bias.
- Create a stronger hardware-aware model than the current flatten-heavy baseline.

Proposed 1D version:

| Stage | Layer | Shape |
|---|---|---|
| Input | raw EEG | `23 x 256` |
| Per-channel temporal filters | `Conv1d(23, 46, kernel=31, groups=23, padding=15)` | `46 x 256` |
| Normalize/activate | BN + ReLU or ELU | `46 x 256` |
| Downsample | AvgPool or MaxPool `4` | `46 x 64` |
| Channel mixing | `Pointwise Conv1d(46, 32, kernel=1)` | `32 x 64` |
| Temporal refinement | `Depthwise Conv1d(32, 32, kernel=15, groups=32, padding=7)` | `32 x 64` |
| Downsample | AvgPool or MaxPool `2` | `32 x 32` |
| Classifier | GlobalAveragePool + FC | `32 -> 2` |

Expected advantage:

- Much smaller FC section because it uses global average pooling.
- Better channel/time separation than a plain Conv1d.
- Easier to quantize and accelerate than LSTM or DenseNet.

Suggested name:

`separable_1dcnn`

### 5.3 Model C: Parallel Multi-Kernel 1D-CNN

Paper basis:

- Li et al. use parallel 1D convolution branches with different receptive fields, then pooling and small FC layers.
- This maps naturally to parallel MAC lanes or separate convolution engines.

Purpose:

- Capture short and longer seizure patterns in parallel.
- Match accelerator thinking: independent conv branches can run in parallel.

Proposed CHB-MIT raw-signal version:

| Stage | Layer | Shape |
|---|---|---|
| Input | raw EEG | `23 x 256` |
| Branch A | `Conv1d(23, 16, kernel=15, padding=7)` | `16 x 256` |
| Branch B | `Conv1d(23, 16, kernel=31, padding=15)` | `16 x 256` |
| Merge | concat channels | `32 x 256` |
| Normalize/activate | BN + ReLU | `32 x 256` |
| Downsample | AvgPool or MaxPool `4` | `32 x 64` |
| Refinement | `Conv1d(32, 32, kernel=5, padding=2)` | `32 x 64` |
| Classifier | GlobalAveragePool + FC | `32 -> 2` |

Expected advantage:

- More paper-aligned than a generic two-layer CNN.
- Receptive fields can be justified: short transient spikes and longer rhythmic patterns.
- FC parameter count remains low.

Suggested name:

`parallel_multikernel_1dcnn`

### 5.4 Model D: DWT + Compact 1D-CNN

Paper basis:

- DWT appears repeatedly in the strongest CHB-MIT papers.
- The DWT+CNN-LSTM paper reports large CHB-MIT performance drops when DWT, CNN, or LSTM are removed.

Purpose:

- Test whether explicit time-frequency decomposition improves CHB-MIT detection.
- Keep the deployed classifier compact even if DWT is used offline or as a hardware frontend.

Recommended first version:

| Stage | Layer |
|---|---|
| Input | `23 x 256` raw window |
| Preprocess | DWT per channel, level 3, db3 or db4 |
| Representation | concatenate coefficients per channel without mixing channels |
| Classifier | compact Conv1D or separable Conv1D |
| Classifier head | global average pooling + FC |

Do not start with the full 765k-parameter CNN-LSTM from the paper for the accelerator target. Use it only as a performance comparator if needed.

Suggested name:

`dwt_compact_1dcnn`

---

## 6. Data Pipeline Required Before Serious Training

### 6.1 Fix CHB-MIT preprocessing scope

Current code processes only the first 5 subjects.

Required change:

- make subject list configurable;
- support all CHB-MIT subjects by default;
- write metadata for every extracted segment:
  - subject ID;
  - EDF file name;
  - start second;
  - end second;
  - label;
  - seizure ID or none;
  - channel list.

Without metadata, subject-wise split and event-level evaluation will be painful.

### 6.2 Add filtering as a real option

Current docs mention bandpass filtering, but code does not apply it.

Recommended config:

```yaml
preprocess:
  sample_rate: 256
  window_sec: 1
  stride_sec: 1
  bandpass:
    enabled: true
    low_hz: 0.5
    high_hz: 40.0
  notch:
    enabled: false
    freq_hz: 60.0
```

Paper alignment:

- Chung et al. used 1-30 Hz for scalp seizure rhythm analysis.
- Channel/frequency selection work indicates higher bands, especially gamma/alpha/beta, can matter.
- A practical first bandpass of 0.5-40 Hz or 1-40 Hz is defensible; if gamma up to 50 Hz is desired, use 0.5-50 Hz.

### 6.3 Do not globally balance the test set

Current preprocessing balances seizure and normal globally before splitting.

For research:

- balancing is acceptable for training batches;
- test/evaluation should preserve continuous timeline or realistic imbalance;
- event metrics need full or near-full non-seizure duration to compute false alarms per hour.

Recommended design:

- save all segments and metadata;
- create a balanced training sampler;
- keep validation/test configurable:
  - balanced segment validation for model selection;
  - continuous event test for final reporting.

---

## 7. Evaluation Stack

Minimum metrics:

| Metric | Why |
|---|---|
| Accuracy | easy comparison, but insufficient |
| Sensitivity/Recall | seizure miss risk |
| Specificity | normal rejection |
| Precision | false seizure cost |
| F1 | class-balanced summary |
| AUC | threshold behavior |
| Event sensitivity | clinical seizure detection |
| False alarms per hour | wearable usability |
| Detection latency | real-time feasibility |
| Params/MACs/weight KB | hardware cost |
| Q16/INT8 accuracy delta | quantization cost |

Recommended post-processing:

- sliding predictions over continuous EEG;
- smooth with majority vote or median/Savitzky-Golay-like filter;
- declare event only after `M` consecutive positive windows;
- merge adjacent detections separated by short gaps;
- compare predicted events with annotated seizure intervals.

This makes the study closer to Shoeb-Guttag, Chung et al., Ali et al., and Lee et al. than to weak balanced-window classification.

---

## 8. Hardware-Aware Research Claims

Strong claim candidates:

1. **Protocol claim:** leakage-safe CHB-MIT evaluation with continuous event-level detection.
2. **Architecture claim:** separable or parallel Conv1D gives a better accuracy/resource tradeoff than plain Conv1D.
3. **Input-efficiency claim:** selected channel/band subsets reduce acquisition and compute cost with controlled loss.
4. **Quantization claim:** Q16 or INT8 inference preserves event-level performance while reducing memory/MAC hardware cost.
5. **Accelerator claim:** layer shapes and memory reuse enable deterministic low-latency FPGA/ASIC inference.

Weak claim candidates to avoid:

- "99% accuracy" on random balanced segments.
- comparing detection directly with prediction papers.
- using the local 178-feature CSV as if it were CHB-MIT.
- deploying LSTM/GAN/DenseNet without hardware resource justification.

---

## 9. Concrete Experiment Roadmap

### Step 0. Data readiness

- Download CHB-MIT raw EDF data.
- Confirm `chb01..chb24` folders and summaries exist.
- Generate a metadata-rich segment dataset.

Output:

- `chbmit_segments.npz`
- `chbmit_segments_metadata.csv`
- preprocessing report with subject/file/seizure counts.

### Step 1. Baseline reproduction

- Train current `baseline_1dcnn`.
- Run random balanced split only as a smoke test.
- Run subject-wise split for realistic baseline.
- Export Q16 weights.

Output:

- FP32 metrics;
- Q16 metrics;
- params/MACs.

### Step 2. Fix final evaluation

- Implement event reconstruction on continuous records.
- Report event sensitivity, false alarms/hour, latency.
- Compare against segment-level metrics.

Output:

- event-level evaluation report.

### Step 3. Architecture variants

Train:

1. `baseline_1dcnn`
2. `separable_1dcnn`
3. `parallel_multikernel_1dcnn`
4. `dwt_compact_1dcnn`

Compare:

- accuracy/sensitivity/specificity/F1/AUC;
- event sensitivity/FAR/h/latency;
- params/MACs/activation memory;
- Q16 accuracy loss.

### Step 4. Channel and band reduction

Run:

- 23 channels;
- 18 common channels;
- 4 selected channels;
- single-channel patient-specific experiments;
- optional frequency band subsets.

Target:

- find Pareto points between performance and hardware cost.

### Step 5. Quantization and accelerator readiness

Run:

- current dynamic Q16 post-training quantization;
- static per-layer Q format;
- optional INT8/QAT if Q16 is stable and hardware target moves to lower precision.

Report:

- per-layer scale;
- saturation/clipping;
- accumulator bit width requirement;
- weight memory;
- activation memory;
- estimated cycles per layer.

---

## 10. Immediate Code Backlog

Highest priority:

1. Add metadata-rich CHB-MIT preprocessing.
2. Add subject-wise split support.
3. Add event-level evaluator.
4. Add model registry with baseline/separable/parallel variants.
5. Add model profiler for params, MACs, activation shapes, and weight memory.

Second priority:

6. Add bandpass option.
7. Add configurable channel subsets.
8. Add DWT feature frontend.
9. Add QAT or INT8 path.
10. Add experiment runner that saves all config/results in `server_results/run_XX`.

---

## 11. Final Recommendation

The next correct implementation move is **not** to immediately add a larger model. The next correct move is:

1. make CHB-MIT data handling scientifically valid;
2. add subject/event-aware evaluation;
3. keep current compact 1D-CNN as baseline;
4. add separable and parallel Conv1D variants;
5. quantify performance vs hardware cost under Q16.

This path matches the project title and gives a stronger journal story than chasing high random-split accuracy.

