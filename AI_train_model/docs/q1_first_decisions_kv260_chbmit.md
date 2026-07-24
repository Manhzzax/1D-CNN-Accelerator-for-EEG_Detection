# Q1 First Decisions for CHB-MIT + KV260

Created: 2026-07-24

Current situation:

- Server is loading/preprocessing CHB-MIT directly and has reached around `chb04`.
- Training hardware is available.
- Deployment/prototype hardware target is AMD/Xilinx Kria KV260.
- The project target is a hardware-aware 1D-CNN accelerator for EEG seizure detection.

---

## 1. First Thing To Determine

Before chasing model accuracy, determine the **exact research claim**:

> Can a compact, fixed-point, KV260-deployable 1D-CNN detect seizures on CHB-MIT with leakage-safe evaluation and useful event-level performance, while reducing compute/memory compared with heavier EEG deep-learning models?

This claim determines everything else:

- task: detection, not prediction;
- dataset: CHB-MIT EDF, not the local 178-point CSV;
- evaluation: subject-aware and event-level, not only balanced-window accuracy;
- architecture: compact Conv1D/separable/parallel Conv1D, not GAN/LSTM/DenseNet as the first deployed model;
- hardware target: KV260 FPGA resource, latency, throughput, and memory.

---

## 2. Decisions To Lock Before Full Training

### D1. Task Definition

Primary task:

- `seizure detection`: ictal vs non-ictal.

Do not mix with:

- `seizure prediction`: preictal vs interictal.

Reason:

- The current repo labels windows from `chbXX-summary.txt` seizure start/end times.
- Prediction requires a separate preictal labeling protocol.

### D2. Dataset Scope

For a Q1-level paper, do not train only on the first 4 or 5 patients except as a debug run.

Recommended:

- debug stage: `chb01..chb05`;
- first full baseline: all available CHB-MIT cases;
- final reporting: clearly list subjects/cases, EDF files, seizure files, seizure count, and total EEG hours.

Required preprocessing output:

- segment tensor file, e.g. `chbmit_segments.npz`;
- metadata file, e.g. `chbmit_segments_metadata.csv`;
- preprocessing report.

Metadata needed per segment:

- subject/case ID;
- EDF file name;
- start second;
- end second;
- label;
- seizure ID;
- channel list;
- whether segment is train/val/test later.

### D3. Window And Stride

Current repo uses:

- `window_sec = 1`;
- `stride_sec = 1`;
- input shape `23 x 256`.

Paper-supported alternatives:

- Shoeb-Guttag uses contiguous 2-second epochs and temporal stacking.
- Chung et al. uses 4-second windows and event-level post-processing.
- Ali et al. uses 5-second non-overlapping windows for continuous event analysis.

Recommended for this project:

- keep `1s` as the hardware baseline because it gives low-latency inference and simple buffering;
- add experiment options for `2s` and `4s`;
- report latency tradeoff explicitly.

### D4. Split Protocol

Minimum:

- random balanced-window split for smoke test only.

Required for paper:

- subject-wise split;
- leave-one-subject-out if feasible;
- patient-specific split as a separate analysis, not mixed with generalized claims.

Rationale:

- Random window split can leak subject/file-specific patterns.
- Q1 reviewers will challenge high accuracy if the split is weak.

### D5. Test Distribution

Do not globally balance the final test set.

Recommended:

- balance training batches or training subset;
- keep final test close to continuous/realistic imbalance;
- compute event-level false alarms per hour from non-seizure duration.

### D6. Metrics

Segment-level:

- accuracy;
- sensitivity/recall;
- specificity;
- precision;
- F1;
- AUC.

Event-level:

- seizure event sensitivity;
- false alarms per hour;
- detection latency;
- missed seizure count;
- number of predicted events.

Hardware-level:

- parameters;
- MACs/inference;
- activation memory;
- weight memory FP32/Q16/INT8;
- latency on KV260;
- LUT/FF/BRAM/URAM/DSP usage;
- power or energy/inference if available.

### D7. KV260 Design Envelope

KV260/K26 facts to use as the implementation envelope:

- device family: Zynq UltraScale+ MPSoC;
- system logic cells: 256K;
- block RAM blocks: 144;
- UltraRAM blocks: 64;
- DSP slices: about 1.2K;
- DDR memory: 4 GB non-ECC DDR4;
- K26 product brief advertises up to 1.4 TOPS AI processing.

Design implication:

- The current model is already small enough to fit.
- The real research value is not simply fitting the model, but showing a disciplined accuracy/resource/latency tradeoff.
- Use the ARM PS for control, DMA, post-processing, and event reconstruction.
- Use PL for Conv1D/FC fixed-point acceleration.

Official sources:

- AMD KV260 DS986 product details: https://docs.amd.com/r/en-US/ds986-kv260-starter-kit/Product-Details
- AMD/Xilinx K26 product brief: https://www.xilinx.com/publications/product-briefs/xilinx-k26-product-brief.pdf

---

## 3. Recommended Architecture Track

### Track A. Baseline First

Run the current model unchanged:

- `Conv1d(23,16,k=5) -> pool`
- `Conv1d(16,32,k=5) -> pool`
- `FC 2048->32->16->2`

Purpose:

- verify preprocessing;
- verify training;
- verify Q16 export;
- establish reference metrics.

### Track B. Hardware-Aware Main Candidate

Build `separable_1dcnn`:

- per-channel temporal Conv1D;
- pointwise channel mixing;
- depthwise temporal refinement;
- global average pooling;
- small FC head.

Why:

- inspired by EEGNet and EEG ConvNet design principles;
- fewer dense parameters;
- better fit for fixed-point FPGA mapping.

### Track C. Accelerator-Friendly Parallel Candidate

Build `parallel_multikernel_1dcnn`:

- branch A kernel 15;
- branch B kernel 31;
- concat;
- compact refinement conv;
- global average pooling.

Why:

- inspired by Li et al. parallel memristive CNN;
- maps cleanly to parallel convolution engines;
- captures short spikes and longer rhythmic activity.

### Track D. DWT Variant

Build `dwt_compact_1dcnn` only after A/B/C:

- DWT per channel;
- compact CNN classifier;
- no LSTM in the first hardware version.

Why:

- DWT is repeatedly useful in CHB-MIT papers;
- full CNN-LSTM is too heavy for the first KV260 accelerator story.

---

## 4. What To Capture While Server Is Preprocessing

Ask the server run to log:

- current subject being processed;
- number of EDF files;
- number of seizure files;
- number of annotated seizures;
- extracted seizure windows;
- extracted normal windows;
- skipped files and why;
- channel mismatches;
- sample rate mismatches;
- total wall-clock time per subject.

This is important because the paper's Methods section needs an auditable preprocessing description.

---

## 5. Q1 Paper Storyline

Recommended storyline:

1. Existing CHB-MIT studies often report strong accuracy but vary in split protocol, channel usage, and event-level reporting.
2. Wearable/edge seizure detection needs low latency, low memory, and low false alarm rate.
3. We propose a hardware-aware compact 1D-CNN family for CHB-MIT detection.
4. We evaluate under leakage-safe subject/event-aware protocols.
5. We show the accuracy/resource tradeoff across baseline, separable, parallel, channel-reduced, and quantized models.
6. We map the selected model to KV260 and report FPGA resources, latency, throughput, and quantization loss.

---

## 6. Immediate Priority

The first implementation priority is:

1. finish CHB-MIT preprocessing with metadata;
2. prevent subject/file leakage in splits;
3. implement event-level evaluation;
4. train the existing baseline;
5. only then modify architecture.

If we skip steps 1-3, high accuracy will be weak evidence for Q1 publication.

