# R2 Lite Accuracy Deep-Research Roadmap

## Scope and decision rule

This document concerns **Track B accuracy research only**. The frozen Track A
accelerator reference, EpiSepNet-5K, is not modified. All candidate selection
uses the locked within-case chronological validation split, raw 17-channel EEG,
and training/validation data only. The historical test cohort is not a model
selection resource.

The R2 Lite target remains a three-seed mean balanced validation-window
accuracy of at least 95%. This is an internal research target, not a literature
threshold: published 94--99% values frequently use different subjects,
re-annotations, window populations, feature pipelines, and patient-specific
splits. A high external number is context, never permission to compare ranks.

## Current evidence

R2 Lite is a depthwise-separable `31/7/3` raw Conv1D model. It uses three
temporal filters per each of 17 channels, one pointwise spatial mixing layer,
two average-pooling operations, global average pooling, and a linear classifier.
The selected seed-42 checkpoint has 4,917 parameters and 91.175% balanced
validation-window accuracy.

| Controlled seed-42 screen | Parameters | Accuracy | Interpretation |
|---|---:|---:|---|
| R2 Lite `31/7/3` + Adam | 4,917 | **91.175%** | Current accuracy reference |
| Temporal filters `m4`, width 32 | 6,022 | 90.328% | More capacity overfits; reject |
| Width 48, temporal filters `m3` | 7,301 | 90.046% | More capacity overfits; reject |
| First kernel 15 / 47 / 63 | 4,101 / 5,733 / 6,549 | 89.277 / 90.457 / 89.302% | Kernel 31 is non-monotonic optimum |
| Parameter-matched residual `31/7/3` | 4,917 | 89.302% | Identity skips do not help |
| Compact multiscale `15+31` | 3,636 | 87.891% | Parallel early branches do not help |
| R2 Lite + AdamW, same `lr=1e-3`, `wd=1e-4` | 4,917 | 90.713% | Lower accuracy and sensitivity; reject |

The best R2 seed-42 checkpoint is epoch 23, selected by minimum validation
loss (0.2474). Its learning rate dropped only after epoch 26, and later loss
never improved the selected value. Increasing early-stopping patience is
therefore not an evidence-based route to a better checkpoint. Selection must
continue to use the predeclared minimum validation-loss rule, rather than
choosing a later epoch only because one sampled accuracy happens to be larger.

## Why R2 plateaus near 91%

### 1. Context is limited before global pooling

The local receptive field of R2 Lite is 102 samples, about 398 ms at 256 Hz.
Global average pooling sees all 2 seconds but discards the order and persistence
of the local features. A 5-second input gives the pooling operator more context,
but it does not make a 398 ms convolutional feature detector into a 5-second
temporal detector. This explains why a 5-second input must be paired, if needed,
with a compact temporal-context head rather than assumed to improve accuracy by
itself.

### 2. The training set is compact despite the 43 GB raw corpus

The current `raw_2s_v2` training artifact contains 5,344 ictal and 5,344
reservoir-sampled normal windows. Its preparation scanned about 1.84 million
eligible normal windows. Class-balanced batches are appropriate for the
balanced validation accuracy endpoint, but they do not expose the classifier to
the full diversity of normal EEG and artefacts. This is a plausible source of
the remaining decision-boundary error; it is not fixed by merely adding model
width.

### 3. Existing screens already exclude several easy explanations

The plateau is not evidence that all CNN improvements are impossible. It does
exclude, for this protocol, larger flat capacity, a longer single first kernel,
early parallel `15+31` branches, identity residuals, AdamW at the matched decay,
and a longer early-stopping tail. The previous score-only TCN is also not a
reason to reject an end-to-end temporal CNN: it operated on frozen scalar scores
rather than raw EEG features and could not correct feature extraction jointly.

## Literature-grounded implications

- **Compact separable temporal/spatial processing remains justified.** EEGNet
  introduced depthwise and separable convolutions specifically to build compact
  EEG models with limited training data. R2 follows this principle rather than
  adding an LSTM or transformer directly. [Lawhern et al., 2018](https://doi.org/10.1088/1741-2552/aace8c)
- **More temporal crops can help CNN training, but must remain training-only.**
  Schirrmeister et al. found cropped training improved many EEG decoding
  subjects. Our one-second stride already supplies overlapping 2-second crops;
  a future crop/augmentation experiment must be restricted to training
  recordings and never leak a recording into validation. [Schirrmeister et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5655781/)
- **Longer context is a credible hypothesis, not a conclusion.** Ali et al.
  use non-overlapping 5-second windows for continuous cross-subject event
  detection, while Chung et al. use 4-second CNN inputs. Both differ materially
  in task and protocol, so they motivate the isolated 5-second ablation only.
  [Ali et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11286169/)
  [Chung et al., 2024](https://doi.org/10.3389/fneur.2024.1389731)
- **Feature and temporal diversity are useful, but expensive.** A recent
  lightweight cross-subject CNN uses PSD features and 61,218 parameters yet
  reports 85.84% accuracy; a strict patient-independent feature-engineering
  study reaches 90.6% with only 15 selected features. These support spectral
  information as a possible later representation ablation, but not a claim that
  it will improve our within-case raw benchmark. [Gu et al., 2026](https://doi.org/10.1038/s41598-026-44536-y)
  [Ghosh et al., 2026](https://doi.org/10.1007/s42452-026-08306-9)
- **Contrastive objectives are a training-time opportunity.** CHB-MIT seizure
  studies report self-supervised or supervised contrastive training to improve
  representation separation, often alongside much heavier attention models.
  A projection head can be discarded after training, retaining R2's inference
  graph and FPGA cost. This requires its own controlled ablation rather than a
  borrowed performance claim. [SLAM, 2023](https://doi.org/10.1016/j.bspc.2023.105464)
  [Li et al., 2026](https://doi.org/10.1007/s10916-026-02395-0)

## Ranked experiment path

| Priority | Controlled change | Why it can help | FPGA effect | Gate |
|---|---|---|---|---|
| P0 | Unchanged Adam R2 Lite on isolated raw 5-second windows | Tests longer aggregate context and a literature-supported window scale | Input buffer and Conv1D activations about 2.5x; same 4,917 weights | First inspect eligible seizure/window counts; then seed-42 training |
| P1 | End-to-end dilated depthwise context head on the 5-second R2 features | Restores temporal order before global pooling; two depthwise `k=3`, dilation 4/8 layers extend the local field to about 1.9 s | About 0.32K added parameters and negligible relative MAC increase; only Conv1D, add, ReLU, pooling | Run only if P0 shows useful 5-second coverage but does not clear the accuracy screen |
| P2 | Supervised-contrastive auxiliary loss on R2's 32-D feature vector | Improves class clustering without changing inference graph; training head is discarded | No inference parameters or FPGA operators added | Predeclare temperature and loss-weight screen; compare against cross-entropy R2 on the same seed |
| P3 | Increase only **training** normal diversity from the 1.84M eligible pool while retaining class-balanced batches | Broadens normal/artifact exposure without changing validation denominator or deployed model | No inference cost; larger server-side NPZ and I/O only | New training prepared artifact; hold validation/test ratios fixed and report candidate counts |
| P4 | Mild, label-preserving training augmentation: amplitude scaling, small time shift, and band-limited noise | Increases crop variability without synthetic seizure generation | No inference cost | One augmentation family at a time; no GAN or validation augmentation |
| P5 | Compact spectral auxiliary features or fixed filter-bank branch | Raw R2 may miss stable band-power cues | Adds fixed DSP and a small feature head; FPGA feasibility must be costed | Only after P0--P4; do not revive the rejected DWT pipeline without a new rationale |

P1 is fundamentally different from the rejected score-TCN: it learns temporal
raw-EEG features end-to-end. It is also more hardware-feasible than LSTM or
Transformer alternatives. A recent lightweight TCN preprint similarly uses
depthwise residual blocks and dilation to represent short and long EEG context;
it is motivation only until independently reproduced. [Pedram et al., 2026 preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6430022)

## What not to do

1. Do not select a model or threshold using the observed historical test set.
2. Do not claim 95% merely because a non-comparable paper reports 95--99%.
   Several such results are patient-specific, feature-engineered, recurrent, or
   use a different window/label cohort.
3. Do not add LSTM, Transformer, or a GAN to the deployed graph during the
   first hardware-aware campaign. They may be AI comparators, but they weaken
   the controlled KV260 contribution.
4. Do not repeat residual, width, long-single-kernel, multiscale-early-branch,
   AdamW, or patience ablations without a new hypothesis.
5. Do not call a 5-second sampled-window accuracy a direct improvement over
   the 2-second result. Report it as a separate context protocol with eligible
   event count, window counts, MACs, input-buffer size, and causal event metrics.

## Decision after the running 5-second preprocessing

1. Verify that the prepared feature shape is `17x1280`, filters and train-only
   normalisation match the stated protocol, and record the seizure/window count
   loss relative to the 2-second artifact.
2. If short-event coverage remains acceptable, train unchanged Adam R2 Lite at
   seed 42 only. This is P0 and does not inspect test recordings.
3. If P0 does not clear the seed-42 2-second reference by at least 0.5 pp,
   retain it as a context ablation and implement P2 before P1. If P0 is
   promising but global pooling is the apparent bottleneck, implement P1.
4. Any candidate that clears the screen must be confirmed with seeds 7 and 123
   before it receives an INT16 export or any FPGA work.
