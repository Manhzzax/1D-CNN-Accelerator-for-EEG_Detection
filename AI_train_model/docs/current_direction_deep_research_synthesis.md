# Deep-Research Synthesis: KV260 Seizure Detector Direction

## Research Question

Can a raw multichannel depthwise-separable 1D-CNN retain defensible seizure
detection performance while providing an INT16 FPGA evidence chain?

The contribution must be a controlled accuracy-efficiency trade-off. It cannot
be inferred from a window-accuracy leaderboard or parameter count alone.

## Immediate Professor-Mandated Gate

Before KV260 implementation work, Track B must exceed **95.0% balanced
validation-window accuracy** without abandoning a hardware-feasible Conv1D
graph. A lucky maximum does not satisfy this gate:

1. A method passes its seed-42 screen only if its validation-loss-selected
   checkpoint reaches at least 95.0% accuracy.
2. It becomes the accuracy winner only if seeds 42, 7, and 123 have a mean
   balanced validation-window accuracy of at least 95.0%.
3. Every seed, its best epoch, AUROC, F1, sensitivity, parameters, MACs, and
   input/activation memory must be reported.

This is an internal accuracy gate imposed for the current research phase. It
does not replace causal event and patient-group-held-out evidence later.

## Evidence From the Existing Corpus

| Evidence stream | Papers | Evidence-based conclusion | Project consequence |
|---|---|---|---|
| Compact raw EEG CNN | EEGNet (M01), Schirrmeister (M02), LMPSeizNet (A13) | Temporal filtering plus spatial mixing with separable convolution is a credible compact EEG representation. | Keep raw separable 1D-CNN as the inference graph. |
| Feature-heavy high accuracy | Kashefi (A04), Cao (A05), Alharthi (A06) | DWT, feature fusion, recurrent layers, and attention can raise reported classification scores but enlarge or move the system boundary. | Use as accuracy context, not as the first KV260 implementation template. |
| Clinical detection protocol | Shoeb (D02), Ali (D03), Chung (A02), Lee (A11), Ghosh (A15) | Continuous EEG requires event sensitivity, FAR/h, and delay. Patient-independent evaluation is harder. | Accuracy screening is allowed, but frozen models require causal continuous and patient-group-held-out evaluation. |
| Hardware seizure systems | Li TBioCAS (H01), Bahr GAP8 (H04), EEGformer TBioCAS (H02), FPGA SNN TBioCAS (H03) | Target-specific latency, energy, and alarm behavior are central results. | Finish with measured KV260 PPA and bit agreement, not merely exported tensors. |
| Accelerator method | Alhammadi HLS (H05), Jacob integer-only quantization (M03), AMD HLS verification (M04) | Quantization requires specified types, accumulator widths, rounding, saturation, requantization, and C/RTL checking. | Define an INT16 arithmetic contract before HLS. |

## Synthesis for the Current Model

### Raw 1D-CNN remains the correct primary family

The `31/7/3` R2 model is EEGNet-like in its essential factorization: temporal
processing per EEG channel followed by learned spatial mixing. It avoids a DWT,
STFT, CSP, LSTM, attention, or spectrogram frontend in the deployed graph.
Conv1D, BatchNorm folding, ReLU, pooling, residual add, and a linear classifier
form a small, regular operator set for deterministic INT16 HLS.

This does not claim raw 1D-CNN is universally more accurate. The DWT/CNN-LSTM
and CNN-Bi-LSTM papers define the high-accuracy context, but their complete
frontend and recurrent cost must be included in an accelerator comparison.

### Five seconds is an accuracy design point, not automatically the deployment winner

Ali et al. use 5 s non-overlapping windows in a cross-subject continuous study,
so a 5 s choice is literature-supported. Locally, R2 rises from
90.200% +/- 0.850% at 2 s to 93.081% +/- 1.096% at 5 s across matched seeds.

However, the 5 s population is different and deployment cost rises: its INT16
input is 42.5 KiB and the graph has about 4.52 MMAC/window. It cannot be called
better for KV260 without causal event results and measured PPA.

### The paper should test two measured design points

| Design point | Role | Evidence required |
|---|---|---|
| EpiSepNet-5K, 2 s, 5,013 parameters | Latency and buffer reference | Existing FP32/INT16-emulator agreement, then full KV260 PPA. |
| EpiSepNet-R2-5K, 5 s, 4,917 parameters | Accuracy and temporal-context candidate | Five-seed accuracy, causal event behavior, INT16 agreement, and identical KV260 PPA. |

If both are measured with the same board, clock, precision, toolchain, and
input boundary, the comparison itself is a contribution: accuracy gained versus
buffering, MACs, latency, and energy. If only one can be implemented, select it
only after the causal and patient-group-held-out gate.

### Clinical metrics constrain architecture selection

EEGformer reports seizure detection probability, FAR/h, onset delay, and MCU
latency/energy rather than presenting isolated window accuracy as the outcome.
[Busia et al., 2024](https://doi.org/10.1109/TBCAS.2024.3357509) The recent
FPGA SNN work likewise reports event detection, FAR/h, inference time, and
energy on its target despite using a different sparse SNN architecture.
[Busia et al., 2025](https://doi.org/10.1109/TBCAS.2025.3575327)

The defensible objective is therefore:

> Maximise patient-group-held-out causal event sensitivity at a prespecified
> FAR/h target, subject to measured KV260 resource, latency, and energy limits.

The internal 95% balanced-window target is useful for screening only. It is not
a clinical or literature-standard acceptance threshold.

## Preserve and Avoid

Preserve:

- raw 17-channel canonical bipolar input;
- depthwise temporal plus pointwise spatial Conv1D;
- train-only normalisation, recording-level provenance, and audit artifacts;
- compact operators compatible with integer C and HLS;
- separate frozen 2 s engineering reference and 5 s accuracy candidate.

Do not add without a new measured hypothesis:

- DWT/STFT/CSP frontend, which weakens the end-to-end INT16 story;
- LSTM/Bi-LSTM/attention, which introduces state and verification burden;
- broad width/depth/kernel retries after locally rejected screens;
- further architecture selection on the same validation cohort after the
  prespecified dilated-R2 screen.

## Ordered Plan

1. Screen the prespecified dilated R2 5 s topology at seed 42. It adds only
   320 parameters and keeps Conv1D/add/ReLU/pooling operators. Replicate seeds
   7 and 123 only if the validation-loss-selected checkpoint reaches 95.0%.
2. P1 failed. Test exactly these remaining accuracy-directed methods in order:
   a supervised-contrastive auxiliary training loss on the 32-D feature vector,
   mild train-only augmentation, then one controlled 5 s capacity check. Each
   receives one seed-42 screen before replication; do not perform broad
   unstructured sweeps. The P2 contract is fixed in
   `docs/supervised_contrastive_accuracy_screen.md`.
3. Do not use hard-negative mining as an accuracy lever. Its role is to reduce
   false alarms after the >=95% candidate has been frozen, and it may lower
   balanced clean-window accuracy by making the negative class more difficult.
4. Freeze the first method with a three-seed mean >=95.0%. Repeat it and R2 5 s
   baseline at five seeds, reporting all runs and recording-block uncertainty.
5. Recreate the frozen protocol using causal IIR preprocessing. Run continuous
   validation, choose one FAR-constrained threshold/policy, then lock it.
6. Run one patient-group-disjoint test only after all choices are frozen.
   Report event sensitivity, FAR/h, delay, seizure-duration strata, and a
   patient/recording block-bootstrap interval.
7. Export tensors and metadata. Implement integer multiplier/shift
   requantization with declared accumulator widths, then verify FP32 -> integer
   reference -> RTL -> board agreement.
8. On KV260, report clock, throughput, end-to-end latency, BRAM, DSP, LUT, FF,
   board power, energy/window, weight bytes, and activation-buffer strategy.

### Controlled P1 Outcome: Dilated R2 Rejected

`run_63_r2_dilated5s_d4_d8_s42` completed the predeclared P1 screen with two
post-pooling depthwise residual blocks (kernel 3, dilations 4 and 8). The
model has 5,237 parameters, 320 more than the 4,917-parameter R2 baseline.
Its validation-loss-selected checkpoint was epoch 8 and obtained 92.159%
accuracy, 97.762% AUROC, 91.969% F1, 89.796% sensitivity, and 94.250%
precision. The matched baseline seed-42 run (`run_60_r2_raw5s_s42`) obtained
92.830% accuracy, 97.969% AUROC, 92.794% F1, 92.320% sensitivity, and
93.272% precision.

Thus P1 changes the operating point toward precision but is worse on the
accuracy-screen objective by 0.671 percentage points and on sensitivity by
2.524 points. It fails the 95% seed-42 gate; do not replicate it at seeds 7
and 123. Preserve it as a negative, controlled ablation and proceed to the
predeclared supervised-contrastive auxiliary-loss screen rather than tuning
its dilations, widths, or patience.

### Controlled P2 Outcome: Supervised Contrastive Loss Improves but Does Not Pass

`run_64_r2_5s_supcon005_t01_s42` retained the 4,917-parameter inference
model and added the training-only objective `CE + 0.05 * SupCon` with
temperature 0.1. The validation-CE-selected epoch-23 checkpoint achieved
93.367% accuracy, 98.112% AUROC, 93.394% F1, 93.770% sensitivity, and
93.021% precision. Relative to the matched baseline seed 42, accuracy rose
0.837 points and sensitivity 1.450 points, while parameter count and
inference graph were unchanged.

This is useful evidence that representation regularisation helps the compact
R2 model, but it does not meet the predeclared 95% seed-42 gate. Do not tune
the SupCon coefficient, temperature, or early-stopping settings and do not
replicate seeds 7/123. Preserve P2 as the leading development screen and move
to the one predeclared mild train-only augmentation experiment.

## Paper Thesis

> We co-design and measure a raw multichannel depthwise-separable EEG detector
> for KV260. A controlled 2 s versus 5 s study exposes the accuracy, buffering,
> and energy trade-off; integer-verified HLS shows whether the compact model
> remains reliable after deployment.

This is stronger than an unmatched highest-accuracy claim and matches TBioCAS
algorithm-hardware co-design framing. [Wei et al.,
2020](https://doi.org/10.1109/TBCAS.2020.2974154)

## Primary Sources

- [Ali et al., 2024: continuous cross-subject CHB-MIT](https://doi.org/10.1098/rsos.230601)
- [Chung et al., 2024: segment and event metrics](https://doi.org/10.3389/fneur.2024.1389731)
- [Lawhern et al., 2018: EEGNet](https://doi.org/10.1088/1741-2552/aace8c)
- [Li et al., 2022: memristive CNN co-design](https://doi.org/10.1109/TBCAS.2022.3185584)
- [Bahr et al., 2021: GAP8 CNN deployment](https://doi.org/10.3390/bios11070203)
- [Busia et al., 2024: EEGformer on MCUs](https://doi.org/10.1109/TBCAS.2024.3357509)
- [Busia et al., 2025: FPGA SNN seizure detector](https://doi.org/10.1109/TBCAS.2025.3575327)
- [Jacob et al., 2018: integer-only quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)
