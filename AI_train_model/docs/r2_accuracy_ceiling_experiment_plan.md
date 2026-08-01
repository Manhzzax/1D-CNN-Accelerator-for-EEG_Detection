# R2 Accuracy-Ceiling Experiment Plan

## Why More Experiments Are Required

The 5-second R2 study has excluded one added post-pooling dilated context
topology and has shown positive but sub-95% effects from SupCon and mild
augmentation. This is insufficient to state that the raw CNN direction has
reached its local ceiling. It has not isolated the first temporal receptive
field or modest spatial capacity under the same 5-second protocol.

The plan is motivated by CNN-only prior work: Wang et al. use a stacked 1D-CNN
with two-second windows and training augmentation; LMPSeizNet uses multiscale
temporal/spatial convolution; Cimr et al. report a deeper CNN without a
handcrafted feature extractor. These motivate tests, not copied accuracy
claims, because their cohorts and splits differ. [Wang et al., 2021](https://doi.org/10.1016/j.neucom.2021.06.048)
[Alsaadan et al., 2024](https://doi.org/10.3390/math12233648)
[Cimr et al., 2023](https://doi.org/10.1016/j.cmpb.2022.107277)

## Fixed Conditions For Every Screen

- Audited 17-channel raw EEG; 5-second windows, 1-second stride;
  `chbmit_prepared_raw_5s_v1`.
- Locked within-case chronological validation; 1:1 sampled validation windows.
- Train-channel z-score, Adam, learning rate 0.001, weight decay 0.0001,
  batch size 128, class-balanced sampler, seed 42, minimum validation CE-loss
  checkpoint, and existing early-stopping rule.
- No test evaluation, continuous event evaluation, quantization, or FPGA
  export during architecture selection.
- Each run records parameters from `model_spec.json`; accuracy is never copied
  from the maximum validation-accuracy epoch.

## Predeclared Ladder

| Order | Run ID | Single changed hypothesis | Continue rule |
|---|---|---|---|
| Done | `run_63_r2_dilated5s_d4_d8_s42` | post-pooling dilated context | Rejected: 92.159% |
| Done | `run_64_r2_5s_supcon005_t01_s42` | training-only SupCon | Positive but below gate: 93.367% |
| Done | `run_65_r2_5s_aug_g10_n02_s42` | mild train-only gain/noise | Positive but below gate: 93.985% |
| Done | `run_66_r2_k15_s42` | first temporal kernel 15 | 93.340% accuracy; positive versus 31-sample baseline, still below gate |
| K47 | `run_67_r2_k47_s42` | first temporal kernel 47 | Run next K value regardless of result |
| K63 | `run_68_r2_k63_s42` | first temporal kernel 63 | Run M15+31 regardless of result |
| M15+31 | `run_69_ms15_31_2x32_s42` | parallel depthwise temporal kernels 15 and 31 | Run W48 regardless of result |
| W48 | `run_70_r2_w48_s42` | spatial width 48 instead of 32 | Consider one combination only under rule below |

The kernel screens retain R2's second/third kernels `7/3`, three temporal
filters per input channel, and 32 spatial filters; only the first kernel
changes. M15+31 uses the existing compact multiscale separable CNN with two
filters per branch and 32 spatial filters, so the model spec supplies its exact
parameter count. W48 retains R2 `31/7/3` with 48 spatial filters.

## Observed Results

All figures below are from the checkpoint selected by minimum validation
cross-entropy, rather than the epoch with maximum validation accuracy.

| Run | First temporal kernel | Parameters | Validation accuracy | AUROC | F1 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `run_60_r2_raw5s_s42` | 31 | 4,917 | 92.830% | 97.969% | 92.794% | Seed-42 R2 baseline |
| `run_66_r2_k15_s42` | 15 | 4,101 | 93.340% | 98.252% | 93.243% | +0.510 percentage points while using 16.6% fewer parameters |

The K15 result supports continuing the predeclared receptive-field screen. It
does not establish superiority over the best observed seed or satisfy the
95.0% selection gate.

## One Conditional Combination Only

If W48 improves on the plain R2 seed-42 baseline (92.830%) but remains below
95%, run exactly one final screen: W48 + the already-fixed SupCon objective
(`0.05`, temperature `0.1`). Otherwise do not combine factors. This prevents
an open-ended hyperparameter search while testing the only two independently
positive directions, capacity and SupCon.

## Stop and Replication Rules

1. A screen reaching at least 95.0% balanced validation accuracy at its
   validation-CE-selected checkpoint earns seeds 7 and 123.
2. A three-seed mean at least 95.0% freezes the development winner.
3. If every screen and the single allowed combination fail, report a **local
   ceiling within this declared compact raw-CNN search space**, not a universal
   CNN maximum. Stop accuracy searching, retain the Pareto R2 candidate, and
   move to causal/event and KV260 evidence.
4. For any winner, run all five predeclared seeds and one untouched
   patient-group-disjoint causal test before a paper-level claim.

The held-out patient test, rather than the development ladder, protects the
final study from repeated-validation selection bias.

## Next Pending Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate chbmit-cnn && cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && git pull origin main && CHBMIT_WINDOW_SEC=5 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_5s_v1 CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn CHBMIT_HIERARCHICAL_TEMPORAL_KERNEL=47 CHBMIT_TRAIN_SEED=42 CHBMIT_RUN_ID=run_67_r2_k47_s42 CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```
