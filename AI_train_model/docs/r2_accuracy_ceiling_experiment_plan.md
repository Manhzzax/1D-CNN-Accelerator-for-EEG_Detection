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
| Done | `run_67_r2_k47_s42` | first temporal kernel 47 | 93.797% accuracy; best kernel screen so far, still below gate |
| Done | `run_68_r2_k63_s42` | first temporal kernel 63 | 93.367% accuracy; below K47, so select K1=47 |
| Done | `run_69_r2_k47_k2_3_s42` | second temporal kernel 3 | 93.045% accuracy; lower than K2=7, but sensitivity reaches 95.005% |
| K2=5 | `run_70_r2_k47_k2_5_s42` | second temporal kernel 5 | Run next K2 value regardless of result |
| K2=11 | `run_71_r2_k47_k2_11_s42` | second temporal kernel 11 | Run next K2 value regardless of result |
| K2=15 | `run_72_r2_k47_k2_15_s42` | second temporal kernel 15 | Run next K2 value regardless of result |
| K2=31 | `run_73_r2_k47_k2_31_s42` | second temporal kernel 31 | Select K2, then screen K3 separately |
| K3 screen | TBD | third temporal kernel around current K3=3 | Use the selected K1/K2 only |
| M15+31 / W48 | TBD | multiscale or width after per-layer kernel screens | Consider only if no per-layer configuration passes the gate |

The completed first-layer screens retained R2's second/third kernels `7/3`,
three temporal filters per input channel, and 32 spatial filters. K47 is now
the selected first-layer kernel. The next K2 screens alter only the second
kernel (`3`, `5`, `11`, `15`, or `31`); `K2=7` is already represented by
`run_67`.
After selecting K2, the third kernel will be screened around its current value
of 3. This one-factor-at-a-time sequence attributes any change in performance
to the altered receptive field. Multiscale and width experiments are deferred
until the compact per-layer kernel space is understood.

## Observed Results

All figures below are from the checkpoint selected by minimum validation
cross-entropy, rather than the epoch with maximum validation accuracy.

| Run | First temporal kernel | Parameters | Validation accuracy | AUROC | F1 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `run_60_r2_raw5s_s42` | 31 | 4,917 | 92.830% | 97.969% | 92.794% | Seed-42 R2 baseline |
| `run_66_r2_k15_s42` | 15 | 4,101 | 93.340% | 98.252% | 93.243% | +0.510 percentage points while using 16.6% fewer parameters |
| `run_67_r2_k47_s42` | 47 | 5,733 | 93.797% | 98.511% | 93.845% | Best kernel screen so far; +0.967 percentage points versus baseline |
| `run_68_r2_k63_s42` | 63 | 6,549 | 93.367% | 98.385% | 93.122% | Larger first-layer field regresses versus K47 |
| `run_69_r2_k47_k2_3_s42` | 47/3/3 | 5,605 | 93.045% | 98.268% | 93.179% | K2=3 trades lower precision for 95.005% sensitivity; below K2=7 overall |

K47 has the best selected-checkpoint accuracy of the completed K1 screens,
with 94.576% sensitivity and 93.125% precision. K63 regresses despite its
larger field, so the data do not support monotonically increasing the first
receptive field. K47 still does not establish superiority over the best
observed seed or satisfy the 95.0% selection gate.

With K1 fixed at 47, shrinking K2 from 7 to 3 reduces selected-checkpoint
accuracy by 0.752 percentage points. This is a sensitivity/precision trade-off
rather than an overall improvement, so K2=7 remains the K2 anchor while K2=5,
K2=11, K2=15, and K2=31 are tested. K2=15 and K2=31 are necessary long-context
screens because K2 follows a 4x temporal pool; their receptive fields map to
approximately 0.53 s and 0.78 s at the output of the K3=3 stack, respectively.

## One Conditional Combination Only

If the selected per-layer kernel configuration improves on the plain R2
seed-42 baseline (92.830%) but remains below 95%, run exactly one final
screen: that configuration plus the already-fixed SupCon objective (`0.05`,
temperature `0.1`). Otherwise do not combine factors. This prevents an
open-ended hyperparameter search while testing one independently positive
training-only objective on the best compact architecture.

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
source ~/miniconda3/etc/profile.d/conda.sh && conda activate chbmit-cnn && cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && git pull origin main && CHBMIT_WINDOW_SEC=5 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_5s_v1 CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn CHBMIT_HIERARCHICAL_TEMPORAL_KERNEL=47 CHBMIT_HIERARCHICAL_SECOND_KERNEL=5 CHBMIT_TRAIN_SEED=42 CHBMIT_RUN_ID=run_70_r2_k47_k2_5_s42 CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```
