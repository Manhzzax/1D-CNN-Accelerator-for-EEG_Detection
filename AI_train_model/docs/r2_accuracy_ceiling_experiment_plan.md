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
| Done | `run_70_r2_k47_k2_5_s42` | second temporal kernel 5 | 93.206% accuracy; below K2=7; stopped at epoch 22 |
| Done | `run_71_r2_k47_k2_9_s42` | second temporal kernel 9 | 91.004% accuracy; rejected |
| Done | `run_72_r2_k47_k2_11_s42` | second temporal kernel 11 | 93.340% accuracy; below K2=7 |
| Done | `run_73_r2_k47_k2_15_s42` | second temporal kernel 15 | 92.401% accuracy; rejected |
| Done | `run_74_r2_k47_k2_31_s42` | second temporal kernel 31 | 92.481% accuracy; rejected |
| Done | `run_75_r2_k47_k2_7_e50_es12_s42` | 50 epochs, ES patience 12 on selected K2=7 | 94.334% at epoch 36; confirms 30-epoch cap was limiting |
| Done | `run_76_r2_k47_k2_5_e50_es12_s42` | 50 epochs, ES patience 12 on early-stopped K2=5 | No improvement; default early stop was appropriate for K2=5 |
| Done | `run_77_r2_k47_k2_11_e50_es6_s42` | 50 epochs, default ES patience 6 on K2=11 | 93.340%; does not overtake K2=7 |
| Done | `run_78_r2_k47_k2_7_e100_es15_s42` | 100 epochs, ES patience 15 on selected K2=7 | No improvement over 50E; terminate schedule tuning |
| K3 screen | TBD | third temporal kernel with K1/K2 fixed at 47/7 | Use 50 epochs and default ES patience 6 |
| K3 screen | TBD | third temporal kernel around current K3=3 | Use the selected K1/K2 only |
| M15+31 / W48 | TBD | multiscale or width after per-layer kernel screens | Consider only if no per-layer configuration passes the gate |

The completed first-layer screens retained R2's second/third kernels `7/3`,
three temporal filters per input channel, and 32 spatial filters. K47 is now
the selected first-layer kernel. The next K2 screens alter only the second
kernel (`3`, `5`, `9`, `11`, `15`, or `31`); `K2=7` is already represented by
`run_67`. The five pending settings are run sequentially under identical
conditions and pushed together; they are independent runs, not a five-way
parallel GPU job.
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
| `run_70_r2_k47_k2_5_s42` | 47/5/3 | 5,669 | 93.206% | 98.299% | 93.108% | Stopped at epoch 22; candidate for extended-patience check |
| `run_71_r2_k47_k2_9_s42` | 47/9/3 | 5,797 | 91.004% | 97.282% | 90.789% | Rejected |
| `run_72_r2_k47_k2_11_s42` | 47/11/3 | 5,861 | 93.340% | 98.385% | 93.305% | Second best K2 screen; completed 30 epochs |
| `run_73_r2_k47_k2_15_s42` | 47/15/3 | 5,989 | 92.401% | 97.625% | 92.172% | Rejected |
| `run_74_r2_k47_k2_31_s42` | 47/31/3 | 6,501 | 92.481% | 98.128% | 92.383% | Rejected |
| `run_75_r2_k47_k2_7_e50_es12_s42` | 47/7/3 | 5,733 | **94.334%** | **98.593%** | **94.320%** | Best current selected checkpoint; epoch 36 |
| `run_76_r2_k47_k2_5_e50_es12_s42` | 47/5/3 | 5,669 | 93.206% | 98.299% | 93.108% | Identical selected result after longer patience; no late recovery |
| `run_77_r2_k47_k2_11_e50_es6_s42` | 47/11/3 | 5,861 | 93.340% | 98.385% | 93.305% | No gain after increasing cap; K2=7 remains selected |
| `run_78_r2_k47_k2_7_e100_es15_s42` | 47/7/3 | 5,733 | 94.334% | 98.593% | 94.320% | Same checkpoint as 50E; extra 3 epochs do not improve result |

K47/K2=7/K3=3 has the best selected-checkpoint accuracy of the completed
kernel screens, with 94.576% sensitivity and 93.125% precision. K63 regresses
despite its larger field, so the data do not support monotonically increasing
the first receptive field. The K2 screen likewise does not support increasing
the later receptive field: K2=7 remains ahead of 3, 5, 9, 11, 15, and 31.
The selected configuration still does not establish superiority over the best
observed seed or satisfy the 95.0% selection gate.

## Extended-Training / Early-Stopping Check

The default training budget is 30 epochs, `ReduceLROnPlateau` uses patience 3
and factor 0.5, and early stopping uses patience 6 with minimum delta 0.001.
This means the first learning-rate reduction can receive only about three more
epochs before early stopping. This is economical but can be too short for a
late recovery after the lower learning rate.

Two controlled 50-epoch runs assessed the former 30-epoch budget and
early-stopping rule. Both retained the scheduler, optimizer, model, data, and
minimum-validation-loss checkpoint rule. `50E-A` reached 94.334% at epoch 36,
an improvement of 0.537 percentage points over its 30-epoch counterpart.
The first 30 validation-loss values were exactly identical across the two
runs. Replaying the default patience-6 rule on the extended trajectory would
stop at epoch 37, after saving the epoch-36 absolute minimum. Therefore the
binding constraint was the 30-epoch cap, not patience 6.

`50E-B` directly tested the early-stopped K2=5 configuration: increasing
patience from 6 to 12 extended training from 22 to 28 epochs but did not change
its selected checkpoint. Thus, patience 6 is not generally too aggressive for
this protocol. Subsequent architecture comparisons use a 50-epoch cap with
the default patience 6; `50E-C` re-evaluates the K2=11 runner-up under that
selected schedule before K2 is frozen.

`50E-C` stopped at epoch 32 and retained the same epoch-26 selected checkpoint
as its 30-epoch counterpart. Therefore K2=7 is frozen for the next layer
screen. The requested 100-epoch/patience-15 run is explicitly exploratory: it
tests late convergence of this frozen candidate but must not be pooled with the
50-epoch architecture screens when choosing K3.

The 100-epoch stress test stopped at epoch 46 and returned exactly the same
epoch-36 checkpoint and metrics as the 50-epoch run. Therefore neither the
100-epoch cap nor patience 15 is adopted. The fixed schedule for the K3
architecture screen is 50 epochs with early-stopping patience 6.

This follows early-stopping literature showing that slower stopping can yield
small generalization gains at a substantially higher training cost, and the
documented intent of `ReduceLROnPlateau` to lower the learning rate after a
metric plateaus. [Prechelt, 1998](https://pubmed.ncbi.nlm.nih.gov/12662814/)
[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)

With K1 fixed at 47, shrinking K2 from 7 to 3 reduces selected-checkpoint
accuracy by 0.752 percentage points. This is a sensitivity/precision trade-off
rather than an overall improvement, so K2=7 remains the K2 anchor. K2=15 and
K2=31 were necessary long-context screens because K2 follows a 4x temporal
pool; their receptive fields map to approximately 0.53 s and 0.78 s at the
output of the K3=3 stack, respectively, but neither improved accuracy.

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
source ~/miniconda3/etc/profile.d/conda.sh && conda activate chbmit-cnn && cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && git pull origin main && CHBMIT_WINDOW_SEC=5 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_5s_v1 CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn CHBMIT_HIERARCHICAL_TEMPORAL_KERNEL=47 CHBMIT_HIERARCHICAL_SECOND_KERNEL=7 CHBMIT_TRAIN_EPOCHS=100 CHBMIT_EARLY_STOPPING_PATIENCE=15 CHBMIT_TRAIN_SEED=42 CHBMIT_RUN_ID=run_78_r2_k47_k2_7_e100_es15_s42 CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```
