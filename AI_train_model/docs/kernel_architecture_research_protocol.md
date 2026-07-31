# Kernel Architecture Decision: 31/15 Versus 31/7/3

## Decision

The first Track B architecture experiment is the professor-proposed
**hierarchical 31/7/3 temporal design**, not a broad sweep of unrelated kernel
lengths. The model is implemented as `hierarchical_separable_1dcnn`.

This is a falsifiable hypothesis: after a long first temporal filter captures
ictal morphology, two shorter filters can refine that representation at lower
temporal resolution. It is not a claim that `31/7/3` is universally optimal
for EEG or CHB-MIT.

## Why test 31/7/3 first

The frozen EpiSepNet-5K has two temporal depthwise layers:

```text
depthwise 31 -> pointwise -> average pool 4
depthwise 15 -> pointwise -> average pool 4 -> global average
```

The proposed hierarchy is:

```text
depthwise 31 -> pointwise -> average pool 4
depthwise  7 -> pointwise -> average pool 4
depthwise  3 -> global average
```

With same-padding convolutions, the nominal local receptive field before global
pooling is the same for both designs:

| Architecture | Receptive-field calculation in input samples | Local receptive field |
|---|---|---:|
| Frozen `31/15` | `31 -> 34 -> 34 + 14*4 = 90 -> 90 + 3*4 = 102` | 102 samples = 398 ms |
| Hierarchical `31/7/3` | `31 -> 34 -> 34 + 6*4 = 58 -> 58 + 3*4 = 70 -> 70 + 2*16 = 102` | 102 samples = 398 ms |

The `4`-sample pools have stride `4`. Global average pooling ultimately
aggregates the complete 2-second window; the table describes the local feature
receptive field before that aggregation.

Therefore this comparison isolates **extra hierarchy and nonlinearity** rather
than simply giving one candidate a longer look-back window.

## Cost calculation

All values below assume the frozen run's 17 channels, 512 samples, 3 temporal
filters per input channel, 32 spatial filters, and two classes. MACs count only
Conv1D and linear multiplications for one window.

| Candidate | Temporal stages | Pointwise stages | Parameters | MACs/window | Role |
|---|---|---:|---:|---:|---|
| `R0` frozen reference | `31/15` | 2 | 5,013 | 1,837,632 | Existing baseline |
| `R1` reduced refinement | `31/7` | 2 | 4,757 | 1,804,864 | Isolates changing 15 to 7 |
| `R2` professor Lite | `31/7/3` | 2 | 4,917 | 1,807,936 | Primary candidate |
| `R3` professor Full | `31/7/3` | 3 | 5,941 | 1,840,704 | Tests final pointwise mixing |

`R2` is only 0.53% lower in parameters and 1.62% lower in MACs than the frozen
reference. `R3` adds 928 parameters and 3,072 MACs, only 0.17% more MACs than
the reference because the third pointwise layer runs after both pooling stages.

## Research basis and limit of the evidence

- EEGNet separates temporal filtering from spatial mixing and uses
  depthwise/separable convolution to remain compact. Its controlled comparison
  also keeps model size fixed when attributing a result to architecture.
  [Lawhern et al.](https://arxiv.org/abs/1611.08024)
- BSDCNN is seizure-prediction work, but it independently motivates raw 1D
  temporal convolution and reduced numerical complexity for hardware-oriented
  EEG processing. [Zhao et al.](https://arxiv.org/abs/2206.07518)
- Multiscale and inception-style EEG networks show that multiple temporal
  resolutions are a legitimate hypothesis, but their task/protocols differ and
  do not prove an accuracy gain here. [Wang et al.](https://arxiv.org/abs/2105.02823),
  [Shyu et al.](https://doi.org/10.1109/ACCESS.2023.3277634)

None of these sources establishes 31, 7, and 3 as a universal optimum. That is
why `R0`--`R3` use the same data, width, loss, and validation split.

## Phase-1 Result: Seed 42

The direct topology screen is complete. These are validation-only, balanced
window metrics, not test or clinical event claims.

| Candidate | Parameters | Accuracy | AUROC | F1 | Sensitivity | Outcome |
|---|---:|---:|---:|---:|---:|---|
| R0 frozen `31/15` | 5,013 | 90.072% | 96.580% | 90.140% | 90.765% | Reference |
| R1 `31/7` | 4,757 | 89.507% | 96.243% | 89.194% | 86.609% | Reject |
| R2 `31/7/3` Lite | 4,917 | **91.175%** | **96.645%** | **91.102%** | 90.354% | Advance |
| R3 `31/7/3` Full | 5,941 | 88.892% | 95.723% | 88.906% | 89.020% | Reject |

R2 improves the frozen seed-42 accuracy by 1.103 percentage points while
using 96 fewer parameters. R3's final pointwise mixing does not help; it
increases the train-validation gap and was early-stopped after its best epoch
14. The selected R2 event sweep does **not** yet meet the locked FAR target:
its best coarse validation point is `10_of_20`, threshold `0.99`, sensitivity
68.97%, FAR `0.526/h`. R2 advances only to validation seed confirmation, not
to deployment or test evaluation.

### R2 Seed Confirmation Status

R2 has now completed seeds 7, 42, and 123 on `chbmit_prepared_raw_2s_v2`:

| Seed | Accuracy | AUROC | F1 | Sensitivity | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 7 | 89.815% | 96.211% | 89.786% | 89.533% | 14 |
| 42 | 91.175% | 96.645% | 91.102% | 90.354% | 23 |
| 123 | 89.610% | 96.146% | 89.734% | 90.816% | 15 |
| Mean +/- sample SD | 90.200% +/- 0.850% | 96.334% +/- 0.272% | 90.207% +/- 0.775% | 90.234% +/- 0.650% | - |

This confirms R2 is trainable but does **not** yet establish a three-seed gain
over R0. The historical R0 reference was produced before the repaired
`raw_2s_v2` prepared artifact; its seed-42 score is not a sufficient matched
three-seed comparator. Re-run R0 `31/15` with seeds 7, 42, and 123 on the same
`raw_2s_v2` artifact before model selection, event comparison, or any
hyperparameter tuning claim.

## Ordered experimental plan

### Phase 0: establish variance

1. Repeat frozen `31/15` architecture with seeds `7` and `123`.
2. Combine them with the existing seed-42 result. Report mean and standard
   deviation; do not change the architecture based on a single lucky seed.

### Phase 1: professor hypothesis screening

Train `R1`, `R2`, and `R3` once with seed `42`, validation only. Use the same
prepared 2-second data, sampled 1:1 validation windows, optimiser, and early
stopping as the frozen source run.

Advance a candidate to Phase 2 only if its validation balanced accuracy is at
least 0.5 percentage points above the seed-42 reference **or** its validation
AUROC and ictal F1 improve without reducing sensitivity by more than 1 point.
This is a screening rule, not a publication claim.

### Phase 2: confirmation and event behaviour

For every Phase-1 winner, run seeds `7` and `123`, then perform validation-only
continuous event evaluation for all three seeds. Rank candidates by mean
balanced-window accuracy first, then report the sensitivity/FAR/h/delay trade-
off. Do not run or inspect the historical test split.

### Phase 2a: Optimization Sensitivity

Run this only after the matched R0/R2 three-seed comparison. Early stopping
selects the best stored checkpoint; increasing its patience cannot improve
that checkpoint. It can only help if a later, lower validation-loss basin is
reached after a learning-rate reduction. The Phase-1 evidence does not justify
tuning R1 or R3: R1 reached epoch 30 without early stopping and still
underperformed, while R3 overfit after epoch 14.

If R2's three-seed mean remains above R0, run these validation-only R2
ablations with seed 42, one factor at a time:

1. schedule: 45 epochs and early-stopping patience `12`;
2. checkpoint criterion: 45 epochs, `min_delta=0`, patience `6`;
3. regularisation: AdamW at weight decay `3e-4`; separately dropout `0.35`.

Every future `training_summary.json` records per-epoch training loss,
validation loss, both accuracies, and actual learning rate. This makes the
scheduler and early-stopping interaction auditable.

Checkpoint selection is stricter than patience: the artifact always stores the
absolute minimum validation-loss epoch, while `min_delta` only determines
whether the early-stopping counter is reset. This prevents a genuine but small
loss improvement from being excluded from the reported validation checkpoint.

### Phase 3: capacity only after hierarchy result

If `R2` or `R3` has a repeatable gain but remains below 95%, increase one axis
at a time on that winning topology:

1. spatial width `32 -> 48`;
2. temporal filters per channel `3 -> 4`;
3. only then test the combined width/depth setting.

Stop an axis if it exceeds 25,000 parameters, worsens validation loss across
three seeds, or causes an unacceptable event-level FAR trade-off.

### Phase 4: only if hierarchy fails

Test a compact parallel multiscale 15/31 variant. This is deliberately later:
it changes topology, activation memory, and branch fusion simultaneously,
whereas `31/7/3` isolates a simpler depth hypothesis. Existing multiscale
results in this repository were not selected as the frozen reference, so they
are not evidence to skip the direct hierarchy ablation.

### Phase 5: deployability

For the selected three-seed candidate only: repeat with causal preprocessing,
export a new INT16 package, implement integer multiplier/shift requantisation,
and run the FPGA flow. The current exporter intentionally supports only the
frozen `separable_1dcnn`; adding export support is a promotion-gate task, not a
reason to alter Track A.

## Server commands for Phase 1

Run from `~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model`. Each
command is validation-only and does not score test recordings.

```bash
CHBMIT_WINDOW_SEC=2 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_2s_v1 CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL=3 CHBMIT_SEPARABLE_SPATIAL_FILTERS=32 CHBMIT_SEPARABLE_TEMPORAL_KERNEL=31 CHBMIT_SEPARABLE_REFINEMENT_KERNEL=7 CHBMIT_TRAIN_SEED=42 CHBMIT_RUN_ID=run_42_r1_31_7_s42 CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```

```bash
CHBMIT_WINDOW_SEC=2 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_2s_v1 CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn CHBMIT_TRAIN_SEED=42 CHBMIT_RUN_ID=run_43_r2_31_7_3_lite_s42 CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```

```bash
CHBMIT_WINDOW_SEC=2 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_2s_v1 CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn CHBMIT_HIERARCHICAL_THIRD_POINTWISE=true CHBMIT_TRAIN_SEED=42 CHBMIT_RUN_ID=run_44_r3_31_7_3_full_s42 CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```

After each completed training run, use this separate one-line validation event
command, changing only the run ID:

```bash
CHBMIT_WINDOW_SEC=2 CHBMIT_MODEL_RUN_ID=run_43_r2_31_7_3_lite_s42 CHBMIT_RUN_ID=run_43_r2_31_7_3_lite_s42 CHBMIT_EVENT_EVAL_SPLITS=val python main.py --mode event_eval
```

Do not execute the three commands in parallel on the same GPU.
