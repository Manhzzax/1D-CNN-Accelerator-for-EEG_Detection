# Paper A Benchmark Evidence

## Purpose

[`chbmit_paper_a_1dcnn_benchmark.csv`](chbmit_paper_a_1dcnn_benchmark.csv)
is the controlled evidence table for the accuracy-focused paper. It has a
standalone `Parameters` column so parameter count is never confused with model
file size, activation memory, quantized tensors, latency, or FPGA resources.

The table is semicolon-delimited to open as columns in the local Excel setup.
It is an evidence inventory, **not** a global leaderboard.

## How to read it

1. `Reported on` is mandatory provenance. `Validation only` means a development
   score and is not comparable with a final test score. `Mean held-out
   cross-validation fold score` means each fold has a held-out fold, but it is
   still not the same as an untouched independent final test set. `Reported
   evaluation` means the accessible primary source does not establish whether
   the number is validation or test; it cannot be ranked against our result.
2. `Patient isolation` reports the strongest evidence supplied by the source.
   `Yes` means an unseen subject is held out in the named evaluation. `No` or
   `Not established` prevents a cross-subject claim, even when the accuracy is
   high.
3. `Seed / repetitions` distinguishes the training random seed from the
   number of repeated runs or outer folds. `NR` means it was not found in the
   primary material currently audited; it must not be invented.
4. A numerical comparison is legitimate only when task, patient isolation,
   split unit, prevalence, channels, window length, preprocessing, and model
   selection procedure are materially comparable. High patient-specific or
   epoch-level CV values do not establish patient-independent performance.
5. The table contains ictal-detection studies only. Prediction studies remain
   outside this benchmark because their labels and clinical endpoint differ.

## Accuracy provenance and fair reporting

The current EpiSepNet-R2 values are **validation-only** and must be labelled
as development results. They must not be claimed to outperform the external
test or outer-CV rows. Two exploratory scores were obtained on the current
locked chronological test partition after development decisions had already
been made: 91.192% raw 1:10 accuracy and 91.511% balanced accuracy for the
47/7/3 seed-42 checkpoint. This partition is now exposed and is deliberately
excluded from Paper A model selection, headline comparison, and final claim.

For Paper A, the final comparable number will be generated only after the
architecture and hyperparameters are frozen. The protocol must use a
predeclared grouped outer evaluation (preferably nested patient-group
five-fold CV): all recordings of an outer-test patient remain unseen; all
normalization and model selection occur inside the corresponding training
patients. Report the mean and standard deviation across outer test folds,
together with five predeclared training seeds per selected architecture. If a
separate untouched external cohort becomes available, it is stronger than
cross-validation and should be reported as the final test.

`Seed` is not a calendar year or semantic label. It initializes pseudorandom
number generators that affect weight initialization, batch order and any
random augmentation. A `split seed` separately controls random fold/split
construction. A paper must state both where relevant. Five fixed training
seeds proposed for Paper A are `7, 42, 123, 314, 2718`; they were selected
before the new evaluation, not because they produced favorable scores.

### Required manuscript table labels

| Claim type | Permitted label | Example |
|---|---|---|
| Development result | `validation accuracy` | `94.334% validation accuracy` |
| Repeated development result | `mean validation accuracy across three seeds` | `93.081 +/- 1.096%` |
| Fold-held-out result | `mean outer-CV test accuracy` | `85.84 +/- 10.13% LOSO test accuracy` |
| Independent cohort | `external test accuracy` | Only after an untouched external dataset is evaluated. |

## Basis for Paper A architecture choices

| Evidence | Design implication |
|---|---|
| LMPSeizNet reports a compact multiscale depthwise-separable CNN with 18,024 parameters. | Test multiscale temporal receptive fields without abandoning Conv1D or the 100K budget. |
| LightSeizureNet uses dilated 1D convolutions, global average pooling and pruning. | Test residual/dilated temporal context as a controlled CNN-only ablation; do not copy its reported accuracy without reproducing its protocol. |
| Gu et al. report a 61,218-parameter lightweight CNN in a cross-subject setting. | Test a meaningful capacity range below 100K; do not assume high window CV accuracy transfers to unseen patients. |
| Adatia et al. report 95% for a multichannel depthwise-separable 1D-CNN. | The 95% objective is a feasible research target, but the missing accessible protocol/parameter details prevent a direct claim. |
| R2 47/7/3 reached 94.334% only on the development validation split. | It is the required baseline for Paper A; its observed current-test probe is exploratory and cannot drive architecture selection. |

## Our rows

- `EpiSepNet-R2-5K three-seed reference` is the reproducibility result:
  93.081 +/- 1.096% balanced validation accuracy across seeds 42, 7 and 123.
- `EpiSepNet-R2-5.7K best development checkpoint` is `run_75` with 47/7/3
  kernels, 5,733 parameters and 94.334% validation accuracy at its
  validation-loss-selected checkpoint. It is one seed, therefore it is not a
  final mean result and must not be visually ranked as a final test result.

The active Paper A candidate must be added only after training artifacts are
available. It should report its actual parameter count from `model_spec.json`,
not a planned budget, and its `Reported on`, `Patient isolation`, and `Seed / repetitions` fields must be completed before any ranking is shown.
