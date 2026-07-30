# Validation-Only Hyperparameter Optimization

## Objective

Select a compact seizure detector using validation continuous EEG only. The test recordings must not be scored while selecting hyperparameters, architecture, threshold, or temporal policy.

The selection objective is the internal clinical screening gate, not a fabricated
single "paper accuracy" target:

1. satisfy validation FAR/h <= 0.50;
2. maximize validation event sensitivity;
3. break ties with lower FAR/h, lower median detection delay, higher AUROC, ictal F1, then 1:1-window accuracy.

A validation pass requires event sensitivity >= 90%, FAR/h <= 0.50, and median delay <= 10 s. Window accuracy, balanced accuracy, AUROC and ictal F1 are mandatory reporting metrics, but are not hard gates: the published accuracy values use different windows, balance ratios, patient selection and validation schemes. It is a screening gate, not a test result.

## First Controlled Sweep

The sweep supports two isolated model families. The same six optimizer/sampling trials are run within a family; architecture and prepared data remain fixed inside that sweep.

| Family | Starting reference | Architecture | Prepared data | Purpose |
|---|---|---|---|---|
| `baseline_mixed` | `run_03_mixed_hardneg` | Baseline 1D-CNN | `chbmit_prepared_mixed_hardneg_v1` | Low-FAR calibration |
| `separable_raw` | `run_06_separable_raw` | Separable 1D-CNN | `chbmit_prepared_v1` | Best current validation accuracy and event sensitivity |

All trials use train-only z-score normalization, the fixed seed, AMP, and the existing early stopping rule.

| Trial | Learning rate | Weight decay | Class-balanced batches |
|---|---:|---:|---|
| A | 1e-3 | 1e-4 | yes |
| B | 3e-4 | 1e-4 | yes |
| C | 3e-4 | 5e-4 | yes |
| D | 1e-3 | 5e-4 | yes |
| E | 3e-4 | 1e-4 | no |
| F | 1e-3 | 1e-4 | no |

This separates optimizer regularization from sampling calibration. The non-balanced trials retain the fixed mixed hard-negative training distribution and test whether class-balanced sampling is causing excessive false alarms.

## Second Controlled Sweep: `separable_raw_refine`

`run_10_separable_hparam_f_lr1e3_wd1e4_nobalance` is the current validation reference: 21/29 events, 0.4552 FAR/h, 15 s median delay, 86.93% 1:1-window accuracy, 0.9426 AUROC and 0.8701 ictal F1. The DWT coefficient-concatenation ablation was worse, so this sweep keeps the raw 17-channel input and the separable 1D-CNN backbone.

Each trial changes one factor from the reference. All use non-balanced batches, train-only z-score, the same seed, AMP and early stopping.

| Trial | Change | Reason |
|---|---|---|
| A | Reference (`lr=1e-3`, `wd=1e-4`) | Reproducibility anchor |
| B | `lr=5e-4` | Less aggressive optimization |
| C | `wd=3e-4` | Stronger regularization |
| D | dropout `0.10` | Test whether `0.25` is over-regularizing |
| E | spatial filters `48` | More cross-channel capacity, still compact |
| F | temporal filters/channel `3` | More per-channel spectral-temporal capacity |

The script records each resolved layer configuration in `model_spec.json`; no test recording is scored.

## Execution

Run one command from `AI_train_model`:

```bash
CHBMIT_SWEEP_FAMILY=separable_raw CHBMIT_SWEEP_ID=run_10_separable_hparam python main.py --mode hyperparameter_sweep
```

For the refinement sweep:

```bash
CHBMIT_SWEEP_FAMILY=separable_raw_refine CHBMIT_SWEEP_ID=run_13_separable_refine python main.py --mode hyperparameter_sweep
```

The script creates one isolated output directory per trial and writes `outputs/<sweep_id>/validation_leaderboard.csv`. It sets `CHBMIT_SKIP_TEST_EVALUATION=1` and `CHBMIT_EVENT_EVAL_SPLITS=val`; therefore it does not produce or inspect test probabilities or test event metrics.

After choosing the validation winner, record its configuration and run a single new final training/evaluation run with a new ID. Only that final run may score the test split. Do not choose another model from its test result.
