# Validation-Only Hyperparameter Optimization

## Objective

Select a compact seizure detector using validation continuous EEG only. The test recordings must not be scored while selecting hyperparameters, architecture, threshold, or temporal policy.

The selection objective is the locked event-level benchmark:

1. satisfy validation FAR/h <= 0.50;
2. satisfy validation 1:1-window accuracy and balanced accuracy >= 90%, and ictal F1 >= 0.85;
3. maximize validation event sensitivity;
4. break ties with higher validation accuracy, lower median detection delay, then lower FAR/h.

A validation pass requires event sensitivity >= 90%, FAR/h <= 0.50, median delay <= 10 s, window accuracy >= 90%, balanced accuracy >= 90%, and ictal F1 >= 0.85. It is a screening gate, not a test result.

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

## Execution

Run one command from `AI_train_model`:

```bash
CHBMIT_SWEEP_FAMILY=separable_raw CHBMIT_SWEEP_ID=run_10_separable_hparam python main.py --mode hyperparameter_sweep
```

The script creates one isolated output directory per trial and writes `outputs/<sweep_id>/validation_leaderboard.csv`. It sets `CHBMIT_SKIP_TEST_EVALUATION=1` and `CHBMIT_EVENT_EVAL_SPLITS=val`; therefore it does not produce or inspect test probabilities or test event metrics.

After choosing the validation winner, record its configuration and run a single new final training/evaluation run with a new ID. Only that final run may score the test split. Do not choose another model from its test result.
