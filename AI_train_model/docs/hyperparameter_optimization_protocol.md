# Validation-Only Hyperparameter Optimization

## Objective

Select a compact seizure detector using validation continuous EEG only. The test recordings must not be scored while selecting hyperparameters, architecture, threshold, or temporal policy.

The selection objective is the locked event-level benchmark:

1. satisfy validation FAR/h <= 0.50;
2. maximize validation event sensitivity;
3. break ties with lower median detection delay, then lower FAR/h.

A validation pass requires sensitivity >= 90%, FAR/h <= 0.50, and median delay <= 10 s. It is a screening gate, not a test result.

## First Controlled Sweep

The first sweep is intentionally limited to six trials around `run_03_mixed_hardneg`, the previous low-FAR reference. The immutable prepared data is `chbmit_prepared_mixed_hardneg_v1`; all trials use the baseline 1D-CNN, train-only z-score normalization, the fixed seed, AMP, and the existing early stopping rule.

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
CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_mixed_hardneg_v1 CHBMIT_SWEEP_ID=run_09_hparam python main.py --mode hyperparameter_sweep
```

The script creates one isolated output directory per trial and writes `outputs/run_09_hparam/validation_leaderboard.csv`. It sets `CHBMIT_SKIP_TEST_EVALUATION=1` and `CHBMIT_EVENT_EVAL_SPLITS=val`; therefore it does not produce or inspect test probabilities or test event metrics.

After choosing the validation winner, record its configuration and run a single new final training/evaluation run with a new ID. Only that final run may score the test split. Do not choose another model from its test result.
