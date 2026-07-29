# Fine Temporal Policy Sweep

## Purpose

The best separable hyperparameter trial detects 21/29 validation events at 0.4552 FAR/h but has 15 s median delay. This sweep does not retrain a model. It reuses that trial's saved validation score stream to find a faster temporal-confirmation policy at the locked FAR target of 0.50/h.

## Search Space

The script evaluates `3_of_6`, `4_of_8`, `5_of_10`, `6_of_12`, `7_of_14`, `8_of_16`, `9_of_18`, and `10_of_20`. Every policy requires positive evidence in 50% of the recent windows. Thresholds are searched from 0.850 through 0.999 in increments of 0.001.

Only candidates with validation FAR/h <= 0.50 are eligible. Among them, selection maximizes event sensitivity, then minimizes median delay, then FAR/h. Segment accuracy and F1 are carried from the frozen source model and cannot improve in this score-only sweep.

## Execution

```bash
CHBMIT_TEMPORAL_SOURCE_RUN_ID=run_10_separable_hparam_f_lr1e3_wd1e4_nobalance CHBMIT_RUN_ID=run_11_separable_temporal_policy python main.py --mode temporal_policy_sweep
```

The input is `continuous_val_scores.npz` from the source run. The script does not load EEG, train a model, or score any test recording. It writes `temporal_policy_validation_sweep.csv` and `temporal_policy_selection.json` to the new run directory.

Do not test the selected policy yet. First inspect the validation trade-off. A later final model run may use the locked policy and evaluate test exactly once.
