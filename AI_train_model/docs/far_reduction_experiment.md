# FAR Reduction Experiment Protocol

This protocol follows the locked CHB-MIT case-wise chronological split. Validation chooses all decision rules. The test recordings are evaluated once per finalized run and are never used to choose a threshold, temporal policy, or mining setting.

## Baseline Temporal Policy

The baseline checkpoint remains in `AI_train_model/outputs`. When its existing continuous-score artifacts are present, this command reuses them and does not rerun EDF inference. Run the validation-selected temporal-policy sweep into a separate artifact directory:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_RUN_ID=run_01_temporal_policy python main.py --mode event_eval
```

The command sweeps every configured threshold and temporal policy. The selected policy maximizes event sensitivity subject to the validation false-alarm target. If no policy reaches the target, the result explicitly records that failure.

## Train-Only Hard Negatives

Mine the highest seizure-score interictal windows only from train recordings. The builder retains the original sampled normal windows and adds unique hard negatives; the default dataset therefore has a 1:3 ictal:normal ratio. The source checkpoint defaults to the baseline in `outputs/` and the result is written to `data/chbmit_prepared_mixed_hardneg_v1`.

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && python main.py --mode mine_hard_negatives
```

The mining summary stores the source checkpoint SHA-256, candidate count, selected ratio, and score range. It must be retained with the experiment artifacts.

## Persistent Temporal Hard Negatives

The first hard-negative run ranks individual normal windows. The next ablation ranks only train-only, fully interictal 10-second score contexts with at least three CNN scores above 0.91, then enforces a 10-second separation within each recording. This targets persistent alarm-like patterns without using validation or test samples for training. It reuses the frozen `run_03_mixed_hardneg` score cache saved by `run_04_score_tcn`; if unavailable it rescans only the locked train recordings.

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && python main.py --mode mine_temporal_hard_negatives
```

The dataset is written to `data/chbmit_prepared_temporal_hardneg_v1`. Train it as a separate run and select its threshold/policy on validation only.

If the strict temporal criterion produces fewer candidates than the requested ratio, the default behavior retains every eligible separated candidate and records `candidate_limited: true` in the summary. Do not lower the criterion merely to fill the ratio; that would change the ablation from persistent alarm-like negatives to ordinary high-score negatives.

## Retrain And Evaluate

Train with the new prepared dataset and write all outputs to a separate run directory:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_mixed_hardneg_v1 CHBMIT_RUN_ID=run_03_mixed_hardneg python main.py --mode train
```

Evaluate that exact checkpoint and scaler continuously, while preserving the same run directory for the new artifacts:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_MODEL_RUN_ID=run_03_mixed_hardneg CHBMIT_RUN_ID=run_03_mixed_hardneg python main.py --mode event_eval
```

Do not commit EDF files, NPZ datasets, checkpoints, or continuous-score arrays. Commit only concise summaries, CSV sweeps, reports, and plots under `server_results/`.
