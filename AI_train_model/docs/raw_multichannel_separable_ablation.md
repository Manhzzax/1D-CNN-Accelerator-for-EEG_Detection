# Raw Multichannel Separable 1D-CNN Ablation

## Question

Can direct, EEG-specific temporal and spatial feature learning improve the validation event-sensitivity/FAR trade-off relative to score post-processing and hard-negative mining alone?

`run_04_score_tcn` and `run_05_temporal_hardneg` are negative ablations for the low-FAR objective. The next experiment therefore changes the raw-signal representation while preserving the locked 17-channel input, preprocessing, split, and event-evaluation protocol.

## Architecture

`separable_1dcnn` accepts `(batch, 17, 256)` raw EEG windows:

1. Depthwise `Conv1d`, 2 filters per channel, kernel 31: a learnable temporal filter bank without mixing electrodes.
2. Pointwise `Conv1d` from 34 to 32 channels: learns spatial/channel mixtures after temporal filtering.
3. Average pool by 4.
4. Depthwise temporal `Conv1d`, kernel 15, then pointwise refinement at 32 channels.
5. Average pool by 4, global average pool, dropout, and a 32-to-2 classifier.

This follows the temporal-then-spatial separation used by EEGNet while using only Conv1D, ReLU, average pooling, and global pooling. It eliminates the flatten-heavy dense head of `baseline_1dcnn`, making it a compact candidate for a future KV260-specific exporter.

## Controls

- Train from the original locked prepared dataset, not `run_05` temporal hard-negative data.
- Train/validation select model checkpoint, threshold, and alarm policy.
- Run continuous test evaluation once after selection.
- The architecture name and parameter count are written to `outputs/<run>/model_spec.json`, so event evaluation reloads the exact model type even if the global config later changes.
- Current Q15 export remains baseline-only; quantization work starts only after a raw architecture proves a validation/event-level benefit.

## First Run

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_RUN_ID=run_06_separable_raw python main.py --mode train
```

Then evaluate exactly that recorded architecture:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_MODEL_RUN_ID=run_06_separable_raw CHBMIT_RUN_ID=run_06_separable_raw python main.py --mode event_eval
```

## Recording-Scale Normalization Ablation

The follow-up `run_07` keeps the same raw separable architecture but computes an unlabeled, channel-wise z-score independently for each recording. Statistics are calculated from the prepared recording windows and persisted with the run; continuous inference reloads exactly those same constants. This tests amplitude/scale robustness after `run_06` showed concentrated test false alarms in a small number of later recordings. It is selected on train/validation only and must be reported as an offline per-recording normalization ablation: a deployed system would need a causal warm-up or running estimator rather than a full-recording statistic.

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_NORMALIZATION_MODE=per_recording_zscore CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_v1 CHBMIT_RUN_ID=run_07_separable_perrecord_z python main.py --mode train
```
