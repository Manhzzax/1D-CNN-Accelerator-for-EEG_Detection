# DWT + Separable 1D-CNN Validation Ablation

## Hypothesis

The raw SeparableCNN reaches 21/29 validation events at 0.4375 FAR/h but misses six events and has 15 s median delay. A compact multiresolution frontend may improve the base classifier's score separation before temporal policy selection.

The model backbone remains `separable_1dcnn`. This experiment changes only its input representation:

`filtered 17-channel EEG window -> level-3 db4 DWT coefficients -> Separable 1D-CNN -> alarm policy`

## Representation Contract

- Input and output shape: `17 x 256` per one-second window.
- Wavelet: Daubechies-4 (`db4`), three decomposition levels, `periodization` boundary mode.
- Coefficient order: `cA3, cD3, cD2, cD1`.
- The concatenated coefficient vector has exactly 256 samples, so the CNN architecture and parameter count remain unchanged at 3,908 trainable parameters.
- DWT is applied after the existing bandpass/notch filtering. Train-only z-score statistics are then fitted to DWT coefficients and applied unchanged to validation inference.

This is a controlled representation ablation, not a change to labels, split, window stride, backbone, or temporal evaluation rules.

## Execution

First build a distinct immutable prepared dataset. This processes the locked recording split but does not change the raw prepared data:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection && git pull origin main && cd AI_train_model && pip install -r requirements.txt && CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_dwt_db4_l3_v1 CHBMIT_FEATURE_REPRESENTATION=dwt_db4_l3 python main.py --mode preprocess
```

Then train the current best optimizer/sampling configuration without scoring test:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_dwt_db4_l3_v1 CHBMIT_RUN_ID=run_12_dwt_separable CHBMIT_TRAIN_LEARNING_RATE=0.001 CHBMIT_TRAIN_WEIGHT_DECAY=0.0001 CHBMIT_CLASS_BALANCED_BATCHES=false CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train && CHBMIT_MODEL_RUN_ID=run_12_dwt_separable CHBMIT_RUN_ID=run_12_dwt_separable CHBMIT_EVENT_EVAL_SPLITS=val python main.py --mode event_eval
```

Success requires improvement over the raw reference on validation: more than 21/29 detected events while FAR/h remains <= 0.50, lower than 15 s median delay, and at least 90% segment accuracy. Do not run test for this ablation.

## Deployment Caveat

The current preparation pipeline uses zero-phase filtering and window-complete DWT, which is suitable for an offline controlled comparison but not a final streaming hardware claim. A winning DWT model must later be repeated with causal filtering/stateful DWT and an explicitly accounted one-second decision latency before FPGA deployment results are reported.
