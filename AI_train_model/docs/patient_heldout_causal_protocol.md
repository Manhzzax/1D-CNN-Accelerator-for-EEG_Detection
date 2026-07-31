# Patient-held-out causal experiment protocol

## Purpose

`EpiSepNet-5K` is frozen as the historical reference on the case-wise split:
raw 17-channel EEG, 2-second windows, `5,013` parameters, and validation
window accuracy `90.0718%`. It remains useful for model-size, FP32/INT16
agreement, and engineering reproducibility. It is not a final
patient-generalisation result because cases from a patient can occur in more
than one split and its preprocessing used zero-phase filtering.

All runs under this protocol use a new manifest, held-out patient groups, and
causal preprocessing. `chb01` and `chb21` are one patient group. No test score
may influence architecture, normalization, threshold, or temporal-policy
selection.

## Fixed protocol

| Item | Requirement |
|---|---|
| Cohort | Audited CHB-MIT manifest; report exact EDF/event counts from `split_plan_summary.json` |
| Split | Patient-group-disjoint, 60/20/20 train/validation/test; deterministic seed `42` |
| Preprocessing | 0.5-45 Hz causal 4th-order IIR bandpass, causal 60 Hz notch, train-only channel z-score |
| Input | 17 canonical bipolar channels, 2 seconds at 256 Hz, stride 1 second |
| Training selection | Validation only; early stopping and all hyperparameter choices use validation only |
| Alarm selection | Validation-only threshold/policy sweep; report event sensitivity, FAR/h, and delay |
| Test use | One final locked evaluation after candidate and alarm policy are frozen |

## Priority experiment sequence

1. `run_30_patient_causal_reference`: reproduce the EpiSepNet-5K capacity on
   the new protocol. This establishes the honest baseline; it is not compared
   numerically with the historical case-wise result as though their splits were
   interchangeable.
2. Capacity ablation: train only `2`, `3`, and `4` depthwise temporal filters
   per input channel. Choose one using validation event sensitivity under
   `FAR/h <= 0.5`, then parameter count and median delay as tie-breakers.
3. Temporal-policy sweep: fit the threshold and `m-of-n` policy on validation
   scores of the selected model. The policy must be persisted before test.
4. Final test: run continuous inference once on the test patients. Export the
   selected model's INT16 tensor package; do not claim KV260 latency, power, or
   resource usage until synthesis and board measurement are available.

The next model change is deliberately narrow. A new backbone, DWT features,
hard-negative mining, or channel-selection method is introduced only after a
controlled ablation shows the compact baseline cannot meet the internal
operating target. This preserves an interpretable result for a journal paper.

## Server commands

Create the held-out manifest:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PATIENT_HELDOUT_PROTOCOL_OUTPUT_DIR=chbmit_protocol_patient_holdout_v1 python main.py --mode plan_patient_heldout
```

Prepare causal, two-second data in a separate directory:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PROTOCOL_OUTPUT_DIR=chbmit_protocol_patient_holdout_v1 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_patient_causal_2s_v1 CHBMIT_FILTER_MODE=causal_iir CHBMIT_WINDOW_SEC=2 python main.py --mode preprocess
```

Train the frozen-capacity reference without evaluating the test windows:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_patient_causal_2s_v1 CHBMIT_WINDOW_SEC=2 CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL=3 CHBMIT_RUN_ID=run_30_patient_causal_reference CHBMIT_SKIP_TEST_EVALUATION=true python main.py --mode train
```

Score continuous validation recordings only and select the alarm policy:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PROTOCOL_OUTPUT_DIR=chbmit_protocol_patient_holdout_v1 CHBMIT_WINDOW_SEC=2 CHBMIT_FILTER_MODE=causal_iir CHBMIT_MODEL_RUN_ID=run_30_patient_causal_reference CHBMIT_RUN_ID=run_30_patient_causal_reference CHBMIT_EVENT_EVAL_SPLITS=val python main.py --mode event_eval
```

Before training any capacity variant, copy its validation artifacts into the
experiment registry. Do not run `event_eval` with `test` until one model and
one validation-selected temporal policy are frozen.
