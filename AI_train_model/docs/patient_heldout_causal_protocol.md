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
   interchangeable. It completed with 56.2975% validation window accuracy,
   17.8791% window sensitivity, and 5/39 event sensitivity at 0.2886 FAR/h.
   Its 93.3% train versus approximately 57% validation accuracy is a
   cross-patient domain-shift failure, not a temporal-policy failure.
2. Source-only subject-adversarial ablation: use the 16 training patient groups
   as domain labels in a gradient-reversal discriminator. Validation and test
   EEG remain completely unseen by that discriminator. The discriminator is
   discarded after training, so inference remains EpiSepNet-5K. Run the fixed
   coefficients `0.02` and `0.05`; do not tune any other architecture parameter
   in this phase. Cross-subject seizure studies motivate explicit feature
   alignment, including shallow/deep alignment and adversarial learning
   ([Wang et al., 2024](https://doi.org/10.1142/S0129065724500552)).
   `run_31` (coefficient .02) reaches 7/39 events at .2741 FAR/h; `run_32`
   (coefficient .05) also reaches 7/39 but at .3174 FAR/h. Thus `run_31` is
   the provisional source-only DG candidate under the pre-specified hierarchy.
3. Causal within-window normalization ablation: standardize every channel from
   the samples in the complete two-second input window. This removes
   patient-specific gain/offset at decision time without fitting any statistic
   on held-out patients, labels, or future samples. Run this without the
   adversarial head first, then combine only if it improves the validation
   frontier.
4. Patient-group-balanced sampling ablation: use an equal-probability
   `class x source-patient-group` sampler during training, while retaining the
   frozen raw input, architecture, causal filter, and train-only z-score.
   This responds to diagnostics showing that aggregate validation performance
   can be concentrated in a single held-out patient group. It is training-only
   and does not change inference size or use held-out patient labels.
   `run_34` improves aggregate validation detection to 9/39 events at 0.3535
   FAR/h, but still detects 0/8 events for `subject_14`; it is therefore an
   intermediate result rather than a final candidate.
5. GroupDRO ablation: retain patient-group-balanced sampling and dynamically
   upweight source patient groups with higher classification loss. This is
   training-only robust optimization; it does not use held-out patient data or
   increase deployment parameters. It is run separately from GRL and without
   window normalization to keep attribution clear.
6. Select one candidate using the validation frontier: highest event
   sensitivity with `FAR/h <= 0.5`, then lower FAR/h, lower median delay, and
   higher validation AUROC as tie-breakers. If neither candidate improves the
   frontier materially, investigate causal calibration/normalization before
   increasing backbone capacity.
7. Temporal-policy sweep: fit the threshold and `m-of-n` policy on validation
   scores of the selected model. The policy must be persisted before test.
8. Final test: run continuous inference once on the test patients. Export the
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

Run the first source-only domain-generalization ablation without test evaluation:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_patient_causal_2s_v1 CHBMIT_WINDOW_SEC=2 CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL=3 CHBMIT_CLASS_BALANCED_BATCHES=false CHBMIT_SUBJECT_ADVERSARIAL=true CHBMIT_SUBJECT_ADVERSARIAL_COEFFICIENT=0.02 CHBMIT_RUN_ID=run_31_patient_dg_grl002 CHBMIT_SKIP_TEST_EVALUATION=true python main.py --mode train
```

Run causal within-window normalization without domain-adversarial training:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_patient_causal_2s_v1 CHBMIT_WINDOW_SEC=2 CHBMIT_NORMALIZATION_MODE=window_channel_zscore CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL=3 CHBMIT_CLASS_BALANCED_BATCHES=false CHBMIT_SUBJECT_ADVERSARIAL=false CHBMIT_RUN_ID=run_33_patient_window_norm CHBMIT_SKIP_TEST_EVALUATION=true python main.py --mode train
```

Run patient-group-balanced sampling without adversarial or window-normalization changes:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_patient_causal_2s_v1 CHBMIT_WINDOW_SEC=2 CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL=3 CHBMIT_CLASS_BALANCED_BATCHES=false CHBMIT_PATIENT_GROUP_BALANCED_BATCHES=true CHBMIT_SUBJECT_ADVERSARIAL=false CHBMIT_RUN_ID=run_34_patient_group_balanced CHBMIT_SKIP_TEST_EVALUATION=true python main.py --mode train
```

Run GroupDRO on the same patient-group-balanced source batches:

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_patient_causal_2s_v1 CHBMIT_WINDOW_SEC=2 CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL=3 CHBMIT_CLASS_BALANCED_BATCHES=false CHBMIT_PATIENT_GROUP_BALANCED_BATCHES=true CHBMIT_SUBJECT_ADVERSARIAL=false CHBMIT_GROUP_DRO=true CHBMIT_GROUP_DRO_ETA=0.1 CHBMIT_RUN_ID=run_35_patient_group_dro CHBMIT_SKIP_TEST_EVALUATION=true python main.py --mode train
```

Before training any additional ablation, copy its validation artifacts into the
experiment registry. Do not run `event_eval` with `test` until one model and
one validation-selected temporal policy are frozen.
