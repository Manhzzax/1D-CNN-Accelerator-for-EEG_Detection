# 1D-CNN Accelerator for EEG Seizure Detection

Research code for a compact 1D-CNN seizure detector on the CHB-MIT scalp EEG
corpus, with an FPGA-oriented deployment path for KV260.

## Current Reference

The current reference model is **EpiSepNet-5K**. Its evidence run is
`run_21_raw_2s_temporal3`; this is a validation-only result under the locked
within-case chronological protocol, and test inference was intentionally
skipped.

| Item | Current value |
|---|---:|
| Input | 17 bipolar EEG channels, 2 s x 256 Hz, 1 s stride |
| Backbone | **EpiSepNet-5K**: raw separable 1D-CNN, 3 temporal filters per channel |
| Trainable parameters | **5,013** |
| Checkpoint size | **28 KB** |
| Validation window accuracy | **90.07%** |
| Validation window sensitivity | 90.76% |
| Validation F1 | 90.14% |
| Validation AUROC | 96.58% |
| Validation average precision | 96.98% |
| Causal event sensitivity | 23/29 = 79.31% |
| Causal false-alarm rate | 0.4671/h |
| Causal median detection delay | 17 s |

The accessible reference checkpoint is
[`AI_train_model/checkpoints/best_model.pth`](AI_train_model/checkpoints/best_model.pth).
Its SHA-256 manifest is stored beside it.

Window accuracy, event sensitivity, false alarms per hour, and detection delay
are different metrics. The 90.07% figure is a balanced validation-window
accuracy, not a clinical event-detection claim. The event operating point is
causal: alarms are timestamped at the end of the 2-second input window.

## Benchmark Context

Published CHB-MIT papers report accuracy values around 94.93% to 98.43% under
different channel selections, window definitions, splits, labels, and often
classification-only evaluation. They are contextual comparators, not directly
comparable targets. The detailed evidence and metric definitions are in
[`AI_train_model/docs/paper_benchmark_comparison.md`](AI_train_model/docs/paper_benchmark_comparison.md)
and
[`AI_train_model/docs/benchmark_definition_and_comparability.md`](AI_train_model/docs/benchmark_definition_and_comparability.md).
The full CHB-MIT literature tables, separated into prediction, continuous
detection, and window-classification evidence, are in
[`AI_train_model/docs/chbmit_literature_benchmark_tables.md`](AI_train_model/docs/chbmit_literature_benchmark_tables.md).
The model-name and experiment-ID convention is in
[`AI_train_model/docs/model_identity.md`](AI_train_model/docs/model_identity.md).

## Repository Layout

```text
AI_train_model/
  checkpoints/                 # Small, versioned reference checkpoint
    best_model.pth
    best_model.pth.sha256
  results/
    reference/                 # Machine-readable summary of the current result
    archive/                   # Previous ablations retained for research evidence
  docs/                        # Protocols, paper knowledge base, and evidence records
  fpga/                        # FPGA package contract and generated fixed-point exports
  config/                      # Default configuration
  src/                         # Data, model, and evaluation implementation
  scripts/                     # Pipeline entry points
  outputs/                     # Generated server output, intentionally ignored by Git
```

## Reproduce The Reference

On the training server, activate `chbmit-cnn`, point `CHBMIT_RAW_DIR` to the
verified CHB-MIT v1.0.0 directory, then run the audit, preprocessing, training,
and validation-only event evaluation. The exact protocol and recorded result
are in
[`AI_train_model/docs/window_duration_ablation_protocol.md`](AI_train_model/docs/window_duration_ablation_protocol.md).

```bash
cd AI_train_model && CHBMIT_WINDOW_SEC=2 CHBMIT_MODEL_ARCHITECTURE=separable_1dcnn CHBMIT_SEPARABLE_TEMPORAL_FILTERS_PER_CHANNEL=3 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_2s_v1 CHBMIT_RUN_ID=run_21_raw_2s_temporal3 CHBMIT_TRAIN_LEARNING_RATE=0.001 CHBMIT_TRAIN_WEIGHT_DECAY=0.0001 CHBMIT_CLASS_BALANCED_BATCHES=false CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```

Do not select architecture, threshold, or temporal policy from test data. The
next research objective is a patient-held-out, end-to-end causal replication
before further accuracy optimization. The historical reference uses a
within-case split and zero-phase offline filtering, so it cannot establish
unseen-patient or real-time performance. The active protocol and ordered
capacity ablation are in
[`AI_train_model/docs/patient_heldout_causal_protocol.md`](AI_train_model/docs/patient_heldout_causal_protocol.md).

## FPGA Export

The selected separable model has a dedicated deployment exporter. It folds
BatchNorm, quantizes weights and calibrated activations, and emits binary
tensors, normalization constants, a layer manifest, and RTL/HLS test vectors.
Run it on the training server after the selected run output is present:

```bash
cd AI_train_model && CHBMIT_FPGA_SOURCE_RUN_ID=run_21_raw_2s_temporal3 CHBMIT_FPGA_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_2s_v1 CHBMIT_FPGA_OUTPUT_DIR=fpga/reference_run_21_int16 python main.py --mode export_fpga
```

The package is a hardware inference contract, not evidence of FPGA deployment.
KV260 synthesis, latency/resource measurement, causal preprocessing, and
hardware-in-the-loop validation remain required.
