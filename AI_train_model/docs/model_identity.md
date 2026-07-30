# Model Identity

## Canonical Name

The current compact seizure-detection model is named **EpiSepNet-5K**:

> **Epi**leptic seizure **Sep**arable **Net**work with approximately 5K
> trainable parameters.

This is the model name used in benchmark tables, figures, FPGA documents, and
future manuscripts. It describes the 17-channel raw, depthwise-separable
1D-CNN architecture, not a clinical product and not a claim of final clinical
validation.

## Provenance

| Identity item | Value |
|---|---|
| Model name | `EpiSepNet-5K` |
| Architecture code | `separable_1dcnn` with 3 temporal filters per input channel |
| Current evidence run | `run_21_raw_2s_temporal3` |
| Input | 17 bipolar channels, 2 s at 256 Hz, 1 s stride |
| Training parameters | 5,013 |
| Fixed-point package | `fpga/reference_run_21_int16/` |
| Current validation window accuracy | 90.0718% |
| Current causal validation event point | 23/29 events, 0.4671 FAR/h, 17 s median delay |

`run_21_raw_2s_temporal3` remains the immutable experiment identifier. It is
needed to reproduce checkpoint, scaler, score arrays, threshold sweep, and
FPGA export, but it must not be used as the model name in figures or prose.

## Versioning Rule

- `EpiSepNet-5K` denotes this fixed architecture and its selected 2-second
  configuration.
- A changed architecture receives a new model name; it is not called
  `EpiSepNet-5K` merely because it has a similar parameter count.
- A new seed, threshold, or prepared data instance receives a new run ID but
  keeps the same model name when architecture and protocol are unchanged.
