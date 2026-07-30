# Causal Window-Duration Ablation

## Motivation

The strongest current causal baseline is the raw separable 1D-CNN with three
temporal filters per channel. Under causal window-end event timestamps, its
validation result is 21/29 detected events, 0.4552 FAR/h and 14 s median
delay. Multiscale kernels improved window AUROC but not event sensitivity.

The next isolated factor is input context. Shoeb-Guttag used 2-second epochs;
Chung et al. uses longer segment context. A 2-second raw window may expose
onset morphology that a 1-second classifier misses while keeping the same 1 s
decision stride and the compact separable backbone.

## Controlled 2-Second Experiment

- `CHBMIT_WINDOW_SEC=2`, 256 Hz, 17 channels, 512 samples/window.
- Stride remains 1 s. Each inference score is timestamped at the 2-second
  window end, so evaluation remains causal.
- Preserve the locked recording split, 30 s interictal guard, filters,
  train-only z-score, raw representation, non-balanced batches and
  separable temporal3 architecture.
- Rebuild a new prepared directory. It is never mixed with the 1-second data.
- Train and score validation only. Do not run test.

## Selection Rule

Compare the fine causal policy sweep against the 1-second baseline. A 2-second
candidate is retained only if it detects at least 21/29 events and improves at
least one of FAR/h below 0.4552 or median delay below 14 s, without worsening
the other metric beyond the baseline. Otherwise reject it and do not expand to
the more expensive 4-second ablation.

The zero-phase filtering remains offline exploratory preprocessing. A final
FPGA claim still requires causal/stateful filtering and a full remeasurement.
