# Policy-Aligned Hard-Negative Mining

## Purpose

`run_13_separable_refine_f_temporal3` with the fine validation policy sweep
detects 21/29 validation events using `7_of_14` at threshold `0.977`, with
0.4493 FAR/h and 13 s median delay. The remaining false alarms are the
interictal patterns most relevant to the deployed decision rule.

This procedure mines them only from the locked **training** recordings. No
validation or test recording is used for mining, selection or normalization.

## Initial Conservative Configuration

- Source model: `run_13_separable_refine_f_temporal3`.
- Policy context: at least `7` threshold hits in a fully interictal `14` window
  context at score threshold `0.977`.
- Requested hard-negative ratio: `0.10` per ictal training window, rather than
  the earlier 1:1 or 2:1 ratios that harmed event sensitivity.
- Minimum separation: 10 s, so one long artifact does not dominate the data.
- Candidate-limited mode is enabled. It records the real available count rather
  than weakening the clinical decision rule to force an arbitrary dataset size.
- A sampling multiplier of `3.0` gives the rare alarm-like negatives additional
  influence inside class-balanced batches without duplicating data files.

## Decision Rule After Mining

First inspect `temporal_hard_negative_mining_summary.json`. Train only if it
contains a nonzero, diverse candidate set. The retrained model must be evaluated
on validation-only continuous EEG with the frozen `7_of_14`, `0.977` policy.

Accept it only if it retains or improves 21/29 event detections while reducing
FAR/h below 0.4493 or reducing median delay below 13 s. Otherwise retain the
unmined `run_13` model and do not consume test recordings.
