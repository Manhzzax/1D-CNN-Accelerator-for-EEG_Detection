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

## Recorded Result: EpiSepNet-5K

**EpiSepNet-5K** is the current **window-classification reference**. Its
evidence run, `run_21_raw_2s_temporal3`, preserves the raw separable temporal3
training configuration and changes only the input window from 1 s to 2 s.
Model selection used the lowest validation loss at epoch 24; no test inference
was run.

| Validation window metric | Value |
|---|---:|
| Accuracy | **90.0718%** |
| Balanced accuracy | 90.0718% |
| Sensitivity | 90.7645% |
| Precision | 89.5243% |
| F1 | 90.1401% |
| AUROC | 96.5802% |
| Average precision | 96.9764% |

The fine validation-only causal policy sweep selected `10_of_20` at threshold
`0.975`. It detected **23/29 events (79.31%)**, with **0.4671 FAR/h** and a
**17 s median detection delay**. Timestamps are at the 2-second window end.

This event-level point meets the internal FAR constraint (`<= 0.5/h`) and
improves event sensitivity over the 1-second causal baseline (21/29). It is
not, however, a predeclared clinical replacement under the selection rule
above: its FAR is slightly higher than `0.4552/h` and its median delay is
higher than 14 s. Therefore EpiSepNet-5K is locked as the window-level accuracy
reference and a promising low-FAR event candidate, not a final clinical claim.

## Planned 5-Second Context Ablation

The 5-second experiment is a **separate protocol**, not an upgrade of the
2-second accuracy result. It is motivated by Ali et al., who used
non-overlapping 5-second windows for continuous cross-subject event detection,
and by Chung et al., who used 4-second CNN inputs. Their cohorts, labels,
channels, representations, and aggregation policies differ from this project,
so their numerical results cannot select the window duration for us.

| Item | Locked 2 s protocol | Planned 5 s ablation |
|---|---:|---:|
| Samples per channel at 256 Hz | 512 | 1,280 |
| Raw input values | 8,704 | 21,760 |
| INT16 input buffer | 17.0 KiB | 42.5 KiB |
| Decision stride | 1 s | 1 s, retained for a causal comparison |
| Relative Conv1D activation/MAC work | 1.0x | approximately 2.5x |
| Timestamp | end of the input window | end of the input window |

The current `31/7/3` hierarchy has a local convolutional receptive field of
102 samples (about 398 ms); global average pooling consumes the full segment.
Moving to 5 s therefore gives the classifier more aggregate context, but does
not by itself give its local filters a five-second receptive field. A 5 s run
also excludes all seizures shorter than a full labelled 5 s window under the
current full-ictal label rule, changes the number and composition of positive
windows, increases onset latency, and creates 4 s overlap at the retained 1 s
stride. Its accuracy is consequently an ablation result, not a directly
comparable replacement for the 2 s result.

### Decision and promotion rule

1. Keep 2 s as the primary accuracy and FPGA reference. It has 17x512 inputs,
   lower buffer and activation cost, and preserves more short ictal events.
2. First finish topology screening at 2 s: parameter-matched residual
   `31/7/3`, then compact multiscale `15+31`.
3. Only the best 2 s topology gets a 5 s reconstruction from the same locked
   recordings, train-only normalisation, and seed-42 screen. Use a new prepared
   directory and never mix 2 s and 5 s windows.
4. Record the eligible-event count, positive-window count, accuracy, AUROC,
   F1, sensitivity, parameters, exact MAC estimate, input-buffer requirement,
   and causal event sensitivity/FAR/h/delay. Promote 5 s only if its gain is
   larger than seed variability and remains hardware-feasible; otherwise retain
   it as a negative context ablation.

This design lets the paper state whether longer context helps this compact raw
CNN, without hiding its latency, short-event coverage, or KV260 memory cost.

## Recorded 5-Second Seed-42 Screen

The isolated raw 5-second preparation completed with the locked recording
split and train-only channel z-score. It retains 5,023/1,862/4,334 ictal
windows in train/validation/test, compared with 5,344/1,949/4,520 at 2 s. The
loss of sampled ictal windows is 6.0% in train and 4.5% in validation, so the
context ablation remains interpretable. It must nevertheless be reported as a
new window population, not as an identical-window comparison.

`run_60_r2_raw5s_s42` keeps the Adam R2 Lite `31/7/3` model and all training
hyperparameters unchanged; only the raw input length changes from 512 to 1,280
samples/channel. The model still has 4,917 trainable parameters. Its selected
checkpoint is epoch 22, the exact minimum validation-loss epoch.

| Validation metric | R2 Lite, 2 s | R2 Lite, 5 s seed 42 | Difference |
|---|---:|---:|---:|
| Accuracy | 91.175% | **92.830%** | +1.655 pp |
| AUROC | 96.645% | **97.969%** | +1.324 pp |
| Average precision | 96.826% | **98.085%** | +1.259 pp |
| F1 | 91.102% | **92.794%** | +1.692 pp |
| Sensitivity | 90.354% | **92.320%** | +1.966 pp |
| Precision | 91.862% | **93.272%** | +1.410 pp |

This passes the predeclared seed-42 screening margin of 0.5 percentage points.
It is promising evidence that longer raw context helps the unchanged compact
backbone. It is not a 95% result, a clinical result, or a direct replacement
for the 2-second benchmark because the fully ictal window set changed. The next
action is a **locked replication** on seeds 7 and 123 with no architecture,
optimizer, data, threshold, or early-stopping changes. No event evaluation,
test evaluation, INT16 export, or 5-second dilated-head development is allowed
until the three-seed result is available.
