# Evidence Record And Next Steps For Q1 Submission

## Locked Experimental Context

All results below use the verified CHB-MIT v1.0.0 EDF corpus, the fixed 17-channel bipolar montage, primary seizure annotations, and a case-wise chronological recording split. Recordings from the same case occur in train, validation, and test, so this is a within-case temporal generalization protocol. It is not a patient-independent claim. The source dataset is [CHB-MIT on PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

## Evidence From The Evaluated Operating Points

| Run | Training strategy | Validation-selected alarm policy | Test event sensitivity | Test FAR/h | Median delay |
|---|---|---:|---:|---:|---:|
| `run_01` | balanced sampled windows | `3_of_5`, threshold 0.430 | 60/62 = 96.77% | 41.26 | 11.0 s |
| `run_03_mixed_hardneg` | original normals plus 2:1 unique hard negatives; class-balanced batches | `5_of_10`, threshold 0.910 | 36/62 = 58.06% | 0.341 | 13.5 s |
| `run_04_score_tcn` | causal 10-score TCN over frozen `run_03` CNN outputs | `5_of_10`, threshold 0.650 | 40/62 = 64.52% | 0.422 | 14.0 s |
| `run_05_temporal_hardneg` | 474 separated persistent train-only hard negatives, sampled at 2x provenance weight | `10_of_20`, threshold 0.980 | 18/62 = 29.03% | 1.232 | 29.0 s |

The policy and threshold for each row were selected on validation only. The continuous test set was then evaluated once for that predeclared selection.

## Validation Error Signature Of `run_03_mixed_hardneg`

Validation records 18 of 29 seizure events and has FAR `0.1537/h`. The 26 false alarms are concentrated: `chb07/chb07_14.edf`, `chb20/chb20_26.edf`, and `chb09/chb09_09.edf` account for 16 alarms. In contrast, the 11 missed seizure events are distributed across `chb06`, `chb13`, `chb14`, `chb16`, `chb18`, `chb20`, `chb21`, and `chb23` recordings. This distinguishes a concentrated false-alarm failure from a distributed seizure-sensitivity failure and motivates timestamp-level review before architecture changes.

Of the 11 missed validation seizures, eight contain at least one ictal window above the selected 0.910 threshold; only three have no window above threshold. The main failure is therefore insufficient temporal persistence of positive evidence under the `5_of_10` alarm rule, with a smaller set of morphology-discrimination failures. The next model must improve temporal consistency of seizure scores without recreating the concentrated false-alarm behavior.

## Causal Score-TCN Ablation

`run_04_score_tcn` trained a causal TCN on ten consecutive frozen-CNN score logits, with all model selection performed on validation. It reached 19/29 validation events at `0.4375` FAR/h and 40/62 test events at `0.4219` FAR/h. Thus, its test operating point is stable relative to validation, but the validation Pareto trade-off is unfavorable for a low-FAR selection: compared with `run_03_mixed_hardneg`, it gains one validation event while increasing false alarms from 26 to 74.

The validation comparison identifies two events recovered by the TCN (`chb18_31` and `chb20_16`) but one newly missed event (`chb03_34`). Ten seizures remain missed, including 47 s, 64 s, and 81 s events. This rejects the hypothesis that short duration alone explains the residual errors. Several events that had isolated high CNN scores in `run_03` are instead assigned sub-threshold TCN scores, while the TCN also increases persistent non-seizure alarms in the same difficult recordings (`chb09_09`, `chb07_14`, `chb20_26`). A score-only TCN with this sampling and architecture is therefore retained as a negative ablation, not selected as the final low-FAR model.

## Persistent Temporal Hard-Negative Ablation

Only 474 separated train-only score contexts met the strict persistent-negative criterion (at least 3 scores above 0.91 in a fully interictal 10-window context), despite 5,451 requested examples. `run_05_temporal_hardneg` retained all such contexts, used a 2x provenance sampling weight without duplicating samples, and early-stopped at epoch 12 (best epoch 6). Its validation-selected policy detected 14/29 events at `0.4138` FAR/h. The reconciled test result is 18/62 events at `1.2318` FAR/h with 29.0 s median delay. This is worse than `run_03` on both test sensitivity and FAR, so persistent score-context mining in this form is a negative ablation.

The reconciliation is based entirely on the saved validation/test score arrays and does not re-run inference or use test results for model selection. Future event-evaluation summaries must be selected from persisted score arrays before publication.

## Per-Recording Z-Score Screening

`run_07_separable_perrecord_z` retained the raw separable architecture but replaced train-fitted channel-wise z-score with unlabeled per-recording z-score. It early-stopped at epoch 14, with its best validation loss at epoch 8 (`0.4590`), substantially worse than the raw separable run's best validation loss (`0.3061`). It was therefore rejected on validation before continuous event evaluation. Its automatically emitted sampled-test window report is exploratory only and is not used for selection. This result does not reject causal recording-scale adaptation as a deployment idea; it rejects this offline prepared-window statistic as the current training transform.

## Paper-Safe Conclusion

Under the locked within-case CHB-MIT protocol, the compact 1D-CNN has a severe false-alarm problem at high event sensitivity. Temporal confirmation alone reduces false alarms but cannot meet the `0.5/h` target for the baseline model. Training with mixed hard negatives plus stronger temporal confirmation reaches the false-alarm target, but misses 26 of 62 test seizure events. Therefore, the current evidence demonstrates a sensitivity-FAR trade-off rather than a clinically ready detector.

Window accuracy must not be used as the primary outcome. The sampled test split is approximately 91% non-seizure windows, so an all-normal classifier would score about 90.9% accuracy while having zero seizure sensitivity. The primary metrics are event sensitivity, false alarms per interictal hour, and detection delay.

## Claims Not Yet Supported

- Patient-independent or unseen-patient generalization.
- Clinical effectiveness, safety, or a universal clinical FAR threshold.
- Real-time behavior on KV260, because current offline preprocessing uses zero-phase filtering.
- Superiority over published work, because confidence intervals, repeated seeds, and a matched external protocol are not available yet.

## Evidence Gates Before Final Model Selection

1. Run validation-only diagnostics by recording and case; identify whether false alarms cluster in specific cases, channels, or time periods.
2. Freeze a development protocol. All new architecture, normalization, and temporal-policy choices use train/validation only.
3. Add a subject-held-out protocol such as leave-one-patient-out or a prespecified patient-held-out test cohort. Report it separately from the current within-case protocol.
4. Repeat each selected configuration with at least three seeds and report bootstrap confidence intervals over seizure events and interictal hours.
5. Compare causal filtering and scaling suitable for KV260 with the current offline filter. Report preprocessing latency, model latency, LUT/BRAM/DSP use, power, and throughput.

## Immediate Research Question

Can a hardware-aware architecture improve the validation event-sensitivity/FAR Pareto frontier beyond the two observed operating points, while preserving the 17-channel input and a deployment-compatible preprocessing chain?

The next controlled experiment must preserve raw multichannel EEG evidence while targeting the validation false-alarm clusters. It should be designed and selected using train/validation only; the test set remains reporting-only.

## Immediate Diagnostic Commands

Generate validation diagnostics for `run_03_mixed_hardneg` before selecting the next experiment:

```bash
CHBMIT_ANALYSIS_RUN_ID=run_03_mixed_hardneg CHBMIT_ANALYSIS_SPLIT=val python main.py --mode event_diagnostics
```

The same command with `CHBMIT_ANALYSIS_SPLIT=test` is descriptive only. It must not be used to select the next model, threshold, or temporal policy.
