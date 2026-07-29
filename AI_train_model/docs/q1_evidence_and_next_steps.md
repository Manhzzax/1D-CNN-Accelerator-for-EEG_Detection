# Evidence Record And Next Steps For Q1 Submission

## Locked Experimental Context

All results below use the verified CHB-MIT v1.0.0 EDF corpus, the fixed 17-channel bipolar montage, primary seizure annotations, and a case-wise chronological recording split. Recordings from the same case occur in train, validation, and test, so this is a within-case temporal generalization protocol. It is not a patient-independent claim. The source dataset is [CHB-MIT on PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

## Evidence From The First Two Operating Points

| Run | Training strategy | Validation-selected alarm policy | Test event sensitivity | Test FAR/h | Median delay |
|---|---|---:|---:|---:|---:|
| `run_01` | balanced sampled windows | `3_of_5`, threshold 0.430 | 60/62 = 96.77% | 41.26 | 11.0 s |
| `run_03_mixed_hardneg` | original normals plus 2:1 unique hard negatives; class-balanced batches | `5_of_10`, threshold 0.910 | 36/62 = 58.06% | 0.341 | 13.5 s |

The policy and threshold for each row were selected on validation only. The continuous test set was then evaluated once for that predeclared selection.

## Validation Error Signature Of `run_03_mixed_hardneg`

Validation records 18 of 29 seizure events and has FAR `0.1537/h`. The 26 false alarms are concentrated: `chb07/chb07_14.edf`, `chb20/chb20_26.edf`, and `chb09/chb09_09.edf` account for 16 alarms. In contrast, the 11 missed seizure events are distributed across `chb06`, `chb13`, `chb14`, `chb16`, `chb18`, `chb20`, `chb21`, and `chb23` recordings. This distinguishes a concentrated false-alarm failure from a distributed seizure-sensitivity failure and motivates timestamp-level review before architecture changes.

Of the 11 missed validation seizures, eight contain at least one ictal window above the selected 0.910 threshold; only three have no window above threshold. The main failure is therefore insufficient temporal persistence of positive evidence under the `5_of_10` alarm rule, with a smaller set of morphology-discrimination failures. The next model must improve temporal consistency of seizure scores without recreating the concentrated false-alarm behavior.

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

The next controlled experiment is a causal score-TCN over ten consecutive frozen-CNN scores. It is an inexpensive temporal-context ablation motivated by the validation evidence: the original CNN produces isolated seizure scores, but the current `5_of_10` rule requires five threshold hits. The TCN is trained from train score streams, selected on validation, and evaluated once on test.

## Immediate Diagnostic Commands

Generate validation diagnostics for `run_03_mixed_hardneg` before selecting the next experiment:

```bash
CHBMIT_ANALYSIS_RUN_ID=run_03_mixed_hardneg CHBMIT_ANALYSIS_SPLIT=val python main.py --mode event_diagnostics
```

The same command with `CHBMIT_ANALYSIS_SPLIT=test` is descriptive only. It must not be used to select the next model, threshold, or temporal policy.
