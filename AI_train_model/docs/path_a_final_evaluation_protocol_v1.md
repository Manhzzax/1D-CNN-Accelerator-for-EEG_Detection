# Path A Final-Evaluation Contract v1

Branch: `research/path-a-final-evaluation-v1`.

## Status of the existing A1.2 result

The existing A1.2 value, **91.096% +/- 12.68%**, is an unweighted mean of
per-case balanced window accuracy over 24 CHB-MIT cases, with seed 42. The
recording-level within-case time split, causal preprocessing, train-only
normalization, and no window overlap across partitions are valid.

It is nevertheless **exploratory, not final**, because A1.0, A1.0b, A1.1 and
A1.2 used test probes from the same cohort and A1.2 was promoted after that
comparison. Re-running that test cannot restore its independence.

The only permitted wording for this result is:

> Exploratory per-case chronological balanced-window result: 91.10% +/- 12.68%
> across 24 CHB-MIT cases, seed 42.

It must not be called a final test result, a 24-patient result, or a clinical
event detector result.

## What this branch adds

1. `research_path_a_final`: a patient-group-aware audit that merges `chb01`
   and `chb21` into `subject_01_21` before pooled metrics and bootstrap CIs.
2. `scripts/run_patient_specific_event_replay.py`: validation-selected causal
   policy replay on continuous test EDFs. It always writes
   `evaluation_kind=exploratory_test_replay` and cannot be a final result.
3. Three server tools in `tools/patient_specific/final/` to audit window
   artifacts, generate event replay artifacts, and aggregate the two.

## Measurement rules

For each case, fit only on train, select checkpoint and event policy only on
validation, then replay full test EDF recordings. Record both:

- Window: full-prevalence accuracy, balanced accuracy, sensitivity,
  specificity, F1, AUROC and AUPRC.
- Continuous replay: event sensitivity, false alarms per interictal hour,
  median/mean detection delay, number of events, and replay hours.

At cohort level, pool counts by patient group, not case. Bootstrap patient
groups for uncertainty. Training seed variation and patient-group variation
are separate quantities and must not be mixed into one standard deviation.

## What is required for a defensible final claim

The architecture, training hyperparameters, checkpoint rule, threshold,
temporal policy, refractory interval, quantization policy and hardware
interface must be frozen before final evaluation. The final data must be
either an external cohort or a newly declared outer holdout that has never
been used for architecture selection. Current A1.2 artifacts fail this last
condition, so these tools report them descriptively only.

## Server order

```bash
bash tools/patient_specific/final/01_audit_a12.sh
bash tools/patient_specific/final/02_replay_a12_events.sh
bash tools/patient_specific/final/03_aggregate_a12_events.sh
```

The second command replays complete EDF recordings and is the expensive step.
It does not train a model. The first and third commands only process saved
artifacts.
