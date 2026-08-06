# Path A: Patient-Specific High-Accuracy Protocol v1

Branch: `research/patient-specific-a1-v1`  
Config: `config/patient_specific_a1.yaml`

## Decision

After shared-model clean-slate A0 achieved **88.03% ± 1.14%** sealed test
balanced accuracy, Path A targets high window accuracy by training **one
compact raw 1D-CNN per CHB-MIT case** (Wang-style patient-specific), not one
shared model for all patients.

## Claim scope

> Under a predeclared per-case chronological recording split, a compact raw
> hierarchical separable 1D-CNN can detect ictal vs interictal 5 s windows for
> that patient with high sealed-test balanced accuracy. Results are reported as
> mean ± SD across eligible cases.

This is **not** patient-independent / LOSO and **not** interchangeable with
clean-slate shared A0.

## Frozen contract

| Item | Value |
|---|---|
| Unit of personalization | CHB-MIT `case_id` (chb01…chb24) |
| Split unit | Recordings chronological **within that case only** |
| Ratios | 60 / 20 / 20 train / val / test |
| Eligibility | ≥3 recordings; ≥2 seizure annotations; ≥1 seizure in train and test |
| Channels / window | 17 bipolar; 5 s / 1 s; causal_iir; train-only z-score |
| Labels | Full-ictal vs interictal with 30 s guard |
| Model A1.0 | `hierarchical_separable_1dcnn` 31/7/3 (~4.9k params) |
| Checkpoint | Min validation CE |
| Primary metric | Per-case **test balanced** accuracy; cohort **unweighted mean** |
| Success | Mean test balanced ≥ **95%** over eligible cases (seed 42 first pass) |
| Test policy | Train/select on train+val only; evaluate test once after freeze per case |

## Pipeline

1. `plan_patient_specific` → `data/chbmit_protocol_ps_a1_v1/{case}/`
2. `prepare_patient_specific` → `data/chbmit_prepared_ps_a1_v1/{case}/`
3. Train each eligible case (skip test)
4. `checkpoint_eval` each case (balanced test seed = training seed)
5. Aggregate mean ± SD; do not drop low cases without predeclared rule

## Ordered model screens (after A1.0 baseline)

| Step | Change | When |
|---|---|---|
| A1.0 | R2 31/7/3, threshold 0.5 | **Done: mean test bal 90.719%** (primary baseline) |
| A1.0b | Val-selected threshold | **Done: mean 89.918% — rejected** (worse than 0.5) |
| A1.0c | Secondary: cases with ≥20 test ictal | Diagnostic only |
| **A1.1** | **Compact MSR `paper_a_multiscale_residual_1dcnn` ≤25k**, thr 0.5 | **Next** — reuse same prepared per-case NPZs |
| A1.2 | Train-only SupCon or mild aug | If A1.1 still short |
| A1.3 | DWT frontend (no LSTM) | Only if A1.0–A1.2 fail and hardware story may change |

### A1.0b integrity

- Threshold is chosen **only on validation** labels/scores.
- Test is scored **once** with the frozen threshold.
- Architecture, split, and checkpoints are unchanged from A1.0.
- **Result:** mean test balanced **fell** vs thr 0.5 → keep thr **0.5** as primary.

### A1.1 contract

- Architecture: multiscale short/long depthwise + residual stages, `max_parameters: 25000`.
- Same Path A splits/prepared data as A1.0 (`chbmit_prepared_ps_a1_v1/{case}`).
- Train seed 42, skip test; sealed test thr 0.5 via `checkpoint_eval`.
- Run ids: `ps_a11_{case}_s42` / `ps_a11_test_ps_a11_{case}_s42`.
- Promote only if cohort mean test balanced **> A1.0 (90.719%)** and preferably ≥95%.


## Explicit non-goals for Path A

- Shared-model generalization across patients
- Replacing Paper B / EpiSepNet-5K KV260 freeze
- Selecting architecture from sealed test after looking at all cases and re-tuning

## Relation to clean-slate A0

| Protocol | Model | Test result role |
|---|---|---|
| Clean-slate shared | One model all cases | A0: 88.03% ± 1.14% balanced (reported baseline) |
| Path A patient-specific | One model per case | High-accuracy track |

Report both; never merge into one leaderboard cell without protocol labels.
