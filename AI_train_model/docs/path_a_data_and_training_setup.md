# Path A: Data Setup and Training Flow (Current)

This document explains **how training data is built and used** for the
patient-specific Path A experiments that produced the frozen results in
`research_v2/reports/path_a/PATH_A_RESULTS_FREEZE_v1.md`.

---

## 1. Big picture (one patient = one dataset = one model)

```text
CHB-MIT EDF (all cases)
        │
        ▼
   Audit manifest (686 EDF, seizure intervals)
        │
        ▼
   For EACH case (chb01 … chb24) independently:
        │
        ├─ Plan: split THAT case's recordings by time → train / val / test (60/20/20)
        ├─ Prepare: cut 5 s windows only from those recordings → NPZ per split
        ├─ Train: fit one 1D-CNN on train+val of that case only
        └─ Test: score sealed test windows of that case once
        │
        ▼
   Cohort metric = unweighted mean of 24 test balanced accuracies
```

**Key idea:** the model never sees EEG from *other* patients during that case’s
training. It only learns “this patient’s earlier recordings → later recordings.”

---

## 2. Step-by-step data setup

### Step A — Audit (once)

- Input: raw CHB-MIT under `data.raw_dir` (server: `/home/ubuntu/Manh/datasets/CHB-MIT/1.0.0`)
- Output: `data/chbmit_audit/recording_manifest.csv`
- Contents: each EDF path, case id, sample count, seizure intervals

### Step B — Plan patient-specific splits

- Command: `python main.py --mode plan_patient_specific`  
  (or `tools/patient_specific/01_plan.sh`)
- Config: `CHBMIT_CONFIG_PATH=config/patient_specific_a1.yaml`
- Output root: `data/chbmit_protocol_ps_a1_v1/`

For **each case** with enough data:

| Rule | Detail |
|---|---|
| Order | Keep recordings in audit/chronological order within the case |
| Split | **60% train / 20% val / 20% test** by recording boundaries |
| Unit | **Whole recording** → train or val or test (never split windows of one EDF across sets) |
| Seizures | Prefer each split to contain seizures; require ≥1 seizure in **train** and **test** |
| Skip | Cases with &lt;3 recordings, &lt;2 seizures total, or no train/test seizure after split |

Each case directory contains:

- `recording_split_manifest.csv` — which EDF is train/val/test  
- `split_plan_summary.json` — counts of recordings/seizures per split  

Plus cohort file: `cohort_summary.json` (eligible vs skipped cases).

**Example (conceptual) for one case:**

```text
chb05 recordings over time:
  [R1][R2][R3][R4][R5][R6][R7][R8][R9][R10]
   |---- train (~60%) ----|-- val --|-- test --|
```

### Step C — Prepare windows

- Command: `python main.py --mode prepare_patient_specific`  
  (or `tools/patient_specific/02_prepare.sh`)
- Output root: `data/chbmit_prepared_ps_a1_v1/{case}/`

For every recording listed in that case’s manifest:

1. Load EDF, map to **17 canonical bipolar** channels  
2. Filter **causal IIR** bandpass 0.5–45 Hz (+ notch as configured)  
3. Cut windows: **5 s**, stride **1 s** (shape per window: **17 × 1280** @ 256 Hz)  
4. Labels:
   - **Ictal (1):** window fully inside a seizure interval  
   - **Normal (0):** outside seizure and outside **30 s guard** around seizures  
5. Sampling ratios into NPZ:
   - train / val: about **1:1** ictal:normal (reservoir sample normals)  
   - test: about **1:10** normals (near natural prevalence for raw acc)

Files per case:

```text
chbmit_train.npz
chbmit_val.npz
chbmit_test.npz
feature_representation.json   # raw, [17, 1280]
preparation_summary.json
test_continuous_recordings.csv
```

Each NPZ stores arrays including `X`, `y`, `recording_id`, `start_sample` so
windows stay traceable to source recordings.

### Step D — Normalization at train/load time (not baked into NPZ as final scale)

When `python main.py --mode train` loads a case:

1. Fit **channel-wise mean/std on train windows only**  
2. Apply the same affine transform to val (and later test)  
3. Save `scaler_mean.npy` / `scaler_scale.npy` next to the run outputs  

→ No leakage of val/test statistics into the scaler.

---

## 3. How one training run uses the data

For case `chbXX`, run id e.g. `ps_a12_chbXX_s42`:

| Setting | Value (current best A1.2) |
|---|---|
| Prepared dir | `data/chbmit_prepared_ps_a1_v1/chbXX` |
| Architecture | `hierarchical_separable_1dcnn` kernels **31/7/3** (~4,917 params) |
| Loss | Cross-entropy **+ 0.05 × SupCon** (train only; no extra inference params) |
| Batching | Class-balanced sampler, batch 128 |
| Optimizer | Adam, lr 1e-3, wd 1e-4 |
| Epochs | 50 (cap), early stop on **val CE**, patience 6 |
| Checkpoint | Epoch with **minimum validation loss** |
| Test during train | **Skipped** (`CHBMIT_SKIP_TEST_EVALUATION=1`) |

After all cases train:

1. `checkpoint_eval` loads `best_model.pth` + same prepared case  
2. Scores **full test** NPZ  
3. Reports:
   - **balanced test accuracy** (equal # normal windows sampled with seed 42) — **primary**  
   - raw test accuracy (prevalence ~1:10) — secondary  
4. Cohort mean = average of 24 balanced numbers (unweighted)

---

## 4. What this setup is *not*

| Not this | Why it matters |
|---|---|
| Shared model train on all patients | That is clean-slate A0, not Path A |
| Random shuffle of all windows | Would leak overlapping 5 s crops across splits |
| Patient-held-out (train A, test B) | Different, harder generalization claim |
| Using test to choose architecture/threshold | Forbidden; thr 0.5 fixed for primary Path A results |

---

## 5. Directory map (server)

```text
AI_train_model/
  config/patient_specific_a1.yaml          # ratios, 5s, causal, model defaults
  data/
    chbmit_audit/                          # global audit
    chbmit_protocol_ps_a1_v1/
      cohort_summary.json
      chb01/recording_split_manifest.csv
      chb02/...
    chbmit_prepared_ps_a1_v1/
      chb01/chbmit_{train,val,test}.npz
      chb02/...
  outputs/                                 # gitignored checkpoints
    ps_a1_chb01_s42/                       # A1.0 train
    ps_a12_chb01_s42/                      # A1.2 SupCon train
    ps_a12_test_ps_a12_chb01_s42/          # sealed test JSON
  research_v2/reports/path_a/              # versioned summaries + freeze doc
```

---

## 6. One-sentence summary

**Data setup:** for each patient, lock a time-ordered recording train/val/test split, cut causal 5 s raw windows with clean ictal/normal labels, balance train/val for learning, keep test sealed; **train one small 1D-CNN per patient** and average their sealed balanced test scores.
