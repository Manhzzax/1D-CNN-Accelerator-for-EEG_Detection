# Clean-Slate Protocol v1 (1D-CNN)

Branch: `research/clean-slate-1dcnn-v1`  
Config: `config/clean_slate_v1.yaml`

## Intent

Rebuild the accuracy track from a locked data contract without using historical
`run_*` metrics as selection evidence. The model family remains a compact raw
multichannel hierarchical separable 1D-CNN.

## Frozen data contract

| Item | Value |
|---|---|
| Split unit | Recording (case-wise chronological) |
| Ratios | Train / val / test = **0.60 / 0.20 / 0.20** |
| Protocol dir | `data/chbmit_protocol_clean_slate_v1` |
| Prepared dir | `data/chbmit_prepared_raw_5s_clean_v1` |
| Channels | 17 canonical bipolar |
| Window / stride | 5 s / 1 s @ 256 Hz → input `[17, 1280]` |
| Labels | Full-ictal positive; normal outside 30 s guard |
| Sampling ratios | Train/val 1:1; test 10:1 normals (NPZ) |
| Normalization | Train-only channel z-score |
| Filter | `causal_iir` (deploy-aligned default) |
| Architecture A0 | `hierarchical_separable_1dcnn` kernels **31/7/3**, ~4.9k params |
| Checkpoint | Minimum **validation CE** |
| Success | **Test balanced window accuracy** mean ≥ **95%** after freeze |
| Selection | Val only; test sealed until one config is frozen |

## What is not this protocol

- Historical `chbmit_prepared_raw_5s_v1` and run_60–85 numbers
- Patient-held-out 60/20/20 patient groups (separate protocol)
- Research V2 FAR / score-replay branches
- Paper B / EpiSepNet-5K 2 s INT16 freeze (untouched)

## Execution order

1. Preflight (conda, CUDA, CHB-MIT root, disk)
2. `plan` → lock recording manifest under clean protocol dir
3. `preprocess` → write clean prepared NPZs
4. Train A0 seed 42 with `CHBMIT_SKIP_TEST_EVALUATION=1`
5. Optional controlled screens (one change at a time), still skip test
6. Freeze config → seeds 42/7/123 → **one test evaluation per seed**
7. Report mean ± SD test balanced accuracy vs 95%

Server helper scripts:

- `tools/clean_slate/00_preflight.sh`
- `tools/clean_slate/01_plan_split.sh`
- `tools/clean_slate/02_prepare_windows.sh`
- `tools/clean_slate/03_train_a0_seed42.sh`

Always:

```bash
export CHBMIT_CONFIG_PATH=config/clean_slate_v1.yaml
```

## Balanced test recipe (for final evaluation)

After freeze, for each training seed:

1. Load the sealed test NPZ (prevalence ~1:10).
2. Keep all ictal windows.
3. Sample an equal number of normal windows with
   `numpy.random.default_rng(training_seed)`.
4. Report accuracy on that 1:1 set as **test balanced accuracy**, plus raw
   full-test accuracy, sensitivity, precision, F1, and AUROC.

## Stop rules

- Do not resplit after seeing metrics.
- Do not choose architecture from test.
- Do not reopen historical val ladders as evidence for this protocol.
- If A0 and at most two predeclared screens fail to justify a freeze, report a
  local ceiling under this contract.
