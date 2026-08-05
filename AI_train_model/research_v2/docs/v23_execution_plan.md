# V2.3 Policy-Aligned Hard-Negative Study

## Question

V2.2-A showed that a 57,446-parameter raw 1D CNN can have good balanced-window
metrics while the calibration-selected operating point fails to transfer its
FAR target in F00 and F02. V2.3 tests one different explanation: the sampled
normal windows do not sufficiently represent persistent, alarm-like
interictal contexts encountered during full-recording replay.

## Controlled Intervention

The inference graph is unchanged from C1: 57,446 parameters, raw causal
17-by-1280 input, 15/47 multiscale depthwise input branches, residual 7/5/3
refinement, and the same optimizer and five seeds. The only changed training
input is a separate derived train cache containing at most one unique,
policy-aligned hard negative per ten ictal windows.

For each fold, the source is the immutable V2.2 C1 seed-42 artifact. Its
calibration-selected policy is pinned in `configs/protocol_v2_3.json`. The
source model scores **training recordings only**. A candidate must satisfy all
of the following:

1. It belongs to a source-policy alarm context.
2. Every endpoint in that trailing policy context is clean interictal and lies
   outside the existing 30-second seizure guard.
3. The selected window is a source threshold hit, is not already one of the
   sampled source normal windows, and is at least 30 seconds from another
   selected window in the same recording.
4. Selection proceeds round-robin across source patient groups, ranked by
   source score, so a small number of noisy patients cannot fill the cache.

The source train z-score tensor is copied into `frozen_train_scaler.npz` and
is used for the H1 run. This isolates hard-negative sampling from a change in
normalization statistics. Hard negatives have sampling weight 3.0; with the
maximum 0.10 ratio, they can contribute at most about 23.1% of normal draws in
the class-balanced sampler. Candidate shortage retains all eligible windows;
zero candidates is a failed intervention, not a reason to lower a threshold.

## Boundaries

- F00, F01, and F02 are already-consumed development folds; their results are
  not final validation.
- The calibration grid and final model's alarm policy are unchanged. The
  source policy is used only to define train-only hard negatives.
- No temporal-evaluation recording, block 5, or block 6 can be used for
  mining, data construction, selection, scoring, or threshold adjustment.
- Re-running `prepare_v23_policy_hardneg_cache.sh` validates and reuses the
  completed cache rather than re-scoring EDF files.

## Evaluation and Decision

Run all five seeds in each predeclared fold. Report seed variation within each
fold and fold variation separately. The endpoint remains event sensitivity at
temporal FAR <= 0.5/h, supported by balanced-window accuracy and AUROC. H1 is
not promotable unless the FAR target transfers consistently across F00--F02;
no final holdout, INT16 calibration, or FPGA claim follows from development
results alone.

## Execution Closure

The three predeclared cache builds were completed before H1 training.  Their
fully clean, policy-aligned hard-negative yields were 8/146 in F00, 41/249 in
F01, and 106/394 in F02.  Because every fold was candidate-limited and the
effective weighted augmentation ranged only from 1.62% to 7.47% of normal
draws, H1 was closed without seed training.  The decision and calculation are
recorded in `reports/v23_feasibility_decision.md`; V2.3's protocol is not
retrospectively modified.
