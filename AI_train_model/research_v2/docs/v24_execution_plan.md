# V2.4 Score-Ranked Hard-Negative Study

## Question

V2.2-A showed that increasing C1 capacity alone did not make a
calibration-selected low-FAR operating point transfer across forward temporal
folds. V2.3 then tested whether fully clean source-policy alarm contexts could
be replayed as hard negatives, but its candidate pool was infeasibly sparse.
V2.4 tests a distinct, predeclared hypothesis: score-ranked clean interictal
windows from training recordings can make the unchanged C1 classifier less
likely to emit high scores on difficult normal EEG.

## Fixed Contract

- Data and split: the read-only V2.1 F00--F02 five-second causal caches and
  their locked forward manifests.
- Source: the immutable V2.2 C1 seed-42 checkpoint and its train scalers,
  checked by SHA-256 for every fold.
- Inference model: C1 raw multiscale residual depthwise-separable 1D CNN,
  exactly 57,446 parameters. No layer or inference operation changes.
- Training: Adam (`lr=3e-4`, `weight_decay=5e-4`), 50/12/12 epoch rule,
  AMP-FP16 training with FP32 evaluation, and seeds 7, 42, 123, 314, 2718.
- Evaluation: unchanged validation-only threshold/policy calibration and one
  temporal-evaluation replay. The endpoint remains event sensitivity at
  observed FAR <= 0.5/h.

## Single Intervention

For each fold, the frozen source scores **only train recordings**. A candidate
window must have a causal endpoint that is clean interictal and outside the
existing 30-second seizure guard, and it must not be one of the original
sampled normal windows. Candidates are ranked by frozen source ictal score,
reduced to a 30-second within-recording separation, then selected in
patient-group round-robin order.

The cache must contain exactly one unique hard negative per ten ictal windows.
These windows have sampling weight three. The original source train scaler is
copied into the derived cache, so new sampling cannot alter normalization
statistics. The cache fails if the full quota is unavailable; no threshold,
policy, separation, guard, ratio, or multiplier may be relaxed afterward.

## Boundaries and Decision

- V2.4 is a new development-only amendment motivated by the closed V2.3
  feasibility result; it does not modify V2.3 retrospectively.
- No temporal-evaluation EEG is used for source scoring or mining. Blocks 5
  and 6 remain sealed.
- Build and audit all three caches before any H2 training. Then run all five
  seeds per fold, keeping fold variation separate from seed variation.
- H2 is not promotable unless calibration-selected FAR <= 0.5/h transfers
  consistently across F00--F02. No final training, INT16 calibration, or FPGA
  claim follows from development evidence.
