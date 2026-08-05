# V2.3 Preregistration

## Fixed Before H1 Training

- V2.1 F00--F02 manifests and causal 5-second/1-second prepared caches.
- C1 seed-42 source checkpoint, scaler hashes, and calibration-selected policy
  for each fold, as listed in `configs/protocol_v2_3.json`.
- H1 architecture, parameter count (57,446), optimizer (Adam 3e-4, 5e-4),
  50/12/12 epoch rule, and seeds 7/42/123/314/2718.
- Candidate definition: fully clean train-only source-policy false-alarm
  context; source threshold hit; source sampled normal exclusion; 30-second
  within-record separation; patient-group round-robin selection.
- Maximum hard-negative ratio 0.10 and sampling multiplier 3.0.
- Original source train scaler, calibration grid, temporal policy grid,
  refractory period, FAR target, and event matching rule.

## Prohibited Actions

- Do not alter source model, source seed, threshold, vote policy, ratio,
  sampling multiplier, selection ranking, architecture, loss, optimizer, or
  seeds after a V2.3 cache or prediction exists.
- Do not mine or inspect temporal-evaluation recordings, block 5, or block 6.
- Do not claim H1 as final validation, patient-independent validation, FPGA
  performance, or clinical readiness.

## Required Result

For each fold and seed, preserve the source/mining summary, model contract,
normalization tensors, validation metrics, calibration sweep, and temporal
event result. Evaluate fold consistency rather than treating 15 fold-by-seed
runs as independent patients.
