# V2.4 Preregistration

## Frozen Before H2 Cache Construction

1. V2.1 F00--F02 manifests, causal five-second/one-second preparation, and
   the prohibition on materializing blocks 5 and 6.
2. The V2.2 C1 seed-42 source artifact, checkpoint/scaler SHA-256 digests,
   source architecture (57,446 parameters), and source recording order for
   each fold.
3. H2 architecture, optimizer, 50/12/12 training rule, five fixed seeds, and
   unchanged validation calibration grid and temporal event evaluator.
4. Candidate eligibility: clean causal interictal endpoint outside the
   existing 30-second guard, exclusion of source-sampled normals, 30-second
   within-record separation, and no temporal-evaluation source scores.
5. Candidate ranking and quota: descending frozen source score within
   patient-group round-robin selection; exactly `0.10 * ictal_windows` unique
   windows; multiplier three; fail the cache if the quota is unavailable.
6. Reuse of the original source train channel z-score tensors.

## Prohibited Actions

- Do not change the source model or source seed, score ranking, quota,
  multiplier, separation, seizure guard, loss, optimizer, architecture,
  training seeds, threshold grid, temporal policy grid, refractory interval,
  or event matching rule after a V2.4 cache exists.
- Do not mine, score, select on, or inspect temporal-evaluation recordings,
  block 5, or block 6.
- Do not claim final validation, patient-independent validation, INT16
  behaviour, FPGA performance, or clinical readiness from H2.

## Required Evidence

Each cache records source artifact/scaler hashes, source score stream hashes,
candidate counts, selected patient-group counts, and selected-score summary.
Each run records its checkpoint, model contract, frozen normalization tensors,
validation metrics, calibration sweep, and one temporal-event result. Report
five-seed variation within a fold and three-fold variation separately.
