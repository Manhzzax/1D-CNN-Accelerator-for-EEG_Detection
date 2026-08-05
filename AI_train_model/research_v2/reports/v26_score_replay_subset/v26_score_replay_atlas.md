# V2.6 Score-Replay Diagnostic

## Scope

This is a counterfactual analysis of score streams created during already
consumed F00--F02 replays. The temporal oracle uses future labels only to
characterize score separability at the declared FAR target. It does not select
a threshold, policy, candidate, or final model. Blocks 5 and 6 remain sealed.

## Selected Policy Versus Temporal Oracle

| Candidate | Fold | Selected-policy FAR passes | Oracle FAR-feasible runs | Selected SEN (%) | Selected FAR/h | Oracle SEN at target (%) | Oracle FAR/h | Diagnostic status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | F00 | 0/1 | 0/1 | 56.52 +/- 0.00 | 2.413 +/- 0.000 | NR | NR | representation_limited_at_declared_grid_for_all_replayed_runs |

## Interpretation Rule

- A selected-policy pass means the calibration-selected policy replayed at FAR <= 0.5/h.
- An oracle-feasible run means at least one *counterfactual* policy in the
  unchanged grid would have met the target on that future block. It cannot be
  deployed retrospectively and must not be used to choose a replacement policy.
- If the oracle is infeasible, the fixed score stream has no operating point in
  the declared grid that reaches the target; this is evidence against score
  separability at that grid, not a proof of a physiological cause.

## Integrity

- Artifact-diagnostic config SHA-256: `446eb1a135cd6c652a1297c1a3a77e534b731b7d23dfd8a1a28b0cfdb73a33f6`
- Score-replay config SHA-256: `0ae009c9a9ef794eb35b2b5249b00d250598357282d67e5bd6e6da5e83d897d6`
- Replayed run records: `1`
- No model was trained and no block-5/block-6 recording was scored.
