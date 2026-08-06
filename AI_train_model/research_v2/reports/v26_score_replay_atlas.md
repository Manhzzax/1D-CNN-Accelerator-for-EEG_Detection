# V2.6 Score-Replay Diagnostic

## Scope

This is a counterfactual analysis of score streams created during already
consumed F00--F02 replays. The temporal oracle uses future labels only to
characterize score separability at the declared FAR target. It does not select
a threshold, policy, candidate, or final model. Blocks 5 and 6 remain sealed.

## Selected Policy Versus Temporal Oracle

| Candidate | Fold | Selected-policy FAR passes | Oracle FAR-feasible runs | Selected SEN (%) | Selected FAR/h | Oracle SEN at target (%) | Oracle FAR/h | Diagnostic status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | F00 | 0/5 | 0/5 | 56.52 +/- 11.91 | 2.648 +/- 0.723 | NR | NR | representation_limited_at_declared_grid_for_all_replayed_runs |
| C1 | F01 | 5/5 | 5/5 | 64.71 +/- 9.07 | 0.279 +/- 0.172 | 74.12 +/- 2.46 | 0.449 +/- 0.036 | temporal_target_feasibility_matches_calibration_selected_policy_for_replayed_runs |
| C1 | F02 | 0/5 | 5/5 | 74.78 +/- 10.83 | 0.900 +/- 0.108 | 70.43 +/- 12.06 | 0.421 +/- 0.054 | calibration_to_temporal_policy_mismatch_possible_but_not_proven |
| H2 | F00 | 2/5 | 5/5 | 74.78 +/- 4.76 | 0.837 +/- 0.686 | 74.78 +/- 6.45 | 0.453 +/- 0.039 | calibration_to_temporal_policy_mismatch_possible_but_not_proven |
| H2 | F01 | 3/5 | 5/5 | 67.65 +/- 10.19 | 0.484 +/- 0.407 | 69.41 +/- 12.06 | 0.376 +/- 0.182 | calibration_to_temporal_policy_mismatch_possible_but_not_proven |
| H2 | F02 | 1/5 | 5/5 | 76.52 +/- 6.59 | 0.757 +/- 0.371 | 78.26 +/- 4.35 | 0.375 +/- 0.074 | calibration_to_temporal_policy_mismatch_possible_but_not_proven |
| G1 | F00 | 0/5 | 2/5 | 61.74 +/- 9.91 | 2.311 +/- 0.832 | 43.48 +/- 0.00 | 0.485 +/- 0.010 | calibration_to_temporal_policy_mismatch_possible_but_not_proven |
| G1 | F01 | 3/5 | 5/5 | 57.06 +/- 12.23 | 0.429 +/- 0.141 | 64.12 +/- 7.61 | 0.464 +/- 0.028 | calibration_to_temporal_policy_mismatch_possible_but_not_proven |
| G1 | F02 | 0/5 | 5/5 | 73.04 +/- 4.76 | 1.003 +/- 0.127 | 60.87 +/- 16.56 | 0.482 +/- 0.020 | calibration_to_temporal_policy_mismatch_possible_but_not_proven |

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
- Replayed run records: `45`
- No model was trained and no block-5/block-6 recording was scored.
