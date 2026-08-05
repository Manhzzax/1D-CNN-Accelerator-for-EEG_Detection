# V2.3 Policy-Aligned Hard-Negative Feasibility Decision

## Decision

`H1_c1_policy_hardneg_57k` is **not trained**.  The predeclared train-only
candidate definition did not yield enough policy-aligned hard negatives in
any development fold to constitute a comparable 10% hard-negative
intervention.  This is a negative feasibility result, not a model-selection
result.  Blocks 5 and 6 remain sealed.

## Audited Cache Results

The immutable V2.2 C1 seed-42 source checkpoint and its persisted
calibration-selected policy were used only to score recordings in each
fold's training partition.  Candidate contexts had to be entirely clean,
outside the 30-second guard, exclude originally sampled normal windows, and
be at least 30 seconds apart.  Selection was patient-group round-robin.

| Fold | Ictal windows | Requested hard negatives | Eligible after separation | Retained ratio | Effective weighted hard-negative share of normal draws |
| --- | ---: | ---: | ---: | ---: |
| F00 | 1,459 | 146 | 8 | 0.55% | 1.62% |
| F01 | 2,490 | 249 | 41 | 1.65% | 4.71% |
| F02 | 3,937 | 394 | 106 | 2.69% | 7.47% |

The final column uses the frozen sampling multiplier of three:
`3 * hard_negatives / (source_normals + 3 * hard_negatives)`.  At the
predeclared 10% maximum, the corresponding share would have been about
23.08%.  All three caches were therefore candidate-limited, reaching only
5.5%, 16.5%, and 26.9% of their requested hard-negative counts.

For F00, the source policy was `7_of_14` at threshold `0.995`.  It produced
29 source-policy alarms in training recordings, but only 10 clean
false-alarm contexts and eight separated eligible windows.  Eligible windows
were distributed across five patient groups rather than coming from one
recording or subject.

## Interpretation

Training five seeds with these fold-dependent and very low augmentation
doses would not test the stated H1 intervention consistently.  A null or
positive result could not distinguish the effect of policy-aligned hard
negatives from the large difference in available sampling mass between F00
and F02.  Increasing the multiplier, lowering the source threshold, changing
the vote policy, reducing the separation, or relaxing the clean-context rule
after observing these caches would be post-hoc tuning and is prohibited by
the V2.3 preregistration.

The result does **not** show that hard-negative learning is generally
ineffective.  It shows that fully clean, persistent, high-confidence
false-alarm contexts are too sparse under this source model and operating
policy to support the specified V2.3 intervention.

## Consequences

1. Preserve the three ignored cache summaries locally and report their
   candidate yield if V2.3 is discussed in the research log.
2. Do not run H1 seed training, calibration, or temporal-evaluation replay
   for V2.3.
3. Do not alter V2.3's frozen mining rule or inspect blocks 5 and 6.
4. Any future hard-negative study requires a separate written protocol with
   a new, predeclared candidate source and feasibility criterion; it cannot
   amend V2.3 retrospectively.
