# Project Contract v1

This clean-room repository implements the Q1 research specification. It has
no scientific dependency on the repository history that preceded its orphan
`main` branch.

## Locked G0--G2 protocol

- Primary dataset: independently acquired CHB-MIT EDF corpus.
- Participant identity: `chb01` and `chb21` are one participant group.
- Input contract: strict 19-channel 10--20 montage, 256 Hz, 4 s windows and
  1 s stride. Missing channels are never interpolated or inferred.
- Split: 23 outer leave-one-subject-out folds. Each outer fold has four
  deterministic, subject-disjoint validation groups and all remaining groups
  in training.
- Preprocessing: causal 0.5--45 Hz band-pass, causal 60 Hz notch, and
  training-recording-only channel z-score.
- Test: all recordings of the held-out participant at natural prevalence.
- Selection: checkpoint, threshold, and temporal policy use validation groups
  only. The outer participant is read once by the final continuous scorer.
- Primary outcomes: event F1, event sensitivity, precision, false positives
  per day, and detection latency. Window metrics are diagnostic only.

The first baseline is the raw temporal 1-D CNN specified in the source Word
document. Quantization, HLS, DPU, and KV260 board work are out of scope until
G2 has passed.

