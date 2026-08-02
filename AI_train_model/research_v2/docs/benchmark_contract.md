# V2 Benchmark Contract

1. The task is binary seizure detection from 17-channel raw CHB-MIT EEG.
2. Every decision uses a 5-second completed window at a one-second stride.
3. The causal endpoint label is positive when the window endpoint belongs to
   the half-open seizure interval `[onset, offset)`.
4. Causal IIR filtering is applied to the full recording before windowing.
   Scaling statistics are fit only on the training partition of the relevant
   temporal fold.
5. Architecture, hyperparameter, policy, and threshold selection occur only in
   inner temporal validation. Outer future partitions are never used for this.
6. Window metrics use the clean, guard-excluded classifier subset. Continuous
   replay uses every complete window in the EDF and reports one-to-one event
   sensitivity, FAR/h, and delay.
7. The five training seeds are `7, 42, 123, 314, 2718`. Dataset sampling uses
   a separate fixed seed. Fold variation and seed variation are reported
   separately.
8. A reported-literature table is descriptive only. Superiority claims use
   only V2 unified reimplementations under this contract.
