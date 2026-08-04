# V2.2-A Preregistration

## Locked Question

Does a raw multiscale residual 1D CNN with 57,446 parameters improve the
separation required for a calibration-selected low-FAR seizure detector when
compared with the completed V2.1 compact-CNN evidence?

## Fixed Elements

- Dataset, 17-channel montage, causal IIR filtering, train-only channel
  z-score, causal endpoint labels, 5-second windows, 1-second stride, and
  30-second interictal guard are unchanged from V2.1.
- Development folds F00, F01, and F02 reuse the audited V2.1 manifests and
  read-only prepared caches.
- Candidate: `C1_multiscale_residual_57k` only.
- Optimizer: Adam learning rate 3e-4, weight decay 5e-4.
- Training seeds: 7, 42, 123, 314, and 2718.
- Training budget: at most 50 epochs, minimum 12 epochs, validation-loss
  patience 12, minimum improvement 0.001.
- Calibration policy grid, 30-second refractory period, one-to-one
  event/alarm matching, and FAR target 0.5/h are unchanged.

## Prohibited Actions

- Do not change architecture, optimizer, seeds, data sampling, threshold grid,
  vote policies, or post-processing after any V2.2-A prediction is observed.
- Do not inspect, prepare, score, or materialize blocks 5 or 6.
- Do not claim a V2.2-A development result as final validation, external
  validation, patient-independent validation, FPGA performance, or clinical
  readiness.

## Required Reporting

For every fold, report balanced-window accuracy and AUROC plus temporal event
sensitivity, FAR/h, delay, patient-group clustered confidence intervals, and
the full five-seed distribution. Seed and temporal-fold variation must remain
separate in tables and confidence statements.
