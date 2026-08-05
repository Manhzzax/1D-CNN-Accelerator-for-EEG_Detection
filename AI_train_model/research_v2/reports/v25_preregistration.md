# V2.5 Preregistration

## Frozen Before Any G1 Training

1. The audited V2.1 patient-group forward manifests and read-only F00--F02
   causal caches; blocks 5 and 6 remain sealed.
2. The C1 57,446-parameter inference graph, all convolution kernels, causal
   preprocessing, endpoint labels, train-only channel z-score, and balanced
   validation windows.
3. Adam learning rate `3e-4`, weight decay `5e-4`, dropout `0.25`, the 50/12/12
   budget, AMP FP16 training with FP32 evaluation, and seeds 7/42/123/314/2718.
4. Equal observed `(class, source patient group)` sampler, GroupDRO objective,
   exponentiated-gradient eta `0.01`, and source patient group as training-only
   metadata.
5. The calibration-only threshold/policy grid, 30-second refractory interval,
   event matching rule, primary FAR target 0.5/h, and one temporal replay.
6. Promotion criteria in `docs/v25_execution_plan.md` and reporting of
   per-patient-group event and false-alarm contributions.

## Prohibited Actions

- Do not alter eta, sampler, group definition, architecture, optimizer,
  regularization, seed list, threshold grid, temporal policy grid, or event
  matching after the first G1 run begins.
- Do not combine GroupDRO with hard-negative mining, MixStyle, contrastive
  loss, subject-adversarial training, or target-patient adaptation.
- Do not use calibration or temporal-evaluation recordings to construct a
  source group, select a training sample, fit normalization, or update a model.
- Do not open blocks 5 or 6, make final-validation or patient-independent
  claims, calibrate INT16 activations, or claim FPGA performance.

## Required Evidence

Each run must persist its checkpoint, C1 parameter count, sampler strata,
GroupDRO weights for every epoch, hyperparameters, normalization tensors,
validation metrics, calibration sweep, and one temporal replay. The final
decision must compare G1 with C1 and H2 on all three folds and show both micro
and patient-group diagnostics.
