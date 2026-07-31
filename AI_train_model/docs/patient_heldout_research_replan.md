# Patient-Held-Out Research Replan

## Decision

The current failure is **cross-patient representation shift**, not a threshold
or temporal-policy defect. On the held-out validation patients, the best
current raw model (`run_34`) detects 9/39 events at 0.3535 FAR/h, but detects
0/8 events for `subject_14`. Its aggregate score is therefore not evidence of
robust patient-independent detection. `run_35` GroupDRO at batch-wise eta 0.1
is a negative ablation: the group distribution concentrated on one source
group and event detection fell to 7/39.

Do not use the locked test patients. Do not report a patient-held-out accuracy
headline from a single seed or compare it numerically to patient-specific
papers.

## Evidence Review

| Source | What is verified | Consequence for this project |
|---|---|---|
| [Ali et al., 2024](https://doi.org/10.1098/rsos.230601) | Continuous, cross-subject CHB-MIT evaluation exposes subject variability, class imbalance, and event-level reporting as essential. Its 5-fold/LOO event sensitivities (72.63%/75.34%) are a strict generalisation context, not a window-accuracy target. | Select and report event sensitivity, FAR/h, delay, macro patient-group sensitivity, and worst patient-group sensitivity. |
| [Ghosh et al., 2026](https://doi.org/10.1007/s42452-026-08306-9) | Patient-exclusive 1 s classification; feature extraction, CSP, feature selection, and tuning are fit inside training folds. A 15-feature KNN/RF system reaches 90.6%/90.5% accuracy. | Build a leakage-safe multi-domain feature baseline before increasing CNN capacity. It diagnoses whether raw 5K CNN representation, rather than only optimisation, is limiting generalisation. |
| [Wang et al., 2024](https://doi.org/10.1142/S0129065724500552) | Cross-subject seizure detection via **unsupervised domain adaptation**: shallow MK-MMD alignment plus deep adversarial alignment. This uses target-domain EEG without target labels. | Do not cite or emulate this as pure patient-held-out domain generalisation. It motivates a separate, explicitly labelled calibration-allowed track after the zero-target experiment. |
| [Chung et al., 2024](https://doi.org/10.3389/fneur.2024.1389731) | High sensitivity/FAR results are patient-specific, use selected cases/channels, and include clinician review/re-annotation. | Device and continuous-metric context only. It supports a later patient-calibration/channel-selection study, not a direct generalisation comparison. |
| [Kashefi Amiri et al., 2025](https://doi.org/10.1038/s41598-025-18479-9) | DWT + CNN-LSTM improves its CHB-MIT classification ablation, but does not provide a protocol-matched continuous patient-held-out result. | Use DWT/time-frequency features as a representation ablation, not as a publishable numerical target or immediate FPGA backbone. |
| [Sagawa et al., 2020](https://arxiv.org/abs/1911.08731) | GroupDRO targets worst-group loss but can fail without appropriate regularisation and optimisation. | Retain only one stable small-eta GroupDRO check; do not perform an open-ended eta search. |
| [Zhou et al., 2021](https://openreview.net/forum?id=6xHJ37MVxxp) | MixStyle mixes source-domain feature statistics during training to improve unseen-domain generalisation. | Candidate training-only method after the feature baseline; it preserves the 5K inference graph but requires a controlled same-class source-domain implementation. |

## Correct Experimental Order

### Stage 0: quantify optimisation variance first

`run_34` is one random seed, not a reliable model ranking. Repeat exactly the
same run with training seeds `314` and `2718`; seed `42` is the existing run.
Keep all data, causal filtering, architecture, sampler, learning rate, batch
size, early stopping, and validation policy selection unchanged.

Report for each seed and as mean/range:

- balanced window accuracy, AUROC, and ictal sensitivity;
- micro event sensitivity and FAR/h;
- macro patient-group event sensitivity;
- minimum patient-group event sensitivity and maximum patient-group FAR/h;
- median delay.

No model is a final candidate if its lowest validation patient group remains at
zero sensitivity, even if the micro total improves. This is a development
selection guardrail, not a clinical acceptance standard.

### Stage 1: one bounded robust-optimisation check

Run GroupDRO only once more at eta `0.01` on the same patient-group-balanced
sampler. The previous eta `0.1` collapsed q-weights because updates occur once
per batch. Persist group weights per epoch. Reject GroupDRO if it does not
improve the seed-42 patient-group frontier over `run_34`; do not sweep eta.

### Stage 2: leakage-safe multi-domain feature baseline

Implement a non-deployment comparator on the exact fixed split:

1. Fit preprocessing and any spatial transform only on source training
   patients.
2. Extract per-channel temporal features (RMS, variance, line length),
   spectral band-power/entropy, wavelet energy, and optional train-fold CSP.
3. Fit feature scaling and selection on train patients only; use validation
   patients only for model/policy selection.
4. Evaluate the continuous validation recordings with the same alarm code.

This baseline is deliberately classical and interpretable. If it does not
outperform the raw 5K CNN under the same held-out protocol, a larger
time-frequency backbone is not justified. If it does, it becomes a teacher or
feature-fusion evidence path, not the immediate KV260 deployable model.

### Stage 3: source-only representation generalisation

Only after Stage 2, test one compact source-only method: same-class,
cross-patient MixStyle in the shallow separable-CNN feature map. It is
training-only, uses neither target patient EEG nor labels, and keeps the
EpiSepNet-5K inference parameter count. Compare it against the repeated
raw-CNN reference, not against patient-specific literature values.

### Stage 4: separate calibration-allowed study

If a deployment workflow can observe an initial seizure-free segment from a
new patient, define a second protocol that allows unlabeled causal running
normalisation or UDA. This must be reported separately from Stage 0--3 because
it is target-data adaptation, as in Wang et al., not zero-target
patient-independent generalisation.

### Stage 5: final Q1 evaluation

After one architecture and one alarm policy are frozen, replace the single
60/20/20 holdout result with outer patient folds/LOPO, keeping all
normalisation, feature selection, calibration, and hyperparameters inside each
training fold. Report fold/patient distributions and confidence intervals. The
current untouched test cohort is used once only after this plan is frozen.

## Frozen Hyperparameters for Stage 0

| Item | Value |
|---|---|
| Representation | Raw canonical 17 bipolar channels |
| Window / stride | 2 s / 1 s |
| Filter | 0.5--45 Hz causal IIR plus causal 60 Hz notch |
| Normalisation | Train-only channel z-score |
| Backbone | EpiSepNet-5K, separable temporal filters per channel = 3, 5,013 parameters |
| Optimizer | Adam, learning rate 0.001, weight decay 0.0001 |
| Batch size | 128 |
| Sampling | Equal observed `(class, source patient group)` strata |
| Early stopping | Validation loss, min 8 epochs, patience 6, min delta 0.001 |
| Training seeds | 42, 314, 2718 |

The only variable in Stage 0 is `CHBMIT_TRAIN_SEED`.
