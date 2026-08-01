# Mild EEG Augmentation Accuracy Screen

## Hypothesis and Scope

P3 is the one predeclared training-only augmentation screen after P2 SupCon
improved the seed-42 result but missed the 95% gate. EEG seizure-detection
literature evaluates jittering and scaling among conventional augmentation
methods, while overlap-based augmentation is already intrinsic to this
pipeline's one-second stride. [Alavi et al., 2025](https://www.sciencedirect.com/science/article/pii/S0010482525008637)
[Turk et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732993/)

The experiment does not synthesize new EDF windows, alter labels, resample
validation/test data, or claim physiological realism. It asks only whether
small acquisition-like perturbations regularise the fixed R2 Lite training
objective.

## Fixed Contract

- Base: raw 5-second R2 Lite `31/7/3`, 4,917 inference parameters.
- Training transform, applied after train-channel z-score only:
  1. one gain per complete window, uniformly sampled from `[0.9, 1.1]` and
     shared across all channels/time samples;
  2. iid Gaussian jitter, standard deviation `0.02` in normalised units.
- The shared gain preserves relative channel amplitudes. Noise is deliberately
  mild (2% of the z-score unit), below broad robustness stress testing.
- No transform is used in validation, test, continuous scoring, quantization,
  tensor export, or FPGA inference.
- No SupCon, GRL, GroupDRO, parameter, model operator, or early-stopping
  change is allowed in this screen.
- Seed: 42. Checkpoint: minimum validation CE loss. Test evaluation: skipped.

## Decision Rule

Replicate only if the selected seed-42 checkpoint reaches 95.0% balanced
validation-window accuracy. Otherwise preserve this single result, do not tune
gain/noise/patience, and proceed to the one bounded capacity check.

## Run Command

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_WINDOW_SEC=5 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_5s_v1 CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn CHBMIT_MILD_EEG_AUGMENTATION=true CHBMIT_MILD_EEG_AUGMENTATION_GAIN_DELTA=0.1 CHBMIT_MILD_EEG_AUGMENTATION_NOISE_STD=0.02 CHBMIT_TRAIN_SEED=42 CHBMIT_RUN_ID=run_65_r2_5s_aug_g10_n02_s42 CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```

## Outcome

`run_65_r2_5s_aug_g10_n02_s42` selected epoch 23 by validation CE loss and
achieved 93.985% accuracy, 98.546% AUROC, 93.893% F1, 92.481% sensitivity,
and 95.349% precision. It improves matched baseline seed-42 accuracy by 1.155
points but is below the 95.0% gate and reduces sensitivity relative to P2.
Do not tune augmentation magnitudes or replicate it; continue through the
bounded architecture ladder in `docs/r2_accuracy_ceiling_experiment_plan.md`.
