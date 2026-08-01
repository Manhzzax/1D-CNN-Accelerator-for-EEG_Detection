# Supervised Contrastive Accuracy Screen

## Purpose

This is the prespecified P2 accuracy screen after the dilated-R2 P1 ablation
failed its matched seed-42 comparison. It tests whether CE training can be
regularised by making same-class EEG window embeddings closer and different
class embeddings farther apart. The hypothesis comes from supervised
contrastive learning, which uses all same-label examples in a batch as
positives rather than only the one-hot classification target. [Khosla et al.,
2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

This is an architecture-independent training-objective ablation, not a claim
that contrastive learning is clinically validated for CHB-MIT.

## Fixed Experimental Contract

- Base model: `hierarchical_separable_1dcnn`, R2 Lite, temporal kernels
  `31/7/3`, no third pointwise convolution, 4,917 inference parameters.
- Input: audited 17-channel raw EEG, 5-second, 1-second stride prepared data
  in `chbmit_prepared_raw_5s_v1`.
- Split, train-only channel z-score, optimiser, scheduler, batch size,
  early-stopping rule, and seed remain unchanged from `run_60_r2_raw5s_s42`.
- Training loss: `cross_entropy + 0.05 * supervised_contrastive_loss`.
- SupCon temperature: `0.1`; embeddings are the existing 32-dimensional
  classifier features and are L2-normalised before pairwise similarity.
- Validation model selection: lowest validation CE loss, exactly as the
  baseline. SupCon is not evaluated on validation or test data.
- Inference graph: unchanged. No projection head, parameter, operation, or
  tensor is exported because of this auxiliary loss.
- Evaluation: validation only; `CHBMIT_SKIP_TEST_EVALUATION=1` remains set.

The implementation omits an anchor if a batch has no other sample of the same
class, avoiding an invalid contrastive denominator. Class-balanced sampling
makes such a case unlikely at batch size 128 but it is handled explicitly.

## Decision Rule

The P2 seed-42 checkpoint must reach at least 95.0% balanced validation-window
accuracy to earn replication at seeds 7 and 123. Otherwise it is recorded as
a negative ablation and only the next predeclared method may be attempted.
Neither a maximum epoch accuracy nor a favorable individual seed overrides
validation-loss checkpoint selection.

## Run Command

```bash
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && CHBMIT_WINDOW_SEC=5 CHBMIT_PREPARED_OUTPUT_DIR=chbmit_prepared_raw_5s_v1 CHBMIT_MODEL_ARCHITECTURE=hierarchical_separable_1dcnn CHBMIT_SUPERVISED_CONTRASTIVE=true CHBMIT_SUPERVISED_CONTRASTIVE_COEFFICIENT=0.05 CHBMIT_SUPERVISED_CONTRASTIVE_TEMPERATURE=0.1 CHBMIT_TRAIN_SEED=42 CHBMIT_RUN_ID=run_64_r2_5s_supcon005_t01_s42 CHBMIT_SKIP_TEST_EVALUATION=1 python main.py --mode train
```

Use the standard result-push command with `r=run_64_r2_5s_supcon005_t01_s42`
after the run completes.

## Outcome

`run_64_r2_5s_supcon005_t01_s42` selected epoch 23 by validation CE loss and
achieved 93.367% accuracy, 98.112% AUROC, 93.394% F1, 93.770% sensitivity,
and 93.021% precision. The `hyperparameters.json` confirms coefficient 0.05,
temperature 0.1, and an inference-parameter delta of zero. Relative to the
matched seed-42 R2 baseline, this is a +0.837 percentage-point accuracy gain
and a +1.450-point sensitivity gain.

It remains below the predeclared 95.0% seed-42 gate. Therefore P2 is a
positive development ablation but not a replication candidate, final model, or
paper headline. Do not tune its coefficient, temperature, or early-stopping
settings on this validation cohort. Proceed once to P3 augmentation under a
separate fixed contract.
