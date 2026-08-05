# V2.5 Design Review: Why Patient-Group Robust Training

## Observed V2 Evidence

The unresolved error is not a lack of window-level capacity alone. C1 has
57,446 parameters and mean AUROC 90.40--94.17% across forward folds, yet its
calibration-selected operating point passes temporal FAR <= 0.5/h in 0/5,
5/5, and 0/5 seeds for F00, F01, and F02. H2 score-ranked hard negatives
changes this to 2/5, 3/5, and 1/5, respectively, but does not establish a
stable fold-consistent operating point.

False alarms are concentrated rather than homogeneous. Aggregating the five
seed replays by patient group, the largest C1 FAR contributors were
`subject_04` in F00 (9.81/h), `subject_08` in F01 (3.60/h), and `subject_05`
in F02 (6.33/h). H2 moved, rather than removed, this concentration: the
largest contributors were `subject_12` in F00 (8.23/h), `subject_05` in F01
(2.98/h), and `subject_05` in F02 (8.13/h). These are diagnostics from already
consumed development replays; they do not inspect a sealed block.

This supports a source-patient robustness hypothesis: reservoir sampling and
mean cross-entropy can optimize the pooled average while leaving uncommon
patient-specific interictal morphology poorly represented. It does not prove
that GroupDRO will work, so V2.5 is one preregistered ablation rather than an
open search.

## External Evidence and Scope

- [Ali et al., 2024](https://doi.org/10.1098/rsos.230601) frames continuous,
  subject-wise CHB-MIT event evaluation as necessary because class imbalance
  and subject variability change the interpretation of segment accuracy. V2.5
  follows this by preserving continuous replay and reporting patient-group
  contributions; it does **not** claim a patient-independent result.
- [Sagawa et al., 2020](https://mlanthology.org/iclr/2020/sagawa2020iclr-distributionally/)
  define GroupDRO as minimizing worst predefined-group loss and show that
  regularization is necessary for worst-group generalization. V2.5 therefore
  retains C1 dropout, weight decay, and early stopping and fixes a small
  batch-wise eta instead of tuning it after results.
- [Zhou et al., 2021](https://mlanthology.org/iclr/2021/zhou2021iclr-domain/)
  motivate training-only source-domain statistic mixing. MixStyle is not added
  to V2.5 because it would be a second, confounded augmentation intervention;
  it remains a possible future protocol only after G1 is closed.
- [Masek et al., 2024](https://doi.org/10.1038/s41598-024-52551-0) show that
  artifacts can drive seizure-detector false alarms and study a separate
  artifact detector. V2.5 deliberately does not add such a detector because
  it would introduce an additional classifier, labels, and hardware pathway.

## Hardware Consequence

The patient identifier, sampler, and GroupDRO state exist only in the training
loop. Inference retains the exact C1 operator sequence, parameter count,
weight shapes, input layout, and INT16 tensor contract. G1 can therefore only
be compared for clinical robustness now; hardware measurements remain
prohibited until a candidate passes development and a final protocol freezes
the model.
