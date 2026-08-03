# V2 Architecture-Screening Decision

**Decision scope:** architecture and optimizer screening used only fold-00
train/inner-validation data. The outer test partitions of all five temporal
folds remain sealed.

## Frozen roles for cross-fold confirmation

| Role | Candidate | Parameters | Fold-00 five-seed evidence |
|---|---|---:|---|
| Primary accuracy-efficient candidate | B4 dilated CNN | 5,237 | 89.70 +/- 0.57% balanced accuracy; 94.88 +/- 0.26% AUROC |
| Discrimination reference | B2 deep matched CNN | 5,622 | 88.66 +/- 0.65% balanced accuracy; 95.24 +/- 0.39% AUROC |
| Compact historical reference | P1 M31 | 4,917 | 88.37 +/- 1.71% balanced accuracy; 94.32 +/- 0.95% AUROC |

B1 is dominated by B2, B3 is larger and lower-performing, B5 is both larger
and less stable, and B0 is a deliberately weak feature-engineered baseline.

## Next stage

Confirm B4 and B2 with their frozen hyperparameters over temporal folds 01-04
using train/validation tensors only. This tests whether the fold-00 selection
is stable under later within-case recordings without using any outer-test
result to choose a model. P1 remains an ablation/reference and does not drive
the deployment choice.

After the cross-fold confirmation, freeze the primary model before the first
outer-test tensors are materialized. Outer test results will be reported for
the predeclared candidates and never used to change their architecture or
hyperparameters.
