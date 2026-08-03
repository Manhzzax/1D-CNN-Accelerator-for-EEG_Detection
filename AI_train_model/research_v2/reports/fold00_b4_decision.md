# Fold 00 B4 Dilated CNN Decision

**Scope:** five fixed seeds on fold 00 inner validation only. This is not an
outer-test, patient-independent, or clinical-performance claim.

| Candidate | Parameters | Balanced accuracy (%) | AUROC (%) | F1 (%) |
|---|---:|---:|---:|---:|
| P1 historical M31 | 4,917 | 88.37 +/- 1.71 | 94.32 +/- 0.95 | 87.94 +/- 2.00 |
| B2 deep matched CNN | 5,622 | 88.66 +/- 0.65 | **95.24 +/- 0.39** | 88.57 +/- 0.60 |
| B4 dilated CNN | **5,237** | **89.70 +/- 0.57** | 94.88 +/- 0.26 | **89.40 +/- 0.70** |

B4 improves mean balanced accuracy by 1.04 percentage points over B2 while
using 385 fewer parameters (6.8% fewer). Its AUROC is 0.36 points lower than
B2, so the two candidates represent different operating points rather than a
single-model win: B4 is the current accuracy-efficient candidate, while B2
remains the discrimination/AUROC reference.

## Decision

B4 is retained on the provisional fold-00 shortlist with B2 and P1. B5 must
still complete the same frozen inner-validation screen before any architecture
is promoted to outer temporal folds. No outer-test data may be opened during
this selection stage.
