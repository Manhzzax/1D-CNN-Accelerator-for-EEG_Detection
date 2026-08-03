# Fold 00 Shortlist After B2

**Scope:** five fixed training seeds on the same inner validation partition.
No candidate has used V2 outer future data.

| Candidate | Parameters | Balanced accuracy (%) | AUROC (%) | F1 (%) | Status |
|---|---:|---:|---:|---:|---|
| P1 M31 31/7/3 | **4,917** | 88.37 +/- 1.71 | 94.32 +/- 0.95 | 87.94 +/- 2.00 | Compact reference |
| B1 vanilla 1D-CNN | 6,338 | 88.44 +/- 0.86 | 94.96 +/- 0.75 | 88.39 +/- 1.04 | Dominated on fold 00 |
| B2 deep matched 1D-CNN | 5,622 | **88.66 +/- 0.65** | **95.24 +/- 0.39** | **88.57 +/- 0.60** | Accuracy reference |

B2 improves mean balanced accuracy by 0.28 points and AUROC by 0.91 points
over M31, while adding 705 parameters (14.3%). It also reduces seed variation
on this fold. Relative to B1, B2 is smaller and has higher means for all three
reported metrics, so B1 is not retained in the provisional shortlist.

## Decision

P1 and B2 define the current Pareto shortlist: P1 minimizes model size, B2
maximizes the observed fold-00 accuracy/AUROC. This is a provisional inner
selection only. B3, B4, and B5 remain predeclared and must be screened before
freezing a shortlist for five-fold evaluation.
