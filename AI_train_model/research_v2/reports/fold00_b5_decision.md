# Fold 00 B5 Residual CNN Decision

**Scope:** five fixed seeds on fold 00 inner validation only. No outer-test
windows were materialized or evaluated.

| Candidate | Parameters | Balanced accuracy (%) | AUROC (%) | F1 (%) |
|---|---:|---:|---:|---:|
| B4 dilated CNN | **5,237** | **89.70 +/- 0.57** | **94.88 +/- 0.26** | **89.40 +/- 0.70** |
| B5 residual multiscale CNN | 57,446 | 88.34 +/- 2.00 | 94.50 +/- 0.84 | 88.21 +/- 1.89 |

The B5 seed-7 screen was favorable, but the fixed five-seed confirmation was
not stable: two seeds produced balanced accuracy near 86%. B5 is 10.97 times
larger than B4 and is lower on every averaged window metric.

## Decision

B5 is retained as a reported baseline but is removed from the provisional
shortlist. Its accuracy variation and accelerator cost are not justified by
the fold-00 result.
