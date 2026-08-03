# Fold 00 B3 Multiscale Decision

**Scope:** five fixed seeds on fold 00 inner validation only.

| Candidate | Parameters | Balanced accuracy (%) | AUROC (%) | F1 (%) |
|---|---:|---:|---:|---:|
| B2 deep matched CNN | **5,622** | **88.66 +/- 0.65** | **95.24 +/- 0.39** | **88.57 +/- 0.60** |
| B3 multiscale CNN | 17,826 | 87.32 +/- 1.31 | 93.89 +/- 1.76 | 87.14 +/- 1.49 |

Although B3 had a favorable seed-7 AUROC, its five-seed average is lower than
B2 by 1.34 balanced-accuracy points and 1.35 AUROC points. Its variation is
also materially larger, while its parameter count is 3.17 times B2's.

## Decision

B3 is removed from the provisional shortlist. The fold-00 multiscale result
does not justify its parameter or accelerator cost. This is an inner
development decision; B3 remains a reported baseline rather than discarded
evidence.
