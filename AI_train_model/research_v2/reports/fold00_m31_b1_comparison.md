# Fold 00 P1/B1 Seed Comparison

**Scope:** five fixed training seeds on fold 00's inner validation partition.
No outer-test tensors or event-level outer replay were used.

| Candidate | Parameters | Balanced accuracy (%) | AUROC (%) | F1 (%) |
|---|---:|---:|---:|---:|
| P1 M31 31/7/3 | **4,917** | 88.37 +/- 1.71 | 94.32 +/- 0.95 | 87.94 +/- 2.00 |
| B1 vanilla 1D-CNN | 6,338 | **88.44 +/- 0.86** | **94.96 +/- 0.75** | **88.39 +/- 1.04** |

B1 changes mean balanced accuracy by only +0.06 percentage points relative to
M31, but raises mean AUROC by +0.64 percentage points and has lower seed
variation on this fold. It uses 1,421 more parameters (28.9% above M31).
Paired seed differences are mixed, so fold 00 alone cannot establish a
superiority claim.

## Decision

Both P1 and B1 remain co-leads for subsequent inner candidate screening. P1
is the preferred compact hardware reference; B1 is the ordinary-convolution
accuracy reference. Neither candidate advances to outer future evaluation
until the remaining predeclared baselines have been screened and the
architecture shortlist is frozen.
