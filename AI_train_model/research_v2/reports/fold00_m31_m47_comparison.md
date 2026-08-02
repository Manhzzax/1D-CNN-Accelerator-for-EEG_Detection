# Fold 00 P1/P2 Seed Comparison

**Scope:** five fixed training seeds on fold 00's inner validation partition.
No outer-test tensors, continuous outer replay, or event metrics were used.

| Candidate | Kernels | Parameters | Balanced accuracy (%) | AUROC (%) | F1 (%) |
|---|---|---:|---:|---:|---:|
| P1 M31 | 31/7/3 | **4,917** | **88.37 +/- 1.71** | **94.32 +/- 0.95** | **87.94 +/- 2.00** |
| P2 M47 | 47/7/3 | 5,733 | 87.96 +/- 0.59 | 94.19 +/- 0.34 | 87.64 +/- 0.58 |

M31's mean balanced-accuracy advantage is 0.41 percentage points and its mean
AUROC advantage is 0.14 percentage points. M47 is more seed-stable on this
single inner partition, but it has 816 additional parameters (16.6% above
M31) and does not show a consistent performance improvement: paired seed
deltas for M31 minus M47 range from -1.55 to +1.61 percentage points.

## Decision

P2 is **not promoted** over P1 for the hardware-aware reference path. This is
not a claim that P1 is the final V2 winner: one inner temporal fold cannot
establish cross-fold robustness. P1 remains the compact reference while the
other predeclared baselines are evaluated under the same sealed protocol.

The lower seed standard deviation of P2 is retained as an observation. It does
not outweigh the absence of a mean performance gain or the added parameter
cost, and it should be revisited only in an aggregate five-fold analysis.
