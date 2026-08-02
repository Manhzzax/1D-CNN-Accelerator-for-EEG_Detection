# Fold 00 M31 Seed Confirmation

**Scope:** inner validation only. The outer future test recordings were not
materialized or evaluated.

| Seed | Balanced accuracy (%) | AUROC (%) | AP (%) | F1 (%) | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 7 | 89.61 | 95.18 | 95.20 | 89.53 | 18 |
| 42 | 89.16 | 94.99 | 95.23 | 88.74 | 22 |
| 123 | 85.87 | 92.83 | 93.27 | 85.13 | 16 |
| 314 | 89.87 | 94.61 | 94.69 | 89.72 | 13 |
| 2718 | 87.35 | 93.99 | 94.52 | 86.59 | 16 |
| Mean +/- sample SD | **88.37 +/- 1.71** | **94.32 +/- 0.95** | **94.58 +/- 0.80** | **87.94 +/- 2.00** | 17.0 |

The candidate is P1, the 4,917-parameter hierarchical depthwise-separable
1D-CNN with kernels 31/7/3. All runs used raw 17-channel, five-second causal
windows; train-only channel z-score scaling; learning rate 0.001; weight decay
0.0005; 50 maximum epochs; and 12/12/0.001 early stopping.

The five checkpoint hashes are distinct and every provenance record points to
protocol commit `a75d3e0`. Best epochs range from 13 to 22, so 50 epochs was a
budget ceiling rather than an instruction to overfit.

## Decision

P1 remains the small-model reference. Its approximately 4.0-point
seed-to-seed balanced-accuracy range on fold 00 means it is not yet a final
architecture claim. The next valid comparison is predeclared P2 (47/7/3) and
the registered baselines on the same fold-local train/validation tensors.
Only after architecture and hyperparameters are frozen may V2 materialize and
evaluate the outer future partitions.
