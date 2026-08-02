# Fold 00 M47 Inner Grid Screen

**Scope:** seed 7 on inner validation only. The outer future test partition
remains sealed.

| Learning rate | Weight decay | Validation loss | Balanced accuracy (%) | AUROC (%) | Parameters | Best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0010 | 0.0001 | 0.31935 | 88.26 | 94.61 | 5,733 | 10 |
| 0.0010 | 0.0005 | 0.31557 | 88.32 | 94.66 | 5,733 | 10 |
| 0.0003 | 0.0001 | 0.31255 | 88.00 | 94.56 | 5,733 | 38 |
| 0.0003 | 0.0005 | **0.30851** | 88.06 | 94.63 | 5,733 | 38 |

The M47 candidate is the registered P2 hierarchical depthwise-separable
1D-CNN with kernels 47/7/3. Its seed-7 hyperparameter selection uses the
lowest checkpoint validation loss, the same loss monitored by early stopping;
balanced accuracy and AUROC are reported without being used as post-hoc
tie-breakers. The selected configuration is `lr=3e-4`, `weight_decay=5e-4`.

The V2 protocol configuration did not encode a separate hyperparameter-ranking
field. This report documents the operational rule before running M47's
five-seed confirmation. The omission is a disclosure item for the manuscript;
it does not permit any outer-test access or a model-superiority claim.
