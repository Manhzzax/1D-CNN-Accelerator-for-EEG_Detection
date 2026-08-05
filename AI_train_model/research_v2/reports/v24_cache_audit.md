# V2.4 Cache Audit and H2 Authorization

## Preflight Result

Server-side protocol tests completed successfully (`20` tests, `OK`) before
cache construction.  All three V2.4 caches satisfy the predeclared exact
hard-negative count; none is candidate-limited.  The cached derived tensors
include only `train`, `val`, and `temporal_eval`, and construction rejects
sealed-test artifacts.

| Fold | Ictal / source normal windows | Requested / retained hard negatives | Retained ratio | Weighted hard-negative normal-draw share |
| --- | ---: | ---: | ---: | ---: |
| F00 | 1,459 / 1,459 | 146 / 146 | 10.0069% | 23.0891% |
| F01 | 2,490 / 2,490 | 249 / 249 | 10.0000% | 23.0769% |
| F02 | 3,937 / 3,937 | 394 / 394 | 10.0076% | 23.0904% |

The slight ratio differences arise only because the fixed quota is an integer
rounded from `0.10 * ictal_windows`.  The sampling multiplier remains three.

## Candidate Pool and Score Audit

| Fold | Raw clean candidates | After 30-second separation | Selected source-score min / median / max |
| --- | ---: | ---: | ---: |
| F00 | 490,659 | 12,539 | 0.850736 / 0.972987 / 1.000000 |
| F01 | 982,000 | 24,979 | 0.896674 / 0.993414 / 0.999991 |
| F02 | 1,483,129 | 37,839 | 0.551194 / 0.992331 / 0.999999 |

Each source score stream was reused from the corresponding V2.3 *train-only*
cache only after validating the locked fold manifest, V2.2 C1 checkpoint,
source scaler, score stream, score-record metadata, and recording order.

| Fold | Reused V2.3 score-stream SHA-256 | V2.3 source-cache-summary SHA-256 |
| --- | --- | --- |
| F00 | `a7c5ad140e881e127b00d375bb166dd7b9bcd6125fb519703d380cdcbc43652a` | `1415d1913afdfb88e9d1f50ec0a5cba65e5d086de464dd377911b1e2b6c90adf` |
| F01 | `7626a0aa32d534add65b0ee5b7919f00eead1668e61c1fa10604279742476a95` | `808af464ae256478d2f280dd08e7f8fb1c2dac91f1defcd3663bb0f70d7090d1` |
| F02 | `3f941b234a6d154e412edaca27560d72042703989ac0d95c9f01c1dfc2fa0ce7` | `975d28e8b4ee7e41239a28d0e44328f6f72f9bbd672344b6277e2195e92f8b9b` |

## Decision

H2 is authorized for the already frozen five seeds in each of F00, F01, and
F02.  This authorization selects neither a fold nor a seed.  Run all fifteen
predeclared runs with no alteration to the cache, architecture, optimizer,
score-ranking rule, quota, multiplier, or calibration policy grid.  Blocks 5
and 6 remain sealed.
