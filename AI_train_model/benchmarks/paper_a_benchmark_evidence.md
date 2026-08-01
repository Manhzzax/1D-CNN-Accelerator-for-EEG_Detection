# Paper A Benchmark Evidence

## Purpose

[`chbmit_paper_a_1dcnn_benchmark.csv`](chbmit_paper_a_1dcnn_benchmark.csv)
is the controlled evidence table for the accuracy-focused paper. It has a
standalone `Parameters` column so parameter count is never confused with model
file size, activation memory, quantized tensors, latency, or FPGA resources.

The table is semicolon-delimited to open as columns in the local Excel setup.
It is an evidence inventory, **not** a global leaderboard.

## How to read it

1. `Direct primary` is suitable for citation after the manuscript author reads
   the original method and protocol. `Direct preprint` is context, not a final
   peer-reviewed comparison. `Secondary verify original` must never be cited
   as if it were primary evidence.
2. A value under `Accuracy (%)` is comparable only if task, split, prevalence,
   channels, window length, and model-selection procedure are materially the
   same. High patient-specific or epoch-level CV values do not establish
   patient-independent performance.
3. `NR` means the source did not provide the metric or parameter count in the
   accessible primary material. It must not be replaced by an estimate.
4. Detection and prediction remain separate tasks. Prediction rows are present
   only as compact 1D-CNN design context.

## Basis for Paper A architecture choices

| Evidence | Design implication |
|---|---|
| LMPSeizNet reports a compact multiscale depthwise-separable CNN with 18,024 parameters. | Test multiscale temporal receptive fields without abandoning Conv1D or the 100K budget. |
| Gu et al. report a 61,218-parameter lightweight CNN in a cross-subject setting. | Test a meaningful capacity range below 100K; do not assume high window CV accuracy transfers to unseen patients. |
| Adatia et al. report 95% for a multichannel depthwise-separable 1D-CNN. | The 95% objective is a feasible research target, but the missing accessible protocol/parameter details prevent a direct claim. |
| Ali et al. use continuous cross-subject event evaluation. | Paper A must retain event sensitivity, FAR/h, and delay alongside window metrics. |
| R2 47/7/3 reached 94.334% only on the development validation split. | It is the required baseline for Paper A; its observed test probe is exploratory and cannot drive architecture selection. |

## Our rows

- `EpiSepNet-R2-5K three-seed reference` is the reproducibility result:
  93.081 +/- 1.096% balanced validation accuracy across seeds 42, 7 and 123.
- `EpiSepNet-R2-5.7K best development checkpoint` is `run_75` with 47/7/3
  kernels, 5,733 parameters and 94.334% validation accuracy at its
  validation-loss-selected checkpoint. It is one seed, therefore it is not a
  final mean result and must not be visually ranked as a final test result.

The active Paper A candidate must be added only after training artifacts are
available. It should report its actual parameter count from `model_spec.json`,
not a planned budget.
