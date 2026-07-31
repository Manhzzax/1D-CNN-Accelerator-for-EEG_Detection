# CHB-MIT AI and Hardware Benchmark

`chbmit_ai_hardware_benchmark.csv` is the single machine-readable literature
benchmark for this repository. It is semicolon-delimited so it opens as columns
in the local Excel configuration. Metric cells use an Excel text prefix and a
decimal point to preserve the reported value without locale conversion.

The table contains 23 rows:

- 2 rows for the frozen EpiSepNet-5K FP32 and INT16 references;
- 16 directly sourced external papers or preprints;
- 5 rows discovered through a published comparison table, explicitly marked
  `Secondary - verify original before citation`.

Filter before making a figure or a claim:

1. Select `Task = Ictal detection` for the classifier/accelerator landscape.
2. Keep `Direct primary` rows for a manuscript table. `Direct preprint` is
   context until peer review. Never cite a `Secondary` row without opening its
   original paper.
3. Compare accuracy only within a stated protocol. Patient-specific,
   within-case, epoch-level CV, and patient-exclusive evaluations are not one
   leaderboard.
4. Use `Parameters / deployed values` only for architecture scale. Use the
   separate KV260 measurement contract for board PPA, power, and latency.

This design deliberately places detection and prediction in one filterable
file rather than pretending that their accuracy values are interchangeable.
The detailed event-level and task-specific narrative remains in
[`../docs/chbmit_literature_benchmark_tables.md`](../docs/chbmit_literature_benchmark_tables.md).

For the hardware-aware paper, the publication figure should have two panels:

- **AI scale context:** direct ictal-detection rows with reported parameter
  count, plotted as accuracy versus log10(parameters), with marker shape for
  evaluation family.
- **KV260 evidence:** EpiSepNet-5K FP32/INT16/FPGA measured points only, using
  accuracy agreement versus actual bytes, latency, energy, and FPGA resources.

Do not globally rank rows from this CSV.
