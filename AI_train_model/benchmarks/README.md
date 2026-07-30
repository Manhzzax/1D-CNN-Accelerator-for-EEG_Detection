# CHB-MIT Benchmark Data

These CSV files are the machine-readable exports of Tables B and C in
`docs/chbmit_literature_benchmark_tables.md`.

They use a semicolon (`;`) delimiter so Excel displays the fields as separate
columns in the project's locale. Import them with `;` as the delimiter in other
tools.

- `chbmit_continuous_detection_benchmark.csv`: event-level seizure-detection
  comparison. Use this file for clinical benchmark figures.
- `chbmit_window_hardware_benchmark.csv`: window-classification and deployment
  context. Do not use it as a substitute for the event-level file.

`source_tier` values:

- `current_internal`: reproducible project result.
- `direct`: result reported in the listed local paper PDF.
- `secondary`: result transcribed from a comparison table in a local paper;
  verify the primary paper before a journal claim.

`NR` means not reported. The evidence run for `EpiSepNet-5K` remains
`run_21_raw_2s_temporal3`.
