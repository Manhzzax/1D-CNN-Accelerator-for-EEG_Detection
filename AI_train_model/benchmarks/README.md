# CHB-MIT Benchmark Data

These compact publication-facing CSV files are the exports of Tables B and C
in `docs/chbmit_literature_benchmark_tables.md`.

They use a semicolon (`;`) delimiter so Excel displays the fields as separate
columns in the project's locale. Numeric metrics use a decimal comma to prevent
Excel from treating a decimal point as a thousands separator. Import them with
`;` as the delimiter in other tools.

- Table B contains ten columns for continuous event detection: method,
  channels, coverage, input/model, event sensitivity, FAR/h, delay, window
  accuracy, evidence/comparability, and a public link.
- Table C contains seven columns for window classification and compactness:
  method, coverage/split, representation/model, accuracy, seizure sensitivity,
  model size/efficiency, and a public link. It groups closely related configurations
  from one study into a single range row where appropriate.

- `chbmit_continuous_detection_benchmark.csv`: event-level seizure-detection
  comparison. Use this file for clinical benchmark figures.
- `chbmit_window_hardware_benchmark.csv`: window-classification and deployment
  context. Do not use it as a substitute for the event-level file.

The `Link` column contains a public DOI, publisher, PhysioNet, or arXiv page.
`N/A (this work)` denotes the current project, which has no published paper.
In Table B, a `Secondary: P02` evidence label means the reported metric was
transcribed from Chung et al.'s comparison table; verify the original study
before using it in a journal claim.

`NR` means not reported. The evidence run for `EpiSepNet-5K` remains
`run_21_raw_2s_temporal3`.
