# CHB-MIT Benchmark Data

These compact publication-facing CSV files are the exports of Tables B and C
in `docs/chbmit_literature_benchmark_tables.md`.

They use a semicolon (`;`) delimiter so Excel displays the fields as separate
columns in the project's locale. Metric display cells use an Excel text prefix
and a decimal point, preventing locale-dependent conversion of `90.0718` into
`900.718` when the file is opened directly. Remove the prefix only after
importing into a tool with an explicitly configured decimal locale.

- Table B contains ten columns for continuous event detection: method,
  channels, coverage, input/model, event sensitivity, FAR/h, delay, window
  accuracy, evidence/comparability, and a public link.
- Table C contains eleven columns for window classification and compactness:
  method, coverage/split, representation/model, accuracy, seizure sensitivity,
  model size/efficiency, a public link, and four ranking fields. It groups
  closely related configurations from one study into a single range row where
  appropriate.

- `chbmit_continuous_detection_benchmark.csv`: event-level seizure-detection
  comparison. Use this file for clinical benchmark figures.
- `chbmit_window_hardware_benchmark.csv`: window-classification and deployment
  context. Do not use it as a substitute for the event-level file.

The `Link` column contains a public DOI, publisher, PhysioNet, or arXiv page.
`N/A (this work)` denotes the current project, which has no published paper.
In Table B, a `Secondary: P02` evidence label means the reported metric was
transcribed from Chung et al.'s comparison table; verify the original study
before using it in a journal claim.

Table C ranking fields are intentionally populated only for `This work
quantization`: FP32 and INT16 use the same validation windows. Higher accuracy
and sensitivity are better; smaller stored package size is better. External
rows remain `NR` because their protocols are not directly comparable.

`NR` means not reported. The evidence run for `EpiSepNet-5K` remains
`run_21_raw_2s_temporal3`.
# Accuracy-Efficiency Context

`chbmit_detection_accuracy_efficiency_context.csv` is the expanded literature
landscape for ictal seizure detection. It uses semicolon delimiters. Accuracy
and sensitivity cells deliberately use an Excel text prefix (`'`) and a decimal
point: this prevents a locale-dependent double-click import from converting
`90.0718` into `900.718`. It must **not** be globally sorted or ranked by
accuracy: each row carries task, split, channel, window, evidence tier and
comparability metadata.

Use it to support the narrow statement that the 90.07% EpiSepNet-5K validation
accuracy lies inside a literature range that includes lower/near-90% results and
that its 5,013 parameters are smaller than several reported compact deep
comparators. It cannot establish cross-paper superiority or a causal
accuracy-versus-size trade-off. The full caveats and final three-table plan are
in [`../docs/academic_validity_audit_and_benchmark_plan.md`](../docs/academic_validity_audit_and_benchmark_plan.md).
