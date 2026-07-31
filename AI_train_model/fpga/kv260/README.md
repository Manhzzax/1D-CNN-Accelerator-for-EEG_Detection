# KV260 Measurement Contract

This directory is reserved for **measured** KV260 artifacts. Do not place
estimated or copied numbers here.

## Input and correctness contract

- Source package: `../reference_run_21_int16/`.
- Input: one signed INT16 `[17, 512]` normalised EEG window in row-major
  channel-first order.
- Functional test: reproduce `test_vectors/expected_logits_i64.txt` before
  measuring performance.
- Arithmetic: use layer-specific scales and signed accumulator widths in the
  source `model_manifest.json`; `ap_int<48>` is the safe first implementation.

## Archive after each implementation

Create `runs/<implementation_id>/` and retain:

- HLS synthesis, Vivado implementation, and timing reports;
- bitstream/XCLBIN and source revision SHA;
- exact clock constraint and parallelism/pipeline pragmas;
- one functional-logit report;
- `measurements.csv` copied from the template with real values only;
- command log describing power and latency collection.

Use [measurement_template.csv](measurement_template.csv) for one row per
implementation. The `measurement_boundary` must be `kernel_only` or
`host_dma_kernel`; do not merge the two without separate rows.
