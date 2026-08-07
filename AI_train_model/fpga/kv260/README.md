# KV260 Measurement Contract

This directory is reserved for **measured** KV260 artifacts. Do not place
estimated or copied numbers here.

## Active implementation track

The frozen H0 implementation is now specified in
[`episepnet_5k/`](episepnet_5k/). It contains the model identity, M1a C golden
implementation, integer arithmetic contract, measurement definitions, and the
ordered HLS-to-board plan. It is based only on
`../reference_run_21_int16/`; it must not be replaced by an active
patient-specific or five-second accuracy candidate.

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
implementation and boundary. The `measurement_boundary` must be
`kernel_only`, `dma_kernel`, or `host_dma_kernel`; do not merge them into one
latency number.
