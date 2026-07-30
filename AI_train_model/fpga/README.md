# FPGA Reference Package

Run `python main.py --mode export_fpga` on the training server after selecting
a separable-model checkpoint. The default source is
`outputs/run_21_raw_2s_temporal3/` and the default package is written to
`fpga/reference_run_21_int16/`.

The exporter folds BatchNorm, performs symmetric per-tensor INT16 weight and
activation quantization, writes INT32 biases, and emits binary tensors in
little-endian row-major order. `model_manifest.json` is the hardware contract;
it specifies tensor shape, scale, layer order, group count, padding, pooling,
and channel order. `normalization.json` contains the exact train-only z-score
constants. `test_vectors/` contains one quantized input and expected integer
logits for RTL/HLS verification.

The current package starts at filtered and normalized EEG windows. The corpus
preprocessing still uses offline zero-phase filters, so causal FPGA filtering
and hardware-in-the-loop event evaluation remain separate work.
