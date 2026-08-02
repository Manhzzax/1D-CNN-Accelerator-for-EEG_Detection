# KV260 DPU Feasibility Audit

Run this audit after a V2 architecture is frozen, before custom HLS work.

1. Record `vitis`, `vivado`, `vitis-ai`, board-image, and DPU `arch.json`
   versions.
2. Export the frozen FP32 model and a deterministic ONNX test vector package.
3. Run the installed Vitis AI Model Inspector/compiler against the actual
   KV260 target. Save its complete operator-support report.
4. Do not claim DPU support from a generic compatibility table. Conv1D support
   must be demonstrated by the target-specific inspector/compiler output.
5. Measure an ARM baseline on the KV260 for causal preprocessing plus model
   inference. Report median/p95 latency, throughput, and board power method.
6. If the target rejects the graph or partitions it unfavorably, freeze the
   DPU finding and use custom HLS as the hardware path. The HLS design must
   reproduce the INT16 golden vectors before synthesis.

The initial quantization target is INT16. INT8 is an additional ablation, not
a substitute for an exact fixed-point reference.
