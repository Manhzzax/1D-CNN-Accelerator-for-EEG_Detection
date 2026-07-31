# Hardware-Aware Biomedical AI / KV260 Paper Plan

## Paper Scope

The selected paper direction is a **hardware-aware biomedical AI accelerator**,
not a claim of clinical state-of-the-art seizure detection.

The central contribution is a co-designed, fixed-point implementation of
EpiSepNet-5K for KV260 that preserves the selected model behaviour while
reducing storage and measuring real board latency, throughput, resources, and
power. The model-quality evidence is necessary, but it supports the
accelerator claim rather than being the sole contribution.

## Frozen Deployment Reference

| Item | Frozen value |
|---|---|
| Model identity | EpiSepNet-5K |
| Evidence run | `run_21_raw_2s_temporal3` |
| Input | 17 canonical bipolar channels, 2 s x 256 Hz (`17 x 512`) |
| Inference graph | Separable 1D-CNN, 5,013 trainable parameters |
| Numeric path | BatchNorm-folded symmetric INT16 weights/activations, INT32 biases |
| Tensor package | `fpga/reference_run_21_int16/` |
| Package size | 10,030 B exported INT16 tensors |
| Software reference | 90.0718% balanced validation-window accuracy; 90.7645% sensitivity |
| INT16 emulation | 90.0462% accuracy; 99.9743% agreement with folded FP32 |

These validation numbers are a locked within-case engineering reference. They
must not be described as patient-independent, clinical, final-test, or
end-to-end FPGA results.

## Required Evidence for Submission

### 1. Functional correctness

The accelerator must consume `test_vectors/input_i16.bin` and match the
expected integer logits in `test_vectors/expected_logits_i64.txt` under the
scales in `test_vectors/manifest.json`.

Report:

- model-package SHA-256 and tensor manifest version;
- integer-logit agreement for the supplied vector;
- agreement and accuracy on a held-out validation-window set using the same
  input boundary;
- FP32 vs folded-float vs INT16-emulator vs FPGA prediction agreement.

The hardware boundary is initially **after causal EEG filtering and z-score
normalisation**. It is invalid to call this end-to-end EEG acceleration until
those stages also run on the board.

### 2. KV260 PPA and timing

Measure the implemented bitstream, not HLS estimates alone.

| Metric | Required measurement |
|---|---|
| Frequency | Achieved post-route clock frequency (MHz) |
| Latency | Batch-1 kernel latency per `17 x 512` window, including DMA where claimed |
| Throughput | Sustained windows/s and equivalent EEG seconds/s |
| Resources | LUT, FF, BRAM/URAM, DSP from implementation report; both count and percent |
| Memory | On-chip and off-chip model/input-buffer footprint |
| Power | Board power or incremental accelerator power with tool and sampling method stated |
| Energy | mJ/window, derived from measured power and latency |

At least 1,000 batch-1 inferences are required for latency statistics. Report
median, p95, and measurement boundary. Do not mix host-to-board transfer time
with kernel-only latency without reporting both separately.

### 3. Fair baselines

All baselines use the same frozen checkpoint, INT16 input contract, one
`17 x 512` window, and batch size one:

1. PyTorch FP32 on the server CPU;
2. PyTorch CUDA FP32/AMP inference on the RTX 8000;
3. KV260 accelerator kernel-only;
4. KV260 end-to-end host plus DMA path.

If available, add a baseline HLS design with no loop pipelining/parallelism to
quantify the proposed dataflow/parallelisation contribution. Comparisons with
published FPGA papers are contextual unless model, FPGA family, precision, and
input boundary match.

### 4. Biomedical validation table

The paper needs one compact model-quality table, but it is not ranked against
patient-specific clinical work:

- balanced-window accuracy, AUROC, sensitivity, F1;
- continuous event sensitivity, FAR/h, and delay on the declared protocol;
- FP32/INT16/FPGA agreement;
- patient-held-out pilot result reported separately with all limitations.

The historical EpiSepNet-5K reference is adequate for hardware verification.
Patient-held-out work remains a validity track and must not block KV260
implementation; it may not be replaced by a favorable accuracy headline.

## What Must Not Be Claimed Yet

- clinical-grade or patient-independent superiority;
- end-to-end FPGA EEG detection;
- FPGA latency, power, or resource use before KV260 implementation reports;
- comparison superiority over papers using patient-specific channels,
  re-annotations, or incompatible data splits.

## Milestones

1. **M0 complete:** audited CHB-MIT, frozen EpiSepNet-5K checkpoint, INT16
   manifest, and test vector.
2. **M1:** reproduce integer logits in C/HLS simulation.
3. **M2:** synthesise/place/route on KV260 and archive implementation reports.
4. **M3:** benchmark batch-1 kernel and DMA latency, throughput, resources,
   and power against CPU/GPU.
5. **M4:** run an FPGA validation-window agreement test and reproducible
   continuous-score replay at the stated input boundary.
6. **M5:** complete a small, explicitly separate patient-held-out validity
   table and freeze the narrative for submission.

M2--M4 determine whether the work reaches hardware-aware Q1 quality. More
CNN hyperparameter searching does not replace those milestones.

The research verification, precise arithmetic gates, and controlled HLS
experiment matrix are in
[`hardware_aware_research_verification.md`](hardware_aware_research_verification.md).
The active IEEE TBioCAS manuscript draft is in
[`../paper/tbiocas_kv260/`](../paper/tbiocas_kv260/).
