# AI Training Pipeline

This directory contains the CHB-MIT preparation, training, continuous
event-evaluation, and quantization code for the EEG detector. The project-level
reference metrics and checkpoint are documented in the
[repository README](../README.md).

## Core Directories

```text
config/       Default reproducible configuration
src/          Dataset, model, evaluation, and utility modules
scripts/      Individual pipeline modes
docs/         Research protocols and evidence records
checkpoints/  Versioned reference checkpoint only
results/      Reference summary and archived experiment artifacts
outputs/      Generated run output; ignored by Git
data/         Generated prepared data and audit artifacts; ignored by Git
```

## Server Setup

```bash
conda activate chbmit-cnn
cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model
pip install -r requirements.txt
```

The CUDA-enabled server should report the Quadro RTX 8000 before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
```

## Primary Pipeline

Set `CHBMIT_RAW_DIR` to the verified CHB-MIT v1.0.0 directory.

```bash
CHBMIT_RAW_DIR=/home/ubuntu/Manh/datasets/CHB-MIT/1.0.0 python main.py --mode audit
```

Use a named `CHBMIT_RUN_ID` for every experiment. Generated output remains in
`outputs/<run_id>/`; after review, retain concise summaries in
`results/reference/` or `results/archive/`. Do not commit raw EDF files,
prepared NPZ datasets, or continuous score arrays.

The historical controlled 2-second experiment is specified in
[docs/window_duration_ablation_protocol.md](docs/window_duration_ablation_protocol.md).
The active Q1-oriented protocol is
[docs/patient_heldout_causal_protocol.md](docs/patient_heldout_causal_protocol.md):
patient-group-disjoint split, causal filtering, validation-only model/policy
selection, and one final test-patient evaluation.
