# SERVER-02 Environment Record

Last verified: 2026-07-28

## Hardware

- CPU: Intel Core i7-8700K @ 3.70 GHz (6 physical cores, 12 logical threads).
- RAM: 31 GiB total; 23 GiB available at inspection.
- Swap: 2.0 GiB total; unused at inspection.
- GPU: NVIDIA GeForce RTX 3090 with 24,576 MiB VRAM.
- NVIDIA driver: 535.309.01.
- Maximum CUDA driver compatibility reported by `nvidia-smi`: CUDA 12.2.
- GPU load at inspection: 0%; only Xorg and GNOME Shell used 26 MiB combined.

## Data State

- CHB-MIT root: `/home/ubuntu/Manh/datasets/CHB-MIT/1.0.0`.
- 686 EDF files match the `RECORDS` manifest exactly.
- `sha256sum -c SHA256SUMS.txt --status` returned exit code 0.

## Python Environment

- Conda environment: `chbmit-cnn`.
- Python: 3.10.
- The initial PyTorch install reports `cuda=False` with a warning that its CUDA runtime requires a newer driver than CUDA 12.2.
- Before GPU training, reinstall PyTorch using CUDA 12.1 wheels, which are compatible with the installed driver:

```bash
python -m pip uninstall -y torch torchvision torchaudio && python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

- Confirm GPU availability afterwards:

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
```

## Execution Policy

- Run EDF/header audit on CPU first; it does not require CUDA.
- Do not train with the legacy random-window split pipeline.
- Use `num_workers: 4` initially, leaving CPU/RAM capacity for EDF I/O and the desktop/session.
