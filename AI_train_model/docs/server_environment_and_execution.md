# SERVER-02 Environment Record

Last verified: 2026-07-28

## Hardware

- CPU: Intel Core i9-10940X @ 3.30 GHz (14 physical cores, 28 logical threads).
- RAM: 188 GiB total; 178 GiB available at inspection.
- Swap: 2.0 GiB total; unused at inspection.
- GPU: NVIDIA Quadro RTX 8000 with 49,152 MiB VRAM.
- NVIDIA driver: 535.230.02.
- Maximum CUDA driver compatibility reported by `nvidia-smi`: CUDA 12.2.
- GPU load at inspection: 0%; desktop processes used 395 MiB VRAM combined.

The earlier RTX 3090/i7-8700K entry referred to a different server and must not be used for SERVER-02 resource or latency claims.

## Data State

- CHB-MIT root: `/home/ubuntu/Manh/datasets/CHB-MIT/1.0.0`.
- 686 EDF files match the `RECORDS` manifest exactly.
- `sha256sum -c SHA256SUMS.txt --status` returned exit code 0.

## Python Environment

- Conda environment: `chbmit-cnn`.
- Python: 3.10.
- PyTorch: `2.5.1+cu121`.
- CUDA is available and detects `Quadro RTX 8000` correctly. CUDA 12.1 PyTorch wheels are compatible with the installed CUDA 12.2 driver.
- Verify when an environment is recreated:

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
```

## Execution Policy

- EDF/header audit passed on 2026-07-29: 686 manifest EDFs, 686 local EDFs, 0 header errors.
- The audit parser currently produced 158 seizure intervals. This does not match the 198 seizures documented by PhysioNet, so label parsing must be reconciled before preprocessing.
- Run EDF/header audit on CPU first; it does not require CUDA.
- Do not train with the legacy random-window split pipeline.
- Use `num_workers: 4` initially, leaving CPU/RAM capacity for EDF I/O and the desktop/session.
