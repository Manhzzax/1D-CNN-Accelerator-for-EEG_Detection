# Server Execution Environment Record

## SERVER-01 (Current Training Host)

Last inspected: 2026-08-01.

- CPU: Intel Core i9-10940X @ 3.30 GHz (14 physical cores, 28 logical
  threads).
- RAM: 251 GiB total; 226 GiB available at inspection.
- Swap: 2.0 GiB total; 1.3 GiB used at inspection.
- GPU: NVIDIA Quadro RTX 8000 with 48 GiB VRAM.
- NVIDIA driver: 595.71.05; `nvitop` reports CUDA driver compatibility 13.2.
- GPU state at inspection: 0% compute utilization and 28.89 GiB shown as
  allocated to a `No Such Process` entry. Treat this as a stale/orphaned CUDA
  context until `nvidia-smi` and `ps` are reconciled. Do not reset the GPU
  while its display server is active. The remaining free VRAM is still ample
  for the present 5K-parameter EEG experiments, but the stale allocation must
  be recorded in run logs.

Before the first run on SERVER-01, verify the repository revision, Conda
packages, CUDA visibility, raw-data root, prepared-data root, and free disk
space in one command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate chbmit-cnn && cd ~/Manh/1D-CNN-Accelerator-for-EEG_Detection/AI_train_model && git pull origin main && python -c "import torch, mne, sklearn; print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available()); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('mne=',mne.__version__); print('sklearn=',sklearn.__version__)" && test -d /home/ubuntu/Manh/datasets/CHB-MIT/1.0.0 && test -d data/chbmit_prepared_raw_5s_v1 && df -h . /home/ubuntu/Manh/datasets/CHB-MIT/1.0.0 && nvidia-smi
```

Do not start a training run until this command succeeds. The raw EDF and
prepared NPZ paths must exist locally on SERVER-01; a repository clone alone
does not contain these ignored data artifacts.

## SERVER-02 (Previous Training Host)

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
