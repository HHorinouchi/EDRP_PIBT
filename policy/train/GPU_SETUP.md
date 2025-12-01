# GPU Setup Guide (Linux + NVIDIA, CUDA 12.x)

This guide helps install a correct PyTorch CUDA build and run the GPU ES trainer (`train_by_gpu.py`) on Linux with NVIDIA GPUs.

IMPORTANT
- Use ONE method (Conda or pip). Do not mix them in the same environment.
- Make sure your driver supports CUDA 12.x. Check with `nvidia-smi` (the driver version row).

## 0) Activate your conda environment

```bash
conda activate edrp
```

## 1) Conda method (RECOMMENDED)

This installs PyTorch and CUDA libraries from official channels.

```bash
# Remove any previous torch packages (optional but recommended to avoid conflicts)
conda remove -y pytorch torchvision torchaudio pytorch-cuda

# Install the CUDA 12.1 build of PyTorch
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); \
print('device_count', torch.cuda.device_count()); \
print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

If the last line prints `cuda_available True` and a proper device name, the install is good.

## 2) pip method (alternative)

Use pip only if you do not use conda to manage your environment.

```bash
# Clean existing (CPU) wheels:
python -m pip uninstall -y torch torchvision torchaudio

# Install CUDA 12.1 wheels from PyTorch index
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Verify
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); \
print('device_count', torch.cuda.device_count()); \
print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

NOTE
- If you copied commands from chat and saw `;&` or `&&` in your terminal, re-type the commands. Those artifacts come from HTML escaping and will break bash. Run one command per line as above.

## 3) Smoke test the GPU trainer

Once PyTorch CUDA is working, run a short training:

```bash
python policy/train/train_by_gpu.py \
  --iterations 5 \
  --population 8 \
  --episodes-per-candidate 2 \
  --eval-episodes 2 \
  --seed 0 \
  --map-name map_3x3 \
  --agent-num 3 \
  --speed 1.0 \
  --time-limit 300 \
  --collision bounceback \
  --task-density 1.0 \
  --workers 2
```

You should see logs like:
```
[GPU-ES Iter 001] ... device=cuda
==== Execution time ====
Device: cuda, Elapsed: ...
```

If it shows `device=cpu`, PyTorch did not detect your GPU. Re-check your install and driver.

## 4) Troubleshooting

- ImportError: iJIT_NotifyEvent (when importing torch)
  - You likely have a mismatched or CPU-only torch wheel. Re-install with the Conda method above:
    ```
    conda remove -y pytorch torchvision torchaudio pytorch-cuda
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
  - Ensure you don’t have leftover CPU wheels in the same env (`pip show torch` vs `conda list | grep torch`). Stick to one package manager.

- `torch.cuda.is_available()` is False
  - Check the driver with `nvidia-smi`. Update to a driver supporting CUDA 12.x.
  - Make sure you installed the CUDA-enabled build as shown above (Conda: `pytorch-cuda=12.1`; pip: `--index-url ... cu121`).
  - On multi-GPU machines, verify device visibility: `CUDA_VISIBLE_DEVICES=0 python ...`.

- RuntimeError related to NCCL
  - Try setting: `export NCCL_P2P_DISABLE=1` or `export NCCL_IB_DISABLE=1` (cluster environments).
  - Ensure the conda `pytorch-cuda` metapackage is installed (Conda method), which pulls correct CUDA libs.

- macOS note
  - CUDA is unsupported on macOS. Apple Silicon uses MPS backend; this repo’s GPU trainer will detect `device=mps` but environment rollouts remain CPU-side.

## 5) Run full GPU training (example)

```bash
python policy/train/train_by_gpu.py \
  --iterations 60 \
  --population 12 \
  --episodes-per-candidate 5 \
  --sigma 0.1 \
  --lr 0.05 \
  --collision-penalty -1000 \
  --log-csv policy/train/train_log_gpu.csv \
  --plot-png policy/train/reward_curve_gpu.png \
  --clip-step-norm 1.0 \
  --best-update-mode moving_avg \
  --best-update-alpha 0.2 \
  --best-update-gap 0.0 \
  --eval-episodes 5 \
  --seed 0 \
  --map-name map_3x3 \
  --agent-num 3 \
  --speed 1.0 \
  --time-limit 300 \
  --collision bounceback \
  --task-density 1.0 \
  --workers 4
```

このとき、学習を回す回数は
`iterations × population × episodes-per-candidate` で算出できるので、上記例なら `60 × 12 × 5 = 3600` エピソード分のロールアウトを GPU で評価することになります。

If you need a Docker recipe with CUDA runtime + PyTorch CUDA, add a request and we can provide a `Dockerfile` tailored for your driver/toolkit versions.
