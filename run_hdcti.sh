#!/usr/bin/env bash
set -e

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE="0"
# Enable this only when TensorFlow reports BFC allocator fragmentation OOM.
# It can segfault with TensorFlow 2.6.x on newer GPUs/WSL.
# export TF_GPU_ALLOCATOR="cuda_malloc_async"
python main.py
