#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE="0"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
unset TF_GPU_ALLOCATOR

exec "${PYTHON_BIN:-python}" tools/run_sdis_checkpoint_calibration.py "$@"
