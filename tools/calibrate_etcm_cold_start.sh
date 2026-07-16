#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export NVIDIA_TF32_OVERRIDE="0"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
unset TF_GPU_ALLOCATOR

PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS=()
if [[ "${CALIBRATION_DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

"$PYTHON_BIN" tools/calibrate_checkpoint_folds.py \
  --config configs/HDCTI_etcm_mention10_cold_start_no_context.conf \
  --checkpoint "saved_model/2026-07-16 15-09-46/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 15-11-48/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 15-14-06/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 15-16-26/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 15-18-23/hdcti_model.ckpt" \
  --output-dir results/checkpoint_calibration/etcm_mention10_cold_start_no_context \
  "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" tools/calibrate_checkpoint_folds.py \
  --config configs/HDCTI_etcm_mention10_cold_start_herb_only.conf \
  --checkpoint "saved_model/2026-07-16 15-23-38/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 15-27-09/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 15-41-12/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 15-55-40/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 16-09-33/hdcti_model.ckpt" \
  --output-dir results/checkpoint_calibration/etcm_mention10_cold_start_herb_only \
  "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" tools/calibrate_checkpoint_folds.py \
  --config configs/HDCTI_etcm_mention10_cold_start_chcr.conf \
  --checkpoint "saved_model/2026-07-16 17-03-31/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 17-16-33/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 17-27-28/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 17-40-15/hdcti_model.ckpt" \
  --checkpoint "saved_model/2026-07-16 17-52-02/hdcti_model.ckpt" \
  --output-dir results/checkpoint_calibration/etcm_mention10_cold_start_chcr \
  "${EXTRA_ARGS[@]}"
