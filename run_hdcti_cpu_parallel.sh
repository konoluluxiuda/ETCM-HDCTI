#!/usr/bin/env bash
set -e

# Force TensorFlow onto CPU for this project run. The -p flag in HDCTI.conf
# runs cross-validation folds in parallel, so keep per-process CPU threads modest.
export HDCTI_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES="-1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

python main.py
