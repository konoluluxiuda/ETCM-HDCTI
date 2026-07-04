#!/usr/bin/env bash
set -e

# Run the same HDCTI.conf experiment on CPU only.
# This changes device placement, not model hyperparameters.
export HDCTI_FORCE_CPU=1
export CUDA_VISIBLE_DEVICES="-1"

python main.py
