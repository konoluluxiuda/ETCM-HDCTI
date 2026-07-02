#!/usr/bin/env bash
set -e

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
python main.py
