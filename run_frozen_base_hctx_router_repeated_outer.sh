#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

exec "${PYTHON_BIN:-python}" \
  tools/run_frozen_base_hctx_router_repeated_outer.py "$@"
