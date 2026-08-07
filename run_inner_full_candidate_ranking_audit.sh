#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_DIR"
exec "$PYTHON_BIN" tools/run_inner_full_candidate_ranking_audit.py "$@"
