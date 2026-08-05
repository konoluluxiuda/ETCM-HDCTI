#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

BASELINE_CONFIG="configs/HDCTI_etcm_mention10_schpt_baseline_pilot.conf"
CANDIDATE_CONFIG="configs/HDCTI_etcm_mention10_schpt_pilot.conf"
EXPECTED_BASELINE_HASH="7349dc86c0c5403a74d7110b8322e341ef61aeba9681d3e3ea530c9d1f99fd56"
EXPECTED_CANDIDATE_HASH="eb84d02941627be019c16d7874474e01f7e5ac978d9b8ea1d269aca33d17e412"

verify_config_hash() {
    local config_path="$1"
    local expected_hash="$2"
    local actual_hash
    actual_hash="$(sha256sum "$config_path" | awk '{print $1}')"
    if [[ "$actual_hash" != "$expected_hash" ]]; then
        printf 'Frozen config hash mismatch: %s\nexpected=%s\nactual=%s\n' \
            "$config_path" "$expected_hash" "$actual_hash" >&2
        exit 2
    fi
}

verify_config_hash "$BASELINE_CONFIG" "$EXPECTED_BASELINE_HASH"
verify_config_hash "$CANDIDATE_CONFIG" "$EXPECTED_CANDIDATE_HASH"

if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'SCHPT paired inner-validation Pilot:\n'
    printf '  Baseline:  %s\n' "$BASELINE_CONFIG"
    printf '  Candidate: %s\n' "$CANDIDATE_CONFIG"
    printf '  Gate: delta AUPR >= 0.003, coverage >= 0.30, nonzero scale\n'
    exit 0
fi
if [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/schpt_pilot_$RUN_TIMESTAMP}"
BASELINE_LOG="$RUN_DIR/01_baseline.log"
CANDIDATE_LOG="$RUN_DIR/02_schpt.log"
mkdir -p "$RUN_DIR"

printf '[1/2] Starting paired Hctx-P + SDIS baseline\n'
./run_hdcti.sh "$BASELINE_CONFIG" 2>&1 | tee "$BASELINE_LOG"
baseline_status=${PIPESTATUS[0]}
if [[ "$baseline_status" -ne 0 ]]; then
    printf 'Baseline failed with status %d.\n' "$baseline_status" >&2
    exit "$baseline_status"
fi

printf '\n[2/2] Starting SCHPT compound-PageRank replacement\n'
./run_hdcti.sh "$CANDIDATE_CONFIG" 2>&1 | tee "$CANDIDATE_LOG"
candidate_status=${PIPESTATUS[0]}
if [[ "$candidate_status" -ne 0 ]]; then
    printf 'SCHPT candidate failed with status %d.\n' "$candidate_status" >&2
    exit "$candidate_status"
fi

python tools/summarize_schpt_pilot.py \
    --baseline-log "$BASELINE_LOG" \
    --candidate-log "$CANDIDATE_LOG" \
    --output-dir "$RUN_DIR"
