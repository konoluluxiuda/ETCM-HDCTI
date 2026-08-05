#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|tcmsuite|baseline|configs/HDCTI_tcmsuite_schpt_baseline_gate1.conf|976c3398489caff8813e5d2ae976e10d92b764e0221eb97a5ea699037ba911d8"
    "TCM-Suite|tcmsuite|candidate|configs/HDCTI_tcmsuite_schpt_gate1.conf|8a6d8c229dbd075cad4e75fe7db1a08c24a094986ffcd029f4bae6619c6bb6d5"
    "TCMSP|tcmsp|baseline|configs/HDCTI_tcmsp_schpt_baseline_gate1.conf|eb322c06e9349e138157caf221a1532c9e154058ea4efbd5b1b08fcd73f3e03b"
    "TCMSP|tcmsp|candidate|configs/HDCTI_tcmsp_schpt_gate1.conf|7026e922a7e34d26c2f893ab904b03dcbc1a6b4e73a2a693b361cd1c4cc7efa1"
    "SymMap2.0|symmap|baseline|configs/HDCTI_symmap_schpt_baseline_gate1.conf|bc891aefd3d438fe750c7c3b11fc7b41b11d21b5747592102ddb289e293b8a7e"
    "SymMap2.0|symmap|candidate|configs/HDCTI_symmap_schpt_gate1.conf|19af2cc8867557e931914fedc9934881891d4f82c6a693da630a3412e2435f72"
    "ETCM2.0-mention10|etcm_mention10|baseline|configs/HDCTI_etcm_mention10_schpt_baseline_pilot.conf|7349dc86c0c5403a74d7110b8322e341ef61aeba9681d3e3ea530c9d1f99fd56"
    "ETCM2.0-mention10|etcm_mention10|candidate|configs/HDCTI_etcm_mention10_schpt_pilot.conf|eb84d02941627be019c16d7874474e01f7e5ac978d9b8ea1d269aca33d17e412"
)

verify_jobs() {
    local job dataset slug variant config_path expected_hash actual_hash
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset slug variant config_path expected_hash <<< "$job"
        actual_hash="$(sha256sum "$config_path" | awk '{print $1}')"
        if [[ "$actual_hash" != "$expected_hash" ]]; then
            printf 'Frozen config hash mismatch: %s\n' "$config_path" >&2
            exit 2
        fi
    done
}

verify_jobs
if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'SCHPT frozen four-dataset Gate 1 jobs:\n'
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset slug variant config_path expected_hash <<< "$job"
        printf '  %-20s %-9s %s\n' "$dataset" "$variant" "$config_path"
    done
    exit 0
fi
RESUME_DIR=""
if [[ "${1:-}" == "--resume" && -n "${2:-}" && $# -eq 2 ]]; then
    RESUME_DIR="$2"
elif [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--dry-run | --resume RUN_DIR]\n' "$0" >&2
    exit 2
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
if [[ -n "$RESUME_DIR" ]]; then
    RUN_DIR="$(realpath "$RESUME_DIR")"
else
    RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/schpt_gate1_$RUN_TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

job_index=0
for job in "${JOBS[@]}"; do
    job_index=$((job_index + 1))
    IFS='|' read -r dataset slug variant config_path expected_hash <<< "$job"
    log_path="$RUN_DIR/$(printf '%02d' "$job_index")_${slug}_${variant}.log"
    if [[ -n "$RESUME_DIR" && -f "$log_path" ]] \
            && grep -q '^Validation-AUPR:' "$log_path"; then
        printf '\n[%d/%d] Reusing completed %s %s\n' \
            "$job_index" "${#JOBS[@]}" "$dataset" "$variant"
        continue
    fi
    printf '\n[%d/%d] Starting %s %s\n' \
        "$job_index" "${#JOBS[@]}" "$dataset" "$variant"
    ./run_hdcti.sh "$config_path" 2>&1 | tee "$log_path"
    status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 ]]; then
        printf '%s %s failed with status %d.\n' \
            "$dataset" "$variant" "$status" >&2
        exit "$status"
    fi
done

python tools/summarize_schpt_gate1.py --run-dir "$RUN_DIR"
