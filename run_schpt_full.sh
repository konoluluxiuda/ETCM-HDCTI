#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|tcmsuite|baseline|configs/HDCTI_tcmsuite_schpt_baseline_full.conf|80ffe66a0793735b9f4c1f34e399ad580b01971d5ebebbd97e9bba3545b56aae"
    "TCM-Suite|tcmsuite|candidate|configs/HDCTI_tcmsuite_schpt_full.conf|06a3a689ea28ba0a6753142c88b36458165bfb9b9420022dbf1f54b8bc8664db"
    "TCMSP|tcmsp|baseline|configs/HDCTI_tcmsp_schpt_baseline_full.conf|4b21b09907295d3ec8a63ad3d6ecdff28b011d3543a36201c8577723d456d4b1"
    "TCMSP|tcmsp|candidate|configs/HDCTI_tcmsp_schpt_full.conf|5fc236d42d8b85f1e78374c516124ff77a56accd9055ee55251b131d42121dbb"
    "SymMap2.0|symmap|baseline|configs/HDCTI_symmap_schpt_baseline_full.conf|bb5a924c511c6354f947ae9798e41be9c665f1d7d699f6ad5a7a475490a335ed"
    "SymMap2.0|symmap|candidate|configs/HDCTI_symmap_schpt_full.conf|e129b0efdcfd6810a4d14075fef919008ed71a74fedd182d4a070fd900d7ef9f"
    "ETCM2.0-mention10|etcm_mention10|baseline|configs/HDCTI_etcm_mention10_schpt_baseline_full.conf|88199fa91be79a9f421a62bd009d869f4bf4c0781f5a1beb9ffb7ea6a16b75ce"
    "ETCM2.0-mention10|etcm_mention10|candidate|configs/HDCTI_etcm_mention10_schpt_full.conf|533c77c55aac3335c8b928a77f912b4ec64013e09f134f10ef68d8dd0c2ad859"
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
if [[ "${1:-}" == "--dry-run" && $# -eq 1 ]]; then
    printf 'SCHPT frozen four-dataset five-fold jobs:\n'
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset slug variant config_path expected_hash <<< "$job"
        printf '  %-20s %-9s %s\n' "$dataset" "$variant" "$config_path"
    done
    exit 0
fi

MODE="run"
RUN_DIR=""
if [[ "${1:-}" == "--resume" && -n "${2:-}" && $# -eq 2 ]]; then
    MODE="resume"
    RUN_DIR="$(realpath "$2")"
elif [[ "${1:-}" == "--summarize-only" && -n "${2:-}" && $# -eq 2 ]]; then
    MODE="summarize"
    RUN_DIR="$(realpath "$2")"
elif [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--dry-run | --resume RUN_DIR | --summarize-only RUN_DIR]\n' "$0" >&2
    exit 2
fi

if [[ "$MODE" == "run" ]]; then
    RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
    RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/schpt_full_$RUN_TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

if [[ "$MODE" == "summarize" ]]; then
    python tools/summarize_schpt_full.py --run-dir "$RUN_DIR"
    exit $?
fi

if [[ ! -f "$RUN_DIR/environment.txt" ]]; then
    {
        printf 'batch_started_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'repository=%s\n' "$REPOSITORY_ROOT"
        printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'python=%s\n' "$(command -v python || printf unknown)"
        python -c 'import platform; print("python_version=" + platform.python_version())' 2>/dev/null || true
        python -c 'import tensorflow as tf; print("tensorflow_version=" + tf.__version__)' 2>/dev/null || true
        printf '\ngit_status:\n'
        git status --short 2>/dev/null || true
    } > "$RUN_DIR/environment.txt"
fi

job_index=0
for job in "${JOBS[@]}"; do
    job_index=$((job_index + 1))
    IFS='|' read -r dataset slug variant config_path expected_hash <<< "$job"
    log_path="$RUN_DIR/$(printf '%02d' "$job_index")_${slug}_${variant}.log"
    if [[ "$MODE" == "resume" && -f "$log_path" ]] \
            && grep -q '^The result of 5-fold cross validation:$' "$log_path"; then
        printf '\n[%d/%d] Reusing completed %s %s\n' \
            "$job_index" "${#JOBS[@]}" "$dataset" "$variant"
        continue
    fi
    printf '\n[%d/%d] Starting %s %s\nConfig: %s\nLog: %s\n' \
        "$job_index" "${#JOBS[@]}" "$dataset" "$variant" \
        "$config_path" "$log_path"
    ./run_hdcti.sh "$config_path" 2>&1 | tee "$log_path"
    status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 ]]; then
        printf '%s %s failed with status %d. Resume with:\n' \
            "$dataset" "$variant" "$status" >&2
        printf '  %q --resume %q\n' "$0" "$RUN_DIR" >&2
        exit "$status"
    fi
done

python tools/summarize_schpt_full.py --run-dir "$RUN_DIR"
exit $?
