#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|tcmsuite|configs/HDCTI_tcmsuite_pair_stratified_hplga_pilot.conf"
    "TCMSP|tcmsp|configs/HDCTI_tcmsp_pair_stratified_hplga_pilot.conf"
    "SymMap2.0|symmap|configs/HDCTI_symmap_pair_stratified_hplga_pilot.conf"
    "ETCM2.0 mention10|etcm_mention10|configs/HDCTI_etcm_mention10_pair_stratified_hplga_pilot.conf"
)

if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'The following frozen HPLGA validation-only pilots will run:\n'
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset slug config_path <<< "$job"
        printf '  %-18s %s\n' "$dataset" "$config_path"
    done
    exit 0
fi

if [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/hplga_pilot_$RUN_TIMESTAMP}"
RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"
ENVIRONMENT_FILE="$RUN_DIR/environment.txt"

mkdir -p "$RUN_DIR"
printf 'dataset\tconfig\texit_code\tstatus\tduration_seconds\tValidation-AUPR\tlog\tconfig_sha256\n' > "$RESULTS_TSV"

{
    printf 'batch_started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'repository=%s\n' "$REPOSITORY_ROOT"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    printf 'python=%s\n' "$(command -v python || printf unknown)"
    python -c 'import platform; print("python_version=" + platform.python_version())' 2>/dev/null || true
    python -c 'import tensorflow as tf; print("tensorflow_version=" + tf.__version__)' 2>/dev/null || true
    printf '\ngit_status:\n'
    git status --short 2>/dev/null || true
} > "$ENVIRONMENT_FILE"

extract_validation_aupr() {
    local log_path="$1"
    awk '
        /^Validation-AUPR:/ {
            value = substr($0, length("Validation-AUPR:") + 1)
            sub(/\(.*/, "", value)
        }
        END {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            print value
        }
    ' "$log_path"
}

write_summary() {
    {
        printf '# HPLGA Four-Dataset Pilot\n\n'
        printf -- '- Result directory: `%s`\n' "$RUN_DIR"
        printf -- '- Protocol: fixed fold 1, inner-validation AUPR, outer test disabled\n'
        printf -- '- Reference: frozen no-dense Hctx-P pilots; no baseline retraining\n\n'
        printf '| Dataset | Status | Validation AUPR | Seconds | Log |\n'
        printf '|---|---|---:|---:|---|\n'
        tail -n +2 "$RESULTS_TSV" | while IFS=$'\t' read -r \
            dataset config_path exit_code status duration aupr log_path config_hash; do
            printf '| %s | %s | %s | %ss | `%s` |\n' \
                "$dataset" "$status" "${aupr:--}" "$duration" "$log_path"
        done
        printf '\nProceed to five folds only when the preregistered Gate 1 criteria pass.\n'
    } > "$SUMMARY_MD"
}

failed_jobs=0
job_index=0
for job in "${JOBS[@]}"; do
    job_index=$((job_index + 1))
    IFS='|' read -r dataset slug config_path <<< "$job"
    log_path="$RUN_DIR/$(printf '%02d' "$job_index")_${slug}.log"
    config_hash="$(sha256sum "$config_path" | awk '{print $1}')"
    start_seconds="$(date +%s)"

    printf '\n[%d/%d] Starting %s HPLGA Pilot\n' \
        "$job_index" "${#JOBS[@]}" "$dataset"
    ./run_hdcti.sh "$config_path" 2>&1 | tee "$log_path"
    exit_code=${PIPESTATUS[0]}
    duration_seconds=$(( $(date +%s) - start_seconds ))
    aupr="$(extract_validation_aupr "$log_path")"

    if [[ "$exit_code" -eq 0 && -n "$aupr" ]]; then
        status='OK'
    elif [[ "$exit_code" -eq 0 ]]; then
        status='PARSE_FAILED'
        failed_jobs=$((failed_jobs + 1))
    else
        status='FAILED'
        failed_jobs=$((failed_jobs + 1))
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$config_path" "$exit_code" "$status" \
        "$duration_seconds" "$aupr" "$log_path" "$config_hash" >> "$RESULTS_TSV"
    write_summary
done

write_summary
printf '\nBatch results: %s\n' "$SUMMARY_MD"
if [[ "$failed_jobs" -gt 0 ]]; then
    printf '%d HPLGA job(s) failed or could not be parsed.\n' "$failed_jobs" >&2
    exit 1
fi
printf 'All HPLGA pilots completed successfully.\n'
