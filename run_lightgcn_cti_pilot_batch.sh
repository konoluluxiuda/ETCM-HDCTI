#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|tcmsuite|configs/LightGCNCTI_tcmsuite_pair_stratified_pilot.conf"
    "TCMSP|tcmsp|configs/LightGCNCTI_tcmsp_pair_stratified_pilot.conf"
    "SymMap2.0|symmap|configs/LightGCNCTI_symmap_pair_stratified_pilot.conf"
    "ETCM2.0 mention10|etcm_mention10|configs/LightGCNCTI_etcm_mention10_pair_stratified_pilot.conf"
)

if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'Frozen LightGCN-CTI single-fold pilots:\n'
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
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/lightgcn_cti_pilot_$RUN_TIMESTAMP}"
RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"
ENVIRONMENT_FILE="$RUN_DIR/environment.txt"

mkdir -p "$RUN_DIR"

if [[ ! -f "$RESULTS_TSV" ]]; then
    printf 'dataset\tconfig\texit_code\tstatus\tduration_seconds\tValidation-AUPR\tlog\tconfig_sha256\n' > "$RESULTS_TSV"
fi

if [[ ! -f "$ENVIRONMENT_FILE" ]]; then
    {
        printf 'batch_started_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'python=%s\n' "$(command -v python || printf unknown)"
        python -c 'import platform; print("python_version=" + platform.python_version())' 2>/dev/null || true
        python -c 'import tensorflow as tf; print("tensorflow_version=" + tf.__version__)' 2>/dev/null || true
    } > "$ENVIRONMENT_FILE"
fi

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
    local temporary_summary="$SUMMARY_MD.tmp"
    {
        printf '# LightGCN-CTI 四库单折 Pilot\n\n'
        printf -- '- 模型角色：same-input pair-only structural baseline\n'
        printf -- '- 图：当前 fold inner-train C-P 正边二部图\n'
        printf -- '- 传播：三层 LightGCN，无特征变换和非线性，均匀层聚合\n'
        printf -- '- 目标：固定正负 pair 上的 BCE adaptation\n'
        printf -- '- 协议：Strict 固定随机折，seed=2026，单折内层验证\n'
        printf -- '- 更新时间：`%s`\n\n' "$(date --iso-8601=seconds)"
        printf '| 数据集 | 状态 | Validation AUPR | 用时 | 日志 |\n'
        printf '|---|---|---:|---:|---|\n'
        tail -n +2 "$RESULTS_TSV" | while IFS=$'\t' read -r \
            dataset config_path exit_code status duration aupr log_path config_hash; do
            printf '| %s | %s | %s | %ss | `%s` |\n' \
                "$dataset" "$status" "${aupr:--}" "$duration" "$log_path"
        done
        printf '\n该结果只用于检查适配基线能否稳定训练，不进入最终论文主表。\n'
    } > "$temporary_summary"
    mv "$temporary_summary" "$SUMMARY_MD"
}

is_completed() {
    local config_path="$1"
    awk -F '\t' -v config="$config_path" '
        NR > 1 && $2 == config && $4 == "OK" { found = 1 }
        END { exit(found ? 0 : 1) }
    ' "$RESULTS_TSV"
}

failed_jobs=0
job_index=0
for job in "${JOBS[@]}"; do
    job_index=$((job_index + 1))
    IFS='|' read -r dataset slug config_path <<< "$job"

    if is_completed "$config_path"; then
        printf '[%d/%d] Skipping completed job: %s\n' \
            "$job_index" "${#JOBS[@]}" "$dataset"
        continue
    fi

    log_path="$RUN_DIR/$(printf '%02d' "$job_index")_${slug}.log"
    start_seconds="$(date +%s)"
    config_hash="$(sha256sum "$config_path" | awk '{print $1}')"
    printf '\n[%d/%d] Starting %s\n' \
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
    printf '%d job(s) failed or could not be parsed.\n' "$failed_jobs" >&2
    exit 1
fi
printf 'All LightGCN-CTI pilot jobs completed successfully.\n'
