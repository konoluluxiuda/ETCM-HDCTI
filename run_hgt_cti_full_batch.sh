#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|tcmsuite|configs/HGTCTI_tcmsuite_pair_stratified_full.conf"
    "TCMSP|tcmsp|configs/HGTCTI_tcmsp_pair_stratified_full.conf"
    "SymMap2.0|symmap|configs/HGTCTI_symmap_pair_stratified_full.conf"
    "ETCM2.0 mention10|etcm_mention10|configs/HGTCTI_etcm_mention10_pair_stratified_full.conf"
)

SUMMARIZE_ONLY=0
if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'Frozen HGT-CTI five-fold jobs will run sequentially:\n'
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset slug config_path <<< "$job"
        printf '  %-18s %s\n' "$dataset" "$config_path"
    done
    exit 0
elif [[ "${1:-}" == "--summarize-only" ]]; then
    SUMMARIZE_ONLY=1
fi

if [[ $# -gt 1 || ( $# -eq 1 && "$SUMMARIZE_ONLY" -ne 1 ) ]]; then
    printf 'Usage: %s [--dry-run|--summarize-only]\n' "$0" >&2
    exit 2
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/hgt_cti_full_$RUN_TIMESTAMP}"
RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"
ENVIRONMENT_FILE="$RUN_DIR/environment.txt"
mkdir -p "$RUN_DIR"

if [[ ! -f "$RESULTS_TSV" ]]; then
    printf 'dataset\tconfig\texit_code\tstatus\tduration_seconds\tAUC\tAUPR\tRecall\tPrecision\tF1-score\tlog\tconfig_sha256\n' > "$RESULTS_TSV"
fi

if [[ ! -f "$ENVIRONMENT_FILE" ]]; then
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
fi

extract_metric() {
    local log_path="$1"
    local metric="$2"
    awk -v metric="$metric" '
        /^The result of 5-fold cross validation:$/ { in_summary = 1; next }
        in_summary && index($0, metric ":") == 1 {
            value = substr($0, length(metric) + 2)
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
        printf '# HGT-CTI 四库正式五折结果\n\n'
        printf -- '- 结果目录：`%s`\n' "$RUN_DIR"
        printf -- '- 更新时间：`%s`\n' "$(date --iso-8601=seconds)"
        printf -- '- 协议：Strict、pair-stratified、seed 2026、固定五折、AUPR 早停、外层测试。\n'
        printf -- '- 模型：两层双头 HGT、六类关系、每关系/目标节点最多 64 个确定性入邻居。\n'
        printf -- '- 原则：四库共享同一模型设置，不进行数据库特定调参。\n\n'
        printf '| 数据集 | 状态 | AUC | AUPR | Recall | Precision | F1-score | 用时 | 日志 |\n'
        printf '|---|---|---:|---:|---:|---:|---:|---:|---|\n'
        tail -n +2 "$RESULTS_TSV" | while IFS=$'\t' read -r \
            dataset config_path exit_code status duration auc aupr \
            recall precision f1 log_path config_hash; do
            printf '| %s | %s | %s | %s | %s | %s | %s | %ss | `%s` |\n' \
                "$dataset" "$status" "${auc:--}" "${aupr:--}" \
                "${recall:--}" "${precision:--}" "${f1:--}" \
                "$duration" "$log_path"
        done
        printf '\n该表报告冻结协议下的描述性结果，不据此声称统计显著性。'
        printf '原始机器可读结果见 `results.tsv`。\n'
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

remove_previous_result() {
    local config_path="$1"
    local temporary_results="$RESULTS_TSV.tmp"
    awk -F '\t' -v config="$config_path" '
        NR == 1 || $2 != config
    ' "$RESULTS_TSV" > "$temporary_results"
    mv "$temporary_results" "$RESULTS_TSV"
}

if [[ "$SUMMARIZE_ONLY" -eq 1 ]]; then
    write_summary
    printf 'Summary refreshed: %s\n' "$SUMMARY_MD"
    exit 0
fi

failed_jobs=0
job_index=0
for job in "${JOBS[@]}"; do
    job_index=$((job_index + 1))
    IFS='|' read -r dataset slug config_path <<< "$job"

    if is_completed "$config_path"; then
        printf '\n[%d/%d] Skipping completed job: %s\n' \
            "$job_index" "${#JOBS[@]}" "$dataset"
        continue
    fi
    if [[ ! -f "$config_path" ]]; then
        printf 'Missing config: %s\n' "$config_path" >&2
        failed_jobs=$((failed_jobs + 1))
        continue
    fi
    remove_previous_result "$config_path"

    log_path="$RUN_DIR/$(printf '%02d' "$job_index")_${slug}.log"
    start_seconds="$(date +%s)"
    config_hash="$(sha256sum "$config_path" | awk '{print $1}')"
    printf '\n[%d/%d] Starting %s\nConfig: %s\nLog: %s\n' \
        "$job_index" "${#JOBS[@]}" "$dataset" "$config_path" "$log_path"
    ./run_hdcti.sh "$config_path" 2>&1 | tee "$log_path"
    exit_code=${PIPESTATUS[0]}
    duration_seconds=$(( $(date +%s) - start_seconds ))
    auc="$(extract_metric "$log_path" 'AUC')"
    aupr="$(extract_metric "$log_path" 'AUPR')"
    recall="$(extract_metric "$log_path" 'Recall')"
    precision="$(extract_metric "$log_path" 'Precision')"
    f1="$(extract_metric "$log_path" 'F1-score')"

    if [[ "$exit_code" -eq 0 && -n "$auc" && -n "$aupr" && -n "$recall" \
        && -n "$precision" && -n "$f1" ]]; then
        status='OK'
    elif [[ "$exit_code" -eq 0 ]]; then
        status='PARSE_FAILED'
        failed_jobs=$((failed_jobs + 1))
    else
        status='FAILED'
        failed_jobs=$((failed_jobs + 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$config_path" "$exit_code" "$status" \
        "$duration_seconds" "$auc" "$aupr" "$recall" "$precision" "$f1" \
        "$log_path" "$config_hash" >> "$RESULTS_TSV"
    write_summary
done

write_summary
printf '\nBatch results: %s\n' "$SUMMARY_MD"
if [[ "$failed_jobs" -gt 0 ]]; then
    printf '%d job(s) failed or could not be parsed. Resume with:\n' "$failed_jobs" >&2
    printf '  HDCTI_BATCH_DIR=%q %q\n' "$RUN_DIR" "$0" >&2
    exit 1
fi
printf 'All frozen HGT-CTI jobs completed successfully.\n'
