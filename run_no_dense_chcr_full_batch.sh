#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|Hctx-P|tcmsuite_herb_only|configs/HDCTI_tcmsuite_pair_stratified_herb_only_no_dense_full.conf"
    "TCM-Suite|Hctx-P+CHCR|tcmsuite_chcr|configs/HDCTI_tcmsuite_pair_stratified_chcr_no_dense_full.conf"
    "TCMSP|Hctx-P|tcmsp_herb_only|configs/HDCTI_tcmsp_pair_stratified_herb_only_no_dense_full.conf"
    "TCMSP|Hctx-P+CHCR|tcmsp_chcr|configs/HDCTI_tcmsp_pair_stratified_chcr_no_dense_full.conf"
    "SymMap2.0|Hctx-P|symmap_herb_only|configs/HDCTI_symmap_pair_stratified_herb_only_no_dense_full.conf"
    "SymMap2.0|Hctx-P+CHCR|symmap_chcr|configs/HDCTI_symmap_pair_stratified_chcr_no_dense_full.conf"
    "ETCM2.0 mention10|Hctx-P|etcm_mention10_herb_only|configs/HDCTI_etcm_mention10_pair_stratified_herb_only_no_dense_full.conf"
    "ETCM2.0 mention10|Hctx-P+CHCR|etcm_mention10_chcr|configs/HDCTI_etcm_mention10_pair_stratified_chcr_no_dense_full.conf"
)

if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'The following jobs will run sequentially:\n'
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset variant slug config_path <<< "$job"
        printf '  %-18s %-12s %s\n' "$dataset" "$variant" "$config_path"
    done
    exit 0
fi

if [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/no_dense_chcr_full_$RUN_TIMESTAMP}"
RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"
ENVIRONMENT_FILE="$RUN_DIR/environment.txt"

mkdir -p "$RUN_DIR"

if [[ ! -f "$RESULTS_TSV" ]]; then
    printf 'dataset\tvariant\tconfig\texit_code\tstatus\tstarted_at\tfinished_at\tduration_seconds\tAUC\tAUPR\tRecall\tPrecision\tF1-score\tlog\tconfig_sha256\n' > "$RESULTS_TSV"
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

write_summary() {
    local temporary_summary="$SUMMARY_MD.tmp"
    {
        printf '# 无稠密注意力 Hctx-P / CHCR 完整五折批处理\n\n'
        printf -- '- 结果目录：`%s`\n' "$RUN_DIR"
        printf -- '- 更新时间：`%s`\n\n' "$(date --iso-8601=seconds)"
        printf '| 数据集 | 模型 | 状态 | AUC | AUPR | Recall | Precision | F1-score | 用时 | 日志 |\n'
        printf '|---|---|---|---:|---:|---:|---:|---:|---:|---|\n'
        tail -n +2 "$RESULTS_TSV" | while IFS=$'\t' read -r \
            dataset variant config_path exit_code status started_at finished_at \
            duration auc aupr recall precision f1 log_path config_hash; do
            printf '| %s | %s | %s | %s | %s | %s | %s | %s | %ss | `%s` |\n' \
                "$dataset" "$variant" "$status" "${auc:--}" "${aupr:--}" \
                "${recall:--}" "${precision:--}" "${f1:--}" "$duration" "$log_path"
        done
        printf '\n原始机器可读结果见 `results.tsv`，环境与 Git 状态见 `environment.txt`。\n'
    } > "$temporary_summary"
    mv "$temporary_summary" "$SUMMARY_MD"
}

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

is_completed() {
    local config_path="$1"
    awk -F '\t' -v config="$config_path" '
        NR > 1 && $3 == config && $5 == "OK" { found = 1 }
        END { exit(found ? 0 : 1) }
    ' "$RESULTS_TSV"
}

failed_jobs=0
job_index=0

for job in "${JOBS[@]}"; do
    job_index=$((job_index + 1))
    IFS='|' read -r dataset variant slug config_path <<< "$job"

    if is_completed "$config_path"; then
        printf '\n[%d/%d] Skipping completed job: %s %s\n' \
            "$job_index" "${#JOBS[@]}" "$dataset" "$variant"
        continue
    fi

    if [[ ! -f "$config_path" ]]; then
        printf 'Missing config: %s\n' "$config_path" >&2
        failed_jobs=$((failed_jobs + 1))
        continue
    fi

    log_path="$RUN_DIR/$(printf '%02d' "$job_index")_${slug}.log"
    started_at="$(date --iso-8601=seconds)"
    start_seconds="$(date +%s)"
    config_hash="$(sha256sum "$config_path" | awk '{print $1}')"

    printf '\n[%d/%d] Starting %s %s\n' \
        "$job_index" "${#JOBS[@]}" "$dataset" "$variant"
    printf 'Config: %s\nLog: %s\n' "$config_path" "$log_path"

    ./run_hdcti.sh "$config_path" 2>&1 | tee "$log_path"
    exit_code=${PIPESTATUS[0]}

    finished_at="$(date --iso-8601=seconds)"
    duration_seconds=$(( $(date +%s) - start_seconds ))
    auc="$(extract_metric "$log_path" 'AUC')"
    aupr="$(extract_metric "$log_path" 'AUPR')"
    recall="$(extract_metric "$log_path" 'Recall')"
    precision="$(extract_metric "$log_path" 'Precision')"
    f1="$(extract_metric "$log_path" 'F1-score')"

    if [[ "$exit_code" -eq 0 && -n "$aupr" ]]; then
        status='OK'
    elif [[ "$exit_code" -eq 0 ]]; then
        status='PARSE_FAILED'
        failed_jobs=$((failed_jobs + 1))
    else
        status='FAILED'
        failed_jobs=$((failed_jobs + 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$variant" "$config_path" "$exit_code" "$status" \
        "$started_at" "$finished_at" "$duration_seconds" "$auc" "$aupr" \
        "$recall" "$precision" "$f1" "$log_path" "$config_hash" >> "$RESULTS_TSV"
    write_summary

    printf '[%d/%d] Finished with status %s in %ss\n' \
        "$job_index" "${#JOBS[@]}" "$status" "$duration_seconds"
done

write_summary

printf '\nBatch results: %s\n' "$SUMMARY_MD"
if [[ "$failed_jobs" -gt 0 ]]; then
    printf '%d job(s) failed or could not be parsed. Re-run with:\n' "$failed_jobs" >&2
    printf '  HDCTI_BATCH_DIR=%q %q\n' "$RUN_DIR" "$0" >&2
    exit 1
fi

printf 'All jobs completed successfully.\n'
