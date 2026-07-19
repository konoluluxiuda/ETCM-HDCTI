#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|SD-only|tcmsuite_sd|configs/HDCTI_tcmsuite_cold_start_sdis_pilot.conf"
    "TCM-Suite|SD+SE|tcmsuite_sdse|configs/HDCTI_tcmsuite_cold_start_sdis_self_excluded_pilot.conf"
    "TCMSP|SD-only|tcmsp_sd|configs/HDCTI_tcmsp_cold_start_sdis_pilot.conf"
    "TCMSP|SD+SE|tcmsp_sdse|configs/HDCTI_tcmsp_cold_start_sdis_self_excluded_pilot.conf"
    "SymMap2.0|SD-only|symmap_sd|configs/HDCTI_symmap_cold_start_sdis_pilot.conf"
    "SymMap2.0|SD+SE|symmap_sdse|configs/HDCTI_symmap_cold_start_sdis_self_excluded_pilot.conf"
    "ETCM2.0 mention10|SD-only|etcm_sd|configs/HDCTI_etcm_mention10_cold_start_sdis_pilot.conf"
    "ETCM2.0 mention10|SD+SE|etcm_sdse|configs/HDCTI_etcm_mention10_cold_start_sdis_self_excluded_pilot.conf"
)

if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'The following inner-validation pilot jobs will run sequentially:\n'
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset variant slug config_path <<< "$job"
        printf '  %-18s %-8s %s\n' "$dataset" "$variant" "$config_path"
    done
    exit 0
fi

if [[ $# -gt 0 ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/sdis_pilot_$RUN_TIMESTAMP}"
RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"
ENVIRONMENT_FILE="$RUN_DIR/environment.txt"

mkdir -p "$RUN_DIR"

if [[ ! -f "$RESULTS_TSV" ]]; then
    printf 'dataset\tvariant\tconfig\texit_code\tstatus\tduration_seconds\tValidation-AUPR\tlog\tconfig_sha256\n' > "$RESULTS_TSV"
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
        printf '# 支持度解耦归纳评分单折 Pilot\n\n'
        printf -- '- 结果目录：`%s`\n' "$RUN_DIR"
        printf -- '- 更新时间：`%s`\n\n' "$(date --iso-8601=seconds)"
        printf '| 数据集 | 变体 | 状态 | Validation AUPR | 用时 | 日志 |\n'
        printf '|---|---|---|---:|---:|---|\n'
        tail -n +2 "$RESULTS_TSV" | while IFS=$'\t' read -r \
            dataset variant config_path exit_code status duration aupr log_path config_hash; do
            printf '| %s | %s | %s | %s | %ss | `%s` |\n' \
                "$dataset" "$variant" "$status" "${aupr:--}" "$duration" "$log_path"
        done
        printf '\n`SD-only` 仅关闭零 C-P 支持的基础 ID 分；`SD+SE` 进一步使用逐层直接自排除 H-C 上下文。\n'
        printf '机器可读结果见 `results.tsv`，运行环境见 `environment.txt`。\n'
    } > "$temporary_summary"
    mv "$temporary_summary" "$SUMMARY_MD"
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
    start_seconds="$(date +%s)"
    config_hash="$(sha256sum "$config_path" | awk '{print $1}')"

    printf '\n[%d/%d] Starting %s %s\n' \
        "$job_index" "${#JOBS[@]}" "$dataset" "$variant"
    printf 'Config: %s\nLog: %s\n' "$config_path" "$log_path"

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

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$variant" "$config_path" "$exit_code" "$status" \
        "$duration_seconds" "$aupr" "$log_path" "$config_hash" >> "$RESULTS_TSV"
    write_summary
done

write_summary
printf '\nBatch results: %s\n' "$SUMMARY_MD"
if [[ "$failed_jobs" -gt 0 ]]; then
    printf '%d job(s) failed or could not be parsed.\n' "$failed_jobs" >&2
    exit 1
fi
printf 'All pilot jobs completed successfully.\n'
