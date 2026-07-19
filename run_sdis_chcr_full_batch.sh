#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|SDIS+CHCR|tcmsuite_sdis_chcr|configs/HDCTI_tcmsuite_cold_start_sdis_chcr_full.conf"
    "TCMSP|SDIS+CHCR|tcmsp_sdis_chcr|configs/HDCTI_tcmsp_cold_start_sdis_chcr_full.conf"
    "SymMap2.0|SDIS+CHCR|symmap_sdis_chcr|configs/HDCTI_symmap_cold_start_sdis_chcr_full.conf"
    "ETCM2.0 mention10|SDIS+CHCR|etcm_sdis_chcr|configs/HDCTI_etcm_mention10_cold_start_sdis_chcr_full.conf"
)

SDIS_SOURCE_DIR="${HDCTI_SDIS_SOURCE_DIR:-$REPOSITORY_ROOT/results/batch_runs/sdis_full_20260718_212240}"
SDIS_RESULTS="$SDIS_SOURCE_DIR/results.tsv"

SUMMARIZE_ONLY=0
if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'Frozen SDIS + CHCR five-fold jobs (SDIS source: %s):\n' "$SDIS_SOURCE_DIR"
    for job in "${JOBS[@]}"; do
        IFS='|' read -r dataset variant slug config_path <<< "$job"
        printf '  %-18s %-10s %s\n' "$dataset" "$variant" "$config_path"
    done
    exit 0
elif [[ "${1:-}" == "--summarize-only" ]]; then
    SUMMARIZE_ONLY=1
fi

if [[ $# -gt 1 || ( $# -eq 1 && "$SUMMARIZE_ONLY" -ne 1 ) ]]; then
    printf 'Usage: %s [--dry-run|--summarize-only]\n' "$0" >&2
    exit 2
fi
if [[ ! -f "$SDIS_RESULTS" ]]; then
    printf 'Missing frozen SDIS results: %s\n' "$SDIS_RESULTS" >&2
    exit 1
fi

RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/sdis_chcr_full_$RUN_TIMESTAMP}"
RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_MD="$RUN_DIR/summary.md"
ENVIRONMENT_FILE="$RUN_DIR/environment.txt"
mkdir -p "$RUN_DIR"

if [[ ! -f "$RESULTS_TSV" ]]; then
    printf 'dataset\tvariant\tconfig\texit_code\tstatus\tduration_seconds\tAUC\tAUPR\tRecall\tPrecision\tF1-score\tlog\tconfig_sha256\n' > "$RESULTS_TSV"
fi

if [[ ! -f "$ENVIRONMENT_FILE" ]]; then
    {
        printf 'batch_started_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'repository=%s\n' "$REPOSITORY_ROOT"
        printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'sdis_source=%s\n' "$SDIS_SOURCE_DIR"
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

lookup_metric() {
    local results_path="$1"
    local dataset="$2"
    local variant="$3"
    local column="$4"
    awk -F '\t' -v dataset="$dataset" -v variant="$variant" -v column="$column" '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == column) target = i
            }
            next
        }
        $1 == dataset && $2 == variant && $5 == "OK" { value = $target }
        END { print value }
    ' "$results_path"
}

numeric_metric() {
    printf '%s' "${1%%(*}"
}

write_summary() {
    local temporary_summary="$SUMMARY_MD.tmp"
    local gate_stats evaluated nondeclining macro_delta worst_delta verdict
    gate_stats="$(awk -F '\t' '
        function numeric(value) {
            sub(/\(.*/, "", value)
            return value + 0
        }
        FNR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == "AUPR") aupr_column = i
            }
            next
        }
        NR == FNR {
            if ($2 == "SDIS" && $5 == "OK") baseline[$1] = numeric($aupr_column)
            next
        }
        $5 == "OK" && ($1 in baseline) {
            delta = numeric($aupr_column) - baseline[$1]
            total += delta
            evaluated += 1
            if (delta >= 0) nondeclining += 1
            if (evaluated == 1 || delta < worst) worst = delta
        }
        END {
            if (evaluated > 0) {
                printf "%d\t%d\t%.6f\t%.6f", evaluated, nondeclining, total / evaluated, worst
            } else {
                printf "0\t0\t0\t0"
            }
        }
    ' "$SDIS_RESULTS" "$RESULTS_TSV")"
    IFS=$'\t' read -r evaluated nondeclining macro_delta worst_delta <<< "$gate_stats"
    verdict='PENDING'
    if [[ "$evaluated" -eq 4 ]]; then
        verdict="$(awk -v nondeclining="$nondeclining" -v macro="$macro_delta" -v worst="$worst_delta" 'BEGIN {
            print (nondeclining >= 3 && macro >= 0 && worst >= -0.005) ? "GO" : "NO-GO"
        }')"
    fi
    {
        printf '# 冻结 SDIS + CHCR 四库 compound cold-start 五折\n\n'
        printf -- '- 结果目录：`%s`\n' "$RUN_DIR"
        printf -- '- SDIS 基准：`%s`\n' "$SDIS_SOURCE_DIR"
        printf -- '- 更新时间：`%s`\n\n' "$(date --iso-8601=seconds)"
        printf '| 数据集 | SDIS AUPR | SDIS+CHCR AUPR | 差值 | AUC | Recall | Precision | F1-score | 状态 |\n'
        printf '|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
        tail -n +2 "$RESULTS_TSV" | while IFS= read -r result_line; do
            IFS=$'\x1f' read -r dataset variant config_path exit_code status \
                duration auc aupr recall precision f1 log_path config_hash <<< \
                "${result_line//$'\t'/$'\x1f'}"
            baseline="$(lookup_metric "$SDIS_RESULTS" "$dataset" 'SDIS' 'AUPR')"
            delta='-'
            if [[ -n "$baseline" && -n "$aupr" ]]; then
                delta="$(awk -v candidate="$(numeric_metric "$aupr")" -v base="$(numeric_metric "$baseline")" 'BEGIN { printf "%+.6f", candidate - base }')"
            fi
            printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
                "$dataset" "${baseline:--}" "${aupr:--}" "$delta" \
                "${auc:--}" "${recall:--}" "${precision:--}" "${f1:--}" "$status"
        done
        printf '\n## 预注册判定\n\n'
        printf -- '- 完成数据库：`%s/4`\n' "$evaluated"
        printf -- '- AUPR 不下降：`%s/4`\n' "$nondeclining"
        printf -- '- Macro AUPR 增量：`%+.6f`\n' "$macro_delta"
        printf -- '- 最差单库增量：`%+.6f`\n' "$worst_delta"
        printf -- '- 结论：**%s**\n' "$verdict"
        printf '\n本实验只检验冻结 CHCR 与 SDIS 的互补性，不修改 gate、donor、margin、loss weight 或数据集特定参数。原始结果见 `results.tsv`。\n'
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

remove_previous_result() {
    local config_path="$1"
    local temporary_results="$RESULTS_TSV.tmp"
    awk -F '\t' -v config="$config_path" '
        NR == 1 || $3 != config
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
    IFS='|' read -r dataset variant slug config_path <<< "$job"
    if is_completed "$config_path"; then
        printf '\n[%d/%d] Skipping completed job: %s\n' "$job_index" "${#JOBS[@]}" "$dataset"
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
    printf '\n[%d/%d] Starting %s %s\nConfig: %s\nLog: %s\n' \
        "$job_index" "${#JOBS[@]}" "$dataset" "$variant" "$config_path" "$log_path"

    ./run_hdcti.sh "$config_path" 2>&1 | tee "$log_path"
    exit_code=${PIPESTATUS[0]}
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dataset" "$variant" "$config_path" "$exit_code" "$status" \
        "$duration_seconds" "$auc" "$aupr" "$recall" "$precision" "$f1" \
        "$log_path" "$config_hash" >> "$RESULTS_TSV"
    write_summary
done

write_summary
printf '\nBatch results: %s\n' "$SUMMARY_MD"
if [[ "$failed_jobs" -gt 0 ]]; then
    printf '%d job(s) failed or could not be parsed.\n' "$failed_jobs" >&2
    exit 1
fi
printf 'All frozen SDIS + CHCR jobs completed successfully.\n'
