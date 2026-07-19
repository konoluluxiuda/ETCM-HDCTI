#!/usr/bin/env bash
set -uo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPOSITORY_ROOT"

JOBS=(
    "TCM-Suite|HerbOnly|tcmsuite_baseline|configs/HDCTI_tcmsuite_cold_start_no_dense_herb_only_full.conf"
    "TCM-Suite|SDIS|tcmsuite_sdis|configs/HDCTI_tcmsuite_cold_start_sdis_full.conf"
    "TCMSP|HerbOnly|tcmsp_baseline|configs/HDCTI_tcmsp_cold_start_no_dense_herb_only_full.conf"
    "TCMSP|SDIS|tcmsp_sdis|configs/HDCTI_tcmsp_cold_start_sdis_full.conf"
    "SymMap2.0|HerbOnly|symmap_baseline|configs/HDCTI_symmap_cold_start_no_dense_herb_only_full.conf"
    "SymMap2.0|SDIS|symmap_sdis|configs/HDCTI_symmap_cold_start_sdis_full.conf"
    "ETCM2.0 mention10|HerbOnly|etcm_baseline|configs/HDCTI_etcm_mention10_cold_start_no_dense_herb_only_full.conf"
    "ETCM2.0 mention10|SDIS|etcm_sdis|configs/HDCTI_etcm_mention10_cold_start_sdis_full.conf"
)
DATASETS=("TCM-Suite" "TCMSP" "SymMap2.0" "ETCM2.0 mention10")

if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'The following paired five-fold jobs will run sequentially:\n'
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
RUN_DIR="${HDCTI_BATCH_DIR:-$REPOSITORY_ROOT/results/batch_runs/sdis_full_$RUN_TIMESTAMP}"
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

lookup_aupr() {
    local dataset="$1"
    local variant="$2"
    awk -F '\t' -v dataset="$dataset" -v variant="$variant" '
        NR > 1 && $1 == dataset && $2 == variant && $5 == "OK" { value = $8 }
        END { print value }
    ' "$RESULTS_TSV"
}

numeric_metric() {
    local value="$1"
    printf '%s' "${value%%(*}"
}

write_summary() {
    local temporary_summary="$SUMMARY_MD.tmp"
    {
        printf '# 支持度解耦归纳评分四库完整五折\n\n'
        printf -- '- 结果目录：`%s`\n' "$RUN_DIR"
        printf -- '- 更新时间：`%s`\n\n' "$(date --iso-8601=seconds)"
        printf '| 数据集 | 模型 | 状态 | AUC | AUPR | Recall | Precision | F1-score | 用时 | 日志 |\n'
        printf '|---|---|---|---:|---:|---:|---:|---:|---:|---|\n'
        tail -n +2 "$RESULTS_TSV" | while IFS=$'\t' read -r \
            dataset variant config_path exit_code status duration auc aupr \
            recall precision f1 log_path config_hash; do
            printf '| %s | %s | %s | %s | %s | %s | %s | %s | %ss | `%s` |\n' \
                "$dataset" "$variant" "$status" "${auc:--}" "${aupr:--}" \
                "${recall:--}" "${precision:--}" "${f1:--}" "$duration" "$log_path"
        done

        printf '\n## AUPR 配对差值\n\n'
        printf '| 数据集 | HerbOnly | SDIS | SDIS - HerbOnly |\n'
        printf '|---|---:|---:|---:|\n'
        for dataset in "${DATASETS[@]}"; do
            baseline="$(lookup_aupr "$dataset" 'HerbOnly')"
            candidate="$(lookup_aupr "$dataset" 'SDIS')"
            if [[ -n "$baseline" && -n "$candidate" ]]; then
                baseline_numeric="$(numeric_metric "$baseline")"
                candidate_numeric="$(numeric_metric "$candidate")"
                delta="$(awk -v candidate="$candidate_numeric" -v baseline="$baseline_numeric" 'BEGIN { printf "%.6f", candidate - baseline }')"
                printf '| %s | %s | %s | %+0.6f |\n' \
                    "$dataset" "$baseline" "$candidate" "$delta"
            else
                printf '| %s | %s | %s | - |\n' \
                    "$dataset" "${baseline:--}" "${candidate:--}"
            fi
        done
        printf '\n原始机器可读结果见 `results.tsv`，环境与 Git 状态见 `environment.txt`。\n'
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
    printf '%d job(s) failed or could not be parsed. Re-run with:\n' "$failed_jobs" >&2
    printf '  HDCTI_BATCH_DIR=%q %q\n' "$RUN_DIR" "$0" >&2
    exit 1
fi
printf 'All paired five-fold jobs completed successfully.\n'
